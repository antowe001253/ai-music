"""
🎤 Modal HifiGAN Vocoder Service
Integrates with existing Diff-SVC pipeline to provide high-quality vocoding
"""

import modal
import torch
import torchaudio
import numpy as np
from pathlib import Path
import tempfile
import logging
import json
import base64
from typing import Dict, Any, Optional
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Modal image with HifiGAN dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
        "huggingface-hub>=0.16.0",
        "transformers>=4.30.0",
        "git+https://github.com/jik876/hifi-gan.git"
    ])
    .run_commands([
        "apt-get update",
        "apt-get install -y wget git ffmpeg"
    ])
)

app = modal.App("hifigan-vocoder", image=image)

# Create persistent volume for model storage
volume = modal.Volume.from_name("hifigan-models", create_if_missing=True)
@app.function(
    gpu="A10G",
    volumes={"/models": volume},
    timeout=1800,  # 30 minutes
    memory=8192,   # 8GB RAM
)
def download_hifigan_models():
    """Download and cache HifiGAN models on Modal"""
    import os
    import wget
    
    model_dir = Path("/models/hifigan")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Universal HifiGAN model (works well for singing voice)
    model_urls = {
        "generator": "https://drive.google.com/uc?id=1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW",
        "config": "https://drive.google.com/uc?id=1Piys37khs68TU4MWjuHHEA__YMTb1HjN"
    }
    
    for name, url in model_urls.items():
        file_path = model_dir / f"{name}.pth" if name == "generator" else model_dir / "config.json"
        if not file_path.exists():
            logger.info(f"Downloading {name}...")
            try:
                wget.download(url, str(file_path))
                logger.info(f"✅ Downloaded {name}")
            except Exception as e:
                logger.error(f"❌ Failed to download {name}: {e}")
                # Fallback to alternative source
                if name == "generator":
                    os.system(f"wget -O {file_path} https://github.com/jik876/hifi-gan/releases/download/v1.0/generator_universal.pth.tar")
    
    volume.commit()
    return str(model_dir)

@app.function(
    gpu="A10G",
    volumes={"/models": volume},
    timeout=600,   # 10 minutes
    memory=8192,
)
def vocode_mel_spectrogram(
    mel_spectrogram_b64: str,
    sample_rate: int = 22050,
    hop_length: int = 256
) -> Dict[str, Any]:
    """
    Convert mel-spectrogram to high-quality audio using HifiGAN
    
    Args:
        mel_spectrogram_b64: Base64 encoded mel-spectrogram numpy array
        sample_rate: Target sample rate
        hop_length: Hop length used in mel spectrogram generation
    
    Returns:
        Dict containing base64 encoded audio and metadata
    """
    try:
        import soundfile as sf
        from hifigan.models import Generator
        from hifigan.utils import load_checkpoint
        
        # Decode mel-spectrogram
        mel_bytes = base64.b64decode(mel_spectrogram_b64)
        mel_spectrogram = np.frombuffer(mel_bytes, dtype=np.float32)
        
        # Reshape to proper mel-spectrogram format (assuming shape info is embedded)
        # This assumes mel shape is stored as first two int32 values
        height = int(np.frombuffer(mel_bytes[:4], dtype=np.int32)[0])
        width = int(np.frombuffer(mel_bytes[4:8], dtype=np.int32)[0])
        mel_spectrogram = mel_spectrogram[2:].reshape(height, width)
        
        logger.info(f"🎵 Processing mel-spectrogram: {mel_spectrogram.shape}")
        
        # Load HifiGAN model
        model_dir = Path("/models/hifigan")
        
        # Load config
        with open(model_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Initialize generator
        generator = Generator(config)
        
        # Load checkpoint
        checkpoint_path = model_dir / "generator.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            generator.load_state_dict(checkpoint['generator'])
        else:
            raise FileNotFoundError("HifiGAN generator checkpoint not found")
        
        # Move to GPU and set to eval mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = generator.to(device)
        generator.eval()
        
        # Convert mel to tensor
        mel_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0).to(device)
        
        # Generate audio
        with torch.no_grad():
            audio = generator(mel_tensor)
            audio = audio.squeeze(0).cpu().numpy()
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            
            # Read back and encode
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"✅ Generated audio: {len(audio)} samples at {sample_rate}Hz")
        
        return {
            "success": True,
            "audio_b64": audio_b64,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate,
            "shape": list(audio.shape),
            "message": "HifiGAN vocoding successful"
        }
        
    except Exception as e:
        logger.error(f"❌ HifiGAN vocoding failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "HifiGAN vocoding failed"
        }

@app.function(
    gpu="A10G",
    volumes={"/models": volume},
    timeout=600,
)
def enhance_diffsvc_output(
    diffsvc_audio_b64: str,
    target_sample_rate: int = 44100
) -> Dict[str, Any]:
    """
    Enhance Diff-SVC output using HifiGAN for better quality
    
    Args:
        diffsvc_audio_b64: Base64 encoded audio from Diff-SVC
        target_sample_rate: Target sample rate for output
    
    Returns:
        Dict containing enhanced audio and metadata
    """
    try:
        import soundfile as sf
        
        # Decode input audio
        audio_bytes = base64.b64decode(diffsvc_audio_b64)
        
        # Save temporarily to read with librosa
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
            tmp_input.write(audio_bytes)
            tmp_input.flush()
            
            # Load audio
            audio, sr = librosa.load(tmp_input.name, sr=None)
            
        logger.info(f"🎵 Processing Diff-SVC audio: {len(audio)} samples at {sr}Hz")
        
        # Convert to mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,
            hop_length=256,
            win_length=1024,
            fmin=0,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        
        # Prepare mel for Modal function
        mel_shape = mel_spectrogram.shape
        mel_with_shape = np.concatenate([
            np.array([mel_shape[0], mel_shape[1]], dtype=np.int32).view(np.float32),
            mel_spectrogram.flatten()
        ])
        mel_b64 = base64.b64encode(mel_with_shape.tobytes()).decode('utf-8')
        
        # Call HifiGAN vocoder
        result = vocode_mel_spectrogram.remote(
            mel_spectrogram_b64=mel_b64,
            sample_rate=target_sample_rate
        )
        
        if result["success"]:
            logger.info("✅ HifiGAN enhancement successful")
            return {
                "success": True,
                "enhanced_audio_b64": result["audio_b64"],
                "original_duration": len(audio) / sr,
                "enhanced_duration": result["duration"],
                "sample_rate": result["sample_rate"],
                "message": "Diff-SVC audio enhanced with HifiGAN"
            }
        else:
            return result
            
    except Exception as e:
        logger.error(f"❌ Enhancement failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Enhancement failed"
        }

# Local client class for integration with your pipeline
class ModalHifiGANClient:
    """Client for Modal HifiGAN service"""
    
    def __init__(self):
        self.app = app
        
    def setup_models(self):
        """Download and setup HifiGAN models on Modal"""
        logger.info("🚀 Setting up HifiGAN models on Modal...")
        model_path = download_hifigan_models.remote()
        logger.info(f"✅ Models ready at: {model_path}")
        return model_path
    
    def enhance_audio(self, audio_file_path: str, output_path: str = None) -> str:
        """
        Enhance audio file using Modal HifiGAN
        
        Args:
            audio_file_path: Path to input audio file
            output_path: Path for output file (optional)
        
        Returns:
            Path to enhanced audio file
        """
        try:
            # Read and encode audio file
            with open(audio_file_path, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            logger.info(f"🎵 Enhancing audio: {audio_file_path}")
            
            # Call Modal function
            result = enhance_diffsvc_output.remote(
                diffsvc_audio_b64=audio_b64,
                target_sample_rate=44100
            )
            
            if result["success"]:
                # Decode and save enhanced audio
                enhanced_audio_bytes = base64.b64decode(result["enhanced_audio_b64"])
                
                if output_path is None:
                    output_path = audio_file_path.replace('.wav', '_enhanced.wav')
                
                with open(output_path, 'wb') as f:
                    f.write(enhanced_audio_bytes)
                
                logger.info(f"✅ Enhanced audio saved: {output_path}")
                logger.info(f"📊 Duration: {result['enhanced_duration']:.2f}s at {result['sample_rate']}Hz")
                
                return output_path
            else:
                raise Exception(f"Modal enhancement failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Audio enhancement failed: {e}")
            raise

if __name__ == "__main__":
    # Test the Modal HifiGAN service
    client = ModalHifiGANClient()
    
    # Setup models (run once)
    client.setup_models()
    
    # Test with a sample file (replace with your actual file)
    test_file = "test_audio.wav"
    if Path(test_file).exists():
        enhanced_file = client.enhance_audio(test_file)
        print(f"🎉 Enhancement complete: {enhanced_file}")
    else:
        print("ℹ️ Create a test_audio.wav file to test the enhancement")
