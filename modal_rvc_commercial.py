"""
Commercial-Grade Modal RVC Voice Conversion Service
Deploy RVC to Modal cloud with proper fairseq for production use
"""

import modal
import torch
import numpy as np
import librosa
import soundfile as sf
import base64
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production-ready Modal image with proper fairseq installation
image = (
    modal.Image.from_registry("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime")
    .apt_install([
        "git", 
        "ffmpeg", 
        "build-essential",
        "cmake",
        "pkg-config",
        "libsndfile1-dev",
        "gcc",
        "g++",
        "libc6-dev",
        "python3-dev",
    ])
    .pip_install([
        # Core dependencies first
        "librosa>=0.10.0",
        "scipy>=1.10.0", 
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
        "gradio>=3.34.0",
        "faiss-cpu>=1.7.3",
        "praat-parselmouth>=0.4.1",
        "pyworld>=0.3.4",
        "torchcrepe>=0.0.20",
        "resampy>=0.4.2",
        "ffmpeg-python>=0.2.0",
        "av>=10.0.0",
        # Fairseq prerequisites
        "cython>=0.29.0",
        "cffi>=1.0.0",
        "omegaconf>=2.0.0",
        "hydra-core>=1.0.0",
        "editdistance>=0.5.0",
        "sacrebleu>=1.4.0",
        "tensorboardX>=2.0",
        "scikit-learn>=0.24.0",
        "regex>=2021.0.0",
        "bitarray>=2.0.0",
    ])
    .run_commands([
        # Install fairseq properly for production
        "pip install fairseq==0.12.2 --no-build-isolation",
        # Clone RVC
        "git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /rvc",
        "cd /rvc && git checkout main",
        # Verify installation
        "python -c 'import fairseq; print(\"✅ Fairseq installed successfully\")'",
        "python -c 'import torch; print(f\"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}\")'",
    ])
)

app = modal.App("rvc-commercial", image=image)

# Production volume for RVC models
rvc_volume = modal.Volume.from_name("rvc-commercial-models", create_if_missing=True)

@app.function(
    gpu="A10G",
    volumes={"/models": rvc_volume},
    timeout=1800,
    memory=16384,  # Increased memory for commercial use
)
def setup_commercial_rvc_models():
    """Download and setup RVC models for commercial use"""
    import os
    import urllib.request
    import zipfile
    
    model_dir = Path("/models/rvc")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🏢 Setting up commercial RVC models...")
    
    # Download base models for RVC
    base_models = {
        "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    }
    
    for model_name, url in base_models.items():
        model_path = model_dir / model_name
        if not model_path.exists():
            logger.info(f"📥 Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(url, str(model_path))
                logger.info(f"✅ Downloaded {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to download {model_name}: {e}")
                raise
    
    # Download Barnabas voice model (commercial quality)
    barnabas_zip_url = "https://huggingface.co/DarkWeBareBears69/My-RVC-Voice-Models/resolve/main/Barnabas.zip"
    barnabas_zip_path = model_dir / "Barnabas.zip"
    
    if not (model_dir / "Barnabas.pth").exists():
        try:
            logger.info("📥 Downloading Barnabas commercial voice model...")
            urllib.request.urlretrieve(barnabas_zip_url, str(barnabas_zip_path))
            logger.info("✅ Downloaded Barnabas.zip")
            
            # Extract the zip file
            logger.info("📦 Extracting Barnabas model files...")
            with zipfile.ZipFile(str(barnabas_zip_path), 'r') as zip_ref:
                zip_ref.extractall(str(model_dir))
            
            # Clean up zip file
            barnabas_zip_path.unlink()
            logger.info("✅ Extracted Barnabas model files")
            
        except Exception as e:
            logger.error(f"❌ Failed to download/extract Barnabas model: {e}")
            raise
    
    # Verify model files
    required_files = ["hubert_base.pt", "rmvpe.pt", "Barnabas.pth"]
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {file}: {size_mb:.1f} MB")
        else:
            logger.error(f"❌ Missing required file: {file}")
            raise FileNotFoundError(f"Required model file not found: {file}")
    
    rvc_volume.commit()
    logger.info("🏢 Commercial RVC models setup complete")
    return str(model_dir)

@app.function(
    gpu="A10G", 
    volumes={"/models": rvc_volume},
    timeout=600,
    memory=16384,
    retries=3,  # Production reliability
)
def commercial_voice_conversion(
    input_audio_b64: str,
    pitch_shift: int = 0,
    filter_radius: int = 3,
    index_rate: float = 0.75,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33
) -> Dict[str, Any]:
    """
    Commercial-grade voice conversion using RVC with Barnabas model
    
    Args:
        input_audio_b64: Base64 encoded input audio
        pitch_shift: Pitch adjustment in semitones
        filter_radius: Smoothing filter
        index_rate: Index influence (0.0-1.0)
        rms_mix_rate: RMS mixing rate
        protect: Protect voiceless consonants
    
    Returns:
        Dict with converted audio and metadata
    """
    try:
        import sys
        sys.path.append('/rvc')
        
        logger.info("🏢 Starting commercial RVC voice conversion...")
        
        # Verify fairseq installation
        try:
            import fairseq
            logger.info(f"✅ Fairseq {fairseq.__version__} loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Fairseq import failed: {e}")
            raise ImportError("Fairseq not available - commercial RVC requires fairseq")
        
        # Import RVC modules with detailed error tracking
        logger.info("📦 Importing RVC modules...")
        try:
            from infer.modules.vc.modules import VC
            logger.info("✅ Successfully imported VC")
        except ImportError as e:
            logger.error(f"❌ Failed to import VC: {e}")
            raise
        
        try:
            from infer.lib.audio import load_audio
            logger.info("✅ Successfully imported load_audio")
        except ImportError as e:
            logger.error(f"❌ Failed to import load_audio: {e}")
            raise
        
        # Decode input audio
        audio_bytes = base64.b64decode(input_audio_b64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
            tmp_input.write(audio_bytes)
            tmp_input.flush()
            
            # Load audio with RVC's loader
            audio = load_audio(tmp_input.name, 16000)
            
        logger.info(f"🎵 Processing audio: {len(audio)} samples")
        
        # Initialize RVC with commercial models
        model_dir = Path("/models/rvc")
        hubert_path = str(model_dir / "hubert_base.pt")
        voice_model_path = str(model_dir / "Barnabas.pth")
        
        # Find the index file (it may have a different name)
        index_files = list(model_dir.glob("*Barnabas*.index"))
        index_model_path = str(index_files[0]) if index_files else ""
        
        # Verify models exist
        if not Path(hubert_path).exists():
            raise FileNotFoundError("Hubert model not found - run setup first")
        if not Path(voice_model_path).exists():
            raise FileNotFoundError("Barnabas voice model not found - run setup first")
        
        logger.info(f"🎤 Using commercial Barnabas voice model: {voice_model_path}")
        logger.info(f"📁 Model file size: {Path(voice_model_path).stat().st_size / (1024 * 1024):.1f} MB")
        
        if index_model_path:
            logger.info(f"📋 Using index file: {Path(index_model_path).name}")
        
        # Initialize VC module
        logger.info("🔧 Initializing commercial VC module...")
        vc = VC()
        vc.get_vc(voice_model_path)
        logger.info("✅ Commercial voice model loaded")
        
        # Convert voice with commercial settings
        logger.info("🎵 Starting commercial RVC voice conversion...")
        logger.info(f"⚙️ Settings: pitch={pitch_shift}, filter={filter_radius}, index_rate={index_rate}")
        
        converted_audio = vc.vc_single(
            sid=0,
            input_audio_path=tmp_input.name,
            f0_up_key=pitch_shift,
            f0_file=None,
            f0_method="rmvpe",
            file_index=index_model_path,
            file_index2="", 
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=0,
            rms_mix_rate=rms_mix_rate,
            protect=protect
        )
        
        logger.info(f"✅ Commercial RVC conversion completed")
        if isinstance(converted_audio, tuple) and len(converted_audio) >= 2:
            logger.info(f"📊 Output: {converted_audio[0]} Hz, {len(converted_audio[1])} samples")
        
        # Save converted audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
            sf.write(tmp_output.name, converted_audio[1], converted_audio[0])
            
            # Read back and encode
            with open(tmp_output.name, 'rb') as f:
                output_bytes = f.read()
                output_b64 = base64.b64encode(output_bytes).decode('utf-8')
        
        logger.info("🏢 Commercial Barnabas RVC conversion complete")
        
        return {
            "success": True,
            "audio_b64": output_b64,
            "sample_rate": converted_audio[0],
            "model_used": "Barnabas (Commercial)",
            "method": "True RVC with Fairseq",
            "settings": {
                "pitch_shift": pitch_shift,
                "filter_radius": filter_radius,
                "index_rate": index_rate,
                "rms_mix_rate": rms_mix_rate,
                "protect": protect
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Commercial RVC conversion failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "method": "Failed - Commercial RVC Error"
        }

# Commercial client class
class CommercialRVCClient:
    """Commercial-grade client for Modal RVC service"""
    
    def __init__(self):
        self.app = app
    
    def setup_models(self):
        """Setup commercial RVC models on Modal"""
        logger.info("🏢 Setting up commercial RVC models on Modal...")
        model_path = setup_commercial_rvc_models.remote()
        logger.info(f"✅ Commercial RVC models ready: {model_path}")
        return model_path
    
    def convert_voice(
        self, 
        input_file: str, 
        output_file: str = None, 
        pitch_shift: int = 0,
        **kwargs
    ):
        """
        Convert voice using commercial-grade RVC
        
        Args:
            input_file: Input audio file path
            output_file: Output file path
            pitch_shift: Pitch adjustment in semitones
            **kwargs: Additional RVC parameters
        """
        try:
            # Read and encode input
            with open(input_file, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            logger.info(f"🏢 Commercial voice conversion: {input_file}")
            
            # Call commercial RVC
            result = commercial_voice_conversion.remote(
                input_audio_b64=audio_b64,
                pitch_shift=pitch_shift,
                **kwargs
            )
            
            if result["success"]:
                # Save converted audio
                if output_file is None:
                    output_file = input_file.replace('.wav', '_commercial_rvc.wav')
                
                converted_bytes = base64.b64decode(result["audio_b64"])
                with open(output_file, 'wb') as f:
                    f.write(converted_bytes)
                
                logger.info(f"✅ Commercial RVC conversion saved: {output_file}")
                logger.info(f"🏢 Method: {result['method']}")
                logger.info(f"🎤 Model: {result['model_used']}")
                
                return output_file
            else:
                raise Exception(f"Commercial RVC failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Commercial voice conversion failed: {e}")
            raise

if __name__ == "__main__":
    # Test commercial RVC service
    client = CommercialRVCClient()
    
    print("🏢 Setting up Commercial Modal RVC...")
    client.setup_models()
    
    print("✅ Commercial Modal RVC ready!")
    print("💼 Usage: client.convert_voice('input.wav', 'output.wav')")
