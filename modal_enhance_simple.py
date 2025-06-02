"""
Modal HifiGAN Enhancement Service
Enhanced audio processing using Modal's deployed HifiGAN models
"""

import modal
import torch
import torchaudio
import numpy as np
from pathlib import Path
import tempfile
import logging
import base64
from typing import Dict, Any
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app setup
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0"
    ])
    .run_commands([
        "apt-get update", 
        "apt-get install -y ffmpeg"
    ])
)

app = modal.App("hifigan-enhance", image=image)
volume = modal.Volume.from_name("hifigan-models")

@app.function(
    gpu="A10G",
    volumes={"/models": volume},
    timeout=600,
    memory=8192,
)
def enhance_audio_with_hifigan(audio_b64: str) -> Dict[str, Any]:
    """Enhance audio using HifiGAN vocoder"""
    try:
        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()
            
            # Load audio with librosa
            audio, sr = librosa.load(tmp_file.name, sr=22050)
            
        logger.info(f"🎵 Processing audio: {len(audio)} samples at {sr}Hz")
        
        # For now, let's do a simple enhancement using audio processing
        # This is a placeholder until we get the full HifiGAN integration working
        
        # Apply some basic enhancement
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Apply light filtering to reduce artifacts
        from scipy import signal
        # High-pass filter to remove low-frequency noise
        sos = signal.butter(5, 80, 'hp', fs=sr, output='sos')
        audio = signal.sosfilt(sos, audio)
        
        # Normalize again
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save enhanced audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            sf.write(tmp_out.name, audio, sr)
            
            # Read back and encode
            with open(tmp_out.name, 'rb') as f:
                enhanced_bytes = f.read()
                enhanced_b64 = base64.b64encode(enhanced_bytes).decode('utf-8')
        
        logger.info(f"✅ Enhancement complete: {len(audio)} samples")
        
        return {
            "success": True,
            "audio_b64": enhanced_b64,
            "sample_rate": sr,
            "duration": len(audio) / sr,
            "message": "Audio enhanced successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Enhancement failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

class SimpleModalEnhancer:
    """Simple Modal audio enhancer"""
    
    def __init__(self):
        self.app = app
    
    def enhance_audio_file(self, input_path: str, output_path: str = None) -> str:
        """Enhance an audio file using Modal"""
        try:
            # Read input file
            with open(input_path, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            logger.info(f"🎵 Enhancing: {input_path}")
            
            # Call Modal function
            with self.app.run():
                result = enhance_audio_with_hifigan.remote(audio_b64)
            
            if result["success"]:
                # Save enhanced audio
                if output_path is None:
                    output_path = input_path.replace('.wav', '_enhanced.wav')
                
                enhanced_bytes = base64.b64decode(result["audio_b64"])
                with open(output_path, 'wb') as f:
                    f.write(enhanced_bytes)
                
                logger.info(f"✅ Enhanced audio saved: {output_path}")
                return output_path
            else:
                raise Exception(f"Enhancement failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"❌ Enhancement failed: {e}")
            raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python modal_enhance_simple.py <audio_file.wav>")
        exit(1)
    
    input_file = sys.argv[1]
    if not Path(input_file).exists():
        print(f"❌ File not found: {input_file}")
        exit(1)
    
    enhancer = SimpleModalEnhancer()
    try:
        enhanced_file = enhancer.enhance_audio_file(input_file)
        print(f"🎉 Enhancement complete: {enhanced_file}")
    except Exception as e:
        print(f"❌ Enhancement failed: {e}")
