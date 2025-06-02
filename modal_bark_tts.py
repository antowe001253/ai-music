"""
Modal Bark TTS Service
Generate singing-style speech with lyrics using Bark TTS on Modal
"""

import modal
import torch
import numpy as np
import base64
import tempfile
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal image with Bark TTS
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0,<2.6.0",  # Use older PyTorch to avoid compatibility issues
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "git+https://github.com/suno-ai/bark.git"
    ])
    .run_commands([
        "apt-get update",
        "apt-get install -y git ffmpeg"
    ])
)

app = modal.App("bark-tts", image=image)

@app.function(
    gpu="A10G",
    timeout=600,
    memory=8192,
)
def generate_singing_speech(
    text: str,
    voice_preset: str = "v2/en_speaker_6"
) -> Dict[str, Any]:
    """
    Generate singing-style speech with Bark TTS
    
    Args:
        text: Text to convert to singing speech
        voice_preset: Bark voice preset to use
    
    Returns:
        Dict with base64 encoded audio and metadata
    """
    try:
        # Import Bark inside function to avoid loading issues
        from bark import SAMPLE_RATE, generate_audio
        import scipy.io.wavfile
        
        logger.info(f"🎤 Generating singing speech: {text}")
        logger.info(f"🎭 Voice preset: {voice_preset}")
        
        # Generate audio with Bark TTS
        audio_array = generate_audio(text, history_prompt=voice_preset)
        
        logger.info(f"✅ Generated audio shape: {audio_array.shape}")
        logger.info(f"🔊 Sample rate: {SAMPLE_RATE}")
        
        # Calculate duration
        duration = len(audio_array) / SAMPLE_RATE
        logger.info(f"⏱️ Duration: {duration:.2f} seconds")
        
        # Normalize and convert to 16-bit
        audio_normalized = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # Save to temporary file and encode
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            scipy.io.wavfile.write(tmp_file.name, SAMPLE_RATE, audio_int16)
            
            # Read back and encode to base64
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info("✅ Bark TTS generation successful")
        
        return {
            "success": True,
            "audio_b64": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "duration": duration,
            "text": text,
            "voice_preset": voice_preset,
            "message": "Bark TTS singing speech generated successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Bark TTS generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Bark TTS generation failed"
        }

# Local client class
class ModalBarkClient:
    """Client for Modal Bark TTS service"""
    
    def __init__(self):
        self.app = app
    
    def generate_singing_speech(self, text: str, output_file: str = None, voice_preset: str = "v2/en_speaker_6"):
        """
        Generate singing-style speech from text
        
        Args:
            text: Text to convert to singing speech
            output_file: Output audio file path
            voice_preset: Bark voice preset
        
        Returns:
            Path to generated audio file
        """
        try:
            logger.info(f"🎤 Generating singing speech: {text}")
            
            # Call Modal function
            with self.app.run():
                result = generate_singing_speech.remote(
                    text=text,
                    voice_preset=voice_preset
                )
            
            if result["success"]:
                # Decode and save audio
                audio_bytes = base64.b64decode(result["audio_b64"])
                
                if output_file is None:
                    output_file = "bark_singing_output.wav"
                
                with open(output_file, 'wb') as f:
                    f.write(audio_bytes)
                
                logger.info(f"✅ Singing speech saved: {output_file}")
                logger.info(f"📊 Duration: {result['duration']:.2f}s at {result['sample_rate']}Hz")
                
                return output_file
            else:
                raise Exception(f"Modal Bark generation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Singing speech generation failed: {e}")
            raise

if __name__ == "__main__":
    # Test Bark TTS on Modal
    client = ModalBarkClient()
    
    # Test with Silent Night lyrics
    test_text = "♪ Silent night, holy night, all is calm, all is bright ♪"
    
    try:
        output_file = client.generate_singing_speech(
            text=test_text,
            output_file="bark_silent_night_modal_test.wav"
        )
        print(f"🎉 Success! Generated: {output_file}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
