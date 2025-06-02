"""
Test Diff-SVC with Vocal Reference Input
This uses your trained model with actual vocal input
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_diffsvc_with_vocal(vocal_input: str, output: str):
    """Test Diff-SVC using actual vocal input"""
    
    logger.info(f"🎤 Testing Diff-SVC with vocal input: {vocal_input}")
    
    # Method 1: Try the working inference approach we discovered
    try:
        # Copy the vocal to raw folder
        raw_folder = Path("raw")
        raw_folder.mkdir(exist_ok=True)
        
        import shutil
        vocal_filename = Path(vocal_input).name
        raw_path = raw_folder / vocal_filename
        shutil.copy2(vocal_input, raw_path)
        
        logger.info(f"📁 Copied vocal to: {raw_path}")
        
        # The key insight: Diff-SVC needs a HUMAN VOCAL INPUT
        # Let's modify the original infer.py to work with our setup
        
        # Method 2: Use a simple conversion approach
        logger.info("🎵 Running Diff-SVC voice conversion...")
        
        cmd = [
            sys.executable, "infer.py",
            "--config", "config.yaml",
            "--input", str(raw_path),
            "--output", output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ Diff-SVC completed: {output}")
            return output
        else:
            logger.error(f"❌ Diff-SVC failed: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Diff-SVC test failed: {e}")
        return None

def create_vocal_reference():
    """Create a simple vocal reference for testing"""
    import librosa
    import soundfile as sf
    import numpy as np
    
    logger.info("🎤 Creating vocal reference for testing...")
    
    # Load one of your melody files
    melody_files = list(Path("outputs/phase3_complete").glob("*/02_vocal_melody.wav"))
    
    if not melody_files:
        logger.error("❌ No melody files found for reference")
        return None
    
    melody_file = melody_files[0]
    audio, sr = librosa.load(melody_file, sr=24000)
    
    # Create vocal-like content (humming/singing style)
    # This simulates what a human vocal input would look like
    
    # Extract harmonic content
    harmonic, _ = librosa.effects.hpss(audio)
    
    # Pitch shift to vocal range
    vocal_like = librosa.effects.pitch_shift(harmonic, sr=sr, n_steps=6)
    
    # Add vocal characteristics
    from scipy import signal
    
    # Bandpass filter for vocal frequencies
    sos = signal.butter(4, [100, 4000], 'bandpass', fs=sr, output='sos')
    vocal_filtered = signal.sosfilt(sos, vocal_like)
    
    # Add some texture
    t = np.linspace(0, len(vocal_filtered)/sr, len(vocal_filtered))
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
    vocal_with_vibrato = vocal_filtered * vibrato
    
    # Normalize
    vocal_final = vocal_with_vibrato / np.max(np.abs(vocal_with_vibrato)) * 0.8
    
    # Save as reference vocal
    ref_path = "reference_vocal.wav"
    sf.write(ref_path, vocal_final, sr)
    
    logger.info(f"✅ Created vocal reference: {ref_path}")
    return ref_path

if __name__ == "__main__":
    # Create a vocal reference
    vocal_ref = create_vocal_reference()
    
    if vocal_ref:
        # Test Diff-SVC with the vocal reference
        output_path = "diffsvc_converted_vocal.wav"
        result = test_diffsvc_with_vocal(vocal_ref, output_path)
        
        if result:
            print(f"🎉 Diff-SVC test successful: {result}")
            print("🎤 Next step: Enhance with Modal HifiGAN")
            print(f"   python modal_enhance_simple.py {result}")
        else:
            print("❌ Diff-SVC test failed")
    else:
        print("❌ Could not create vocal reference")
