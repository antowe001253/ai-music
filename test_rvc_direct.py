#!/usr/bin/env python3
"""
Simple test for Modal RVC using direct modal run
"""

import subprocess
import base64
import sys
from pathlib import Path

def test_modal_rvc_direct():
    """Test Modal RVC using direct modal run command"""
    
    # Find a test audio file
    test_files = [
        "sentence_singing_1.wav",
        "test_vocal_synthesis.wav", 
        "reference_vocal.wav"
    ]
    
    input_file = None 
    for test_file in test_files:
        if Path(test_file).exists():
            input_file = test_file
            break
    
    if not input_file:
        print("❌ No test audio files found")
        return False
    
    print(f"🎵 Testing with: {input_file}")
    
    try:
        # Read and encode the audio file
        with open(input_file, 'rb') as f:
            audio_data = f.read()
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        print(f"📦 Encoded audio: {len(audio_b64)} characters")
        
        # Create a temporary Python script to pass the data
        test_script = f'''
import base64
import tempfile
from modal_rvc_service import convert_voice_with_rvc

# Test data
audio_b64 = "{audio_b64}"
result = convert_voice_with_rvc.remote(
    input_audio_b64=audio_b64,
    pitch_shift=2
)
print("Result:", result)
'''
        
        # Write the test script
        with open('temp_rvc_test.py', 'w') as f:
            f.write(test_script)
        
        print("🚀 Running Modal RVC test...")
        
        # Run via modal
        result = subprocess.run([
            'bash', '-c', 
            'source venv/bin/activate && modal run temp_rvc_test.py'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Modal RVC test successful!")
            print("Output:", result.stdout)
            return True
        else:
            print("❌ Modal RVC test failed")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up
        if Path('temp_rvc_test.py').exists():
            Path('temp_rvc_test.py').unlink()

if __name__ == "__main__":
    print("🧪 Testing Modal RVC Service (Direct)")
    print("=" * 50)
    
    success = test_modal_rvc_direct()
    
    if success:
        print("🎉 Modal RVC is WORKING!")
    else:
        print("⚠️ Modal RVC test failed")
    
    print("=" * 50)
