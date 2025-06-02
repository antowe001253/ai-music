#!/usr/bin/env python3
"""
Test RVC Modal Service
"""

from modal_rvc_service import ModalRVCClient
import os

def test_rvc():
    """Test the RVC Modal service"""
    
    print("🚀 Testing Modal RVC Service...")
    
    # Initialize client
    client = ModalRVCClient()
    
    # Find a test audio file
    test_files = [
        "sentence_singing_1.wav",
        "real_vocals_test.wav", 
        "reference_vocal.wav",
        "better_vocal_reference.wav"
    ]
    
    test_file = None
    for f in test_files:
        if os.path.exists(f):
            test_file = f
            break
    
    if not test_file:
        print("❌ No test audio file found")
        return
    
    print(f"🎵 Using test file: {test_file}")
    
    try:
        # Test voice conversion
        output_file = client.convert_voice(
            input_file=test_file,
            output_file=f"{test_file}_rvc_test.wav",
            pitch_shift=2  # Raise pitch by 2 semitones
        )
        
        print(f"✅ RVC conversion successful!")
        print(f"📁 Output saved: {output_file}")
        
    except Exception as e:
        print(f"❌ RVC test failed: {e}")

if __name__ == "__main__":
    test_rvc()
