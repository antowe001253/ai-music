#!/usr/bin/env python3
"""
Test Modal RVC Voice Conversion Service
"""

import sys
import os
from pathlib import Path
from modal_rvc_service import ModalRVCClient

def test_modal_rvc():
    """Test the Modal RVC service with existing audio files"""
    
    # Initialize client
    print("🚀 Initializing Modal RVC Client...")
    client = ModalRVCClient()
    
    # Test files (use your existing audio files)
    test_files = [
        "sentence_singing_1.wav",
        "test_vocal_synthesis.wav", 
        "reference_vocal.wav"
    ]
    
    # Find available test file
    input_file = None
    for test_file in test_files:
        if Path(test_file).exists():
            input_file = test_file
            break
    
    if not input_file:
        print("❌ No test audio files found")
        return False
    
    print(f"🎵 Using test file: {input_file}")
    
    try:
        # Test voice conversion
        print("🎤 Testing RVC voice conversion...")
        output_file = client.convert_voice(
            input_file=input_file,
            pitch_shift=2  # Shift up 2 semitones
        )
        
        print(f"✅ RVC conversion successful!")
        print(f"📁 Output: {output_file}")
        
        # Check output file
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            print(f"📊 Output file size: {file_size:,} bytes")
            
            if file_size > 1000:  # At least 1KB
                print("✅ RVC Modal service is working correctly!")
                return True
            else:
                print("⚠️ Output file seems too small")
                return False
        else:
            print(f"❌ Output file not created: {output_file}")
            return False
            
    except Exception as e:
        print(f"❌ RVC test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Modal RVC Service")
    print("=" * 50)
    
    success = test_modal_rvc()
    
    if success:
        print("🎉 Modal RVC setup is COMPLETE and WORKING!")
        print("💡 You can now use RVC for voice conversion in the cloud")
    else:
        print("⚠️ RVC test had issues - check the logs above")
    
    print("=" * 50)
