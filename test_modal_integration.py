#!/usr/bin/env python3
"""
Test script for Modal HifiGAN integration
Run this to verify everything is working correctly
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from pipeline.modal_enhanced_interface import ModalEnhancedInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_modal_setup():
    """Test Modal HifiGAN setup"""
    print("🚀 Testing Modal HifiGAN Integration")
    print("=" * 50)
    
    # Initialize interface
    interface = ModalEnhancedInterface()
    
    # Test setup
    print("1. Setting up Modal HifiGAN...")
    setup_success = interface.setup_modal_hifigan()
    
    if setup_success:
        print("✅ Modal HifiGAN setup successful!")
        return True
    else:
        print("❌ Modal HifiGAN setup failed!")
        return False

def test_audio_enhancement():
    """Test audio enhancement with sample file"""
    print("\n2. Testing audio enhancement...")
    
    # Look for your existing Diff-SVC output
    possible_files = [
        "session_1748802833/03_vocals_diffsvc.wav",
        "outputs/automated_pipeline/session_*/vocals_only.wav",
        "temp/test_audio.wav"
    ]
    
    test_file = None
    for pattern in possible_files:
        files = list(Path(".").glob(pattern))
        if files:
            test_file = str(files[0])
            break
    
    if not test_file:
        print("⚠️ No test audio file found.")
        print("💡 To test enhancement, place a .wav file in the project root and run:")
        print("   python test_modal_integration.py path/to/your/audio.wav")
        return False
    
    print(f"🎵 Testing with: {test_file}")
    
    try:
        interface = ModalEnhancedInterface()
        interface.setup_modal_hifigan()
        
        enhanced_file = interface.enhance_existing_audio(test_file)
        print(f"✅ Enhancement successful!")
        print(f"📁 Enhanced file: {enhanced_file}")
        return True
        
    except Exception as e:
        print(f"❌ Enhancement failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎤 Modal HifiGAN Integration Test")
    print("=" * 40)
    
    # Test 1: Setup
    setup_ok = test_modal_setup()
    
    if not setup_ok:
        print("\n❌ Setup failed. Please check:")
        print("1. Modal is installed: pip install modal")
        print("2. Modal is authenticated: modal setup")
        print("3. Internet connection is available")
        return
    
    # Test 2: Enhancement (if file provided)
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        if Path(test_file).exists():
            print(f"\n🎵 Testing enhancement with: {test_file}")
            try:
                interface = ModalEnhancedInterface()
                enhanced_file = interface.enhance_existing_audio(test_file)
                print(f"✅ Success! Enhanced file: {enhanced_file}")
            except Exception as e:
                print(f"❌ Enhancement failed: {e}")
        else:
            print(f"❌ File not found: {test_file}")
    else:
        test_audio_enhancement()
    
    print("\n🎉 Modal HifiGAN integration test complete!")
    print("\n📝 Next steps:")
    print("1. Test with your Diff-SVC output:")
    print("   python test_modal_integration.py session_1748802833/03_vocals_diffsvc.wav")
    print("2. Integrate with your main pipeline using the ModalEnhancedInterface class")

if __name__ == "__main__":
    main()
