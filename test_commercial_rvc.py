#!/usr/bin/env python3
"""
Quick RVC Test with Simplified Setup
For immediate commercial voice conversion testing
"""

from modal_rvc_service import ModalRVCClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_current_rvc():
    """Test the current RVC service with your MusicGen melody"""
    
    print("🎤 Testing Current RVC Service")
    print("=" * 40)
    
    client = ModalRVCClient()
    
    input_file = "/Users/alexntowe/Projects/AI/Diff-SVC/outputs/phase3_complete/session_1748797646/03_vocals_melody_enhanced.wav"
    
    # Test multiple pitch variations for commercial use
    test_cases = [
        ("nativity_commercial_original.wav", 0, "Original pitch"),
        ("nativity_commercial_higher.wav", 3, "Higher voice (+3)"),  
        ("nativity_commercial_deeper.wav", -2, "Deeper voice (-2)"),
        ("nativity_commercial_child.wav", 5, "Child-like (+5)"),
    ]
    
    results = []
    
    for output_file, pitch, description in test_cases:
        try:
            print(f"\n🎵 Converting: {description}")
            
            result = client.convert_voice(
                input_file=input_file,
                output_file=output_file,
                pitch_shift=pitch
            )
            
            print(f"✅ Success: {output_file}")
            results.append(output_file)
            
        except Exception as e:
            print(f"❌ Failed: {description} - {e}")
    
    print(f"\n🎄 Commercial Test Complete!")
    print(f"📁 Generated {len(results)} voice variations:")
    for i, file in enumerate(results, 1):
        print(f"  {i}. {file}")
    
    if results:
        print(f"\n💼 These files are ready for commercial use")
        print(f"🎵 Based on your MusicGen nativity melody")
        print(f"🎤 Converted using RVC voice processing")
    
    return results

if __name__ == "__main__":
    test_current_rvc()
