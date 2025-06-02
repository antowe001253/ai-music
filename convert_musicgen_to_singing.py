#!/usr/bin/env python3
"""
Convert MusicGen-generated melody to singing voice using RVC Modal service
"""

from modal_rvc_service import ModalRVCClient
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_musicgen_to_singing():
    """Convert MusicGen melody to multiple singing voice variations"""
    
    # Path to your MusicGen file
    input_file = "/Users/alexntowe/Projects/AI/Diff-SVC/outputs/phase3_complete/session_1748797646/03_vocals_melody_enhanced.wav"
    
    if not os.path.exists(input_file):
        logger.error(f"❌ Input file not found: {input_file}")
        return []
    
    logger.info(f"🎵 Converting MusicGen melody to singing voices...")
    logger.info(f"📁 Input: {input_file}")
    
    # Initialize RVC client
    client = ModalRVCClient()
    
    # Create different voice variations
    voice_variations = [
        ("musicgen_nativity_original.wav", 0),        # Original pitch
        ("musicgen_nativity_gentle.wav", -1),         # Slightly lower, gentle
        ("musicgen_nativity_bright.wav", 2),          # Brighter, more uplifting
        ("musicgen_nativity_warm.wav", -3),           # Warmer, deeper
        ("musicgen_nativity_angelic.wav", 4),         # Higher, angelic
        ("musicgen_nativity_child.wav", 6),           # Child-like voice
    ]
    
    results = []
    
    for output_file, pitch_shift in voice_variations:
        try:
            logger.info(f"🎤 Creating: {output_file} (pitch: {pitch_shift:+d})")
            
            result_file = client.convert_voice(
                input_file=input_file,
                output_file=output_file,
                pitch_shift=pitch_shift
            )
            
            results.append(result_file)
            logger.info(f"✅ Created: {result_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create {output_file}: {e}")
    
    return results

if __name__ == "__main__":
    print("🎼 Converting MusicGen Nativity Melody to Singing Voice 🎼")
    print("=" * 60)
    
    results = convert_musicgen_to_singing()
    
    print(f"\n🎄 Conversion Complete! Created {len(results)} singing voices:")
    for i, file in enumerate(results, 1):
        print(f"  {i}. {file}")
    
    print("\n🌟 Your MusicGen melody has been transformed into")
    print("   beautiful singing voices about the birth of Jesus! 🌟")
