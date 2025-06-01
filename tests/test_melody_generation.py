#!/usr/bin/env python3
"""
Test script for Melody Generation System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.music_generation import MelodyGenerationSystem
import numpy as np
import scipy.io.wavfile

def test_melody_generation():
    """Test melody generation system"""
    print("🎼 Testing Melody Generation System")
    print("=" * 50)
    
    # Initialize system
    melody_system = MelodyGenerationSystem()
    
    # Test loading melody model (this will download ~3.3GB)
    print("Loading MusicGen Melody model (this may take a while)...")
    success = melody_system.load_melody_model()
    
    if not success:
        print("❌ Failed to load melody model")
        return False
    
    # Test melody generation
    print("\nTesting melody generation...")
    try:
        audio_values, sample_rate = melody_system.generate_melody_conditioned(
            text_description="happy upbeat pop melody",
            duration=5  # Short duration for testing
        )
        
        if audio_values is not None:
            print(f"✅ Generated melody: shape {audio_values.shape}")
            
            # Save generated melody
            output_path = "/Users/alexntowe/Projects/AI/Diff-SVC/test_melody_output.wav"
            audio_data = audio_values[0, 0].numpy()
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            scipy.io.wavfile.write(output_path, sample_rate, audio_data)
            print(f"✅ Melody saved to: {output_path}")
            
            return True
        else:
            print("❌ Failed to generate melody")
            return False
            
    except Exception as e:
        print(f"❌ Error during melody generation: {e}")
        return False

if __name__ == "__main__":
    success = test_melody_generation()
    
    if success:
        print("\n✅ Melody Generation System is working!")
        print("Ready for Steps 6-8 integration!")
    else:
        print("\n❌ Melody Generation System test failed.")
