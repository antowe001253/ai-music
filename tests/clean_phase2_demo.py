#!/usr/bin/env python3
"""
Clean Phase 2 Integration Demo
Creates a cleaner version without synthesis artifacts
"""

import sys
import os
import numpy as np
import scipy.io.wavfile
from pathlib import Path

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

def create_clean_demo():
    """Create a cleaner integration demo using actual MusicGen output"""
    print("🧹 CREATING CLEAN PHASE 2 INTEGRATION DEMO")
    print("=" * 60)
    
    # Step 1: Use actual MusicGen output as base
    from pipeline.audio_processing import AudioProcessingSuite
    
    processor = AudioProcessingSuite()
    
    # Load the clean MusicGen output
    base_audio, sr = processor.load_audio('../outputs/generated_music/test_musicgen_output.wav')
    print(f"📁 Loaded clean base audio: {len(base_audio)} samples at {sr} Hz")
    
    # Step 2: Demonstrate prompt intelligence on the actual audio
    from prompt_intelligence import PromptIntelligence
    
    intelligence = PromptIntelligence()
    params = intelligence.parse_prompt("upbeat electronic techno with synthesizer")
    
    print(f"🧠 Prompt Analysis:")
    print(f"  Genre: {params.genre}")
    print(f"  Tempo: {params.tempo} BPM") 
    print(f"  Instruments: {', '.join(params.instruments)}")
    
    # Step 3: Analyze the actual audio
    tempo_info = processor.detect_tempo_and_beats(base_audio, sr)
    key_info = processor.detect_key(base_audio, sr)
    
    print(f"\\n🎵 Audio Analysis:")
    print(f"  Detected BPM: {tempo_info['bpm']:.1f}")
    print(f"  Detected Key: {key_info['key']}")
    print(f"  Beat markers: {len(tempo_info['beat_times'])}")
    
    # Step 4: Apply clean professional processing
    from advanced_audio_processing import AdvancedAudioProcessor
    
    processor_adv = AdvancedAudioProcessor(sr)
    
    # Clean processing settings
    clean_settings = {
        'eq': {
            'presence': 2.0,      # Enhance clarity
            'brilliance': 1.0,    # Add sparkle
            'bass': -1.0          # Reduce muddiness
        },
        'compression': {
            'threshold': -18,
            'ratio': 3
        },
        'gain': 0.8
    }
    
    master_settings = {
        'master_eq': {'presence': 1.5},
        'master_compression': {'threshold': -6, 'ratio': 2},
        'target_lufs': -16.0
    }
    
    print(f"\\n🎚️ Applying clean processing:")
    print(f"  EQ: Presence +2dB, Brilliance +1dB, Bass -1dB")
    print(f"  Compression: 3:1 ratio at -18dB threshold")
    print(f"  Master: Light compression and EQ")
    
    # Process the track
    tracks_for_mixing = [(base_audio, clean_settings)]
    final_mix = processor_adv.mix_tracks(tracks_for_mixing, master_settings)
    
    # Step 5: Save clean result
    output_path = '../outputs/generated_music/phase2_clean_demo.wav'
    output_full_path = os.path.join(os.path.dirname(__file__), output_path)
    
    # Ensure it's properly normalized and clean
    final_mix_clean = np.clip(final_mix, -0.95, 0.95)  # Prevent clipping
    final_mix_int16 = (final_mix_clean * 32767).astype(np.int16)
    
    scipy.io.wavfile.write(output_full_path, sr, final_mix_int16)
    
    print(f"\\n🎵 Clean Result:")
    print(f"  Duration: {len(final_mix) / sr:.1f} seconds")
    print(f"  Peak level: {np.max(np.abs(final_mix_clean)):.3f}")
    print(f"  RMS level: {np.sqrt(np.mean(final_mix_clean**2)):.3f}")
    print(f"  Output: {output_path}")
    
    print(f"\\n✅ CLEAN PHASE 2 DEMO COMPLETE!")
    print(f"🎯 Demonstrates professional processing without synthesis artifacts")

if __name__ == "__main__":
    create_clean_demo()