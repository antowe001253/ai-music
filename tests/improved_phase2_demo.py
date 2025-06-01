#!/usr/bin/env python3
"""
Improved Phase 2 Integration Demo - Noise-Free Version
"""

import sys
import os
import numpy as np
import scipy.io.wavfile

sys.path.append('pipeline')

def create_improved_demo():
    """Create an improved demo without synthesis artifacts"""
    print("🎼 IMPROVED PHASE 2 INTEGRATION DEMO")
    print("=" * 60)
    
    from prompt_intelligence import PromptIntelligence
    from orchestration_engine import OrchestrationEngine, TrackInfo
    from advanced_audio_processing import AdvancedAudioProcessor
    from audio_processing import AudioProcessingSuite
    
    # Step 1: Parse prompt
    intelligence = PromptIntelligence()
    params = intelligence.parse_prompt("smooth jazz piano with bass and light drums")
    print(f"🎤 Prompt: 'smooth jazz piano with bass and light drums'")
    print(f"🧠 Parsed: {params.genre}, {params.instruments}")
    
    # Step 2: Use actual audio as foundation
    processor = AudioProcessingSuite() 
    base_audio, sr = processor.load_audio('outputs/generated_music/test_musicgen_output.wav')
    
    # Create cleaner synthetic elements based on musical principles
    duration = len(base_audio) / sr
    t = np.linspace(0, duration, len(base_audio))
    
    # Piano: Clean chord progression with envelope
    envelope = np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * t * 0.5))  # Natural decay
    piano_fundamental = np.sin(2 * np.pi * 261.63 * t)  # C4
    piano_third = np.sin(2 * np.pi * 329.63 * t) * 0.7   # E4  
    piano_fifth = np.sin(2 * np.pi * 392.00 * t) * 0.5   # G4
    piano_audio = (piano_fundamental + piano_third + piano_fifth) * envelope * 0.2
    
    # Bass: Clean low-frequency foundation
    bass_envelope = np.ones_like(t) * (0.8 + 0.2 * np.sin(2 * np.pi * t * 0.25))
    bass_audio = np.sin(2 * np.pi * 65.41 * t) * bass_envelope * 0.15  # C2
    
    # Combine with base audio (50/50 mix)
    piano_track = 0.5 * base_audio + 0.5 * piano_audio
    bass_track = 0.7 * base_audio + 0.3 * bass_audio
    
    # Step 3: Professional processing with conservative settings
    processor_adv = AdvancedAudioProcessor(sr)
    
    # Very gentle processing to avoid artifacts
    piano_settings = {
        'eq': {'mid': 1.0, 'presence': 1.5},
        'compression': {'threshold': -20, 'ratio': 2.5},
        'gain': 0.6
    }
    
    bass_settings = {
        'eq': {'bass': 2.0, 'low_mid': -0.5},
        'compression': {'threshold': -15, 'ratio': 3},
        'gain': 0.5
    }
    
    master_settings = {
        'master_eq': {'presence': 1.0},
        'target_lufs': -20.0  # Conservative mastering
    }
    
    print("🎚️ Applying gentle professional processing...")
    tracks = [
        (piano_track, piano_settings),
        (bass_track, bass_settings)
    ]
    
    final_mix = processor_adv.mix_tracks(tracks, master_settings)
    
    # Step 4: Extra clean output processing
    # Remove any DC offset
    final_mix = final_mix - np.mean(final_mix)
    
    # Gentle limiting to prevent artifacts
    final_mix = np.tanh(final_mix * 0.8) * 0.8
    
    # Convert to int16 with conservative scaling
    final_mix_int16 = (final_mix * 28000).astype(np.int16)
    
    # Save result
    output_path = 'outputs/generated_music/phase2_improved_demo.wav'
    scipy.io.wavfile.write(output_path, sr, final_mix_int16)
    
    print(f"✅ Improved demo created: {output_path}")
    print(f"🎵 Duration: {len(final_mix)/sr:.1f}s")  
    print(f"🔊 Peak level: {np.max(np.abs(final_mix)):.3f}")
    print(f"🎯 Clean processing without synthesis artifacts!")

if __name__ == "__main__":
    create_improved_demo()