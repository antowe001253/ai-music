#!/usr/bin/env python3
"""
Phase 2 Integration Demo
Demonstrates the complete intelligence and orchestration pipeline
"""

import sys
import os
import numpy as np
import scipy.io.wavfile
from pathlib import Path

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

def run_integration_demo():
    """Demonstrate the complete Phase 2 pipeline"""
    print("🎼 PHASE 2 INTEGRATION DEMO")
    print("=" * 60)
    print("Demonstrating: Prompt → Intelligence → Orchestration → Processing")
    print()
    
    # Step 1: Parse user prompt
    from prompt_intelligence import PromptIntelligence
    
    user_prompt = "upbeat pop song with piano, guitar, and vocals in C major at 120 BPM"
    print(f"🎤 User Prompt: '{user_prompt}'")
    
    intelligence = PromptIntelligence()
    params = intelligence.parse_prompt(user_prompt)
    
    print(f"🧠 Extracted Parameters:")
    print(f"  • Genre: {params.genre}")
    print(f"  • Tempo: {params.tempo} BPM")
    print(f"  • Mood: {params.mood}")
    print(f"  • Key: {params.key}")
    print(f"  • Instruments: {', '.join(params.instruments)}")
    print()
    
    # Step 2: Create mock tracks (simulating MusicGen output)
    from orchestration_engine import OrchestrationEngine, TrackInfo
    
    sample_rate = 44100
    duration = 8.0  # 8 seconds
    samples = int(duration * sample_rate)
    
    # Generate synthetic tracks
    t = np.linspace(0, duration, samples)
    
    # Piano track (C major chord progression)
    piano_audio = (np.sin(2*np.pi*261.63*t) + np.sin(2*np.pi*329.63*t) + np.sin(2*np.pi*392.00*t)) * 0.3
    piano_track = TrackInfo(
        audio_data=piano_audio, sample_rate=sample_rate, tempo=120.0,
        key='C_major', duration=duration, track_type='piano'
    )
    
    # Guitar track (slightly detuned for richness)
    guitar_audio = np.sin(2*np.pi*196.00*t) * 0.4 * np.exp(-t/4)  # G note with decay
    guitar_track = TrackInfo(
        audio_data=guitar_audio, sample_rate=sample_rate, tempo=118.0,
        key='C_major', duration=duration, track_type='guitar'
    )
    
    # Vocal track (melody line)
    vocal_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C major scale
    vocal_audio = np.zeros_like(t)
    for i, freq in enumerate(vocal_freqs):
        start_idx = int(i * samples // len(vocal_freqs))
        end_idx = int((i + 1) * samples // len(vocal_freqs))
        vocal_audio[start_idx:end_idx] = np.sin(2*np.pi*freq*t[start_idx:end_idx]) * 0.5
    
    vocal_track = TrackInfo(
        audio_data=vocal_audio, sample_rate=sample_rate, tempo=121.0,
        key='C_major', duration=duration, track_type='vocal'
    )
    
    tracks = [piano_track, guitar_track, vocal_track]
    print(f"🎹 Created {len(tracks)} synthetic tracks:")
    for i, track in enumerate(tracks):
        print(f"  • Track {i+1}: {track.track_type} ({track.tempo} BPM, {track.key})")
    print()
    
    # Step 3: Orchestration planning
    engine = OrchestrationEngine()
    plan = engine.create_orchestration_plan(params, tracks)
    
    print(f"🎼 Orchestration Plan:")
    print(f"  • Target Tempo: {plan.target_tempo} BPM")
    print(f"  • Target Key: {plan.target_key}")
    print(f"  • Song Structure: {len(plan.song_structure)} sections")
    for section, (start, end) in list(plan.song_structure.items())[:4]:
        print(f"    - {section}: {start}s - {end}s")
    print(f"  • Timing Grid: {len(plan.timing_grid)} beat markers")
    print()
    
    # Step 4: Synchronize tracks
    print("🔄 Synchronizing tracks...")
    synced_tracks = []
    for track in tracks:
        synced_audio = engine.synchronize_tempo(track, plan.target_tempo)
        synced_tracks.append((synced_audio, track.track_type))
        print(f"  • {track.track_type}: {track.tempo} → {plan.target_tempo} BPM")
    print()
    
    # Step 5: Advanced processing and mixing
    from advanced_audio_processing import AdvancedAudioProcessor
    
    processor = AdvancedAudioProcessor(sample_rate)
    
    print("🎚️ Applying advanced processing...")
    
    # Define processing for each track type
    track_processing = [
        (synced_tracks[0][0], {  # Piano
            'eq': {'low_mid': 2.0, 'presence': 1.5},
            'compression': {'threshold': -18, 'ratio': 3},
            'reverb': {'room_size': 0.4, 'wet_level': 0.2},
            'gain': 0.7
        }),
        (synced_tracks[1][0], {  # Guitar  
            'eq': {'mid': 1.0, 'high_mid': 2.0},
            'compression': {'threshold': -15, 'ratio': 4},
            'reverb': {'room_size': 0.6, 'wet_level': 0.3},
            'gain': 0.6
        }),
        (synced_tracks[2][0], {  # Vocals
            'eq': {'presence': 3.0, 'brilliance': 1.5},
            'compression': {'threshold': -12, 'ratio': 6},
            'reverb': {'room_size': 0.3, 'wet_level': 0.15},
            'gain': 0.8
        })
    ]
    
    # Master settings
    master_settings = {
        'master_eq': {'presence': 2.0, 'brilliance': 1.0},
        'master_compression': {'threshold': -6, 'ratio': 2},
        'target_lufs': -16.0
    }
    
    # Mix everything
    final_mix = processor.mix_tracks(track_processing, master_settings)
    
    print("  • Piano: EQ (low-mid +2dB, presence +1.5dB), Compression 3:1, Reverb")
    print("  • Guitar: EQ (mid +1dB, high-mid +2dB), Compression 4:1, Reverb")  
    print("  • Vocals: EQ (presence +3dB, brilliance +1.5dB), Compression 6:1, Light Reverb")
    print("  • Master: EQ (presence +2dB), Compression 2:1, Mastering to -16 LUFS")
    print()
    
    # Step 6: Save result
    output_path = "../outputs/generated_music/phase2_integration_demo.wav"
    output_full_path = os.path.join(os.path.dirname(__file__), output_path)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
    
    # Normalize and save
    final_mix_normalized = np.clip(final_mix, -1.0, 1.0)
    final_mix_int16 = (final_mix_normalized * 32767).astype(np.int16)
    
    scipy.io.wavfile.write(output_full_path, sample_rate, final_mix_int16)
    
    print(f"🎵 Final Result:")
    print(f"  • Duration: {len(final_mix) / sample_rate:.1f} seconds")
    print(f"  • Sample Rate: {sample_rate} Hz")
    print(f"  • Dynamic Range: {np.max(final_mix) - np.min(final_mix):.3f}")
    print(f"  • Output File: {output_path}")
    print()
    
    print("✅ PHASE 2 INTEGRATION DEMO COMPLETE!")
    print("🎯 Successfully demonstrated:")
    print("   • Intelligent prompt parsing")
    print("   • Multi-track orchestration")
    print("   • Tempo and key synchronization")
    print("   • Professional audio processing")
    print("   • Advanced mixing and mastering")

if __name__ == "__main__":
    run_integration_demo()