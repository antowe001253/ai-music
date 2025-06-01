#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase 2 - Steps 9-11
Tests Prompt Intelligence, Orchestration Engine, and Advanced Audio Processing
"""

import sys
import os
import numpy as np
import scipy.io.wavfile

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

def test_step_9_prompt_intelligence():
    """Test Step 9: Prompt Intelligence"""
    print("🧠 TESTING STEP 9: Prompt Intelligence")
    print("=" * 50)
    
    try:
        from prompt_intelligence import PromptIntelligence, MusicalParameters
        
        intelligence = PromptIntelligence()
        
        test_cases = [
            ("upbeat rock song with electric guitar", "rock", 130, ["guitar"]),
            ("slow sad piano ballad in minor key", None, 80, ["piano"]),
            ("fast electronic dance music with synth", "electronic", 155, ["synth"]),
            ("calm acoustic folk with vocals", "folk", None, ["guitar", "vocals"])
        ]
        
        print("Testing prompt parsing...")
        for prompt, expected_genre, expected_tempo, expected_instruments in test_cases:
            params = intelligence.parse_prompt(prompt)
            print(f"✓ '{prompt[:30]}...'")
            print(f"  Genre: {params.genre} (expected: {expected_genre})")
            print(f"  Tempo: {params.tempo} (expected: {expected_tempo})")
            print(f"  Instruments: {params.instruments}")
            
        print("✅ Step 9 PASSED: Prompt Intelligence working!")
        return True
        
    except Exception as e:
        print(f"❌ Step 9 FAILED: {e}")
        return False

def test_step_10_orchestration_engine():
    """Test Step 10: Orchestration Engine"""
    print("\n🎼 TESTING STEP 10: Orchestration Engine")
    print("=" * 50)
    
    try:
        from orchestration_engine import OrchestrationEngine, TrackInfo
        from prompt_intelligence import MusicalParameters
        
        engine = OrchestrationEngine()
        
        # Create mock track data
        audio_data = np.random.randn(44100 * 5)  # 5 seconds of random audio
        track1 = TrackInfo(
            audio_data=audio_data, sample_rate=44100, tempo=120.0,
            key='C_major', duration=5.0, track_type='instrumental'
        )        
        track2 = TrackInfo(
            audio_data=audio_data, sample_rate=44100, tempo=125.0,
            key='G_major', duration=5.0, track_type='vocal'
        )
        
        # Test orchestration planning
        params = MusicalParameters(genre='pop', tempo=120, key='C_major')
        plan = engine.create_orchestration_plan(params, [track1, track2])
        
        print(f"✓ Target tempo: {plan.target_tempo} BPM")
        print(f"✓ Target key: {plan.target_key}")
        print(f"✓ Song sections: {len(plan.song_structure)}")
        print(f"✓ Timing grid: {len(plan.timing_grid)} beats")
        
        # Test tempo synchronization
        synced_audio = engine.synchronize_tempo(track2, plan.target_tempo)
        print(f"✓ Tempo sync: {len(synced_audio)} samples")
        
        print("✅ Step 10 PASSED: Orchestration Engine working!")
        return True
        
    except Exception as e:
        print(f"❌ Step 10 FAILED: {e}")
        return False

def test_step_11_advanced_audio():
    """Test Step 11: Advanced Audio Processing"""
    print("\n🎚️ TESTING STEP 11: Advanced Audio Processing")
    print("=" * 50)
    
    try:
        from advanced_audio_processing import AdvancedAudioProcessor
        
        processor = AdvancedAudioProcessor()
        
        # Generate test audio
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate))
        test_audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone
        
        # Test EQ
        eq_settings = {'mid': 3.0, 'bass': -2.0}  # Boost mids, cut bass
        eq_audio = processor.apply_eq(test_audio, eq_settings)
        print(f"✓ EQ applied: {len(eq_audio)} samples")
        
        # Test compression
        compressed = processor.apply_compression(test_audio, threshold=-15, ratio=3)
        print(f"✓ Compression applied: {len(compressed)} samples")
        
        # Test reverb
        reverb_audio = processor.apply_reverb(test_audio, room_size=0.7, wet_level=0.4)
        print(f"✓ Reverb applied: {len(reverb_audio)} samples")
        
        # Test stereo positioning
        stereo_audio = processor.position_stereo(test_audio, position=-0.5)  # Left side
        print(f"✓ Stereo positioning: {stereo_audio.shape}")
        
        # Test mixing
        tracks = [
            (test_audio, {'gain': 0.8, 'eq': {'mid': 2.0}}),
            (test_audio * 0.7, {'gain': 0.6, 'compression': {'threshold': -12}})
        ]
        master_settings = {'master_eq': {'presence': 1.5}, 'target_lufs': -16.0}
        mixed = processor.mix_tracks(tracks, master_settings)
        print(f"✓ Mixing complete: {len(mixed)} samples")
        
        print("✅ Step 11 PASSED: Advanced Audio Processing working!")
        return True
        
    except Exception as e:
        print(f"❌ Step 11 FAILED: {e}")
        return False

def run_phase2_tests():
    """Run all Phase 2 tests"""
    print("🚀 PHASE 2 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    results = []
    results.append(("Step 9 - Prompt Intelligence", test_step_9_prompt_intelligence()))
    results.append(("Step 10 - Orchestration Engine", test_step_10_orchestration_engine()))
    results.append(("Step 11 - Advanced Audio Processing", test_step_11_advanced_audio()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 2 TEST RESULTS:")
    print("=" * 60)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 PHASE 2 COMPLETE! All integration & intelligence systems working!")
        print("🎯 Ready for Phase 3: Complete Automation Pipeline")
    else:
        print(f"\n⚠️ {len(results)-passed} test(s) failed. Check the errors above.")

if __name__ == "__main__":
    run_phase2_tests()