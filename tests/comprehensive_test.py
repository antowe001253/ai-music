#!/usr/bin/env python3
"""
Quick Test Suite for Steps 6-8 Implementation
"""

def run_all_tests():
    print("🧪 COMPREHENSIVE TEST: Steps 6-8 Implementation")
    print("=" * 60)
    
    results = []
    
    # Test 1: MusicGen Generation
    print("\n1️⃣ Testing MusicGen (Step 6)...")
    try:
        from transformers import MusicgenForConditionalGeneration, AutoProcessor
        import scipy.io.wavfile
        import numpy as np
        
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        
        inputs = processor(text=["happy pop song"], padding=True, return_tensors="pt")
        audio_values = model.generate(**inputs, max_new_tokens=256)
        
        # Save test output
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        test_file = "/Users/alexntowe/Projects/AI/Diff-SVC/comprehensive_test_output.wav"
        scipy.io.wavfile.write(test_file, sampling_rate, audio_data)
        
        print(f"✅ MusicGen: Generated {len(audio_data)/sampling_rate:.1f}s audio")
        results.append(("MusicGen", True, test_file))
        
    except Exception as e:
        print(f"❌ MusicGen failed: {e}")
        results.append(("MusicGen", False, None))
        test_file = None
    
    # Test 2: Audio Processing
    print("\n2️⃣ Testing Audio Processing (Step 7)...")
    if test_file:
        try:
            from audio_processing_suite import AudioProcessingSuite
            
            processor = AudioProcessingSuite()
            audio, sr = processor.load_audio(test_file)
            
            # Quick tests
            tempo_info = processor.detect_tempo_and_beats(audio, sr)
            key_info = processor.detect_key(audio, sr)
            
            print(f"✅ Audio Processing: BPM={tempo_info['bpm']:.1f}, Key={key_info['key']}")
            results.append(("Audio Processing", True, None))
            
        except Exception as e:
            print(f"❌ Audio Processing failed: {e}")
            results.append(("Audio Processing", False, None))
    else:
        print("⏭️  Skipping audio processing (no test file)")
        results.append(("Audio Processing", False, "No input file"))
    
    # Test 3: Melody System
    print("\n3️⃣ Testing Melody System (Step 8)...")
    try:
        from melody_generation_system import MelodyGenerationSystem
        
        melody_system = MelodyGenerationSystem()
        print("✅ Melody System: Initialized successfully")
        results.append(("Melody System", True, None))
        
    except Exception as e:
        print(f"❌ Melody System failed: {e}")
        results.append(("Melody System", False, None))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    for name, success, extra in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if extra:
            print(f"     → {extra}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Steps 6-8 implementation is working!")
        print("🚀 Ready to proceed to Step 9 (Diff-SVC integration)")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Check the errors above.")

if __name__ == "__main__":
    run_all_tests()