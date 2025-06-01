#!/usr/bin/env python3
"""
🎵 FRESH START: Simple Music Generation System
Clean implementation of Phase 1 & 2 from scratch

This script will:
1. Test basic music generation capabilities
2. Create working audio files
3. Build a foundation for Phase 3 integration
"""

import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
import time

class FreshMusicSystem:
    def __init__(self):
        self.device = self._get_device()
        self.sample_rate = 32000  # Standard rate
        self.output_dir = Path("outputs/fresh_generation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎵 Fresh Music System Initialized")
        print(f"🖥️ Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        
    def _get_device(self):
        """Get best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def test_basic_generation(self):
        """Test 1: Basic audio generation without AI models"""
        print("\n🧪 Test 1: Basic Audio Generation")
        print("=" * 35)
        
        try:
            # Generate a simple musical scale
            duration = 10  # seconds
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            
            # C major scale frequencies
            notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
            
            audio = np.zeros_like(t)
            note_duration = duration / len(notes)
            
            for i, freq in enumerate(notes):
                start_time = i * note_duration
                end_time = (i + 1) * note_duration
                
                start_idx = int(start_time * self.sample_rate)
                end_idx = int(end_time * self.sample_rate)
                
                if end_idx > len(t):
                    end_idx = len(t)
                
                note_t = t[start_idx:end_idx] - start_time
                note = 0.3 * np.sin(2 * np.pi * freq * note_t)
                
                # Add envelope
                envelope = np.exp(-2 * note_t / note_duration)
                note *= envelope
                
                audio[start_idx:end_idx] = note
            
            # Save test
            test_file = self.output_dir / "test1_basic_scale.wav"
            self._save_audio(audio, test_file)
            
            print(f"✅ Generated musical scale: {duration}s")
            print(f"💾 Saved: {test_file}")
            return True
            
        except Exception as e:
            print(f"❌ Basic generation failed: {e}")
            return False
    
    def test_musicgen_simple(self):
        """Test 2: Try simple MusicGen generation"""
        print("\n🧪 Test 2: Simple MusicGen Test")
        print("=" * 32)
        
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            print("📥 Loading MusicGen Small...")
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            
            if self.device != "cpu":
                model = model.to(self.device)
            
            print("✅ Model loaded successfully")
            
            # Very simple generation
            inputs = processor(
                text=["upbeat music"],
                return_tensors="pt"
            ).to(self.device)
            
            print("🎵 Generating...")
            start_time = time.time()
            
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=128,  # Very short
                    do_sample=True,
                    temperature=1.0,
                )
            
            generation_time = time.time() - start_time
            
            audio_np = audio_values[0, 0].cpu().numpy()
            duration = len(audio_np) / model.config.audio_encoder.sampling_rate
            
            print(f"✅ Generated: {duration:.1f}s in {generation_time:.1f}s")
            
            # Save
            test_file = self.output_dir / "test2_musicgen_simple.wav"
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / np.max(np.abs(audio_np)) * 0.9
            
            wavfile.write(
                str(test_file), 
                model.config.audio_encoder.sampling_rate,
                (audio_np * 32767).astype(np.int16)
            )
            
            print(f"💾 Saved: {test_file}")
            
            # Check if it's actual music or noise
            # Simple test: check if there's variation in the signal
            signal_variation = np.std(audio_np)
            if signal_variation > 0.01:
                print(f"🎵 Signal variation: {signal_variation:.4f} - Looks like music!")
                return True
            else:
                print(f"⚠️ Signal variation: {signal_variation:.4f} - Might be noise")
                return False
                
        except Exception as e:
            print(f"❌ MusicGen test failed: {e}")
            return False
    
    def test_christmas_carol_generation(self):
        """Test 3: Generate Christmas carol specifically"""
        print("\n🧪 Test 3: Christmas Carol Generation")
        print("=" * 37)
        
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Christmas carol prompts
            prompts = [
                "Christmas carol instrumental music",
                "gentle Christmas melody with bells",
                "traditional holiday music"
            ]
            
            results = []
            
            for i, prompt in enumerate(prompts):
                print(f"🎄 Generating: '{prompt}'")
                
                inputs = processor(
                    text=[prompt],
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=1.0,
                    )
                
                audio_np = audio_values[0, 0].cpu().numpy()
                duration = len(audio_np) / model.config.audio_encoder.sampling_rate
                
                # Save
                test_file = self.output_dir / f"test3_christmas_{i+1}.wav"
                if np.max(np.abs(audio_np)) > 0:
                    audio_np = audio_np / np.max(np.abs(audio_np)) * 0.9
                
                wavfile.write(
                    str(test_file),
                    model.config.audio_encoder.sampling_rate,
                    (audio_np * 32767).astype(np.int16)
                )
                
                # Analyze
                signal_variation = np.std(audio_np)
                is_music = signal_variation > 0.01
                
                print(f"  ✅ Generated: {duration:.1f}s")
                print(f"  🎵 Music quality: {'Good' if is_music else 'Poor'}")
                print(f"  💾 Saved: {test_file}")
                
                results.append(is_music)
            
            success_rate = sum(results) / len(results)
            print(f"\n📊 Christmas carol success: {success_rate*100:.1f}%")
            
            return success_rate >= 0.5  # At least 50% should be music
            
        except Exception as e:
            print(f"❌ Christmas carol test failed: {e}")
            return False
    
    def _save_audio(self, audio_np, file_path):
        """Save audio array to file"""
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.9
        
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(str(file_path), self.sample_rate, audio_int16)
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🎵" * 60)
        print("🎵 FRESH START: MUSIC GENERATION TESTS")
        print("🎵" * 60)
        
        tests = [
            ("Basic Audio Generation", self.test_basic_generation),
            ("MusicGen Simple Test", self.test_musicgen_simple),
            ("Christmas Carol Generation", self.test_christmas_carol_generation)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results[test_name] = False
        
        # Summary
        print(f"\n{'🎯' * 60}")
        print("🎯 TEST RESULTS SUMMARY")
        print(f"{'🎯' * 60}")
        
        passed = 0
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        print(f"\n📊 Overall: {passed}/{len(tests)} tests passed")
        
        if passed == len(tests):
            print("🎉 ALL TESTS PASSED! Ready for Phase 2!")
        elif passed >= len(tests) // 2:
            print("⚠️ PARTIAL SUCCESS - Some issues to resolve")
        else:
            print("❌ MAJOR ISSUES - Need troubleshooting")
        
        print(f"\n📁 Generated files in: {self.output_dir}")
        return results

if __name__ == "__main__":
    system = FreshMusicSystem()
    results = system.run_all_tests()
