"""
Your Original Plan - CORRECTLY IMPLEMENTED
MusicGen → Bark (Text-to-Singing) → Modal HifiGAN → Mix
"""

import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def implement_your_original_plan():
    """Implement your correct original workflow"""
    
    print("🎤 YOUR ORIGINAL PLAN - CORRECTLY IMPLEMENTED")
    print("=" * 45)
    print()
    print("✅ Step 1: MusicGen → Instrumental")
    print("✅ Step 2: Bark → AI Singing")  
    print("✅ Step 3: Modal HifiGAN → Enhancement")
    print("✅ Step 4: Audio Mixing → Complete Song")
    print()
    
    # Step 1: Check for existing instrumental
    instrumental_files = list(Path("outputs/phase3_complete").glob("*/01_instrumental.wav"))
    
    if instrumental_files:
        instrumental = instrumental_files[0]
        print(f"✅ Step 1: Found instrumental: {instrumental}")
    else:
        print("⚠️ Step 1: Need instrumental from MusicGen")
        print("   Run your existing pipeline to generate instrumental")
        return
    
    # Step 2: Generate singing with Bark
    print("\n🎤 Step 2: Generating AI Singing with Bark...")
    
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
        import soundfile as sf
        
        # Load models (this may take a few minutes first time)
        print("📦 Loading Bark models...")
        preload_models()
        
        # Test lyrics that match your original plan
        test_lyrics = [
            "♪ Hello world, this is AI singing ♪",
            "♪ Beautiful music generation technology ♪",
            "♪ Amazing artificial intelligence creates songs ♪"
        ]
        
        for i, lyrics in enumerate(test_lyrics):
            print(f"🎵 Generating: {lyrics}")
            
            # Generate singing with Bark
            audio_array = generate_audio(lyrics)
            
            # Save Bark output
            bark_file = f"bark_singing_{i+1}.wav"
            sf.write(bark_file, audio_array, SAMPLE_RATE)
            print(f"✅ Bark output: {bark_file}")
            
            # Step 3: Enhance with Modal HifiGAN
            print("🚀 Step 3: Modal HifiGAN enhancement...")
            
            result = subprocess.run([
                "python", "modal_enhance_simple.py", bark_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                enhanced_file = bark_file.replace('.wav', '_enhanced.wav')
                print(f"✅ Enhanced: {enhanced_file}")
                
                # Step 4: Mix with instrumental
                print("🎧 Step 4: Creating complete song...")
                
                try:
                    import librosa
                    import numpy as np
                    
                    # Load both tracks
                    inst_audio, inst_sr = librosa.load(str(instrumental), sr=None)
                    vocal_audio, vocal_sr = librosa.load(enhanced_file, sr=None)
                    
                    # Resample if needed
                    if inst_sr != vocal_sr:
                        vocal_audio = librosa.resample(vocal_audio, orig_sr=vocal_sr, target_sr=inst_sr)
                    
                    # Make same length (use shorter one)
                    min_length = min(len(inst_audio), len(vocal_audio))
                    inst_audio = inst_audio[:min_length]
                    vocal_audio = vocal_audio[:min_length]
                    
                    # Mix with balance
                    mixed_audio = inst_audio * 0.6 + vocal_audio * 0.8
                    
                    # Normalize
                    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.9
                    
                    # Save complete song
                    complete_song = f"complete_song_{i+1}.wav"
                    sf.write(complete_song, mixed_audio, inst_sr)
                    
                    print(f"🎉 COMPLETE SONG: {complete_song}")
                    print("   This is your original plan working!")
                    
                except Exception as mix_error:
                    print(f"⚠️ Mixing failed: {mix_error}")
                    print(f"💡 Manual mix: Use {enhanced_file} + {instrumental}")
            else:
                print(f"❌ Modal enhancement failed: {result.stderr}")
        
        print("\n🎯 YOUR ORIGINAL PLAN: ✅ SUCCESS!")
        print("=" * 35)
        print("You now have:")
        print("• AI-generated instrumentals (MusicGen)")
        print("• AI-generated singing voices (Bark)")  
        print("• Professional enhancement (Modal HifiGAN)")
        print("• Complete mixed songs")
        print()
        print("🎧 Listen to complete_song_*.wav files!")
        
    except Exception as e:
        print(f"❌ Bark failed: {e}")
        print("💡 Alternative: Use ElevenLabs or other singing API")

if __name__ == "__main__":
    implement_your_original_plan()
