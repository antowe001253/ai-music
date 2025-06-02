"""
Proper AI Singing Voice Generation
Using text-to-singing models instead of voice conversion
"""

import subprocess
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_singing_with_bark(lyrics: str, melody_file: str, output_file: str):
    """Generate singing using Bark TTS with singing capability"""
    
    logger.info(f"🎤 Generating singing with Bark TTS...")
    
    try:
        # Install bark if not present
        subprocess.run(["pip", "install", "bark"], check=True, capture_output=True)
        
        from bark import SAMPLE_RATE, generate_audio, preload_models
        from scipy.io.wavfile import write as write_wav
        
        # Download/cache the model
        logger.info("📦 Loading Bark models...")
        preload_models()
        
        # Create singing prompt
        singing_prompt = f"♪ {lyrics} ♪"
        
        logger.info(f"🎵 Generating: {singing_prompt}")
        
        # Generate audio
        audio_array = generate_audio(singing_prompt)
        
        # Save the result
        write_wav(output_file, SAMPLE_RATE, audio_array)
        
        logger.info(f"✅ Bark singing saved: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Bark failed: {e}")
        return None

def generate_singing_with_eleven_labs():
    """Guide for using ElevenLabs for singing"""
    
    print("""
🎤 ElevenLabs AI Singing (Commercial Option):

1. Go to elevenlabs.io
2. Sign up for account
3. Use their AI singing feature
4. Upload your melody
5. Add lyrics
6. Generate singing voice

This will give you professional-quality singing voice.
    """)

def generate_singing_with_suno():
    """Guide for using Suno AI"""
    
    print("""
🎵 Suno AI (Free Option):

1. Go to suno.ai  
2. Create account
3. Input lyrics + style description
4. Generate complete songs with singing

Example prompt: "Pop ballad with emotional female vocals, lyrics: [your lyrics]"
    """)

def create_simple_singing_synthesis(lyrics: str, melody_file: str, output_file: str):
    """Create simple singing synthesis as proof of concept"""
    
    logger.info("🎼 Creating simple singing synthesis...")
    
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        from scipy import signal
        
        # Load melody
        melody, sr = librosa.load(melody_file, sr=22050)
        
        # Create phoneme-like segments based on lyrics
        words = lyrics.split()
        segments_per_word = len(melody) // (len(words) * sr)  # Rough timing
        
        vocal_synth = np.zeros_like(melody)
        
        # Create different vowel sounds for each word
        vowel_formants = {
            'A': [730, 1090, 2440],  # "ah" sound
            'E': [270, 2290, 3010],  # "eh" sound  
            'I': [390, 1990, 2550],  # "ih" sound
            'O': [570, 840, 2410],   # "oh" sound
            'U': [440, 1020, 2240],  # "uh" sound
        }
        
        vowels = ['A', 'E', I', 'O', 'U']
        
        for i, word in enumerate(words):
            start_idx = i * segments_per_word * sr
            end_idx = min((i + 1) * segments_per_word * sr, len(melody))
            
            if start_idx < len(melody):
                # Choose vowel based on word
                vowel = vowels[i % len(vowels)]
                formants = vowel_formants[vowel]
                
                # Create formant synthesis for this segment
                segment = melody[start_idx:end_idx]
                vowel_sound = np.zeros_like(segment)
                
                for formant_freq in formants:
                    # Create resonance at formant frequency
                    nyquist = sr / 2
                    if formant_freq < nyquist - 200:
                        low = (formant_freq - 50) / nyquist
                        high = (formant_freq + 50) / nyquist
                        
                        b, a = signal.butter(4, [low, high], btype='band')
                        formant_signal = signal.filtfilt(b, a, segment)
                        vowel_sound += formant_signal * 0.3
                
                vocal_synth[start_idx:end_idx] = vowel_sound
        
        # Add vocal characteristics
        # Vibrato
        t = np.linspace(0, len(vocal_synth)/sr, len(vocal_synth))
        vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
        vocal_synth = vocal_synth * vibrato
        
        # Normalize
        vocal_synth = vocal_synth / np.max(np.abs(vocal_synth)) * 0.8
        
        # Save result
        sf.write(output_file, vocal_synth, sr)
        
        logger.info(f"✅ Simple singing synthesis saved: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Simple synthesis failed: {e}")
        return None

if __name__ == "__main__":
    lyrics = "Hello world, this is a test, singing voice generation"
    melody_file = "outputs/phase3_complete/session_1748797646/02_vocal_melody.wav"
    
    print("🎤 AI Singing Voice Generation Options:")
    print()
    
    # Try Bark
    bark_output = generate_singing_with_bark(lyrics, melody_file, "bark_singing.wav")
    
    if bark_output:
        print(f"✅ Bark singing generated: {bark_output}")
        
        # Enhance with Modal HifiGAN
        print("🚀 Enhancing with Modal HifiGAN...")
        subprocess.run(["python", "modal_enhance_simple.py", bark_output])
    else:
        print("❌ Bark not available")
    
    # Create simple version
    simple_output = create_simple_singing_synthesis(lyrics, melody_file, "simple_singing.wav")
    
    if simple_output:
        print(f"✅ Simple singing generated: {simple_output}")
        
        # Enhance with Modal HifiGAN  
        subprocess.run(["python", "modal_enhance_simple.py", simple_output])
    
    # Show commercial options
    generate_singing_with_eleven_labs()
    generate_singing_with_suno()
