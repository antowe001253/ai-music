"""
Generate Singing with Full Sentences
Create proper lyrical content instead of just letter sounds
"""

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
import sys

def create_sentence_singing(melody_file: str, lyrics: str, output_file: str):
    """Create singing with full sentence structure"""
    
    print(f"🎤 Creating singing with lyrics: '{lyrics}'")
    
    # Load melody
    audio, sr = librosa.load(melody_file, sr=24000)
    
    # Parse lyrics into words and phonemes
    words = lyrics.lower().split()
    print(f"📝 Words: {words}")
    
    # Estimate phonemes per word (simplified)
    phoneme_mapping = {
        'hello': ['h', 'e', 'l', 'o'],
        'world': ['w', 'er', 'l', 'd'],
        'this': ['th', 'i', 's'],
        'is': ['i', 's'], 
        'a': ['a'],
        'test': ['t', 'e', 's', 't'],
        'of': ['o', 'f'],
        'singing': ['s', 'i', 'ng', 'i', 'ng'],
        'voice': ['v', 'oy', 's'],
        'ai': ['a', 'i'],
        'music': ['m', 'u', 's', 'i', 'k'],
        'generation': ['j', 'e', 'n', 'e', 'r', 'a', 'sh', 'un'],
        'beautiful': ['b', 'yu', 't', 'i', 'f', 'ul'],
        'song': ['s', 'o', 'ng'],
        'melody': ['m', 'e', 'l', 'o', 'd', 'y'],
        'computer': ['k', 'om', 'p', 'yu', 't', 'er'],
        'amazing': ['a', 'm', 'a', 'z', 'i', 'ng'],
        'technology': ['t', 'e', 'k', 'n', 'o', 'l', 'o', 'j', 'y']
    }
    
    # Build phoneme sequence
    all_phonemes = []
    for word in words:
        if word in phoneme_mapping:
            all_phonemes.extend(phoneme_mapping[word])
        else:
            # Fallback: split word into letters
            all_phonemes.extend(list(word))
        
        # Add slight pause between words
        all_phonemes.append('_')  # Silence marker
    
    print(f"🔤 Phonemes: {all_phonemes}")
    
    # Create detailed vowel and consonant sounds
    sound_library = {
        # Vowels with formant frequencies [F1, F2, F3]
        'a': [730, 1090, 2440],    # "ah" in "father"
        'e': [530, 1840, 2480],    # "eh" in "bed"  
        'i': [270, 2290, 3010],    # "ee" in "beat"
        'o': [570, 840, 2410],     # "oh" in "boat"
        'u': [300, 870, 2240],     # "oo" in "boot"
        'er': [490, 1350, 1690],   # "er" in "bird"
        'oy': [400, 1000, 2540],   # "oy" in "boy"
        'yu': [250, 1750, 2600],   # "you"
        'om': [500, 1000, 2500],   # "om" 
        
        # Consonants (approximate formant regions)
        'h': [500, 1500, 2500],    # Breathy
        'l': [360, 1300, 2500],    # Liquid
        'w': [300, 610, 2150],     # Glide
        'd': [400, 1700, 2600],    # Stop
        'th': [400, 1600, 2200],   # Fricative
        's': [200, 1400, 2800],    # Sibilant
        'f': [300, 1400, 2500],    # Fricative  
        'v': [300, 1400, 2500],    # Voiced fricative
        't': [400, 1800, 2600],    # Stop
        'n': [400, 1500, 2500],    # Nasal
        'ng': [280, 1300, 2500],   # Velar nasal
        'sh': [300, 1800, 2400],   # "sh"
        'j': [300, 1500, 2500],    # "j"
        'r': [400, 1300, 1600],    # "r"
        'k': [400, 1800, 2600],    # "k"  
        'm': [400, 1300, 2500],    # "m"
        'b': [400, 1400, 2600],    # "b"
        'p': [400, 1400, 2600],    # "p"
        'g': [400, 1600, 2600],    # "g"
        'z': [300, 1600, 2800],    # "z"
        'y': [300, 2200, 3000],    # "y"
        '_': [0, 0, 0]             # Silence
    }
    
    # Generate audio for each phoneme
    n_phonemes = len(all_phonemes)
    phoneme_duration = len(audio) / n_phonemes
    
    singing_audio = np.zeros_like(audio)
    
    for i, phoneme in enumerate(all_phonemes):
        start_idx = int(i * phoneme_duration)
        end_idx = int((i + 1) * phoneme_duration)
        
        if start_idx >= len(audio):
            break
            
        # Get the melody segment for this phoneme
        melody_segment = audio[start_idx:end_idx]
        
        if phoneme == '_':
            # Silence between words
            phoneme_audio = np.zeros_like(melody_segment) 
            # Add subtle breath sound
            if len(melody_segment) > 0:
                breath = np.random.normal(0, 0.02, len(melody_segment))
                # Filter to breath frequencies
                nyquist = sr / 2
                b, a = signal.butter(2, [100/nyquist, 1000/nyquist], btype='band')
                breath_filtered = signal.filtfilt(b, a, breath)
                phoneme_audio = breath_filtered * 0.3
        else:
            # Get formants for this phoneme
            formants = sound_library.get(phoneme, sound_library['a'])
            
            # Generate formant synthesis
            phoneme_audio = np.zeros_like(melody_segment)
            
            for j, formant_freq in enumerate(formants):
                if formant_freq > 0 and formant_freq < sr/2 - 100:
                    # Create resonance around formant frequency
                    bandwidth = 80 + j * 20  # Wider bandwidth for higher formants
                    low_freq = max(formant_freq - bandwidth, 50)
                    high_freq = min(formant_freq + bandwidth, sr/2 - 50)
                    
                    # Apply bandpass filter
                    nyquist = sr / 2
                    low = low_freq / nyquist
                    high = high_freq / nyquist
                    
                    if 0 < low < high < 1:
                        try:
                            b, a = signal.butter(4, [low, high], btype='band')
                            formant_signal = signal.filtfilt(b, a, melody_segment)
                            
                            # Weight formants (F1 strongest, F2 medium, F3 weakest)
                            weights = [1.0, 0.7, 0.4]
                            weight = weights[j] if j < len(weights) else 0.2
                            
                            phoneme_audio += formant_signal * weight
                        except:
                            # Fallback
                            phoneme_audio += melody_segment * 0.3
            
            # Add phoneme-specific characteristics
            if phoneme in ['s', 'sh', 'f', 'th']:
                # Add noise for fricatives
                noise = np.random.normal(0, 0.1, len(melody_segment))
                b, a = signal.butter(2, [2000/nyquist, 8000/nyquist], btype='band')
                fricative_noise = signal.filtfilt(b, a, noise)
                phoneme_audio += fricative_noise * 0.2
            
            elif phoneme in ['m', 'n', 'ng']:
                # Nasal resonance
                b, a = signal.butter(2, [200/nyquist, 500/nyquist], btype='band')
                nasal_resonance = signal.filtfilt(b, a, melody_segment)
                phoneme_audio += nasal_resonance * 0.4
        
        # Apply envelope (natural attack/decay)
        if len(phoneme_audio) > 10:
            attack_len = min(len(phoneme_audio) // 8, 50)
            decay_len = min(len(phoneme_audio) // 6, 80)
            
            envelope = np.ones_like(phoneme_audio)
            
            # Attack
            if attack_len > 0:
                envelope[:attack_len] = np.linspace(0.1, 1, attack_len)
            
            # Decay
            if decay_len > 0:
                envelope[-decay_len:] = np.linspace(1, 0.2, decay_len)
            
            phoneme_audio *= envelope
        
        # Place in final audio
        end_idx = min(end_idx, len(singing_audio))
        singing_audio[start_idx:end_idx] = phoneme_audio[:end_idx-start_idx]
    
    # Add overall vocal characteristics
    
    # 1. Vibrato
    t = np.linspace(0, len(singing_audio)/sr, len(singing_audio))
    vibrato_rate = 5.5  # Hz
    vibrato_depth = 0.03
    vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    singing_audio = singing_audio * vibrato
    
    # 2. Overall formant enhancement
    # Boost vocal frequencies
    nyquist = sr / 2
    b, a = signal.butter(2, [200/nyquist, 4000/nyquist], btype='band')
    singing_audio = signal.filtfilt(b, a, singing_audio)
    
    # 3. Add subtle reverb (room ambience)
    reverb_delay = int(0.03 * sr)  # 30ms delay
    if len(singing_audio) > reverb_delay:
        reverb = np.zeros_like(singing_audio)
        reverb[reverb_delay:] = singing_audio[:-reverb_delay] * 0.15
        singing_audio = singing_audio + reverb
    
    # Normalize
    singing_audio = singing_audio / np.max(np.abs(singing_audio) + 1e-8) * 0.8
    
    # Save result
    sf.write(output_file, singing_audio, sr)
    print(f"✅ Sentence singing saved: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Test sentences
    test_sentences = [
        "Hello world, this is a test",
        "AI singing voice generation", 
        "Beautiful music from computer",
        "Amazing technology singing song"
    ]
    
    # Find melody file
    melody_files = list(Path("outputs/phase3_complete").glob("*/02_vocal_melody.wav"))
    if not melody_files:
        print("❌ No melody files found")
        exit(1)
    
    melody_file = melody_files[0]
    print(f"📁 Using melody: {melody_file}")
    
    for i, sentence in enumerate(test_sentences):
        print(f"\n🎤 Creating sentence {i+1}: '{sentence}'")
        
        output_file = f"sentence_singing_{i+1}.wav"
        create_sentence_singing(str(melody_file), sentence, output_file)
        
        # Enhance with Modal HifiGAN
        print("🚀 Enhancing with Modal HifiGAN...")
        import subprocess
        result = subprocess.run([
            "python", "modal_enhance_simple.py", output_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            enhanced_file = output_file.replace('.wav', '_enhanced.wav')
            print(f"✅ Enhanced: {enhanced_file}")
            print(f"🎧 Listen to: {enhanced_file}")
        else:
            print(f"⚠️ Enhancement failed: {result.stderr}")
    
    print(f"\n🎉 Generated {len(test_sentences)} sentence-based singing files!")
    print("🎧 These should sound more like actual singing with words!")
