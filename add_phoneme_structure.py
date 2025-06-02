"""
Add Phoneme Structure to Make Vocals Intelligible
Transform the vocal-like sounds into more structured singing
"""

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import sys
from pathlib import Path

def add_phoneme_structure(input_audio: str, output_audio: str, lyrics: str = "la la la"):
    """Add phoneme-like structure to existing vocal audio"""
    
    print(f"🎤 Adding phoneme structure to: {input_audio}")
    print(f"📝 Lyrics pattern: {lyrics}")
    
    # Load the vocal audio
    audio, sr = librosa.load(input_audio, sr=24000)
    
    # Create syllable timing based on lyrics
    syllables = lyrics.replace(" ", "").lower()
    n_syllables = len(syllables)
    
    if n_syllables == 0:
        syllables = "lalala"
        n_syllables = 6
    
    # Divide audio into syllable segments
    segment_length = len(audio) // n_syllables
    
    structured_audio = np.zeros_like(audio)
    
    # Define vowel formant frequencies
    vowel_formants = {
        'a': [730, 1090, 2440],   # "ah"
        'e': [530, 1840, 2480],   # "eh" 
        'i': [270, 2290, 3010],   # "ee"
        'o': [570, 840, 2410],    # "oh"
        'u': [300, 870, 2240],    # "oo"
        'l': [400, 1200, 2600],   # "l" sound
    }
    
    print(f"🔤 Processing {n_syllables} syllables...")
    
    for i, char in enumerate(syllables):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, len(audio))
        
        if start_idx < len(audio):
            # Get segment
            segment = audio[start_idx:end_idx]
            
            # Get formants for this character
            formants = vowel_formants.get(char, vowel_formants['a'])
            
            # Apply formant filtering
            enhanced_segment = np.zeros_like(segment)
            
            for j, formant_freq in enumerate(formants):
                if formant_freq < sr // 2 - 100:
                    # Create bandpass filter around formant
                    low_freq = max(formant_freq - 80, 50)
                    high_freq = min(formant_freq + 80, sr // 2 - 50)
                    
                    # Normalize frequencies
                    nyquist = sr / 2
                    low = low_freq / nyquist
                    high = high_freq / nyquist
                    
                    if low < high < 1.0:
                        try:
                            b, a = signal.butter(4, [low, high], btype='band')
                            formant_signal = signal.filtfilt(b, a, segment)
                            
                            # Weight formants (first formant strongest)
                            weight = 1.0 / (j + 1)
                            enhanced_segment += formant_signal * weight
                        except:
                            # Fallback if filter fails
                            enhanced_segment += segment * 0.5
            
            # Add consonant-like articulation for 'l'
            if char == 'l':
                # Add slight noise burst at beginning
                noise_length = min(len(enhanced_segment) // 10, 200)
                noise = np.random.normal(0, 0.1, noise_length)
                
                # Filter noise to consonant frequencies
                b, a = signal.butter(4, [2000 / nyquist, 6000 / nyquist], btype='band')
                consonant_noise = signal.filtfilt(b, a, noise)
                
                enhanced_segment[:noise_length] += consonant_noise * 0.3
            
            # Apply amplitude envelope (attack-sustain-release)
            envelope = np.ones_like(enhanced_segment)
            attack_len = len(enhanced_segment) // 10
            release_len = len(enhanced_segment) // 5
            
            # Attack
            if attack_len > 0:
                envelope[:attack_len] = np.linspace(0, 1, attack_len)
            
            # Release  
            if release_len > 0:
                envelope[-release_len:] = np.linspace(1, 0.1, release_len)
            
            enhanced_segment *= envelope
            
            # Place in output
            structured_audio[start_idx:end_idx] = enhanced_segment
    
    # Add overall vocal characteristics
    # Breath noise between syllables
    for i in range(n_syllables - 1):
        breath_start = (i + 1) * segment_length - segment_length // 20
        breath_end = (i + 1) * segment_length + segment_length // 20
        
        if breath_start > 0 and breath_end < len(structured_audio):
            breath_length = breath_end - breath_start
            breath = np.random.normal(0, 0.05, breath_length)
            
            # Filter breath to natural frequencies
            nyquist = sr / 2
            b, a = signal.butter(2, [100 / nyquist, 1000 / nyquist], btype='band')
            breath_filtered = signal.filtfilt(b, a, breath)
            
            structured_audio[breath_start:breath_end] += breath_filtered * 0.2
    
    # Overall smoothing
    b, a = signal.butter(2, 8000 / (sr / 2), btype='low')
    structured_audio = signal.filtfilt(b, a, structured_audio)
    
    # Normalize
    structured_audio = structured_audio / np.max(np.abs(structured_audio)) * 0.8
    
    # Save result
    sf.write(output_audio, structured_audio, sr)
    print(f"✅ Structured vocal saved: {output_audio}")
    
    return output_audio

if __name__ == "__main__":
    # Test with your best vocal output
    input_files = [
        "better_diffsvc_output_proper_audio_method1_enhanced.wav",
        "diffsvc_output_with_mel_proper_audio_method1_enhanced.wav"
    ]
    
    lyrics_tests = [
        "hello world",
        "la la la la",
        "singing voice test"
    ]
    
    for input_file in input_files:
        if Path(input_file).exists():
            print(f"\n🎤 Processing: {input_file}")
            
            for i, lyrics in enumerate(lyrics_tests):
                output_file = input_file.replace('.wav', f'_structured_{i+1}.wav')
                add_phoneme_structure(input_file, output_file, lyrics)
                
                # Enhance with Modal HifiGAN
                print(f"🚀 Enhancing structured version...")
                import subprocess
                result = subprocess.run([
                    "python", "modal_enhance_simple.py", output_file
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    enhanced_file = output_file.replace('.wav', '_enhanced.wav')
                    print(f"✅ Enhanced structured vocal: {enhanced_file}")
            
            break  # Process first available file
