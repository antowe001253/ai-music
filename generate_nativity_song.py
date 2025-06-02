#!/usr/bin/env python3
"""
Generate a singing voice about the birth of Jesus
Uses DiffSVC for melody generation + RVC for voice conversion
"""

import numpy as np
import torch
import librosa
import soundfile as sf
from modal_rvc_service import ModalRVCClient
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_nativity_melody():
    """Create a Christmas/Nativity melody"""
    
    # Christmas carol-inspired melody (Silent Night style)
    # Notes: C4, D4, E4, F4, G4, A4, B4, C5
    notes = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25
    }
    
    # Nativity song melody - "Silent Night" inspired
    melody_sequence = [
        ('G4', 1.5), ('A4', 0.5), ('G4', 1.0),  # "Silent night"
        ('E4', 2.0),                             # "holy"
        ('G4', 1.5), ('A4', 0.5), ('G4', 1.0),  # "night, all is"
        ('E4', 2.0),                             # "calm"
        
        ('D4', 1.5), ('D4', 0.5), ('B4', 2.0),  # "all is bright"
        ('C5', 1.5), ('C5', 0.5), ('G4', 2.0),  # "round yon virgin"
        
        ('A4', 1.5), ('A4', 0.5), ('C5', 1.0), ('B4', 0.5), ('A4', 0.5),  # "mother and child"
        ('G4', 1.5), ('A4', 0.5), ('G4', 1.0),   # "holy infant"
        ('E4', 3.0),                              # "so tender and mild"
        
        ('D4', 1.5), ('D4', 0.5), ('F4', 1.0), ('D4', 1.0),  # "sleep in heavenly"
        ('C4', 4.0),                              # "peace"
    ]
    
    sample_rate = 22050
    total_duration = sum(duration for _, duration in melody_sequence)
    
    # Generate sine wave melody
    audio = np.zeros(int(total_duration * sample_rate))
    current_time = 0
    
    for note, duration in melody_sequence:
        freq = notes[note]
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, False)
        
        # Create richer tone with harmonics
        wave = (np.sin(2 * np.pi * freq * t) * 0.6 +
                np.sin(2 * np.pi * freq * 2 * t) * 0.2 +
                np.sin(2 * np.pi * freq * 3 * t) * 0.1)
        
        # Apply envelope
        envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 8))
        wave *= envelope
        
        start_idx = int(current_time * sample_rate)
        end_idx = start_idx + samples
        audio[start_idx:end_idx] = wave
        current_time += duration
    
    # Add slight reverb effect
    from scipy import signal
    impulse_response = np.exp(-np.linspace(0, 2, sample_rate // 4)) * np.random.normal(0, 0.01, sample_rate // 4)
    audio = signal.convolve(audio, impulse_response, mode='same')
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sample_rate

def add_vocal_characteristics(audio, sr):
    """Add vocal formants and breathing to make it more voice-like"""
    
    # Add formant filtering (vocal tract simulation)
    from scipy import signal
    
    # Formant frequencies for vowel sounds
    formant1 = 800   # First formant
    formant2 = 1200  # Second formant
    formant3 = 2500  # Third formant
    
    # Create formant filters
    nyquist = sr / 2
    
    # Bandpass filters for formants
    for freq, gain in [(formant1, 1.2), (formant2, 1.0), (formant3, 0.8)]:
        b, a = signal.butter(2, [freq-100, freq+100], btype='band', fs=sr)
        formant_signal = signal.filtfilt(b, a, audio) * gain
        audio += formant_signal * 0.3
    
    # Add subtle vibrato
    t = np.linspace(0, len(audio)/sr, len(audio))
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
    audio *= vibrato
    
    # Add breath-like noise at phrase boundaries
    breath_positions = [0.2, 4.0, 8.0, 12.0, 16.0]  # seconds
    for pos in breath_positions:
        if pos * sr < len(audio):
            breath_length = int(0.1 * sr)  # 100ms breath
            breath = np.random.normal(0, 0.02, breath_length)
            breath *= np.exp(-np.linspace(0, 3, breath_length))
            
            start_idx = int(pos * sr)
            end_idx = min(start_idx + breath_length, len(audio))
            audio[start_idx:end_idx] += breath[:end_idx-start_idx]
    
    return audio

def generate_nativity_singing():
    """Generate complete nativity singing voice"""
    
    logger.info("🎵 Creating nativity melody...")
    
    # Create base melody
    melody, sr = create_nativity_melody()
    
    # Add vocal characteristics
    vocal_melody = add_vocal_characteristics(melody, sr)
    
    # Save base melody
    base_file = "nativity_base_melody.wav"
    sf.write(base_file, vocal_melody, sr)
    logger.info(f"💿 Base melody saved: {base_file}")
    
    # Convert with RVC for singing voice
    logger.info("🎤 Converting to singing voice with RVC...")
    
    client = ModalRVCClient()
    
    try:
        # Try different pitch shifts for different voice characters
        voice_variations = [
            ("nativity_singing_gentle.wav", 0),      # Original pitch
            ("nativity_singing_higher.wav", 3),      # Higher, more angelic
            ("nativity_singing_warm.wav", -2),       # Lower, warmer
            ("nativity_singing_child.wav", 5),       # Child-like voice
        ]
        
        results = []
        
        for output_file, pitch_shift in voice_variations:
            try:
                result = client.convert_voice(
                    input_file=base_file,
                    output_file=output_file,
                    pitch_shift=pitch_shift
                )
                results.append(result)
                logger.info(f"✅ Created: {output_file} (pitch: {pitch_shift:+d})")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to create {output_file}: {e}")
        
        logger.info(f"🎄 Generated {len(results)} nativity singing voices!")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ RVC conversion failed: {e}")
        return [base_file]  # Return base melody if RVC fails

if __name__ == "__main__":
    print("🌟 Generating Nativity Song - The Birth of Jesus 🌟")
    print("=" * 50)
    
    results = generate_nativity_singing()
    
    print("\n🎄 Nativity Song Generation Complete! 🎄")
    print("Generated files:")
    for i, file in enumerate(results, 1):
        print(f"  {i}. {file}")
    
    print("\n🎵 This melody tells the story of Jesus's birth")
    print("   in the style of 'Silent Night' - a peaceful,")
    print("   holy song celebrating the Nativity! 🌟")
