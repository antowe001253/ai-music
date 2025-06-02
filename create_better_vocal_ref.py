"""
Create Better Vocal Reference for Diff-SVC
Generate vocal-like input that Diff-SVC can properly process
"""

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import sys

def create_realistic_vocal_reference(melody_file: str, output_file: str):
    """Create a realistic vocal reference from melody"""
    
    print(f"🎤 Creating realistic vocal reference...")
    
    # Load melody
    audio, sr = librosa.load(melody_file, sr=24000)
    
    print(f"📊 Input audio: {len(audio)} samples at {sr}Hz")
    
    # Extract harmonic content (vocal-like)
    harmonic, percussive = librosa.effects.hpss(audio, margin=8.0)
    
    # Create vocal formants (simulate human vocal tract)
    vocal_synth = np.zeros_like(harmonic)
    
    # Generate multiple vocal formants
    formant_freqs = [400, 800, 1200, 2400, 3200]  # Typical vowel formants
    formant_gains = [1.0, 0.8, 0.6, 0.4, 0.3]
    
    for freq, gain in zip(formant_freqs, formant_gains):
        # Create bandpass filter for each formant
        nyquist = sr / 2
        low = max(freq - 100, 50) / nyquist
        high = min(freq + 100, nyquist - 100) / nyquist
        
        if low < high and high < 1.0:
            b, a = signal.butter(4, [low, high], btype='band')
            formant_signal = signal.filtfilt(b, a, harmonic)
            vocal_synth += formant_signal * gain
    
    # Add pitch modulation (vibrato)
    t = np.linspace(0, len(vocal_synth)/sr, len(vocal_synth))
    vibrato_rate = 5.0  # Hz
    vibrato_depth = 0.01  # 1% pitch variation
    
    # Create vibrato modulation
    vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    
    # Apply amplitude modulation
    vocal_with_vibrato = vocal_synth * vibrato
    
    # Add breath-like texture
    breath_noise = np.random.normal(0, 0.02, len(vocal_with_vibrato))
    b, a = signal.butter(2, [200/nyquist, 4000/nyquist], btype='band')
    breath_filtered = signal.filtfilt(b, a, breath_noise)
    
    # Combine vocal elements
    final_vocal = vocal_with_vibrato + breath_filtered * 0.1
    
    # Add dynamics (volume changes like singing)
    envelope_freq = 0.8  # Slow changes
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * envelope_freq * t)
    final_vocal = final_vocal * envelope
    
    # Normalize
    final_vocal = final_vocal / np.max(np.abs(final_vocal)) * 0.8
    
    # Save the vocal reference
    sf.write(output_file, final_vocal, sr)
    print(f"✅ Realistic vocal reference saved: {output_file}")
    
    return output_file

def test_with_new_reference():
    """Test the complete pipeline with better vocal reference"""
    
    # Find a melody file
    melody_files = list(Path("outputs/phase3_complete").glob("*/02_vocal_melody.wav"))
    if not melody_files:
        print("❌ No melody files found")
        return
    
    melody_file = melody_files[0]
    print(f"📁 Using melody: {melody_file}")
    
    # Create better vocal reference
    better_vocal = create_realistic_vocal_reference(str(melody_file), "better_vocal_reference.wav")
    
    # Test with Diff-SVC
    print("\n🎤 Testing with Diff-SVC...")
    import subprocess
    
    result = subprocess.run([
        "python", "fixed_infer.py", 
        better_vocal, 
        "better_diffsvc_output.wav"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Diff-SVC completed!")
        
        # Convert mel to audio
        print("🔄 Converting mel to audio...")
        mel_result = subprocess.run([
            "python", "proper_mel_convert.py",
            "better_diffsvc_output_mel.npy"
        ], capture_output=True, text=True)
        
        if mel_result.returncode == 0:
            print("🎉 Complete pipeline test successful!")
            print("🎧 Check these files:")
            print("   - better_vocal_reference.wav (input)")
            print("   - better_diffsvc_output_proper_audio_method1.wav (converted)")
            print("   - better_diffsvc_output_proper_audio_method1_enhanced.wav (enhanced)")
        else:
            print(f"❌ Mel conversion failed: {mel_result.stderr}")
    else:
        print(f"❌ Diff-SVC failed: {result.stderr}")

if __name__ == "__main__":
    from pathlib import Path
    
    if len(sys.argv) > 1:
        melody_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "better_vocal_reference.wav"
        create_realistic_vocal_reference(melody_file, output_file)
    else:
        test_with_new_reference()
