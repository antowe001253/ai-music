"""
Proper Diff-SVC Mel-spectrogram to Audio Conversion
Uses the correct Diff-SVC processing pipeline
"""

import numpy as np
import torch
import librosa
import soundfile as sf
import json
import sys
from pathlib import Path

def convert_diffsvc_mel_properly(mel_file: str, output_file: str):
    """Convert Diff-SVC mel-spectrogram using proper parameters"""
    
    print(f"🎤 Converting Diff-SVC mel-spectrogram properly...")
    
    # Load mel-spectrogram and metadata
    mel_numpy = np.load(mel_file)
    meta_file = mel_file.replace('_mel.npy', '_meta.json')
    
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Original mel shape: {mel_numpy.shape}")
    
    # Remove batch dimension and transpose to correct format
    if len(mel_numpy.shape) == 3:
        mel_numpy = mel_numpy[0]  # Shape: [time, mels]
    
    # Diff-SVC uses specific parameters
    sr = 24000  # Diff-SVC sample rate
    hop_length = 128  # Diff-SVC hop length  
    win_length = 512  # Diff-SVC window length
    n_fft = 1024
    n_mels = 128
    
    print(f"🎼 Using Diff-SVC parameters:")
    print(f"   Sample rate: {sr}Hz")
    print(f"   Hop length: {hop_length}")
    print(f"   Mel bins: {n_mels}")
    
    try:
        # Method 1: Use librosa's mel inversion with correct parameters
        print("🔄 Method 1: Librosa mel inversion...")
        
        # Transpose mel to [mels, time] format for librosa
        mel_transposed = mel_numpy.T
        
        # Convert from log scale to linear
        mel_linear = np.exp(mel_transposed)
        
        # Use librosa's mel inversion
        audio_reconstructed = librosa.feature.inverse.mel_to_audio(
            mel_linear,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window='hann',
            center=True,
            pad_mode='reflect',
            power=1.0,  # Magnitude spectrogram
            n_iter=32,
            length=None
        )
        
        # Normalize audio
        audio_reconstructed = audio_reconstructed / np.max(np.abs(audio_reconstructed)) * 0.8
        
        # Save result
        method1_file = output_file.replace('.wav', '_method1.wav')
        sf.write(method1_file, audio_reconstructed, sr)
        print(f"✅ Method 1 saved: {method1_file}")
        
        return method1_file
        
    except Exception as e1:
        print(f"❌ Method 1 failed: {e1}")
        
        try:
            # Method 2: Manual mel filter inversion
            print("🔄 Method 2: Manual mel filter inversion...")
            
            # Create mel filter bank
            mel_basis = librosa.filters.mel(
                sr=sr,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=0,
                fmax=sr//2,
                htk=False
            )
            
            # Convert mel to linear spectrogram
            mel_linear = np.exp(mel_numpy.T)  # Convert from log and transpose
            
            # Pseudo-inverse to get approximate magnitude spectrogram
            mag_spec = np.dot(np.linalg.pinv(mel_basis), mel_linear)
            
            # Ensure positive values
            mag_spec = np.maximum(mag_spec, 0.01)
            
            # Use Griffin-Lim with proper parameters
            audio_gl = librosa.griffinlim(
                mag_spec,
                n_iter=60,  # More iterations
                hop_length=hop_length,
                win_length=win_length,
                window='hann',
                center=True,
                dtype=np.float32,
                length=None,
                pad_mode='reflect',
                momentum=0.99,
                init='random',
                random_state=None
            )
            
            # Normalize
            audio_gl = audio_gl / np.max(np.abs(audio_gl)) * 0.8
            
            method2_file = output_file.replace('.wav', '_method2.wav')
            sf.write(method2_file, audio_gl, sr)
            print(f"✅ Method 2 saved: {method2_file}")
            
            return method2_file
            
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            
            # Method 3: Simple approach
            print("🔄 Method 3: Simple reconstruction...")
            
            # Very basic conversion
            mel_simple = np.exp(mel_numpy.T)
            
            # Create a simple harmonic series
            n_frames = mel_simple.shape[1]
            audio_length = n_frames * hop_length
            audio_simple = np.zeros(audio_length)
            
            # Generate audio from mel bins
            for mel_bin in range(min(64, n_mels)):  # Use lower mel bins
                freq = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr//2)[mel_bin]
                
                # Generate sine wave for this frequency
                t = np.arange(audio_length) / sr
                sine_wave = np.sin(2 * np.pi * freq * t)
                
                # Modulate amplitude based on mel values
                amplitude_envelope = np.repeat(mel_simple[mel_bin], hop_length)[:audio_length]
                amplitude_envelope = amplitude_envelope / np.max(amplitude_envelope + 1e-8)
                
                audio_simple += sine_wave * amplitude_envelope * 0.01
            
            # Normalize
            audio_simple = audio_simple / np.max(np.abs(audio_simple) + 1e-8) * 0.5
            
            method3_file = output_file.replace('.wav', '_method3.wav')
            sf.write(method3_file, audio_simple, sr)
            print(f"✅ Method 3 saved: {method3_file}")
            
            return method3_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python proper_mel_convert.py mel_file.npy [output.wav]")
        exit(1)
    
    mel_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else mel_file.replace('_mel.npy', '_proper_audio.wav')
    
    if not Path(mel_file).exists():
        print(f"❌ Mel file not found: {mel_file}")
        exit(1)
    
    try:
        result = convert_diffsvc_mel_properly(mel_file, output_file)
        print(f"🎤 Conversion complete: {result}")
        print("🎧 Test this audio file!")
        
        # Now enhance with Modal HifiGAN
        print("\n🚀 Enhancing with Modal HifiGAN...")
        import subprocess
        enhanced_result = subprocess.run([
            "python", "modal_enhance_simple.py", result
        ], capture_output=True, text=True)
        
        if enhanced_result.returncode == 0:
            enhanced_file = result.replace('.wav', '_enhanced.wav')
            print(f"✅ Modal enhancement complete: {enhanced_file}")
            print("🎤 This should be your best quality singing voice!")
        else:
            print(f"⚠️ Modal enhancement failed: {enhanced_result.stderr}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
