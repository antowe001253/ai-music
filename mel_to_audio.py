"""
Convert Diff-SVC Mel-spectrogram to Audio using Modal HifiGAN
This takes the saved mel-spectrogram and converts it to high-quality singing voice
"""

import numpy as np
import librosa
import soundfile as sf
import json
import tempfile
import base64
import sys
from pathlib import Path

# Add the modal enhancement
sys.path.append(str(Path(__file__).parent))
from modal_enhance_simple import SimpleModalEnhancer

def mel_to_audio_with_modal(mel_file: str, output_file: str):
    """Convert Diff-SVC mel-spectrogram to audio using Modal HifiGAN"""
    
    print(f"🎤 Converting Diff-SVC mel-spectrogram to audio...")
    print(f"📁 Mel file: {mel_file}")
    print(f"📁 Output: {output_file}")
    
    # Load mel-spectrogram and metadata
    mel_numpy = np.load(mel_file)
    meta_file = mel_file.replace('_mel.npy', '_meta.json')
    
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Mel shape: {metadata['shape']}")
    print(f"🎵 Sample rate: {metadata['sample_rate']}Hz")
    
    # Remove batch dimension if present
    if len(mel_numpy.shape) == 3:
        mel_numpy = mel_numpy[0]  # Remove batch dimension
    
    # Convert mel-spectrogram to audio using Griffin-Lim first (as intermediate)
    try:
        print("🎼 Converting mel to audio with Griffin-Lim...")
        
        # Convert from log scale
        mel_linear = np.exp(mel_numpy.T)  # Transpose and convert from log
        
        # Create mel filter bank
        mel_basis = librosa.filters.mel(
            sr=metadata['sample_rate'],
            n_fft=1024,
            n_mels=metadata['n_mels'],
            fmin=0,
            fmax=metadata['sample_rate']//2
        )
        
        # Approximate linear spectrogram
        linear_spec = np.dot(np.linalg.pinv(mel_basis), mel_linear)
        
        # Griffin-Lim reconstruction
        audio_gl = librosa.griffinlim(
            linear_spec,
            hop_length=metadata.get('hop_length', 256),
            win_length=1024,
            n_iter=64  # More iterations for better quality
        )
        
        # Save intermediate Griffin-Lim result
        gl_file = output_file.replace('.wav', '_griffinlim.wav')
        sf.write(gl_file, audio_gl, metadata['sample_rate'])
        print(f"✅ Griffin-Lim audio saved: {gl_file}")
        
        # Now enhance with Modal HifiGAN
        print("🚀 Enhancing with Modal HifiGAN...")
        enhancer = SimpleModalEnhancer()
        enhanced_file = enhancer.enhance_audio_file(gl_file, output_file)
        
        print(f"🎉 Final enhanced audio: {enhanced_file}")
        return enhanced_file
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        
        # Fallback: try direct mel-spectrogram enhancement
        print("🔄 Trying direct mel enhancement...")
        try:
            # Create a simple audio from mel using basic ISTFT
            # This is a simplified approach
            mel_db = librosa.power_to_db(np.exp(mel_numpy.T))
            audio_simple = librosa.feature.inverse.mel_to_audio(
                mel_db,
                sr=metadata['sample_rate'],  
                hop_length=metadata.get('hop_length', 256),
                n_fft=1024
            )
            
            # Save and enhance
            temp_file = output_file.replace('.wav', '_temp.wav')
            sf.write(temp_file, audio_simple, metadata['sample_rate'])
            
            # Enhance with Modal
            enhancer = SimpleModalEnhancer()
            enhanced_file = enhancer.enhance_audio_file(temp_file, output_file)
            
            print(f"🎉 Enhanced audio (fallback method): {enhanced_file}")
            return enhanced_file
            
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mel_to_audio.py mel_file.npy [output.wav]")
        exit(1)
    
    mel_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else mel_file.replace('_mel.npy', '_final_singing.wav')
    
    if not Path(mel_file).exists():
        print(f"❌ Mel file not found: {mel_file}")
        exit(1)
    
    try:
        result = mel_to_audio_with_modal(mel_file, output_file)
        print(f"🎤 SUCCESS! Your Diff-SVC singing voice is ready: {result}")
        print("🎧 Play this file to hear the AI-generated singing voice!")
    except Exception as e:
        print(f"❌ Failed: {e}")
