#!/usr/bin/env python3
"""
Test script for MusicGen functionality using transformers
"""

import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile
import numpy as np
import os

def test_musicgen():
    """Test MusicGen model functionality"""
    print("Testing MusicGen functionality...")
    
    try:
        # Load the processor and model
        print("Loading MusicGen model and processor...")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        
        print(f"✅ Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Test basic generation
        print("\nTesting basic music generation...")
        inputs = processor(
            text=["upbeat techno"],
            padding=True,
            return_tensors="pt",
        )
        
        # Generate audio (shorter duration for testing)
        print("Generating audio... (this may take a moment)")
        audio_values = model.generate(**inputs, max_new_tokens=256)        
        print(f"✅ Generated audio shape: {audio_values.shape}")
        
        # Save the audio
        sampling_rate = model.config.audio_encoder.sampling_rate
        output_path = "/Users/alexntowe/Projects/AI/Diff-SVC/test_musicgen_output.wav"
        
        # Convert to numpy and normalize
        audio_data = audio_values[0, 0].cpu().numpy()
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        scipy.io.wavfile.write(output_path, sampling_rate, audio_data)
        print(f"✅ Audio saved to: {output_path}")
        print(f"Sample rate: {sampling_rate} Hz")
        print(f"Duration: {len(audio_data) / sampling_rate:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing MusicGen: {str(e)}")
        return False

def get_model_info():
    """Get information about available MusicGen models"""
    print("\nAvailable MusicGen models:")
    models = [
        "facebook/musicgen-small",      # ~300MB, fastest
        "facebook/musicgen-medium",     # ~1.5GB
        "facebook/musicgen-large",      # ~3.3GB
        "facebook/musicgen-melody",     # ~3.3GB, melody-conditioned
    ]
    
    for model in models:
        print(f"  - {model}")

if __name__ == "__main__":
    print("🎵 MusicGen Test Script")
    print("=" * 50)
    
    get_model_info()
    
    # Test with the smallest model first
    success = test_musicgen()
    
    if success:
        print("\n✅ MusicGen is working correctly!")
        print("You can now use MusicGen for music generation in your pipeline.")
    else:
        print("\n❌ MusicGen test failed. Check the error messages above.")