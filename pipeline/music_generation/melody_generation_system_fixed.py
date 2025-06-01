#!/usr/bin/env python3
"""
FIXED Melody Generation System for Automated Music Pipeline
Uses stable model to avoid the issues we discovered
"""

import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import librosa
import numpy as np
import scipy.io.wavfile
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from audio_processing import AudioProcessingSuite
except ImportError:
    print("Warning: AudioProcessingSuite not found, some features may be limited")
    AudioProcessingSuite = None

class MelodyGenerationSystem:
    def __init__(self, device="auto"):
        """
        Initialize melody generation system with FIXED model selection
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        self.device = self._get_device(device)
        self.melody_model = None
        self.melody_processor = None
        
        if AudioProcessingSuite:
            self.audio_processor = AudioProcessingSuite()
        else:
            self.audio_processor = None
        
    def _get_device(self, device):
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_melody_model(self):
        """Load MusicGen model - FIXED to use stable version"""
        print(f"Loading STABLE MusicGen model on {self.device}...")
        
        # Try models in order of stability
        models_to_try = [
            "facebook/musicgen-small",     # Most stable
            "facebook/musicgen-medium",    # Backup option
            "facebook/musicgen-melody"     # Original (problematic)
        ]
        
        for model_name in models_to_try:
            try:
                print(f"🔄 Trying {model_name}...")
                
                self.melody_processor = AutoProcessor.from_pretrained(model_name)
                self.melody_model = MusicgenForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Stable dtype
                    low_cpu_mem_usage=True      # Memory optimization
                )
                
                # Check for vocab mismatch before moving to device
                if hasattr(self.melody_processor.tokenizer, 'vocab_size') and hasattr(self.melody_model.config, 'text_encoder'):
                    tokenizer_vocab = self.melody_processor.tokenizer.vocab_size
                    model_vocab = self.melody_model.config.text_encoder.vocab_size
                    
                    if tokenizer_vocab != model_vocab:
                        print(f"⚠️ Vocab mismatch detected in {model_name}: {tokenizer_vocab} vs {model_vocab}")
                        if model_name != "facebook/musicgen-small":
                            print("🔄 Trying next model...")
                            continue
                        else:
                            print("🔧 Applying vocab fix...")
                            # Apply vocab fix for small model - EXACT same as session_508d302d
                            while len(self.melody_processor.tokenizer) < model_vocab:
                                self.melody_processor.tokenizer.add_tokens([f'<extra_token_{len(self.melody_processor.tokenizer)}>'])
                
                # Move to device if not CPU
                if self.device != "cpu":
                    self.melody_model = self.melody_model.to(self.device)
                
                print(f"✅ {model_name} loaded successfully on {self.device}")
                self.current_model = model_name
                return True
                
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
                if model_name == models_to_try[-1]:
                    print("❌ All models failed to load")
                    return False
                continue
        
        return False
    
    def generate_melody_conditioned(self, text_description, melody_audio=None, duration=10):
        """
        Generate melody-conditioned music with FIXED parameters
        
        Args:
            text_description: Text description of the desired music
            melody_audio: Optional melody audio to condition on
            duration: Duration in seconds
        """
        if self.melody_model is None:
            print("❌ Melody model not loaded. Call load_melody_model() first.")
            return None
        
        try:
            print(f"🎵 Generating music: '{text_description}'")
            
            # Simplified input processing to avoid tokenization issues
            safe_description = text_description.lower().replace("song", "music")
            if len(safe_description.split()) > 5:
                safe_description = " ".join(safe_description.split()[:5])
            
            # Process input
            inputs = self.melody_processor(
                text=[safe_description],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use standard parameters that produce good audio quality
            sample_rate = self.melody_model.config.audio_encoder.sampling_rate
            print(f"📊 Generating audio at {sample_rate}Hz for ~{duration}s")
            
            with torch.no_grad():
                # Use simple, standard generation parameters that work
                audio_values = self.melody_model.generate(
                    **inputs,
                    max_new_tokens=1500,  # EXACT same as session_508d302d
                    do_sample=True,
                    temperature=1.0,
                    top_k=250,
                    top_p=0.0,
                    guidance_scale=3.0,  # This was in session_508d302d
                    num_beams=1,
                    pad_token_id=self.melody_processor.tokenizer.pad_token_id or self.melody_processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Process generated audio
            audio_np = audio_values[0, 0].cpu().numpy()
            
            # Normalize
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / np.max(np.abs(audio_np)) * 0.9
            
            print(f"✅ Generated {len(audio_np)/sample_rate:.1f}s of audio")
            
            return {
                'audio': audio_np,
                'sample_rate': sample_rate,
                'description': safe_description,
                'model': self.current_model,
                'duration': len(audio_np) / sample_rate
            }
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_generated_audio(self, audio_data, output_path, sample_rate=None):
        """Save generated audio to file"""
        if audio_data is None:
            return False
        
        try:
            if isinstance(audio_data, dict):
                audio_np = audio_data['audio']
                sample_rate = audio_data['sample_rate']
            else:
                audio_np = audio_data
                sample_rate = sample_rate or 32000
            
            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Save
            scipy.io.wavfile.write(str(output_path), sample_rate, audio_int16)
            print(f"💾 Audio saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False

# Test function
if __name__ == "__main__":
    print("🧪 Testing Fixed Melody Generation System...")
    
    system = MelodyGenerationSystem()
    
    if system.load_melody_model():
        print("✅ Model loaded successfully!")
        
        # Test generation
        result = system.generate_melody_conditioned("Christmas carol music", duration=15)
        
        if result:
            print("✅ Generation successful!")
            
            # Save test output
            output_path = Path("test_fixed_melody.wav")
            if system.save_generated_audio(result, output_path):
                print(f"🎵 Test audio saved to: {output_path}")
        else:
            print("❌ Generation failed")
    else:
        print("❌ Model loading failed")
