#!/usr/bin/env python3
"""
Modal Cloud Setup for Diff-SVC - FIXED Version
Handles vocab size mismatch properly
"""

import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("diff-svc-music-fixed")

# Create the container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "transformers>=4.35.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "accelerate>=0.20.0"
    ])
    .apt_install(["ffmpeg"])
    .env({"CUDA_LAUNCH_BLOCKING": "1"})
)

@app.function(
    image=image,
    gpu="A10G",
    memory=16000,
    timeout=1800,
    volumes={}
)
def generate_music_fixed(prompt: str, duration: int = 30):
    """
    Fixed version that handles vocab mismatch
    """
    import torch
    import numpy as np
    import scipy.io.wavfile as wavfile
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    from pathlib import Path
    
    work_dir = Path("/tmp/music_work")
    work_dir.mkdir(exist_ok=True)
    
    try:
        print(f"🚀 FIXED music generation starting...")
        print(f"📝 Prompt: {prompt}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Device: {device}")
        
        # Use a different model that doesn't have vocab mismatch
        print("📥 Loading AudioCraft MusicGen model...")
        
        # Try musicgen-medium which often has better compatibility
        model_name = "facebook/musicgen-medium"
        
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            ).to(device)
            
            print(f"✅ Model loaded: {model_name}")
            
        except Exception as model_error:
            print(f"❌ musicgen-medium failed: {model_error}")
            print("🔄 Trying musicgen-small...")
            
            # Fallback to small with manual vocab fix
            model_name = "facebook/musicgen-small"
            processor = AutoProcessor.from_pretrained(model_name)
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            
            # Manual vocab size fix for small model
            if hasattr(processor.tokenizer, 'vocab_size') and hasattr(model.config.text_encoder, 'vocab_size'):
                tokenizer_vocab = processor.tokenizer.vocab_size
                model_vocab = model.config.text_encoder.vocab_size
                
                if tokenizer_vocab != model_vocab:
                    print(f"🔧 Fixing vocab mismatch: {tokenizer_vocab} -> {model_vocab}")
                    
                    # Resize tokenizer to match model
                    processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    while len(processor.tokenizer) < model_vocab:
                        processor.tokenizer.add_tokens([f'<extra_token_{len(processor.tokenizer)}>'])
                    
                    print(f"✅ Vocab fixed: tokenizer now has {len(processor.tokenizer)} tokens")
            
            model = model.to(device)
        
        # Simple prompt to avoid tokenization issues
        safe_prompt = prompt.replace("song", "music").replace("vocal", "instrumental")
        if len(safe_prompt.split()) > 5:
            safe_prompt = " ".join(safe_prompt.split()[:5])  # Limit words
        
        print(f"🎵 Using prompt: '{safe_prompt}'")
        
        # Process input carefully
        inputs = processor(
            text=[safe_prompt],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=128  # Limit input length
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Conservative generation
        sample_rate = model.config.audio_encoder.sampling_rate
        max_new_tokens = min(512, int(duration * 32))  # Conservative token count
        
        print(f"📊 Generating {max_new_tokens} tokens at {sample_rate}Hz")
        
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=250,
                top_p=0.0,  # Disable nucleus sampling
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        print("✅ Generation successful!")
        
        # Process audio
        audio_np = audio_values[0, 0].cpu().numpy()
        
        # Normalize
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.95
        
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Save
        output_path = work_dir / "generated_music.wav"
        wavfile.write(str(output_path), sample_rate, audio_int16)
        
        with open(output_path, 'rb') as f:
            audio_data = f.read()
        
        print(f"💾 Audio saved: {len(audio_data)} bytes")
        
        return {
            "success": True,
            "prompt": prompt,
            "safe_prompt": safe_prompt,
            "model": model_name,
            "device": device,
            "files": {"audio": audio_data},
            "metadata": {
                "sample_rate": sample_rate,
                "duration": len(audio_np) / sample_rate,
                "file_size": len(audio_data)
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.local_entrypoint()
def main(prompt: str = "Christmas carol song", duration: int = 30):
    """Fixed version entrypoint"""
    print("🔧 Running VOCAB-FIXED version")
    print("=" * 50)
    
    result = generate_music_fixed.remote(prompt, duration)
    
    if result["success"]:
        print("✅ Fixed generation successful!")
        print(f"🎵 Model: {result['model']}")
        print(f"📝 Prompt: {result['safe_prompt']}")
        print(f"⏱️ Duration: {result['metadata']['duration']:.1f}s")
        
        output_dir = Path("fixed_output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "christmas_carol.wav", 'wb') as f:
            f.write(result["files"]["audio"])
        
        print(f"💾 Music saved to: {output_dir}/christmas_carol.wav")
        print("🎉 SUCCESS! The vocab mismatch issue has been resolved!")
        
    else:
        print(f"❌ Still failed: {result['error']}")

if __name__ == "__main__":
    print("Run with: modal run modal_music_pipeline_vocab_fixed.py")
