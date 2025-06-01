#!/usr/bin/env python3
"""
Modal Cloud CPU-Only Music Generation - Final Stable Version
Uses CPU to avoid CUDA device-side assert issues completely
"""

import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("diff-svc-cpu-stable")

# Create CPU-focused image (no GPU complications)
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
)

@app.function(
    image=image,
    cpu=8.0,  # Use CPU instead of GPU
    memory=32000,  # More RAM for CPU processing
    timeout=3600,  # Longer timeout for CPU
    volumes={}
)
def generate_music_cpu_stable(prompt: str, duration: int = 30):
    """
    CPU-only version that avoids all CUDA issues
    """
    import torch
    import numpy as np
    import scipy.io.wavfile as wavfile
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    from pathlib import Path
    
    work_dir = Path("/tmp/cpu_music")
    work_dir.mkdir(exist_ok=True)
    
    try:
        print(f"🖥️ CPU-ONLY music generation starting...")
        print(f"📝 Prompt: {prompt}")
        print(f"⏱️ Duration: {duration}s")
        print(f"🔧 Using CPU to avoid CUDA issues")
        
        # Force CPU usage
        device = "cpu"
        
        # Use the most stable model
        model_name = "facebook/musicgen-small"
        print(f"📥 Loading {model_name} on CPU...")
        
        # Load everything on CPU explicitly
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None  # Don't auto-assign device
        )
        
        # Ensure model is on CPU
        model = model.to("cpu")
        model.eval()  # Set to evaluation mode
        
        print(f"✅ Model loaded on CPU")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # Simple, safe prompt
        clean_prompt = prompt.replace("song", "music").lower()
        if len(clean_prompt.split()) > 3:
            clean_prompt = " ".join(clean_prompt.split()[:3])
        
        print(f"🎵 Using prompt: '{clean_prompt}'")
        
        # Process input
        inputs = processor(
            text=[clean_prompt],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=64  # Short input
        )
        
        # Ensure inputs are on CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        print("📊 Input processed")
        for key, tensor in inputs.items():
            print(f"  {key}: {tensor.shape}")
        
        # Very conservative generation for CPU
        sample_rate = model.config.audio_encoder.sampling_rate
        max_new_tokens = min(256, int(duration * 16))  # Very conservative for CPU
        
        print(f"🎵 Generating {max_new_tokens} tokens at {sample_rate}Hz on CPU...")
        print("⏳ This may take a few minutes on CPU...")
        
        # CPU generation with minimal parameters
        with torch.no_grad():
            torch.set_num_threads(8)  # Use multiple CPU threads
            
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=250,
                num_beams=1,
                pad_token_id=processor.tokenizer.eos_token_id,  # Safe fallback
                use_cache=True,
                guidance_scale=3.0  # Reasonable guidance
            )
        
        print("✅ CPU generation completed!")
        
        # Process generated audio
        if audio_values is not None and len(audio_values) > 0:
            audio_np = audio_values[0, 0].cpu().numpy()
            
            print(f"🔍 Generated audio shape: {audio_np.shape}")
            print(f"🔍 Audio range: [{audio_np.min():.6f}, {audio_np.max():.6f}]")
            
            # Check for issues
            if np.isnan(audio_np).any():
                print("⚠️ Found NaN values, cleaning...")
                audio_np = np.nan_to_num(audio_np)
            
            if np.isinf(audio_np).any():
                print("⚠️ Found Inf values, cleaning...")
                audio_np = np.nan_to_num(audio_np)
            
            # Normalize safely
            max_abs = np.max(np.abs(audio_np))
            if max_abs > 0:
                audio_np = audio_np / max_abs * 0.9
            else:
                print("⚠️ Generated silent audio")
            
            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Save audio
            output_path = work_dir / "cpu_generated_music.wav"
            wavfile.write(str(output_path), sample_rate, audio_int16)
            
            # Read file
            with open(output_path, 'rb') as f:
                audio_data = f.read()
            
            actual_duration = len(audio_np) / sample_rate
            print(f"💾 Audio saved: {len(audio_data)} bytes, {actual_duration:.1f}s")
            
            return {
                "success": True,
                "prompt": prompt,
                "clean_prompt": clean_prompt,
                "model": model_name,
                "device": "cpu",
                "files": {"audio": audio_data},
                "metadata": {
                    "sample_rate": sample_rate,
                    "duration": actual_duration,
                    "file_size": len(audio_data),
                    "tokens_generated": max_new_tokens,
                    "method": "cpu_stable"
                }
            }
        else:
            return {"success": False, "error": "No audio generated"}
            
    except Exception as e:
        print(f"❌ CPU generation error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.local_entrypoint()
def main(prompt: str = "Christmas carol song", duration: int = 30):
    """CPU-stable version entrypoint"""
    print("🖥️ Running CPU-STABLE version (no CUDA issues!)")
    print("=" * 60)
    
    result = generate_music_cpu_stable.remote(prompt, duration)
    
    if result["success"]:
        print("🎉 CPU generation SUCCESS!")
        print(f"🎵 Model: {result['model']}")
        print(f"📝 Prompt: {result['clean_prompt']}")
        print(f"⏱️ Duration: {result['metadata']['duration']:.1f}s")
        print(f"💾 File size: {result['metadata']['file_size']} bytes")
        
        output_dir = Path("cpu_stable_output")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"christmas_carol_cpu.wav"
        output_path = output_dir / filename
        
        with open(output_path, 'wb') as f:
            f.write(result["files"]["audio"])
        
        print(f"💾 Christmas carol saved to: {output_path}")
        print("🎵 No more CUDA errors - CPU generation is stable!")
        print("\n🎊 SUCCESS! Your Christmas carol music is ready!")
        
    else:
        print(f"❌ CPU generation failed: {result['error']}")

if __name__ == "__main__":
    print("Run with: modal run modal_music_pipeline_cpu_stable.py")
