#!/usr/bin/env python3
"""
Modal Cloud Setup - Mac Environment Replication
Tries to replicate the working Mac environment on Modal
"""

import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("diff-svc-mac-replica")

# Create image that matches typical Mac development environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.1.0",  # Pin to stable version
        "torchaudio==2.1.0",
        "transformers==4.35.2",  # Pin to stable version
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "scipy==1.11.4",
        "numpy==1.24.4",
        "accelerate==0.24.1"
    ])
    .apt_install(["ffmpeg"])
    .env({"PYTORCH_ENABLE_MPS_FALLBACK": "1"})  # Mac-like fallback
)

@app.function(
    image=image,
    gpu="T4",  # Try smaller, more stable GPU
    memory=16000,
    timeout=1800,
    volumes={}
)
def generate_music_mac_style(prompt: str, duration: int = 30):
    """
    Replicate the Mac environment that was working
    """
    import torch
    import numpy as np
    import scipy.io.wavfile as wavfile
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    from pathlib import Path
    
    work_dir = Path("/tmp/mac_style")
    work_dir.mkdir(exist_ok=True)
    
    try:
        print(f"🍎 Mac-style generation starting...")
        print(f"📝 Prompt: {prompt}")
        
        # Try to replicate Mac-like device handling
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🖥️ Using CUDA (like MPS on Mac)")
        else:
            device = "cpu"
            print(f"🖥️ Using CPU (Mac fallback)")
        
        # Use exact same model that worked on Mac
        model_name = "facebook/musicgen-small"
        print(f"📥 Loading {model_name} (Mac-compatible version)...")
        
        # Load with Mac-like settings
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Mac typically uses float32
            low_cpu_mem_usage=True,     # Mac memory optimization
        )
        
        model = model.to(device)
        print(f"✅ Model loaded successfully")
        
        # Use the exact same approach that worked on Mac
        print(f"🎵 Processing: '{prompt}'")
        
        # Simple, Mac-like input processing
        inputs = processor(
            text=prompt,
            return_tensors="pt",
        ).to(device)
        
        # Mac-style generation parameters (what typically works locally)
        sample_rate = model.config.audio_encoder.sampling_rate
        max_new_tokens = int(duration * sample_rate / model.config.audio_encoder.hop_length)
        
        print(f"🎵 Generating {max_new_tokens} tokens...")
        
        # Use Mac-like generation settings
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=250,
                top_p=0.0,
                guidance_scale=3.0,
                num_beams=1,
            )
        
        print("✅ Generation completed!")
        
        # Process like on Mac
        audio_np = audio_values[0, 0].cpu().numpy()
        
        # Mac-style normalization
        audio_np = audio_np / np.max(np.abs(audio_np)) * 0.9
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Save
        output_path = work_dir / "mac_style_music.wav"
        wavfile.write(str(output_path), sample_rate, audio_int16)
        
        with open(output_path, 'rb') as f:
            audio_data = f.read()
        
        return {
            "success": True,
            "prompt": prompt,
            "model": model_name,
            "device": device,
            "files": {"audio": audio_data},
            "metadata": {
                "sample_rate": sample_rate,
                "duration": len(audio_np) / sample_rate,
                "approach": "mac_replica"
            }
        }
        
    except Exception as e:
        print(f"❌ Mac-style generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.local_entrypoint()  
def main(prompt: str = "Christmas carol song", duration: int = 30):
    """Mac environment replica"""
    print("🍎 Running MAC ENVIRONMENT REPLICA")
    print("=" * 50)
    
    result = generate_music_mac_style.remote(prompt, duration)
    
    if result["success"]:
        print("✅ Mac-style generation SUCCESS!")
        print(f"🎵 Approach: {result['metadata']['approach']}")
        print(f"⏱️ Duration: {result['metadata']['duration']:.1f}s")
        
        output_dir = Path("mac_replica_output")  
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "christmas_carol_mac_style.wav", 'wb') as f:
            f.write(result["files"]["audio"])
        
        print(f"💾 Music saved to: {output_dir}/christmas_carol_mac_style.wav")
        print("🍎 Mac environment successfully replicated on Modal!")
        
    else:
        print(f"❌ Mac-style failed: {result['error']}")

if __name__ == "__main__":
    print("Run with: modal run modal_music_pipeline_mac_replica.py")
