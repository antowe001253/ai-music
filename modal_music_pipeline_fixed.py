#!/usr/bin/env python3
"""
Modal Cloud Setup for Diff-SVC Automated Music Pipeline - Fixed Version
Simplified approach to avoid CUDA device-side assert errors
"""

import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("diff-svc-music-pipeline")

# Create the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "transformers>=4.35.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "accelerate>=0.20.0",
        "PyYAML>=6.0.0"
    ])
    .apt_install(["ffmpeg", "git"])
)

# Define volume for model storage
models_volume = modal.Volume.from_name("music-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",  # Fixed deprecation warning
    memory=16000,
    timeout=1800,
    volumes={"/models": models_volume}
)
def generate_music_on_modal(prompt: str, duration: int = 30):
    """
    Generate music using a more stable approach to avoid CUDA errors.
    """
    import torch
    import numpy as np
    import scipy.io.wavfile as wavfile
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    import tempfile
    from pathlib import Path
    
    # Create working directory
    work_dir = Path("/tmp/diff_svc_work")
    work_dir.mkdir(exist_ok=True)
    
    try:
        print(f"🚀 Starting music generation on Modal GPU")
        print(f"📝 Prompt: {prompt}")
        print(f"⏱️ Duration: {duration}s")
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {device}")
        
        # Load model with more conservative settings
        print("📥 Loading MusicGen model...")
        
        # Use the smaller, more stable model
        model_name = "facebook/musicgen-small"  # Smaller model is more stable
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for stability
            device_map="auto"
        )
        
        print("✅ MusicGen loaded successfully!")
        
        # Simple, safe prompt processing
        safe_prompt = f"instrumental {prompt.replace('song', 'music')}"
        print(f"🎵 Using prompt: '{safe_prompt}'")
        
        # Process input with error handling
        try:
            inputs = processor(
                text=[safe_prompt],
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device safely
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
        except Exception as input_error:
            print(f"❌ Input processing failed: {input_error}")
            # Fallback to basic input
            inputs = processor(
                text=["instrumental music"],
                return_tensors="pt"
            )
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Calculate conservative token count
        sample_rate = model.config.audio_encoder.sampling_rate
        max_new_tokens = min(int(duration * sample_rate / model.config.audio_encoder.hop_length), 1024)
        
        print(f"📊 Sample rate: {sample_rate}, Max tokens: {max_new_tokens}")
        
        # Generate with very conservative settings
        print("🎵 Generating audio...")
        
        with torch.no_grad():
            try:
                # Ultra-conservative generation
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,  # Higher temperature for stability
                    num_beams=1,      # No beam search
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                print("✅ Audio generation successful!")
                
            except Exception as gen_error:
                print(f"❌ Generation failed: {gen_error}")
                print("🔄 Trying minimal fallback generation...")
                
                # Minimal fallback
                audio_values = model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
                print("✅ Fallback generation successful!")
        
        # Process generated audio
        if audio_values is not None and len(audio_values) > 0:
            # Extract audio from the generated tokens
            audio_np = audio_values[0, 0].cpu().numpy()
            
            # Normalize audio
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / np.max(np.abs(audio_np)) * 0.95
            
            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Save files
            instrumental_path = work_dir / "instrumental.wav"
            wavfile.write(str(instrumental_path), sample_rate, audio_int16)
            
            print(f"💾 Saved instrumental: {instrumental_path}")
            
            # Read file data
            with open(instrumental_path, 'rb') as f:
                instrumental_data = f.read()
            
            return {
                "success": True,
                "prompt": prompt,
                "used_prompt": safe_prompt,
                "duration": duration,
                "device_used": device,
                "files": {
                    "instrumental": instrumental_data,
                    "complete_song": instrumental_data  # Same for now
                },
                "metadata": {
                    "sample_rate": sample_rate,
                    "audio_length_seconds": len(audio_np) / sample_rate,
                    "model_used": model_name
                }
            }
        else:
            raise Exception("No audio generated")
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "prompt": prompt,
            "duration": duration
        }

@app.local_entrypoint()
def main(prompt: str = "Christmas carol song", duration: int = 30):
    """
    Local entrypoint to run music generation from command line.
    """
    print("🚀 Running Fixed Diff-SVC Music Pipeline on Modal")
    print("=" * 50)
    
    # Run the generation
    result = generate_music_on_modal.remote(prompt, duration)
    
    if result["success"]:
        print("✅ Generation successful!")
        print(f"📝 Prompt: {result['prompt']}")
        print(f"🖥️ Device: {result['device_used']}")
        print(f"⏱️ Audio length: {result['metadata']['audio_length_seconds']:.1f}s")
        print(f"🤖 Model: {result['metadata']['model_used']}")
        
        # Save files locally
        output_dir = Path("modal_output")
        output_dir.mkdir(exist_ok=True)
        
        for file_type, file_data in result["files"].items():
            file_path = output_dir / f"{file_type}.wav"
            with open(file_path, 'wb') as f:
                f.write(file_data)
            print(f"💾 Saved: {file_path}")
        
        print(f"\n🎵 Your generated music is in: {output_dir}/")
        print("🎉 Modal generation completed successfully!")
        
    else:
        print("❌ Generation failed!")
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    print("To run on Modal, use:")
    print("modal run modal_music_pipeline_fixed.py --prompt 'Christmas carol song'")
