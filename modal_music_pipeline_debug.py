#!/usr/bin/env python3
"""
Modal Cloud Setup for Diff-SVC - Debug Version
Implements proper CUDA debugging as per PyTorch documentation
"""

import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("diff-svc-music-debug")

# Create the container image with debugging environment
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
    .env({"CUDA_LAUNCH_BLOCKING": "1"})  # Enable synchronous CUDA for debugging
)

@app.function(
    image=image,
    gpu="A10G",
    memory=16000,
    timeout=1800,
    volumes={}
)
def debug_generate_music(prompt: str, duration: int = 30):
    """
    Debug version with proper CUDA error handling
    """
    import torch
    import numpy as np
    import scipy.io.wavfile as wavfile
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    from pathlib import Path
    
    # Set debugging environment
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    work_dir = Path("/tmp/debug_work")
    work_dir.mkdir(exist_ok=True)
    
    try:
        print(f"🔧 Debug Mode: CUDA_LAUNCH_BLOCKING=1")
        print(f"📝 Prompt: {prompt}")
        print(f"⏱️ Duration: {duration}s")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Device: {device}")
        
        if device == "cuda":
            print(f"🔍 CUDA Device: {torch.cuda.get_device_name()}")
            print(f"💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Load model with explicit matching
        print("📥 Loading MusicGen model...")
        model_name = "facebook/musicgen-small"
        
        # Load tokenizer and model separately for debugging
        processor = AutoProcessor.from_pretrained(model_name)
        print(f"✅ Processor loaded - vocab_size: {processor.tokenizer.vocab_size}")
        
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for stability
        )
        
        # Debug model configuration
        print(f"✅ Model loaded")
        print(f"🔍 Model vocab_size: {model.config.text_encoder.vocab_size}")
        print(f"🔍 Model max_length: {model.config.max_length}")
        
        # Verify tokenizer/model compatibility
        if hasattr(processor.tokenizer, 'vocab_size') and hasattr(model.config, 'text_encoder'):
            tokenizer_vocab = processor.tokenizer.vocab_size
            model_vocab = model.config.text_encoder.vocab_size
            print(f"🔍 Tokenizer vocab: {tokenizer_vocab}, Model vocab: {model_vocab}")
            
            if tokenizer_vocab != model_vocab:
                print("⚠️ WARNING: Tokenizer/Model vocab size mismatch!")
                return {
                    "success": False,
                    "error": f"Vocab mismatch: tokenizer={tokenizer_vocab}, model={model_vocab}",
                    "debug": True
                }
        
        # Move to device after verification
        model = model.to(device)
        print(f"✅ Model moved to {device}")
        
        # Prepare input with debugging
        safe_prompt = f"instrumental music"  # Ultra-simple prompt
        print(f"🎵 Using prompt: '{safe_prompt}'")
        
        # Process input with detailed debugging
        print("🔍 Processing input...")
        inputs = processor(
            text=[safe_prompt],
            padding=True,
            return_tensors="pt",
        )
        
        # Debug input tensors before moving to device
        print("🔍 Input tensor shapes:")
        for key, tensor in inputs.items():
            print(f"  {key}: {tensor.shape}, dtype: {tensor.dtype}")
            
            # Check for problematic values
            if torch.isnan(tensor).any():
                print(f"❌ Found NaN in {key}")
                return {"success": False, "error": f"NaN in input {key}", "debug": True}
            
            if torch.isinf(tensor).any():
                print(f"❌ Found Inf in {key}")
                return {"success": False, "error": f"Inf in input {key}", "debug": True}
            
            # Check token ID ranges
            if key == "input_ids":
                max_id = tensor.max().item()
                min_id = tensor.min().item()
                print(f"  Token ID range: {min_id} to {max_id}")
                
                if min_id < 0 or max_id >= processor.tokenizer.vocab_size:
                    print(f"❌ Token ID out of bounds: range=[{min_id}, {max_id}], vocab_size={processor.tokenizer.vocab_size}")
                    return {"success": False, "error": f"Token out of bounds", "debug": True}
        
        # Move inputs to device safely
        if device == "cuda":
            print("🔄 Moving inputs to CUDA...")
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                print("✅ Inputs moved to CUDA successfully")
            except Exception as cuda_error:
                print(f"❌ Failed to move inputs to CUDA: {cuda_error}")
                return {"success": False, "error": f"CUDA input transfer failed: {cuda_error}", "debug": True}
        
        # Conservative generation parameters
        sample_rate = model.config.audio_encoder.sampling_rate
        max_new_tokens = min(256, int(duration * sample_rate / model.config.audio_encoder.hop_length))
        print(f"📊 Sample rate: {sample_rate}, Max tokens: {max_new_tokens}")
        
        # Generate with minimal parameters
        print("🎵 Starting generation...")
        
        with torch.no_grad():
            try:
                # Most conservative generation possible
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding only
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=False,  # Disable caching to avoid memory issues
                    output_scores=False,
                    return_dict_in_generate=False
                )
                
                print("✅ Generation successful!")
                
            except RuntimeError as cuda_error:
                print(f"❌ CUDA Runtime Error during generation: {cuda_error}")
                
                # Try CPU fallback
                print("🔄 Attempting CPU fallback...")
                try:
                    model_cpu = model.to("cpu")
                    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                    
                    audio_values = model_cpu.generate(
                        **inputs_cpu,
                        max_new_tokens=128,  # Even smaller for CPU
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                    
                    print("✅ CPU fallback successful!")
                    device = "cpu"  # Update device info
                    
                except Exception as cpu_error:
                    print(f"❌ CPU fallback also failed: {cpu_error}")
                    return {
                        "success": False, 
                        "error": f"Both CUDA and CPU failed. CUDA: {cuda_error}, CPU: {cpu_error}",
                        "debug": True
                    }
        
        # Process generated audio
        if audio_values is not None and len(audio_values) > 0:
            print("🔍 Processing generated audio...")
            
            # Debug audio tensor
            print(f"Audio tensor shape: {audio_values.shape}")
            print(f"Audio tensor dtype: {audio_values.dtype}")
            
            audio_np = audio_values[0, 0].cpu().numpy()
            print(f"Audio numpy shape: {audio_np.shape}")
            print(f"Audio range: [{audio_np.min():.6f}, {audio_np.max():.6f}]")
            
            # Check for problematic values
            if np.isnan(audio_np).any():
                print("❌ Generated audio contains NaN values")
                return {"success": False, "error": "Generated audio contains NaN", "debug": True}
            
            if np.isinf(audio_np).any():
                print("❌ Generated audio contains Inf values")
                return {"success": False, "error": "Generated audio contains Inf", "debug": True}
            
            # Normalize safely
            max_abs = np.max(np.abs(audio_np))
            if max_abs > 0:
                audio_np = audio_np / max_abs * 0.95
            
            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Save file
            output_path = work_dir / "debug_output.wav"
            wavfile.write(str(output_path), sample_rate, audio_int16)
            
            with open(output_path, 'rb') as f:
                audio_data = f.read()
            
            print(f"💾 Audio saved: {len(audio_data)} bytes")
            
            return {
                "success": True,
                "prompt": prompt,
                "device_used": device,
                "model": model_name,
                "files": {"audio": audio_data},
                "metadata": {
                    "sample_rate": sample_rate,
                    "duration": len(audio_np) / sample_rate,
                    "debug_mode": True
                }
            }
        else:
            return {"success": False, "error": "No audio generated", "debug": True}
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "debug": True}

@app.local_entrypoint()
def main(prompt: str = "Christmas carol song", duration: int = 30):
    """Debug version entrypoint"""
    print("🔧 Running DEBUG version of music generation")
    print("=" * 50)
    
    result = debug_generate_music.remote(prompt, duration)
    
    if result["success"]:
        print("✅ Debug generation successful!")
        
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "debug_audio.wav", 'wb') as f:
            f.write(result["files"]["audio"])
        
        print(f"🎵 Debug audio saved to: {output_dir}/debug_audio.wav")
        
    else:
        print(f"❌ Debug generation failed: {result['error']}")

if __name__ == "__main__":
    print("Run with: modal run modal_music_pipeline_debug.py")
