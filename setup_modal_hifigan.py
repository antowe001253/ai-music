"""
Modal HifiGAN Setup Script
Separate script to deploy and setup HifiGAN models on Modal
"""

import modal
import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Modal image with HifiGAN dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
        "huggingface-hub>=0.16.0",
        "transformers>=4.30.0",
        "wget"
    ])
    .run_commands([
        "apt-get update",
        "apt-get install -y wget git ffmpeg"
    ])
)

app = modal.App("hifigan-vocoder-setup", image=image)
volume = modal.Volume.from_name("hifigan-models", create_if_missing=True)

@app.function(
    gpu="A10G",
    volumes={"/models": volume},
    timeout=1800,
    memory=8192,
)
def setup_hifigan_models():
    """Download and setup HifiGAN models"""
    import os
    import wget
    import json
    
    model_dir = Path("/models/hifigan")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("📦 Setting up HifiGAN models...")
    
    # Download Universal HifiGAN model (better for singing voice)
    try:
        # Download generator model
        generator_path = model_dir / "generator_universal.pth.tar"
        if not generator_path.exists():
            logger.info("📥 Downloading HifiGAN generator...")
            os.system(f"wget -O {generator_path} https://github.com/jik876/hifi-gan/releases/download/v1.0/generator_universal.pth.tar")
        
        # Create a basic config file
        config_path = model_dir / "config.json"
        if not config_path.exists():
            logger.info("📝 Creating HifiGAN config...")
            config = {
                "resblock": "1",
                "num_gpus": 1,
                "batch_size": 16,
                "learning_rate": 0.0002,
                "adam_b1": 0.8,
                "adam_b2": 0.99,
                "lr_decay": 0.999,
                "seed": 1234,
                "upsample_rates": [8, 8, 2, 2],
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "upsample_initial_channel": 512,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "segment_size": 8192,
                "num_mels": 80,
                "num_freq": 1025,
                "n_fft": 1024,
                "hop_size": 256,
                "win_size": 1024,
                "sampling_rate": 22050,
                "fmin": 0,
                "fmax": 8000,
                "fmax_for_loss": None
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Commit the volume
        volume.commit()
        
        logger.info("✅ HifiGAN models setup complete!")
        return {
            "success": True,
            "generator_path": str(generator_path),
            "config_path": str(config_path),
            "message": "HifiGAN models ready"
        }
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    print("🚀 Deploying HifiGAN setup to Modal...")
    
    with app.run():
        result = setup_hifigan_models.remote()
        
        if result["success"]:
            print("✅ HifiGAN models setup successfully!")
            print(f"📁 Generator: {result['generator_path']}")
            print(f"📁 Config: {result['config_path']}")
        else:
            print(f"❌ Setup failed: {result['error']}")
