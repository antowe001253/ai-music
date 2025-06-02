#!/usr/bin/env python3
"""
Fixed Diff-SVC Inference - Works with your actual model setup
"""

import io
import time
from pathlib import Path
import librosa
import numpy as np
import soundfile
from infer_tools import infer_tool
from infer_tools import slicer
from infer_tools.infer_tool import Svc
from utils.hparams import hparams

def run_inference(input_file, output_file, key_shift=0):
    """Run Diff-SVC inference with your actual model"""
    
    # Fixed paths for your setup
    project_name = "base_model"  # Your actual model folder name
    model_path = './checkpoints/base_model/model_ckpt_steps_100000.ckpt'  # Your actual model
    config_path = './config.yaml'  # Your root config
    
    print(f"🎤 Diff-SVC Inference Starting...")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"🎼 Key shift: {key_shift}")
    print(f"📦 Model: {model_path}")
    print(f"⚙️ Config: {config_path}")
    
    # Validate files exist
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input not found: {input_file}")
    
    # Create necessary directories
    infer_tool.mkdir(["./raw", "./results"])
    
    # Initialize model
    print("🚀 Loading Diff-SVC model...")
    model = Svc(project_name, config_path, hubert_gpu=True, model_path=model_path)
    
    # Process the file
    print("🎵 Processing audio...")
    
    # Copy input to raw for processing
    import shutil
    raw_file = f"./raw/{Path(input_file).name}"
    shutil.copy2(input_file, raw_file)
    
    # Run the actual inference with mel-spectrogram saving
    try:
        import torch
        import pickle
        
        print("🎵 Running Diff-SVC inference...")
        
        # Get the batch data (this is what contains the mel-spectrogram)
        batch = model.pre(raw_file, 20, True, 0.05)  # acc=20, use_crepe=True, thre=0.05
        
        # Apply key shift
        batch['f0'] = batch['f0'] + (key_shift / 12)
        batch['f0'][batch['f0'] > np.log2(hparams['f0_max'])] = 0
        
        print("🧠 Running diffusion model...")
        
        # Run the diffusion model (this generates the mel-spectrogram)
        with torch.no_grad():
            spk_embed = batch.get('spk_embed') if not hparams['use_spk_id'] else batch.get('spk_ids')
            outputs = model.model(
                batch['hubert'].to(model.dev), 
                spk_embed=spk_embed, 
                mel2ph=batch['mel2ph'].to(model.dev), 
                f0=batch['f0'].to(model.dev), 
                uv=batch['uv'].to(model.dev),
                energy=batch['energy'].to(model.dev),
                ref_mels=batch["mels"].to(model.dev),
                infer=True
            )
        
        # Extract the mel-spectrogram
        mel_out = model.model.out2mel(outputs['mel_out'])
        
        print("💾 Saving mel-spectrogram...")
        
        # Save mel-spectrogram as numpy array
        mel_numpy = mel_out.cpu().numpy()
        mel_file = output_file.replace('.wav', '_mel.npy')
        np.save(mel_file, mel_numpy)
        print(f"✅ Mel-spectrogram saved: {mel_file}")
        
        # Save mel-spectrogram metadata
        mel_meta = {
            'shape': mel_numpy.shape,
            'sample_rate': hparams["audio_sample_rate"],
            'hop_length': hparams.get('hop_size', 256),
            'n_mels': hparams.get('audio_num_mel_bins', 128)
        }
        meta_file = output_file.replace('.wav', '_meta.json')
        import json
        with open(meta_file, 'w') as f:
            json.dump(mel_meta, f, indent=2)
        print(f"✅ Metadata saved: {meta_file}")
        
        # Try to convert mel to audio using vocoder
        print("🎤 Attempting vocoding...")
        try:
            if model.vocoder is not None:
                # Use the model's vocoder
                audio_output = model.vocoder.spec2wav(mel_out)
                if audio_output is not None:
                    audio_np = audio_output.cpu().numpy()
                    soundfile.write(output_file, audio_np, hparams["audio_sample_rate"])
                    print(f"✅ Audio saved with model vocoder: {output_file}")
                else:
                    raise Exception("Vocoder returned None")
            else:
                raise Exception("No vocoder available")
                
        except Exception as vocoder_error:
            print(f"⚠️ Vocoder failed: {vocoder_error}")
            
            # Use Griffin-Lim as fallback
            print("🔄 Trying Griffin-Lim fallback...")
            try:
                # Convert mel-spectrogram to linear spectrogram
                mel_basis = librosa.filters.mel(
                    sr=hparams["audio_sample_rate"],
                    n_fft=hparams.get('fft_size', 1024),
                    n_mels=hparams.get('audio_num_mel_bins', 128),
                    fmin=hparams.get('fmin', 0),
                    fmax=hparams.get('fmax', 8000)
                )
                
                # Convert to linear scale
                mel_linear = np.exp(mel_numpy[0])  # Remove batch dimension and convert from log
                
                # Invert mel filter to get approximate linear spectrogram
                linear_spec = np.dot(np.linalg.pinv(mel_basis), mel_linear)
                
                # Use Griffin-Lim to reconstruct audio
                audio_reconstructed = librosa.griffinlim(
                    linear_spec,
                    hop_length=hparams.get('hop_size', 256),
                    win_length=hparams.get('win_size', 1024),
                    n_iter=32
                )
                
                # Save the audio
                soundfile.write(output_file, audio_reconstructed, hparams["audio_sample_rate"])
                print(f"✅ Audio saved with Griffin-Lim: {output_file}")
                print("⚠️ Note: Griffin-Lim quality is poor - use Modal HifiGAN for better results!")
                
            except Exception as griffin_error:
                print(f"❌ Griffin-Lim also failed: {griffin_error}")
                print("💡 But mel-spectrogram is saved! Use Modal HifiGAN to convert it to audio.")
        
        return mel_file  # Return the mel-spectrogram file path
        
        print(f"✅ Diff-SVC inference complete: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Diff-SVC inference failed: {e}")
        raise

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python fixed_infer.py input.wav output.wav [key_shift]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_shift = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    try:
        result = run_inference(input_file, output_file, key_shift)
        print(f"🎉 Success! Output: {result}")
    except Exception as e:
        print(f"❌ Failed: {e}")
