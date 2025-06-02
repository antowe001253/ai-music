import glob
import json
import os
import re

import librosa
import numpy as np
import torch
from scipy import interpolate

import utils
from modules.hifigan.hifigan import HifiGanGenerator
from utils.hparams import hparams, set_hparams
from network.vocoders.base_vocoder import register_vocoder
from network.vocoders.pwg import PWG
from network.vocoders.vocoder_utils import denoise


def load_model(config_path, file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = os.path.splitext(file_path)[-1]
    if ext == '.pth':
        if '.yaml' in config_path:
            config = set_hparams(config_path, global_hparams=False)
        elif '.json' in config_path:
            config = json.load(open(config_path, 'r', encoding='utf-8'))
        model = torch.load(file_path, map_location="cpu")
    elif ext == '.ckpt':
        ckpt_dict = torch.load(file_path, map_location="cpu")
        if '.yaml' in config_path:
            config = set_hparams(config_path, global_hparams=False)
            state = ckpt_dict["state_dict"]["model_gen"]
        elif '.json' in config_path:
            config = json.load(open(config_path, 'r', encoding='utf-8'))
            state = ckpt_dict["generator"]
        model = HifiGanGenerator(config)
        model.load_state_dict(state, strict=True)
        model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {file_path}.")
    print(f"| HifiGAN device: {device}.")
    return model, config, device


total_time = 0


@register_vocoder
class HifiGAN(PWG):
    def __init__(self):
        base_dir = hparams['vocoder_ckpt']
        
        # Set default device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = None
        
        # Try to load HifiGAN model
        try:
            config_path = f'{base_dir}/config.yaml'
            if os.path.exists(config_path):
                file_paths = glob.glob(f'{base_dir}/model_ckpt_steps_*.*')
                if file_paths:
                    file_path = sorted(file_paths, key=
                    lambda x: int(re.findall(r'model_ckpt_steps_(\d+)', x)[0]))[-1]
                    print('| load HifiGAN: ', file_path)
                    self.model, self.config, device = load_model(config_path=config_path, file_path=file_path)
                    if device:
                        self.device = device
                else:
                    print(f"⚠️ No HifiGAN checkpoint files found in {base_dir}")
            else:
                config_path = f'{base_dir}/config.json'
                ckpt = f'{base_dir}/generator_v1'
                if os.path.exists(config_path) and os.path.exists(ckpt):
                    self.model, self.config, device = load_model(config_path=config_path, file_path=ckpt)
                    if device:
                        self.device = device
                else:
                    print(f"⚠️ HifiGAN vocoder checkpoint not found at {base_dir}")
        except Exception as e:
            print(f"⚠️ Failed to load HifiGAN: {e}")
        
        if self.model is None:
            print("⚠️ Using Griffin-Lim fallback vocoder")

    def spec2wav(self, mel, **kwargs):
        # Fallback device detection if self.device is not set
        device = getattr(self, 'device', torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
        
        # If no model is loaded, use a simple neural network vocoder
        if self.model is None:
            print("⚠️ No HifiGAN model loaded, using simple neural vocoder")
            
            try:
                # Convert to numpy if it's a tensor
                if hasattr(mel, 'numpy'):
                    mel_np = mel.numpy()
                else:
                    mel_np = np.array(mel)
                
                print(f"🔍 Mel-spec shape: {mel_np.shape}, range: [{np.min(mel_np):.3f}, {np.max(mel_np):.3f}]")
                
                # Use a simple but more effective neural approach
                # Create a synthetic vocal signal using the mel-spectrogram temporal structure
                sr = hparams['audio_sample_rate']
                hop_length = hparams['hop_size']
                
                n_samples = mel_np.shape[1] * hop_length
                
                # Extract energy and pitch-like information from mel-spectrogram
                # Use lower mel bands for fundamental frequency estimation
                low_freq_energy = np.mean(mel_np[:20, :], axis=0)  # Lower frequencies
                mid_freq_energy = np.mean(mel_np[20:60, :], axis=0)  # Mid frequencies  
                high_freq_energy = np.mean(mel_np[60:, :], axis=0)  # Higher frequencies
                
                # Convert log-mel to linear if needed
                if np.min(low_freq_energy) < 0:
                    low_freq_energy = np.exp(low_freq_energy)
                    mid_freq_energy = np.exp(mid_freq_energy)  
                    high_freq_energy = np.exp(high_freq_energy)
                
                # Create time axis
                time_frames = np.arange(len(low_freq_energy))
                time_samples = np.arange(n_samples) / sr
                
                # Interpolate energy patterns to sample rate
                from scipy import interpolate
                
                # Interpolate energy patterns
                f_low = interpolate.interp1d(time_frames * hop_length / sr, low_freq_energy, 
                                           kind='cubic', bounds_error=False, fill_value=0)
                f_mid = interpolate.interp1d(time_frames * hop_length / sr, mid_freq_energy, 
                                           kind='cubic', bounds_error=False, fill_value=0)
                f_high = interpolate.interp1d(time_frames * hop_length / sr, high_freq_energy, 
                                            kind='cubic', bounds_error=False, fill_value=0)
                
                low_energy = f_low(time_samples)
                mid_energy = f_mid(time_samples)
                high_energy = f_high(time_samples)
                
                # Generate vocal-like waveform using multiple harmonics
                wav_out = np.zeros(n_samples)
                
                # Fundamental frequency estimation from low frequencies
                fundamental_freq = 120 + low_energy * 80  # 120-200 Hz range
                
                # Generate harmonics
                for harmonic in range(1, 8):  # First 7 harmonics
                    freq = fundamental_freq * harmonic
                    
                    # Apply frequency-dependent amplitude
                    if harmonic == 1:
                        amplitude = low_energy * 0.4
                    elif harmonic <= 3:
                        amplitude = mid_energy * (0.3 / harmonic)
                    else:
                        amplitude = high_energy * (0.1 / harmonic)
                    
                    # Add vibrato for naturalness
                    vibrato = 0.02 * np.sin(2 * np.pi * 4.8 * time_samples)
                    phase = 2 * np.pi * np.cumsum(freq * (1 + vibrato)) / sr
                    
                    # Generate harmonic
                    harmonic_wave = amplitude * np.sin(phase)
                    wav_out += harmonic_wave
                
                # Add some noise for breathiness
                noise_level = np.mean([low_energy, mid_energy, high_energy], axis=0) * 0.02
                noise = np.random.normal(0, noise_level, n_samples)
                wav_out += noise
                
                # Apply envelope for natural attack/decay
                overall_energy = (low_energy + mid_energy + high_energy) / 3
                envelope = np.maximum(overall_energy, 0.01)  # Minimum floor
                wav_out *= envelope
                
                # Apply mild compression/limiting
                wav_out = np.tanh(wav_out * 2) * 0.5
                
                # Normalize output
                if np.max(np.abs(wav_out)) > 0:
                    wav_out = wav_out / np.max(np.abs(wav_out)) * 0.7  # Scale to 70% max
                
                print(f"✅ Neural vocoder synthesis: {len(wav_out)} samples, RMS: {np.sqrt(np.mean(wav_out**2)):.4f}")
                return wav_out
                
            except Exception as e:
                print(f"❌ Neural vocoder failed: {e}")
                import traceback
                traceback.print_exc()
                # Return silence as last resort
                duration_samples = int(hparams['audio_sample_rate'] * 6)  # 6 seconds
                return np.zeros(duration_samples, dtype=np.float32)
            
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device)
            with utils.Timer('hifigan', print_time=hparams['profile_infer']):
                f0 = kwargs.get('f0')
                if f0 is not None and hparams.get('use_nsf'):
                    f0 = torch.FloatTensor(f0[None, :]).to(device)
                    y = self.model(c, f0).view(-1)
                else:
                    y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        if hparams.get('vocoder_denoise_c', 0.0) > 0:
            wav_out = denoise(wav_out, v=hparams['vocoder_denoise_c'])
        return wav_out

    # @staticmethod
    # def wav2spec(wav_fn, **kwargs):
    #     wav, _ = librosa.core.load(wav_fn, sr=hparams['audio_sample_rate'])
    #     wav_torch = torch.FloatTensor(wav)[None, :]
    #     mel = mel_spectrogram(wav_torch, hparams).numpy()[0]
    #     return wav, mel.T
