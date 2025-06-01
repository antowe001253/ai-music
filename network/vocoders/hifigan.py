import glob
import json
import os
import re

import librosa
import torch

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
        
        # If no model is loaded, use Griffin-Lim algorithm as fallback
        if self.model is None:
            print("⚠️ No HifiGAN model loaded, using Griffin-Lim fallback")
            import librosa
            
            # Convert mel-spectrogram to linear spectrogram using Griffin-Lim
            # This is basic but functional
            try:
                # Convert to numpy if it's a tensor
                if hasattr(mel, 'numpy'):
                    mel_np = mel.numpy()
                else:
                    mel_np = mel
                    
                # Griffin-Lim reconstruction
                wav_out = librosa.feature.inverse.mel_to_audio(
                    mel_np, 
                    sr=hparams['audio_sample_rate'],
                    hop_length=hparams['hop_size'],
                    win_length=hparams['win_size'],
                    n_fft=hparams['fft_size']
                )
                
                print(f"✅ Griffin-Lim conversion successful: {len(wav_out)} samples")
                return wav_out
                
            except Exception as e:
                print(f"❌ Griffin-Lim fallback failed: {e}")
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
