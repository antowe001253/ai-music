#!/usr/bin/env python3
"""
Advanced Audio Processing - Step 11 Implementation
Professional mixing, EQ, compression, reverb, and mastering
"""

import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt
from typing import Dict, List, Optional, Tuple
import librosa

class AdvancedAudioProcessor:
    def __init__(self, sample_rate: int = 44100):
        """Initialize advanced audio processor"""
        self.sample_rate = sample_rate
        
        # EQ frequency bands (in Hz)
        self.eq_bands = {
            'sub_bass': (20, 60), 'bass': (60, 250), 'low_mid': (250, 500),
            'mid': (500, 2000), 'high_mid': (2000, 4000), 'presence': (4000, 8000),
            'brilliance': (8000, 20000)
        }
        
        # Stereo positioning presets
        self.stereo_positions = {
            'center': 0.0, 'left_slight': -0.3, 'left': -0.7, 'left_wide': -1.0,
            'right_slight': 0.3, 'right': 0.7, 'right_wide': 1.0
        }
    
    def apply_eq(self, audio: np.ndarray, eq_settings: Dict[str, float]) -> np.ndarray:
        """Apply EQ to audio"""
        processed_audio = audio.copy()
        
        for band, gain_db in eq_settings.items():
            if band in self.eq_bands and abs(gain_db) > 0.1:
                low_freq, high_freq = self.eq_bands[band]
                processed_audio = self._apply_band_filter(
                    processed_audio, low_freq, high_freq, gain_db
                )
        
        return processed_audio
    
    def apply_compression(self, audio: np.ndarray, threshold: float = -20, 
                         ratio: float = 4) -> np.ndarray:
        """Apply dynamic range compression (simplified)"""
        threshold_linear = 10 ** (threshold / 20)
        
        # Simple compression
        compressed = audio.copy()
        mask = np.abs(compressed) > threshold_linear
        compressed[mask] = np.sign(compressed[mask]) * (
            threshold_linear + (np.abs(compressed[mask]) - threshold_linear) / ratio
        )
        
        return compressed    
    def apply_reverb(self, audio: np.ndarray, room_size: float = 0.5, 
                     damping: float = 0.5, wet_level: float = 0.3) -> np.ndarray:
        """Apply reverb effect"""
        # Simple reverb using delay lines
        delay_samples = int(room_size * 0.1 * self.sample_rate)
        
        if delay_samples > 0:
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * damping
            
            # Mix dry and wet signals
            reverb_audio = (1 - wet_level) * audio + wet_level * delayed
            return reverb_audio
        
        return audio
    
    def position_stereo(self, audio: np.ndarray, position: float) -> np.ndarray:
        """Position audio in stereo field"""
        if len(audio.shape) == 1:
            # Convert mono to stereo
            stereo = np.zeros((len(audio), 2))
            
            # Calculate left/right gains
            if position <= 0:  # Left side
                left_gain = 1.0
                right_gain = 1.0 + position
            else:  # Right side
                left_gain = 1.0 - position
                right_gain = 1.0
            
            stereo[:, 0] = audio * left_gain   # Left channel
            stereo[:, 1] = audio * right_gain  # Right channel
            
            return stereo
        
        return audio
    
    def master_audio(self, audio: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
        """Master audio track (simplified mastering)"""
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            normalized = audio / max_val
        else:
            normalized = audio
        
        # Apply soft limiting
        limited = np.tanh(normalized * 0.9) * 0.95
        
        return limited
    
    def _apply_band_filter(self, audio: np.ndarray, low_freq: float, 
                          high_freq: float, gain_db: float) -> np.ndarray:
        """Apply band-specific EQ filter"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = min(high_freq / nyquist, 0.99)
        
        # Create bandpass filter
        b, a = butter(4, [low, high], btype='band')
        
        # Filter the audio
        filtered = filtfilt(b, a, audio)
        
        # Apply gain
        gain_linear = 10 ** (gain_db / 20)
        
        # Mix original and filtered audio
        return audio + filtered * (gain_linear - 1)    
    def mix_tracks(self, tracks: List[Tuple[np.ndarray, Dict]], master_settings: Dict) -> np.ndarray:
        """Mix multiple tracks with individual processing"""
        if not tracks:
            return np.array([])
        
        # Find the longest track for output length
        max_length = max(len(track[0]) for track in tracks)
        
        # Initialize mix bus
        mix = np.zeros(max_length)
        
        # Process and mix each track
        for audio, settings in tracks:
            processed = audio.copy()
            
            # Pad shorter tracks
            if len(processed) < max_length:
                processed = np.pad(processed, (0, max_length - len(processed)))
            
            # Apply individual track processing
            if 'eq' in settings:
                processed = self.apply_eq(processed, settings['eq'])
            
            if 'compression' in settings:
                comp_settings = settings['compression']
                processed = self.apply_compression(
                    processed, 
                    comp_settings.get('threshold', -20),
                    comp_settings.get('ratio', 4)
                )
            
            if 'reverb' in settings:
                rev_settings = settings['reverb']
                processed = self.apply_reverb(
                    processed,
                    rev_settings.get('room_size', 0.5),
                    rev_settings.get('damping', 0.5),
                    rev_settings.get('wet_level', 0.3)
                )
            
            # Apply gain
            gain = settings.get('gain', 1.0)
            processed *= gain
            
            # Add to mix
            mix += processed
        
        # Apply master processing
        if 'master_eq' in master_settings:
            mix = self.apply_eq(mix, master_settings['master_eq'])
        
        if 'master_compression' in master_settings:
            comp_settings = master_settings['master_compression']
            mix = self.apply_compression(
                mix,
                comp_settings.get('threshold', -6),
                comp_settings.get('ratio', 2)
            )
        
        # Final mastering
        mix = self.master_audio(mix, master_settings.get('target_lufs', -14.0))
        
        return mix