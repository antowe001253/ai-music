#!/usr/bin/env python3
"""
Advanced Audio Processing Suite for Automated Music Pipeline
Handles tempo detection, BPM matching, key detection, pitch shifting, and audio mixing
"""

import librosa
import numpy as np
import scipy.signal
from pydub import AudioSegment
import soundfile as sf
from pathlib import Path
import json

class AudioProcessingSuite:
    def __init__(self, sample_rate=44100):
        """
        Initialize audio processing suite
        
        Args:
            sample_rate: Default sample rate for processing
        """
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path, target_sr=None):
        """Load audio file using librosa"""
        if target_sr is None:
            target_sr = self.sample_rate
            
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def detect_tempo_and_beats(self, audio, sr):
        """
        Detect tempo (BPM) and beat locations
        
        Returns:
            dict: tempo info including BPM, beat times, and confidence
        """
        try:
            # Extract tempo using librosa
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, units='time')
            
            # Get onset strength for more detailed analysis
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)            
            # Dynamic programming beat tracker for more accurate results
            tempo_dyn, beats_dyn = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, units='time'
            )
            
            return {
                'bpm': float(tempo),
                'bpm_dynamic': float(tempo_dyn),
                'beat_times': beats.tolist(),
                'beat_times_dynamic': beats_dyn.tolist(),
                'onset_strength': onset_env
            }
        except Exception as e:
            print(f"Error in tempo detection: {e}")
            return None
    
    def detect_key(self, audio, sr):
        """
        Detect musical key using chroma features
        
        Returns:
            dict: key information including key name and confidence
        """
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Calculate key profile correlation
            # Krumhansl-Schmuckler key profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            # Normalize profiles
            major_profile = major_profile / np.sum(major_profile)
            minor_profile = minor_profile / np.sum(minor_profile)
            
            # Average chroma over time
            chroma_mean = np.mean(chroma, axis=1)
            
            # Calculate correlations for all keys
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            major_correlations = []
            minor_correlations = []
            
            for i in range(12):
                # Rotate profiles to match key
                major_shifted = np.roll(major_profile, i)
                minor_shifted = np.roll(minor_profile, i)
                
                # Calculate correlation
                major_corr = np.corrcoef(chroma_mean, major_shifted)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_shifted)[0, 1]
                
                major_correlations.append(major_corr)
                minor_correlations.append(minor_corr)
            
            # Find best key
            best_major_idx = np.argmax(major_correlations)
            best_minor_idx = np.argmax(minor_correlations)
            best_major_corr = major_correlations[best_major_idx]
            best_minor_corr = minor_correlations[best_minor_idx]
            
            if best_major_corr > best_minor_corr:
                key = f"{keys[best_major_idx]} major"
                confidence = best_major_corr
            else:
                key = f"{keys[best_minor_idx]} minor"
                confidence = best_minor_corr
            
            return {
                'key': key,
                'confidence': float(confidence),
                'chroma_mean': chroma_mean.tolist()
            }
        except Exception as e:
            print(f"Error in key detection: {e}")
            return None    
    def pitch_shift(self, audio, sr, n_steps):
        """
        Pitch shift audio by n semitones
        
        Args:
            audio: Input audio
            sr: Sample rate
            n_steps: Number of semitones to shift (positive = higher, negative = lower)
        """
        try:
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        except Exception as e:
            print(f"Error in pitch shifting: {e}")
            return audio
    
    def time_stretch(self, audio, rate):
        """
        Time stretch audio without changing pitch
        
        Args:
            audio: Input audio
            rate: Stretch factor (>1 = faster, <1 = slower)
        """
        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except Exception as e:
            print(f"Error in time stretching: {e}")
            return audio
    
    def match_tempo(self, audio, sr, target_bpm):
        """
        Match audio tempo to target BPM
        
        Args:
            audio: Input audio
            sr: Sample rate
            target_bpm: Target BPM
        """
        try:
            # Detect current tempo
            current_tempo = librosa.beat.tempo(y=audio, sr=sr)[0]
            
            # Calculate stretch rate
            stretch_rate = target_bpm / current_tempo
            
            # Apply time stretching
            stretched_audio = self.time_stretch(audio, stretch_rate)
            
            return stretched_audio, stretch_rate
        except Exception as e:
            print(f"Error in tempo matching: {e}")
            return audio, 1.0
    
    def extract_melody(self, audio, sr):
        """
        Extract predominant melody using librosa
        
        Returns:
            dict: melody information including F0 contour and times
        """
        try:
            # Extract F0 (fundamental frequency) using pyin algorithm
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            
            # Get time axis
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
            
            return {
                'f0': f0,
                'voiced_flag': voiced_flag,
                'voiced_probabilities': voiced_probs,
                'times': times,
                'notes': [librosa.hz_to_note(freq) if not np.isnan(freq) else None for freq in f0]
            }
        except Exception as e:
            print(f"Error in melody extraction: {e}")
            return None