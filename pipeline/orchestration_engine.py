#!/usr/bin/env python3
"""
Orchestration Engine - Step 10 Implementation
Coordinates tempo, key, and structure across multiple audio tracks
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from prompt_intelligence import MusicalParameters
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from audio_processing import AudioProcessingSuite

@dataclass
class TrackInfo:
    """Information about an individual track"""
    audio_data: np.ndarray
    sample_rate: int
    tempo: float
    key: str
    duration: float
    track_type: str  # 'instrumental', 'vocal', 'drums', etc.

@dataclass 
class OrchestrationPlan:
    """Plan for coordinating multiple tracks"""
    target_tempo: int
    target_key: str
    song_structure: Dict[str, Tuple[float, float]]  # section -> (start, end) times
    track_assignments: Dict[str, List[str]]  # section -> track types
    timing_grid: np.ndarray  # Beat grid for alignment

class OrchestrationEngine:
    def __init__(self):
        """Initialize orchestration engine"""
        self.audio_processor = AudioProcessingSuite()
        
        # Song structure templates  
        self.structure_templates = {
            'pop': {
                'intro': (0, 8), 'verse1': (8, 24), 'chorus1': (24, 40),
                'verse2': (40, 56), 'chorus2': (56, 72), 'bridge': (72, 88),
                'chorus3': (88, 104), 'outro': (104, 112)
            },
            'rock': {
                'intro': (0, 8), 'verse1': (8, 32), 'chorus1': (32, 48),
                'verse2': (48, 72), 'chorus2': (72, 88), 'solo': (88, 104),
                'chorus3': (104, 120), 'outro': (120, 128)
            }
        }    
    def create_orchestration_plan(self, params: MusicalParameters, tracks: List[TrackInfo]) -> OrchestrationPlan:
        """
        Create orchestration plan for multiple tracks
        
        Args:
            params: Musical parameters from prompt intelligence
            tracks: List of track information
            
        Returns:
            OrchestrationPlan: Coordination plan for all tracks
        """
        # Determine target tempo (use fastest track as reference)
        target_tempo = params.tempo or max(track.tempo for track in tracks if track.tempo > 0)
        
        # Determine target key (use most common or first track's key)
        target_key = params.key or tracks[0].key if tracks else 'C_major'
        
        # Get song structure template
        genre = params.genre or 'pop'
        structure = self.structure_templates.get(genre, self.structure_templates['pop'])
        
        # Create timing grid based on target tempo
        timing_grid = self._create_timing_grid(target_tempo, max(track.duration for track in tracks))
        
        # Assign tracks to sections
        track_assignments = self._assign_tracks_to_sections(structure, tracks)
        
        return OrchestrationPlan(
            target_tempo=target_tempo,
            target_key=target_key,
            song_structure=structure,
            track_assignments=track_assignments,
            timing_grid=timing_grid
        )
    
    def synchronize_tempo(self, track: TrackInfo, target_tempo: int) -> np.ndarray:
        """Synchronize track tempo to target"""
        if abs(track.tempo - target_tempo) < 5:  # Close enough
            return track.audio_data
        
        # Calculate stretch ratio
        stretch_ratio = track.tempo / target_tempo
        
        # Time stretch the audio
        return self.audio_processor.time_stretch(track.audio_data, stretch_ratio)
    
    def transpose_key(self, track: TrackInfo, target_key: str) -> np.ndarray:
        """Transpose track to target key"""
        if track.key == target_key:
            return track.audio_data
        
        # Calculate semitone difference (simplified)
        # This would need more sophisticated key detection/transposition
        semitones = self._calculate_key_distance(track.key, target_key)
        
        if semitones != 0:
            return self.audio_processor.pitch_shift(track.audio_data, track.sample_rate, semitones)
        
        return track.audio_data
    
    def _create_timing_grid(self, tempo: int, duration: float) -> np.ndarray:
        """Create beat grid for tempo synchronization"""
        beats_per_second = tempo / 60.0
        num_beats = int(duration * beats_per_second)
        return np.linspace(0, duration, num_beats)
    
    def _assign_tracks_to_sections(self, structure: Dict, tracks: List[TrackInfo]) -> Dict[str, List[str]]:
        """Assign tracks to song sections based on type"""
        assignments = {}
        for section in structure.keys():
            if section in ['intro', 'outro']:
                assignments[section] = ['instrumental']
            elif section in ['verse1', 'verse2']:
                assignments[section] = ['instrumental', 'vocal']
            elif section in ['chorus1', 'chorus2', 'chorus3']:
                assignments[section] = ['instrumental', 'vocal', 'drums']
            else:
                assignments[section] = ['instrumental']
        return assignments
    
    def _calculate_key_distance(self, from_key: str, to_key: str) -> int:
        """Calculate semitone distance between keys (simplified)"""
        # This is a simplified version - real implementation would be more complex
        key_to_semitone = {
            'C_major': 0, 'C#_major': 1, 'D_major': 2, 'D#_major': 3,
            'E_major': 4, 'F_major': 5, 'F#_major': 6, 'G_major': 7,
            'G#_major': 8, 'A_major': 9, 'A#_major': 10, 'B_major': 11,
            'A_minor': 9, 'B_minor': 11, 'C_minor': 0, 'D_minor': 2,
            'E_minor': 4, 'F_minor': 5, 'G_minor': 7
        }
        
        from_semitone = key_to_semitone.get(from_key, 0)
        to_semitone = key_to_semitone.get(to_key, 0)
        
        return to_semitone - from_semitone