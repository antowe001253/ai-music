#!/usr/bin/env python3
"""
Prompt Intelligence System - Step 9 Implementation
Parses natural language prompts and extracts musical parameters
"""

import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class MusicalParameters:
    """Container for extracted musical parameters"""
    genre: Optional[str] = None
    tempo: Optional[int] = None
    key: Optional[str] = None
    mood: Optional[str] = None
    energy: Optional[str] = None
    instruments: List[str] = None
    structure: Optional[str] = None
    dynamics: Optional[str] = None
    
    def __post_init__(self):
        if self.instruments is None:
            self.instruments = []

class PromptIntelligence:
    def __init__(self):
        """Initialize prompt intelligence system with musical knowledge"""
        # Genre detection keywords
        self.genres = {
            'rock': ['rock', 'metal', 'punk', 'grunge', 'alternative'],
            'pop': ['pop', 'mainstream', 'radio', 'catchy', 'commercial'],
            'electronic': ['electronic', 'edm', 'techno', 'house', 'synth'],
            'jazz': ['jazz', 'swing', 'bebop', 'smooth jazz', 'fusion'],
            'classical': ['classical', 'orchestral', 'symphony', 'baroque'],
            'folk': ['folk', 'acoustic', 'country', 'americana', 'christmas', 'carol'],
            'blues': ['blues', 'soul', 'r&b', 'rhythm and blues'],
            'ambient': ['ambient', 'atmospheric', 'soundscape', 'chill']
        }
        
        # Tempo mapping (keyword -> BPM range)
        self.tempos = {
            'slow': (['slow', 'ballad', 'relaxed', 'chill'], (70, 90)),
            'moderate': (['moderate', 'walking', 'medium'], (90, 120)),
            'upbeat': (['upbeat', 'energetic', 'driving'], (120, 140)),
            'fast': (['fast', 'dance', 'exciting'], (140, 170))
        }
        
        # Mood detection
        self.moods = {
            'happy': ['happy', 'joyful', 'cheerful', 'bright', 'uplifting'],
            'sad': ['sad', 'melancholy', 'dark', 'gloomy', 'tragic'],
            'energetic': ['energetic', 'powerful', 'intense', 'aggressive'],
            'calm': ['calm', 'peaceful', 'serene', 'gentle', 'soft'],
            'peaceful': ['christmas', 'carol', 'peaceful', 'warm', 'cozy']
        }        
        # Key detection and musical theory
        self.keys = {
            'major': ['major', 'bright', 'happy', 'cheerful'],
            'minor': ['minor', 'sad', 'dark', 'emotional']
        }
        
        # Instrument detection
        self.instruments = {
            'piano': ['piano', 'keys', 'keyboard'],
            'guitar': ['guitar', 'acoustic', 'electric'],
            'drums': ['drums', 'percussion', 'beats'],
            'bass': ['bass', 'bassline'],
            'synth': ['synthesizer', 'synth', 'electronic'],
            'vocals': ['vocals', 'singing', 'voice'],
            'strings': ['violin', 'strings', 'orchestra'],
            'bells': ['bells', 'chimes', 'christmas', 'carol'],
            'organ': ['organ', 'church']
        }
    
    def parse_prompt(self, prompt: str) -> MusicalParameters:
        """
        Parse a natural language prompt and extract musical parameters
        
        Args:
            prompt: Natural language description (e.g., "upbeat rock song with guitar")
            
        Returns:
            MusicalParameters: Extracted musical parameters
        """
        prompt_lower = prompt.lower()
        params = MusicalParameters()
        
        # Extract genre
        params.genre = self._extract_genre(prompt_lower)
        
        # Extract tempo
        params.tempo = self._extract_tempo(prompt_lower)
        
        # Extract mood
        params.mood = self._extract_mood(prompt_lower)
        
        # Extract instruments
        params.instruments = self._extract_instruments(prompt_lower)
        
        # Extract key preference
        params.key = self._extract_key_preference(prompt_lower)
        
        return params
    
    def _extract_genre(self, prompt: str) -> Optional[str]:
        """Extract genre from prompt"""
        for genre, keywords in self.genres.items():
            for keyword in keywords:
                if keyword in prompt:
                    return genre
        return None
    
    def _extract_tempo(self, prompt: str) -> Optional[int]:
        """Extract tempo from prompt"""
        for tempo_category, (keywords, bpm_range) in self.tempos.items():
            for keyword in keywords:
                if keyword in prompt:
                    # Return middle of BPM range
                    return (bpm_range[0] + bpm_range[1]) // 2
        return None
    
    def _extract_mood(self, prompt: str) -> Optional[str]:
        """Extract mood from prompt"""
        for mood, keywords in self.moods.items():
            for keyword in keywords:
                if keyword in prompt:
                    return mood
        return None
    
    def _extract_instruments(self, prompt: str) -> List[str]:
        """Extract instruments from prompt"""
        found_instruments = []
        for instrument, keywords in self.instruments.items():
            for keyword in keywords:
                if keyword in prompt:
                    found_instruments.append(instrument)
                    break
        return found_instruments
    
    def _extract_key_preference(self, prompt: str) -> Optional[str]:
        """Extract key preference from prompt"""
        if any(keyword in prompt for keyword in self.keys['minor']):
            return 'minor'
        elif any(keyword in prompt for keyword in self.keys['major']):
            return 'major'
        return None