#!/usr/bin/env python3
"""
Test script for Audio Processing Suite
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.audio_processing import AudioProcessingSuite
import json

def test_audio_processing():
    """Test the audio processing suite with the MusicGen output"""
    print("🎵 Testing Audio Processing Suite")
    print("=" * 50)
    
    # Initialize audio processing suite
    audio_processor = AudioProcessingSuite()
    
    # Test file path
    test_file = "/Users/alexntowe/Projects/AI/Diff-SVC/test_musicgen_output.wav"
    
    try:
        # Load audio
        print("Loading audio file...")
        audio, sr = audio_processor.load_audio(test_file)
        
        if audio is None:
            print("❌ Failed to load audio file")
            return False
            
        print(f"✅ Audio loaded: {len(audio)} samples at {sr} Hz")
        print(f"Duration: {len(audio) / sr:.2f} seconds")
        
        # Test tempo detection
        print("\n1. Testing tempo and beat detection...")
        tempo_info = audio_processor.detect_tempo_and_beats(audio, sr)
        if tempo_info:
            print(f"✅ Detected BPM: {tempo_info['bpm']:.1f}")
            print(f"✅ Dynamic BPM: {tempo_info['bpm_dynamic']:.1f}")
            print(f"✅ Found {len(tempo_info['beat_times'])} beats")
        else:
            print("❌ Tempo detection failed")
        
        # Test key detection
        print("\n2. Testing key detection...")
        key_info = audio_processor.detect_key(audio, sr)
        if key_info:
            print(f"✅ Detected key: {key_info['key']}")
            print(f"✅ Confidence: {key_info['confidence']:.3f}")
        else:
            print("❌ Key detection failed")
            
        # Test melody extraction
        print("\n3. Testing melody extraction...")
        melody_info = audio_processor.extract_melody(audio, sr)