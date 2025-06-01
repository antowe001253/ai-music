#!/usr/bin/env python3
"""
MusicGen Integration for Automated Music Pipeline
Generates instrumental music that can be used with Diff-SVC for vocal synthesis
"""

import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile
import numpy as np
import os
import argparse
import json
from pathlib import Path

class MusicGenPipeline:
    def __init__(self, model_name="facebook/musicgen-small", device="auto"):
        """
        Initialize MusicGen pipeline
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.processor = None
        self.model = None
        
    def _get_device(self, device):
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the MusicGen model and processor"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        
        if self.device != "cpu":
            self.model = self.model.to(self.device)
            
        print(f"✅ Model loaded successfully on {self.device}")
        return True