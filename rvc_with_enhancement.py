#!/usr/bin/env python3
"""
🎤 RVC + HifiGAN Enhancement Pipeline
Convert voice with Barnabas RVC, then enhance with Modal HifiGAN for crystal clear quality
"""

import logging
from pathlib import Path
from modal_rvc_service import ModalRVCClient
from pipeline.modal_hifigan import ModalHifiGANClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RVCEnhancementPipeline:
    """Complete RVC + Enhancement pipeline"""
    
    def __init__(self):
        self.rvc_client = ModalRVCClient()
        self.hifigan_client = ModalHifiGANClient()
        
    def setup