"""
Modal HifiGAN Integration for User Interface
Extends the existing user interface with Modal HifiGAN enhancement
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.modal_hifigan import ModalHifiGANClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalEnhancedInterface:
    """Enhanced interface with Modal HifiGAN integration"""
    
    def __init__(self):
        self.modal_client = None
        self.use_modal_enhancement = False
        
    def setup_modal_hifigan(self) -> bool:
        """Initialize Modal HifiGAN client"""
        try:
            logger.info("🚀 Initializing Modal HifiGAN...")
            self.modal_client = ModalHifiGANClient()
            
            # Setup models on first run
            logger.info("📦 Setting up HifiGAN models on Modal...")
            self.modal_client.setup_models()
            
            self.use_modal_enhancement = True
            logger.info("✅ Modal HifiGAN ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Modal HifiGAN setup failed: {e}")
            logger.info("📝 Falling back to local processing")
            self.use_modal_enhancement = False
            return False    
    def enhance_vocals_with_modal(self, vocals_path: str) -> str:
        """Enhance vocals using Modal HifiGAN"""
        if not self.use_modal_enhancement or not self.modal_client:
            logger.warning("⚠️ Modal enhancement not available, using original audio")
            return vocals_path
        
        try:
            logger.info("🎤 Enhancing vocals with Modal HifiGAN...")
            enhanced_path = self.modal_client.enhance_audio(vocals_path)
            logger.info(f"✅ Modal enhancement complete: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            logger.error(f"❌ Modal enhancement failed: {e}")
            logger.info("📝 Using original vocals")
            return vocals_path
    
    def enhance_existing_audio(self, audio_path: str, output_path: str = None) -> str:
        """Enhance any existing audio file with Modal HifiGAN"""
        if not self.use_modal_enhancement:
            if not self.setup_modal_hifigan():
                raise Exception("Modal HifiGAN setup failed")
        
        logger.info(f"🎵 Enhancing audio file: {audio_path}")
        enhanced_path = self.modal_client.enhance_audio(audio_path, output_path)
        return enhanced_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Modal HifiGAN Audio Enhancement")
    parser.add_argument("audio_file", nargs='?', help="Path to audio file to enhance")
    parser.add_argument("--output", "-o", help="Output path for enhanced audio")
    parser.add_argument("--setup-only", action="store_true", help="Only setup Modal models")
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = ModalEnhancedInterface()
    
    if args.setup_only:
        print("🚀 Setting up Modal HifiGAN models...")
        success = interface.setup_modal_hifigan()
        if success:
            print("✅ Setup complete!")
        else:
            print("❌ Setup failed!")
            exit(1)
    elif args.audio_file:
        # Setup and enhance audio
        if interface.setup_modal_hifigan():
            try:
                enhanced_file = interface.enhance_existing_audio(args.audio_file, args.output)
                print(f"🎉 Enhanced audio saved: {enhanced_file}")
            except Exception as e:
                print(f"❌ Enhancement failed: {e}")
                exit(1)
        else:
            print("❌ Modal setup failed!")
            exit(1)
    else:
        print("❌ Please provide an audio file or use --setup-only")
        parser.print_help()
        exit(1)
