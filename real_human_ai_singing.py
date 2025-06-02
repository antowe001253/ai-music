"""
Real Human-like AI Singing Voice Generation
Using actual TTS/Singing models instead of synthetic formants
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_and_test_bark():
    """Install and test Bark for singing"""
    try:
        logger.info("🚀 Installing Bark AI TTS...")
        subprocess.run([sys.executable, "-m", "pip", "install", "bark"], check=True)
        
        logger.info("🎤 Testing Bark singing...")
        
        # Test Bark singing
        from bark import SAMPLE_RATE, generate_audio, preload_models
        import soundfile as sf
        
        # Preload models
        logger.info("📦 Loading Bark models (this may take a few minutes)...")
        preload_models()
        
        # Test sentences with singing notation
        test_lyrics = [
            "♪ Hello world, singing with AI ♪",
            "♪ This is a beautiful song ♪", 
            "♪ Amazing technology creates music ♪"
        ]
        
        for i, lyrics in enumerate(test_lyrics):
            logger.info(f"🎵 Generating: {lyrics}")
            
            # Generate with Bark
            audio_array = generate_audio(lyrics)
            
            # Save result
            output_file = f"bark_singing_{i+1}.wav"
            sf.write(output_file, audio_array, SAMPLE_RATE)
            
            logger.info(f"✅ Bark singing saved: {output_file}")
            
            # Enhance with Modal HifiGAN
            logger.info("🚀 Enhancing with Modal HifiGAN...")
            result = subprocess.run([
                "python", "modal_enhance_simple.py", output_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                enhanced_file = output_file.replace('.wav', '_enhanced.wav')
                logger.info(f"✅ Enhanced: {enhanced_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Bark failed: {e}")
        return False

def install_and_test_coqui_tts():
    """Install and test Coqui TTS"""
    try:
        logger.info("🚀 Installing Coqui TTS...")
        subprocess.run([sys.executable, "-m", "pip", "install", "coqui-tts"], check=True)
        
        logger.info("🎤 Testing Coqui TTS...")
        
        from TTS.api import TTS
        
        # Initialize TTS with a good voice model
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        
        test_sentences = [
            "Hello world, this is AI singing",
            "Beautiful music generation technology", 
            "Amazing artificial intelligence voice"
        ]
        
        for i, text in enumerate(test_sentences):
            output_file = f"coqui_tts_{i+1}.wav"
            
            logger.info(f"🎵 Generating: {text}")
            tts.tts_to_file(text=text, file_path=output_file)
            
            logger.info(f"✅ Coqui TTS saved: {output_file}")
            
            # Enhance with Modal HifiGAN
            result = subprocess.run([
                "python", "modal_enhance_simple.py", output_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                enhanced_file = output_file.replace('.wav', '_enhanced.wav')
                logger.info(f"✅ Enhanced: {enhanced_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Coqui TTS failed: {e}")
        return False

def setup_eleven_labs_guide():
    """Guide for using ElevenLabs (best quality)"""
    
    print("""
🎤 ELEVENLABS AI - BEST QUALITY OPTION
=====================================

ElevenLabs provides the highest quality AI singing voices.

SETUP:
1. Go to elevenlabs.io
2. Sign up (free tier available)
3. Use their Voice Design or choose existing voices
4. Generate singing audio
5. Download the results

FEATURES:
✅ Extremely realistic human voices
✅ Singing capability  
✅ Multiple languages
✅ Custom voice cloning
✅ Professional quality output

COST: ~$5/month for basic plan

This will give you genuine human-sounding singing voice.
    """)

def setup_suno_ai_guide():
    """Guide for using Suno AI (full songs)"""
    
    print("""
🎵 SUNO AI - COMPLETE SONG GENERATION  
====================================

Suno AI creates complete songs with lyrics and music.

SETUP:
1. Go to suno.ai
2. Create free account
3. Enter lyrics and style description
4. Generate complete songs

EXAMPLE PROMPTS:
- "Pop song about AI technology, upbeat female vocals"
- "Ballad about friendship, emotional male voice"
- "Electronic dance track with robotic vocals"

FEATURES:
✅ Complete song generation
✅ Multiple genres and styles
✅ Professional quality
✅ Lyrics + music + vocals
✅ Free tier available

This creates full songs, not just vocals.
    """)

def try_system_tts():
    """Try system built-in TTS as fallback"""
    
    logger.info("🎤 Trying system TTS...")
    
    test_sentences = [
        "Hello world, this is a test of AI singing",
        "Beautiful music from artificial intelligence",
        "Technology creates amazing vocal sounds"
    ]
    
    try:
        # macOS
        if sys.platform == "darwin":
            for i, text in enumerate(test_sentences):
                output_file = f"system_tts_{i+1}.wav"
                
                # Use say command with better voice
                cmd = [
                    "say", "-v", "Samantha", "-o", output_file, text
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ System TTS saved: {output_file}")
                    
                    # Convert to proper format and enhance
                    # Convert AIFF to WAV
                    convert_cmd = [
                        "ffmpeg", "-i", output_file, "-y", 
                        output_file.replace('.wav', '_converted.wav')
                    ]
                    subprocess.run(convert_cmd, capture_output=True)
                    
                    # Enhance with Modal
                    converted_file = output_file.replace('.wav', '_converted.wav')
                    if Path(converted_file).exists():
                        subprocess.run([
                            "python", "modal_enhance_simple.py", converted_file
                        ], capture_output=True)
            
            return True
        
    except Exception as e:
        logger.error(f"❌ System TTS failed: {e}")
    
    return False

if __name__ == "__main__":
    print("🎤 REAL HUMAN AI SINGING VOICE GENERATION")
    print("=" * 50)
    print()
    
    success = False
    
    # Try Bark first (best open-source option)
    print("1️⃣ Trying Bark AI (realistic singing)...")
    if install_and_test_bark():
        print("✅ Bark successful! Check bark_singing_*_enhanced.wav files")
        success = True
    else:
        print("❌ Bark failed")
    
    # Try Coqui TTS
    if not success:
        print("\n2️⃣ Trying Coqui TTS...")
        if install_and_test_coqui_tts():
            print("✅ Coqui TTS successful! Check coqui_tts_*_enhanced.wav files")
            success = True
        else:
            print("❌ Coqui TTS failed")
    
    # Try system TTS as fallback
    if not success:
        print("\n3️⃣ Trying system TTS...")
        if try_system_tts():
            print("✅ System TTS successful! Check system_tts_*_enhanced.wav files")
            success = True
        else:
            print("❌ System TTS failed")
    
    # Show commercial options
    print("\n" + "="*50)
    setup_eleven_labs_guide()
    print("\n" + "="*50)  
    setup_suno_ai_guide()
    
    if success:
        print("\n🎉 SUCCESS! You now have real human-like AI voices!")
        print("🎧 Listen to the generated files - they should sound much more human!")
    else:
        print("\n💡 RECOMMENDATION: Use ElevenLabs or Suno AI for best results")
        print("🎤 The commercial options will give you genuine human singing voice")
