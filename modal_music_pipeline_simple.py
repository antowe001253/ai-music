#!/usr/bin/env python3
"""
Modal Cloud Music Generation - Alternative Approach
Uses a different, more stable music generation method
"""

import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("diff-svc-alternative")

# Simple, stable image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pretty_midi>=0.2.10",
        "mido>=1.2.10"
    ])
    .apt_install(["ffmpeg", "timidity", "fluid-soundfont-gm"])
)

@app.function(
    image=image,
    cpu=4.0,
    memory=8000,
    timeout=1800,
    volumes={}
)
def generate_simple_music(prompt: str, duration: int = 30):
    """
    Generate simple procedural music based on prompt
    Much more reliable than complex AI models
    """
    import numpy as np
    import scipy.io.wavfile as wavfile
    from pathlib import Path
    import math
    
    work_dir = Path("/tmp/simple_music")
    work_dir.mkdir(exist_ok=True)
    
    try:
        print(f"🎵 Generating simple music for: {prompt}")
        print(f"⏱️ Duration: {duration}s")
        
        sample_rate = 44100
        total_samples = int(duration * sample_rate)
        
        # Create time array
        t = np.linspace(0, duration, total_samples, False)
        
        # Generate Christmas carol-style music based on prompt
        audio = np.zeros(total_samples)
        
        if "christmas" in prompt.lower() or "carol" in prompt.lower():
            # Christmas carol melody (simple version of "Silent Night" style)
            print("🎄 Generating Christmas carol melody...")
            
            # Main melody notes (frequencies in Hz)
            notes = [
                523.25,  # C5
                587.33,  # D5
                659.25,  # E5
                698.46,  # F5
                783.99,  # G5
                880.00,  # A5
                987.77,  # B5
                1046.50  # C6
            ]
            
            # Simple Christmas melody pattern
            melody_pattern = [0, 0, 4, 0, 2, 1, 0, 0, 4, 4, 6, 4, 2, 2, 0]
            note_duration = duration / len(melody_pattern)
            
            for i, note_idx in enumerate(melody_pattern):
                start_time = i * note_duration
                end_time = (i + 1) * note_duration
                
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                if end_sample > total_samples:
                    end_sample = total_samples
                
                note_samples = end_sample - start_sample
                note_t = np.linspace(0, note_duration, note_samples, False)
                
                # Generate note with harmonics
                freq = notes[note_idx]
                note_wave = (
                    0.5 * np.sin(2 * np.pi * freq * note_t) +           # Fundamental
                    0.25 * np.sin(2 * np.pi * freq * 2 * note_t) +      # 2nd harmonic
                    0.125 * np.sin(2 * np.pi * freq * 3 * note_t)       # 3rd harmonic
                )
                
                # Add envelope (fade in/out)
                envelope = np.ones(note_samples)
                fade_samples = min(1000, note_samples // 10)
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
                
                note_wave *= envelope
                audio[start_sample:end_sample] += note_wave
            
            # Add simple harmony/bass line
            print("🎼 Adding harmony...")
            harmony_notes = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5
            harmony_pattern = [0, 2, 1, 0, 3, 2, 0, 0, 2, 1, 3, 2, 1, 1, 0]
            
            for i, note_idx in enumerate(harmony_pattern):
                start_time = i * note_duration
                end_time = (i + 1) * note_duration
                
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                if end_sample > total_samples:
                    end_sample = total_samples
                
                note_samples = end_sample - start_sample
                note_t = np.linspace(0, note_duration, note_samples, False)
                
                freq = harmony_notes[note_idx]
                harmony_wave = 0.3 * np.sin(2 * np.pi * freq * note_t)
                
                # Envelope
                envelope = np.ones(note_samples)
                fade_samples = min(500, note_samples // 20)
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
                
                harmony_wave *= envelope
                audio[start_sample:end_sample] += harmony_wave
        
        else:
            # Generic instrumental music
            print("🎶 Generating generic instrumental music...")
            
            # Simple chord progression
            base_freq = 220  # A3
            chord_freqs = [
                [base_freq, base_freq * 5/4, base_freq * 3/2],      # A major
                [base_freq * 4/3, base_freq * 5/3, base_freq * 2],  # D major  
                [base_freq * 3/2, base_freq * 15/8, base_freq * 9/4], # E major
                [base_freq, base_freq * 5/4, base_freq * 3/2]       # A major
            ]
            
            chord_duration = duration / 4
            
            for chord_idx, chord in enumerate(chord_freqs):
                start_time = chord_idx * chord_duration
                end_time = (chord_idx + 1) * chord_duration
                
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                if end_sample > total_samples:
                    end_sample = total_samples
                
                chord_samples = end_sample - start_sample
                chord_t = np.linspace(0, chord_duration, chord_samples, False)
                
                chord_wave = np.zeros(chord_samples)
                for freq in chord:
                    chord_wave += 0.2 * np.sin(2 * np.pi * freq * chord_t)
                
                # Add some rhythm
                rhythm_envelope = np.abs(np.sin(2 * np.pi * 2 * chord_t))
                chord_wave *= rhythm_envelope
                
                audio[start_sample:end_sample] += chord_wave
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Apply gentle reverb effect
        print("🔊 Adding reverb...")
        reverb_delay = int(0.1 * sample_rate)  # 0.1 second delay
        reverb_audio = np.zeros_like(audio)
        reverb_audio[reverb_delay:] = audio[:-reverb_delay] * 0.3
        audio = audio + reverb_audio
        
        # Final normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save audio
        output_path = work_dir / "simple_music.wav"
        wavfile.write(str(output_path), sample_rate, audio_int16)
        
        with open(output_path, 'rb') as f:
            audio_data = f.read()
        
        print(f"✅ Generated {duration}s of music: {len(audio_data)} bytes")
        
        return {
            "success": True,
            "prompt": prompt,
            "method": "procedural_generation",
            "files": {"audio": audio_data},
            "metadata": {
                "sample_rate": sample_rate,
                "duration": duration,
                "file_size": len(audio_data),
                "type": "christmas_carol" if "christmas" in prompt.lower() else "instrumental"
            }
        }
        
    except Exception as e:
        print(f"❌ Simple generation error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.local_entrypoint()
def main(prompt: str = "Christmas carol song", duration: int = 30):
    """Simple, reliable music generation"""
    print("🎵 Running SIMPLE PROCEDURAL music generation")
    print("=" * 50)
    print("This approach avoids AI model issues entirely!")
    
    result = generate_simple_music.remote(prompt, duration)
    
    if result["success"]:
        print("🎉 Simple generation SUCCESS!")
        print(f"🎵 Method: {result['method']}")
        print(f"📝 Prompt: {result['prompt']}")
        print(f"⏱️ Duration: {result['metadata']['duration']}s")
        print(f"🎼 Type: {result['metadata']['type']}")
        
        output_dir = Path("simple_output")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"simple_christmas_carol.wav"
        output_path = output_dir / filename
        
        with open(output_path, 'wb') as f:
            f.write(result["files"]["audio"])
        
        print(f"💾 Music saved to: {output_path}")
        print("🎵 No AI model issues - simple procedural generation works!")
        print("\n🎊 SUCCESS! Your Christmas carol music is ready!")
        
    else:
        print(f"❌ Simple generation failed: {result['error']}")

if __name__ == "__main__":
    print("Run with: modal run modal_music_pipeline_simple.py")
