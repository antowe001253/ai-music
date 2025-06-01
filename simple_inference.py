#!/usr/bin/env python3
"""
Simple Diff-SVC Inference Script
Usage: python simple_inference.py input.wav output.wav [pitch_shift]
"""

import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python simple_inference.py input.wav output.wav [pitch_shift]")
        print("  input.wav    - Input audio file") 
        print("  output.wav   - Output audio file")
        print("  pitch_shift  - Pitch shift in semitones (optional, default: 0)")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pitch_shift = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        return
        
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"🎵 Processing: {input_path}")
    print(f"📁 Output: {output_path}")
    if pitch_shift != 0:
        print(f"🎼 Pitch shift: {pitch_shift:+d} semitones")
    
    # Use the existing infer.py script
    cmd = f"python infer.py --config config.yaml --input '{input_path}' --output '{output_path}'"
    if pitch_shift != 0:
        cmd += f" --transpose {pitch_shift}"
    
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print(f"✨ Inference completed! Output saved to: {output_path}")
    else:
        print("❌ Error during inference")

if __name__ == "__main__":
    main()
