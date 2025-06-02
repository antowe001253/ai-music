"""
Summary of Generated Vocal Files
Listen to these in order to test the pipeline progression
"""

import os
from pathlib import Path

def summarize_vocal_files():
    """Create a summary of all generated vocal files"""
    
    print("🎤 VOCAL GENERATION PIPELINE RESULTS")
    print("=" * 50)
    
    files_to_check = [
        # Original references
        ("reference_vocal.wav", "🎼 Original vocal reference"),
        ("better_vocal_reference.wav", "🎵 Improved vocal reference with formants"),
        
        # Diff-SVC outputs (raw)
        ("better_diffsvc_output_proper_audio_method1.wav", "🤖 Diff-SVC output (Griffin-Lim)"),
        
        # Modal HifiGAN enhanced
        ("better_diffsvc_output_proper_audio_method1_enhanced.wav", "✨ Modal HifiGAN enhanced"),
        
        # Structured versions (with phonemes)
        ("better_diffsvc_output_proper_audio_method1_enhanced_structured_1.wav", "🗣️ Structured: 'hello world'"),
        ("better_diffsvc_output_proper_audio_method1_enhanced_structured_1_enhanced.wav", "🎤 Final: 'hello world' + Modal"),
        
        ("better_diffsvc_output_proper_audio_method1_enhanced_structured_2.wav", "🗣️ Structured: 'la la la'"),
        ("better_diffsvc_output_proper_audio_method1_enhanced_structured_2_enhanced.wav", "🎤 Final: 'la la la' + Modal"),
        
        ("better_diffsvc_output_proper_audio_method1_enhanced_structured_3.wav", "🗣️ Structured: 'singing voice test'"),
        ("better_diffsvc_output_proper_audio_method1_enhanced_structured_3_enhanced.wav", "🎤 Final: 'singing voice test' + Modal"),
    ]
    
    print("\n🎧 RECOMMENDED LISTENING ORDER:")
    print()
    
    existing_files = []
    for filename, description in files_to_check:
        if Path(filename).exists():
            file_size = Path(filename).stat().st_size
            duration_est = file_size / (44100 * 2 * 2)  # Rough estimate
            
            existing_files.append((filename, description, duration_est))
            print(f"{len(existing_files):2d}. {description}")
            print(f"    📁 {filename}")
            print(f"    📊 ~{duration_est:.1f}s")
            print()
    
    print("🎯 WHAT TO EXPECT:")
    print()
    print("• Files 1-2: Input references (should sound vocal-like)")
    print("• File 3: Raw Diff-SVC (robotic but vocal)")  
    print("• File 4: Modal enhanced (cleaner, more natural)")
    print("• Files 5+: Structured versions (should sound more like words/syllables)")
    print()
    
    print("🔍 LISTENING TIPS:")
    print()
    print("✅ SUCCESS indicators:")
    print("  - Vocal timbre (sounds like voice, not instruments)")
    print("  - Pitch following melody")
    print("  - Syllable-like rhythm in structured versions")
    print()
    print("❌ If still too distorted:")
    print("  - The base Diff-SVC model may need different training data")
    print("  - Try commercial options (ElevenLabs, Suno AI)")
    print("  - Consider other open-source singing models")
    print()
    
    print("🚀 BEST RESULTS LIKELY IN:")
    print("  - better_diffsvc_output_proper_audio_method1_enhanced_structured_2_enhanced.wav")
    print("  - (The 'la la la' version with all enhancements)")
    print()
    
    return existing_files

if __name__ == "__main__":
    files = summarize_vocal_files()
    
    print(f"📁 Total files generated: {len(files)}")
    print("\n🎤 Your AI singing voice pipeline is complete!")
    print("🎧 Test these files to evaluate the results.")
