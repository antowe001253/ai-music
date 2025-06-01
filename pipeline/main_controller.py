#!/usr/bin/env python3
"""
Main Controller - Phase 3: Complete Automated Music Pipeline
Integrates Phase 1-2 intelligence with Diff-SVC for end-to-end automation

Usage:
    python pipeline/main_controller.py "upbeat electronic dance track with emotional vocals"
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
import json
import numpy as np

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.prompt_intelligence import PromptIntelligence
from pipeline.orchestration_engine import OrchestrationEngine
from pipeline.advanced_audio_processing import AdvancedAudioProcessor
from pipeline.music_generation.melody_generation_system_fixed import MelodyGenerationSystem

class AutomatedMusicPipeline:
    """
    Main controller for complete automated music generation pipeline.
    Integrates all Phase 1-2 components with Diff-SVC for vocal synthesis.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the complete automation pipeline."""
        self.config_path = config_path
        self.project_root = Path(__file__).parent.parent
        self.outputs_dir = self.project_root / "outputs" / "automated_pipeline"
        self.temp_dir = self.project_root / "temp"
        
        # Create output directories
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.logger.info("🚀 Initializing Automated Music Pipeline...")
        self._initialize_components()
        
        # Track processing state
        self.current_session = None
        
    def _setup_logging(self):
        """Set up comprehensive logging system."""
        log_file = self.outputs_dir / "pipeline.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Phase 2 Intelligence Components
            self.prompt_intelligence = PromptIntelligence()
            self.orchestration_engine = OrchestrationEngine()
            self.audio_processor = AdvancedAudioProcessor()
            
            # Phase 1 Generation Components  
            self.melody_system = MelodyGenerationSystem()
            
            # Load MusicGen model
            self.logger.info("Loading MusicGen model...")
            success = self.melody_system.load_melody_model()
            if not success:
                self.logger.warning("MusicGen model failed to load, using fallback")
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def generate_complete_song(self, prompt: str, **kwargs) -> Dict:
        """
        Generate complete song from single prompt.
        
        Args:
            prompt: Natural language description of desired music
            **kwargs: Additional options (vocal_style, output_format, etc.)
            
        Returns:
            Dict with paths to generated files and metadata
        """
        session_id = str(uuid.uuid4())[:8]
        
        # Create session directory early to save intermediate files
        session_dir = self.outputs_dir / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session = {
            'id': session_id,
            'session_dir': session_dir,
            'prompt': prompt,
            'start_time': time.time()
        }
        
        self.logger.info(f"🎵 Starting complete song generation")
        self.logger.info(f"📝 Prompt: {prompt}")
        self.logger.info(f"🔖 Session ID: {session_id}")
        self.logger.info(f"📁 Session directory: {session_dir}")
        
        try:
            # Step 1: Parse prompt and extract musical parameters
            self.logger.info("🧠 Step 1: Analyzing prompt...")
            prompt_params = self.prompt_intelligence.parse_prompt(prompt)
            self._log_prompt_analysis(prompt_params)
            
            # Step 2: Generate instrumental track
            self.logger.info("🎹 Step 2: Generating instrumental track...")
            instrumental_path = self._generate_instrumental(prompt_params, session_id)
            
            # Step 3: Generate and extract melody
            self.logger.info("🎼 Step 3: Generating vocal melody...")
            melody_info = self._generate_melody(prompt_params, session_id)
            
            # Step 4: Synthesize vocals using Diff-SVC
            self.logger.info("🎤 Step 4: Synthesizing vocals...")
            vocal_path = self._synthesize_vocals(melody_info, prompt_params, session_id)
            
            # Step 5: Mix and master final track
            self.logger.info("🎚️ Step 5: Mixing and mastering...")
            final_mix = self._create_final_mix(instrumental_path, vocal_path, prompt_params, session_id)
            
            # Step 6: Package results
            results = self._package_results(final_mix, session_id, prompt, prompt_params)
            
            self.logger.info(f"✨ Complete song generation successful!")
            self.logger.info(f"📁 Output directory: {results['output_dir']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Song generation failed: {e}")
            self._cleanup_session(session_id)
            raise
            
    def _log_prompt_analysis(self, params):
        """Log detailed prompt analysis results."""
        self.logger.info("📊 Prompt Analysis Results:")
        # Handle both dict and MusicalParameters object
        if hasattr(params, 'genre'):
            self.logger.info(f"   🎵 Genre: {params.genre}")
            self.logger.info(f"   ⏱️ Tempo: {params.tempo} BPM")
            self.logger.info(f"   🎭 Mood: {params.mood}")
            self.logger.info(f"   🎹 Key: {params.key}")
            self.logger.info(f"   🎸 Instruments: {', '.join(params.instruments)}")
            self.logger.info(f"   🎤 Vocal Style: {getattr(params, 'vocal_style', 'auto')}")
        else:
            self.logger.info(f"   🎵 Genre: {params.genre or 'unknown'}")
            self.logger.info(f"   ⏱️ Tempo: {params.tempo or 'auto'} BPM")
            self.logger.info(f"   🎭 Mood: {params.mood or 'neutral'}")
            self.logger.info(f"   🎹 Key: {params.key or 'auto'}")
            self.logger.info(f"   🎸 Instruments: {', '.join(params.instruments or [])}")
            self.logger.info(f"   🎤 Vocal Style: {params.vocal_style or 'auto'}")
        
    def _generate_instrumental(self, params, session_id: str) -> str:
        """Generate instrumental background track."""
        self.logger.info("🎹 Generating instrumental background...")
        
        # Create instrumental-specific prompt
        instrumental_prompt = self._create_instrumental_prompt(params)
        self.logger.info(f"   📝 Instrumental prompt: {instrumental_prompt}")
        
        # Generate using MusicGen
        output_path = self.temp_dir / f"{session_id}_instrumental.wav"
        
        try:
            # Default duration
            duration = 30
            
            # Use melody_conditioned method for instrumental generation
            result = self.melody_system.generate_melody_conditioned(
                text_description=instrumental_prompt,
                duration=duration
            )
            
            if result is None:
                raise Exception("MusicGen generation returned None")
            
            # Handle both dict and tuple returns for compatibility
            if isinstance(result, dict):
                audio_values = result['audio']
                sample_rate = result['sample_rate']
            else:
                audio_values, sample_rate = result
            
            if audio_values is None:
                raise Exception("MusicGen generation returned None")
            
            # Save instrumental - handle different audio formats
            if isinstance(audio_values, np.ndarray):
                # Fixed version returns numpy array directly
                audio_to_save = audio_values
            else:
                # Original version returns tensor with batch dimension
                audio_to_save = audio_values[0]
                
            # Save to both temp AND session directory
            temp_path = str(output_path)
            session_path = str(self.current_session['session_dir'] / f"step2_instrumental.wav")
            
            self._save_audio(audio_to_save, sample_rate, temp_path)
            self._save_audio(audio_to_save, sample_rate, session_path)
            
            self.logger.info(f"   💾 Saved instrumental to: {session_path}")
            self.logger.info(f"   📊 Generated: {duration}s instrumental track")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Instrumental generation failed: {e}")
            raise
            
    def _get_param_value(self, params, key, default=None):
        """Get parameter value, handling both dict and MusicalParameters object"""
        if hasattr(params, key):
            return getattr(params, key, default)
        else:
            return getattr(params, key, default)
    
    def _create_instrumental_prompt(self, params) -> str:
        """Create optimized prompt for instrumental generation."""
        # Use helper method for consistent parameter access
        genre = self._get_param_value(params, 'genre', '')
        mood = self._get_param_value(params, 'mood', '')
        instruments = self._get_param_value(params, 'instruments', [])
        
        # Build instrumental-focused prompt
        prompt_parts = []
        
        if genre:
            prompt_parts.append(f"{genre} instrumental")
        if mood:
            prompt_parts.append(f"{mood} mood")
        if instruments:
            inst_str = ", ".join(instruments)
            prompt_parts.append(f"featuring {inst_str}")
            
        prompt_parts.append("no vocals, instrumental only")
        
        return ", ".join(prompt_parts)
        
    def _generate_melody(self, params, session_id: str) -> Dict:
        """Generate vocal melody line."""
        self.logger.info("🎼 Generating vocal melody...")
        
        # Create melody-specific prompt  
        melody_prompt = self._create_melody_prompt(params)
        self.logger.info(f"   📝 Melody prompt: {melody_prompt}")
        
        try:
            # Generate melody using MusicGen Melody
            output_path = self.temp_dir / f"{session_id}_melody.wav"
            
            # Default duration
            duration = 30
            
            # Use melody_conditioned method
            result = self.melody_system.generate_melody_conditioned(
                text_description=melody_prompt,
                duration=duration
            )
            
            if result is None:
                raise Exception("MusicGen melody generation returned None")
            
            # Handle both dict and tuple returns for compatibility
            if isinstance(result, dict):
                audio_values = result['audio']
                sample_rate = result['sample_rate']
            else:
                audio_values, sample_rate = result
            
            if audio_values is None:
                raise Exception("MusicGen melody generation returned None")
            
            # Save melody - handle different audio formats
            if isinstance(audio_values, np.ndarray):
                # Fixed version returns numpy array directly
                audio_to_save = audio_values
            else:
                # Original version returns tensor with batch dimension
                audio_to_save = audio_values[0]
                
            # Save to both temp AND session directory
            temp_path = str(output_path)
            session_path = str(self.current_session['session_dir'] / f"step3_vocal_melody.wav")
            
            self._save_audio(audio_to_save, sample_rate, temp_path)
            self._save_audio(audio_to_save, sample_rate, session_path)
            
            self.logger.info(f"   💾 Saved vocal melody to: {session_path}")
            self.logger.info(f"   📊 Melody: Generated vocal melody line")
            
            return {
                'path': str(output_path),
                'analysis': {'avg_pitch': 440.0},  # Placeholder
                'target_key': params.key,
                'target_tempo': params.tempo
            }
            
        except Exception as e:
            self.logger.error(f"❌ Melody generation failed: {e}")
            raise
            
    def _create_melody_prompt(self, params) -> str:
        """Create optimized prompt for melody generation."""
        # Use helper method for consistent parameter access
        genre = self._get_param_value(params, 'genre', '')
        mood = self._get_param_value(params, 'mood', '')
        vocal_style = self._get_param_value(params, 'vocal_style', '')
        
        # Build melody-focused prompt
        prompt_parts = []
        
        if vocal_style:
            prompt_parts.append(f"{vocal_style} vocal melody")
        elif genre:
            prompt_parts.append(f"{genre} vocal melody")
        else:
            prompt_parts.append("vocal melody")
            
        if mood:
            prompt_parts.append(f"{mood} feeling")
            
        prompt_parts.append("clear melodic line, single voice")
        
        return ", ".join(prompt_parts)
        
    def _synthesize_vocals(self, melody_info: Dict, params: Dict, session_id: str) -> str:
        """Synthesize vocals using Diff-SVC."""
        self.logger.info("🎤 Synthesizing vocals with Diff-SVC...")
        
        melody_path = melody_info['path']
        output_path = self.temp_dir / f"{session_id}_vocals.wav"
        
        try:
            # Calculate pitch adjustment based on key analysis
            pitch_shift = self._calculate_vocal_pitch_shift(melody_info, params)
            
            if pitch_shift != 0:
                self.logger.info(f"   🎼 Applying pitch shift: {pitch_shift:+d} semitones")
            
            # Run Diff-SVC inference
            self._run_diffsvc_inference(melody_path, str(output_path), pitch_shift)
            
            # Verify output was created
            if not os.path.exists(output_path):
                raise Exception("Diff-SVC inference did not generate output file")
                
            self.logger.info(f"   ✅ Vocals synthesized successfully")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Vocal synthesis failed: {e}")
            raise
            
    def _calculate_vocal_pitch_shift(self, melody_info: Dict, params: Dict) -> int:
        """Calculate optimal pitch shift for vocals based on target key."""
        # Basic pitch adjustment logic
        # Can be enhanced with more sophisticated key matching
        target_key = self._get_param_value(params, 'key')
        if not target_key:
            return 0
            
        # Simplified pitch shift calculation
        # In production, this would use more sophisticated music theory
        return 0  # For now, no automatic pitch shift
        
    def _run_diffsvc_inference(self, input_path: str, output_path: str, pitch_shift: int = 0):
        """Run Diff-SVC inference using the existing system."""
        cmd_parts = [
            "python", str(self.project_root / "infer.py"),
            "--config", self.config_path,
            "--input", f"'{input_path}'", 
            "--output", f"'{output_path}'"
        ]
        
        if pitch_shift != 0:
            cmd_parts.extend(["--transpose", str(pitch_shift)])
            
        cmd = " ".join(cmd_parts)
        self.logger.info(f"   🖥️ Running: {cmd}")
        
        result = os.system(cmd)
        if result != 0:
            raise Exception(f"Diff-SVC inference failed with exit code {result}")
            
    def _create_final_mix(self, instrumental_path: str, vocal_path: str, params: Dict, session_id: str) -> Dict:
        """Create final mixed and mastered track."""
        self.logger.info("🎚️ Creating final mix...")
        
        try:
            # Load tracks
            instrumental = self.audio_processor.load_audio(instrumental_path)
            vocals = self.audio_processor.load_audio(vocal_path)
            
            # Synchronize timing and tempo
            self.logger.info("   ⏱️ Synchronizing tracks...")
            synchronized_tracks = self.orchestration_engine.synchronize_tracks([
                {'audio': instrumental, 'type': 'instrumental'},
                {'audio': vocals, 'type': 'vocals'}
            ])
            
            # Apply track-specific processing
            self.logger.info("   🎛️ Processing individual tracks...")
            processed_instrumental = self.audio_processor.process_instrumental_track(
                synchronized_tracks[0]['audio'], params
            )
            processed_vocals = self.audio_processor.process_vocal_track(
                synchronized_tracks[1]['audio'], params
            )
            
            # Create stereo mix
            self.logger.info("   🎧 Creating stereo mix...")
            final_mix = self.audio_processor.create_stereo_mix({
                'vocals': processed_vocals,
                'instrumental': processed_instrumental
            })
            
            # Master the final track
            self.logger.info("   ✨ Mastering final track...")
            mastered = self.audio_processor.master_track(final_mix, params)
            
            # Save all versions
            output_paths = self._save_mix_versions(mastered, processed_vocals, processed_instrumental, session_id)
            
            return output_paths
            
        except Exception as e:
            self.logger.error(f"❌ Final mix creation failed: {e}")
            raise
            
    def _save_mix_versions(self, final_mix, vocals, instrumental, session_id: str) -> Dict:
        """Save different versions of the mix."""
        session_dir = self.outputs_dir / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)
        
        paths = {}
        
        # Save final mix
        final_path = session_dir / "complete_song.wav"
        self.audio_processor.save_audio(final_mix, str(final_path))
        paths['final_mix'] = str(final_path)
        
        # Save individual tracks
        vocal_path = session_dir / "vocals_only.wav"
        self.audio_processor.save_audio(vocals, str(vocal_path))
        paths['vocals'] = str(vocal_path)
        
        instrumental_path = session_dir / "instrumental_only.wav"  
        self.audio_processor.save_audio(instrumental, str(instrumental_path))
        paths['instrumental'] = str(instrumental_path)
        
        self.logger.info(f"   💾 Saved all tracks to: {session_dir}")
        
        return paths
        
    def _package_results(self, mix_paths: Dict, session_id: str, prompt: str, params: Dict) -> Dict:
        """Package final results with metadata."""
        session_dir = self.outputs_dir / f"session_{session_id}"
        
        # Create metadata
        metadata = {
            'session_id': session_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'prompt': prompt,
            'parameters': params,
            'files': mix_paths,
            'pipeline_version': 'Phase 3.0'
        }
        
        # Save metadata
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Create results summary
        results = {
            'session_id': session_id,
            'output_dir': str(session_dir),
            'files': mix_paths,
            'metadata': metadata,
            'success': True
        }
        
        return results
        
    def _save_audio(self, audio_tensor, sample_rate, output_path):
        """Save audio tensor to file."""
        import scipy.io.wavfile as wavfile
        import numpy as np
        
        # Convert tensor to numpy if needed
        if hasattr(audio_tensor, 'numpy'):
            audio_np = audio_tensor.numpy()
        else:
            audio_np = np.array(audio_tensor)
            
        # Ensure audio is in correct format
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)  # Convert to mono if stereo
            
        # Normalize to prevent clipping
        audio_np = audio_np / np.max(np.abs(audio_np)) * 0.95
        
        # Convert to int16
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Save as WAV
        wavfile.write(output_path, sample_rate, audio_int16)
        
    def _cleanup_session(self, session_id: str):
        """Clean up temporary files for a session."""
        temp_files = list(self.temp_dir.glob(f"{session_id}_*"))
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except:
                pass
                
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict]:
        """Generate multiple songs in batch."""
        self.logger.info(f"🔄 Starting batch generation for {len(prompts)} prompts...")
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            self.logger.info(f"📝 Processing {i}/{len(prompts)}: {prompt}")
            try:
                result = self.generate_complete_song(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"❌ Batch item {i} failed: {e}")
                results.append({'success': False, 'error': str(e), 'prompt': prompt})
                
        self.logger.info(f"✅ Batch complete: {sum(1 for r in results if r.get('success'))} successful")
        return results

def main():
    """Command line interface for the automated pipeline."""
    if len(sys.argv) < 2:
        print("🎵 Automated Music Pipeline - Phase 3")
        print("\nUsage:")
        print("  python pipeline/main_controller.py \"your music prompt here\"")
        print("\nExamples:")
        print("  python pipeline/main_controller.py \"upbeat electronic dance track with emotional vocals\"")
        print("  python pipeline/main_controller.py \"peaceful jazz ballad with soft female vocals\"")
        print("  python pipeline/main_controller.py \"energetic rock song with powerful vocals\"")
        return
        
    prompt = sys.argv[1]
    
    try:
        # Initialize pipeline
        pipeline = AutomatedMusicPipeline()
        
        # Generate song
        results = pipeline.generate_complete_song(prompt)
        
        # Display results
        print(f"\n✨ Song generation completed successfully!")
        print(f"📁 Output directory: {results['output_dir']}")
        print(f"🎵 Files generated:")
        for file_type, path in results['files'].items():
            print(f"   {file_type}: {path}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Generation cancelled by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
