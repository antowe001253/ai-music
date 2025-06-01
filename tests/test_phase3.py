#!/usr/bin/env python3
"""
Phase 3 Integration Test Suite
Comprehensive testing for the complete automated pipeline

Tests:
- Main controller functionality
- Quality enhancement system
- User interface components
- Caching system
- Style presets
- End-to-end pipeline
"""

import os
import sys
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.main_controller import AutomatedMusicPipeline
from pipeline.quality_enhancement import QualityEnhancementSystem, QualityMetrics
from pipeline.user_interface import CacheManager, StylePresetManager, UserInterface

class TestMainController(unittest.TestCase):
    """Test the main automation controller."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Mock the pipeline components to avoid heavy initialization
        with patch('pipeline.main_controller.PromptIntelligence'), \
             patch('pipeline.main_controller.OrchestrationEngine'), \
             patch('pipeline.main_controller.AdvancedAudioProcessor'), \
             patch('pipeline.main_controller.MelodyGenerationSystem'):
            
            self.pipeline = AutomatedMusicPipeline()
            
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        self.assertTrue(hasattr(self.pipeline, 'prompt_intelligence'))
        self.assertTrue(hasattr(self.pipeline, 'orchestration_engine'))
        self.assertTrue(hasattr(self.pipeline, 'audio_processor'))
        self.assertTrue(hasattr(self.pipeline, 'melody_system'))
        
    def test_logging_setup(self):
        """Test logging configuration."""
        self.assertIsNotNone(self.pipeline.logger)
        self.assertTrue(self.pipeline.outputs_dir.exists())
        
    @patch('pipeline.main_controller.AutomatedMusicPipeline._generate_instrumental')
    @patch('pipeline.main_controller.AutomatedMusicPipeline._generate_melody')
    @patch('pipeline.main_controller.AutomatedMusicPipeline._synthesize_vocals')
    @patch('pipeline.main_controller.AutomatedMusicPipeline._create_final_mix')
    def test_complete_generation_workflow(self, mock_mix, mock_vocals, mock_melody, mock_instrumental):
        """Test complete song generation workflow."""
        # Mock the pipeline steps
        mock_instrumental.return_value = str(self.temp_dir / "instrumental.wav")
        mock_melody.return_value = {
            'path': str(self.temp_dir / "melody.wav"),
            'analysis': {'avg_pitch': 440.0}
        }
        mock_vocals.return_value = str(self.temp_dir / "vocals.wav")
        mock_mix.return_value = {
            'final_mix': str(self.temp_dir / "final.wav"),
            'vocals': str(self.temp_dir / "vocals_only.wav"),
            'instrumental': str(self.temp_dir / "instrumental_only.wav")
        }
        
        # Create mock files
        for filename in ["instrumental.wav", "melody.wav", "vocals.wav", "final.wav"]:
            (self.temp_dir / filename).touch()
            
        # Mock prompt intelligence
        with patch.object(self.pipeline, 'prompt_intelligence') as mock_intelligence:
            mock_intelligence.parse_prompt.return_value = {
                'genre': 'electronic',
                'tempo': 128,
                'mood': 'energetic'
            }
            
            # Test generation
            result = self.pipeline.generate_complete_song("test electronic track")
            
            self.assertTrue(result['success'])
            self.assertIn('session_id', result)
            self.assertIn('files', result)
            
    def test_prompt_analysis_logging(self):
        """Test prompt analysis logging."""
        test_params = {
            'genre': 'rock',
            'tempo': 120,
            'mood': 'powerful',
            'key': 'E',
            'instruments': ['guitar', 'drums'],
            'vocal_style': 'powerful'
        }
        
        # This should not raise an exception
        self.pipeline._log_prompt_analysis(test_params)
        
    def test_instrumental_prompt_creation(self):
        """Test instrumental prompt creation."""
        params = {
            'genre': 'jazz',
            'mood': 'relaxed',
            'instruments': ['piano', 'saxophone']
        }
        
        prompt = self.pipeline._create_instrumental_prompt(params)
        
        self.assertIn('jazz', prompt)
        self.assertIn('relaxed', prompt)
        self.assertIn('piano', prompt)
        self.assertIn('saxophone', prompt)
        self.assertIn('no vocals', prompt)
        
    def test_melody_prompt_creation(self):
        """Test melody prompt creation."""
        params = {
            'genre': 'pop',
            'mood': 'emotional',
            'vocal_style': 'smooth'
        }
        
        prompt = self.pipeline._create_melody_prompt(params)
        
        self.assertIn('smooth vocal melody', prompt)
        self.assertIn('emotional', prompt)

class TestQualityEnhancement(unittest.TestCase):
    """Test the quality enhancement system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.quality_system = QualityEnhancementSystem()
        
        # Create mock audio file
        self.test_audio_path = self.temp_dir / "test_audio.wav"
        self._create_mock_audio_file()
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def _create_mock_audio_file(self):
        """Create a mock audio file for testing."""
        import numpy as np
        import soundfile as sf
        
        # Generate simple test audio
        duration = 2.0  # seconds
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Stereo sine wave
        left = np.sin(2 * np.pi * 440 * t) * 0.5
        right = np.sin(2 * np.pi * 880 * t) * 0.5
        audio = np.column_stack([left, right])
        
        sf.write(str(self.test_audio_path), audio, sample_rate)
        
    def test_quality_metrics_dataclass(self):
        """Test QualityMetrics dataclass."""
        metrics = QualityMetrics(
            overall_score=8.5,
            dynamic_range=12.0,
            frequency_balance=7.5,
            stereo_width=0.6,
            vocal_clarity=8.0,
            background_separation=7.0,
            artifacts_score=9.0,
            loudness_lufs=-16.0,
            peak_level=-3.0
        )
        
        self.assertEqual(metrics.overall_score, 8.5)
        self.assertEqual(metrics.dynamic_range, 12.0)
        
    def test_quality_validation(self):
        """Test audio quality validation."""
        metrics = self.quality_system.validate_audio_quality(str(self.test_audio_path))
        
        self.assertIsInstance(metrics, QualityMetrics)
        self.assertGreaterEqual(metrics.overall_score, 0)
        self.assertLessEqual(metrics.overall_score, 10)
        self.assertGreaterEqual(metrics.dynamic_range, 0)
        self.assertGreaterEqual(metrics.stereo_width, 0)
        self.assertLessEqual(metrics.stereo_width, 1)
        
    def test_quality_thresholds(self):
        """Test quality threshold configuration."""
        thresholds = self.quality_system.quality_thresholds
        
        self.assertIn('minimum_overall', thresholds)
        self.assertIn('minimum_dynamic_range', thresholds)
        self.assertIn('target_lufs', thresholds)
        
    def test_quality_enhancement(self):
        """Test audio quality enhancement."""
        # This will use the mock audio file
        enhanced_path = self.quality_system.enhance_audio_quality(str(self.test_audio_path))
        
        self.assertTrue(os.path.exists(enhanced_path))
        self.assertNotEqual(enhanced_path, str(self.test_audio_path))
        
    def test_quality_report_generation(self):
        """Test quality report generation."""
        metrics = self.quality_system.validate_audio_quality(str(self.test_audio_path))
        report_path = self.quality_system.create_quality_report(
            str(self.test_audio_path), metrics, "test_session"
        )
        
        self.assertTrue(os.path.exists(report_path))
        
        # Verify report content
        with open(report_path, 'r') as f:
            report = json.load(f)
            
        self.assertIn('quality_metrics', report)
        self.assertIn('quality_assessment', report)
        self.assertIn('recommendations', report)

class TestCacheManager(unittest.TestCase):
    """Test the caching system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_manager = CacheManager(self.temp_dir / "cache")
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_cache_initialization(self):
        """Test cache manager initialization."""
        self.assertTrue(self.cache_manager.cache_dir.exists())
        self.assertIsInstance(self.cache_manager.cache_index, dict)
        
    def test_cache_key_generation(self):
        """Test cache key generation."""
        prompt1 = "electronic dance music"
        params1 = {'genre': 'electronic', 'tempo': 128}
        
        prompt2 = "Electronic Dance Music"  # Different case
        params2 = {'genre': 'electronic', 'tempo': 128}
        
        key1 = self.cache_manager._generate_cache_key(prompt1, params1)
        key2 = self.cache_manager._generate_cache_key(prompt2, params2)
        
        self.assertEqual(key1, key2)  # Should be the same (case insensitive)
        
    def test_cache_miss(self):
        """Test cache miss scenario."""
        result = self.cache_manager.get_cached_result("nonexistent prompt", {})
        self.assertIsNone(result)
        
    def test_cache_hit_miss_cycle(self):
        """Test complete cache cycle."""
        prompt = "test music"
        params = {'genre': 'test'}
        result_path = str(self.temp_dir / "test_result")
        metadata = {'test': 'data'}
        
        # Create mock result file
        Path(result_path).mkdir(parents=True, exist_ok=True)
        
        # Should be cache miss initially
        cached = self.cache_manager.get_cached_result(prompt, params)
        self.assertIsNone(cached)
        
        # Cache the result
        self.cache_manager.cache_result(prompt, params, result_path, metadata)
        
        # Should be cache hit now
        cached = self.cache_manager.get_cached_result(prompt, params)
        self.assertIsNotNone(cached)
        self.assertTrue(cached['cached'])
        self.assertEqual(cached['path'], result_path)

class TestStylePresetManager(unittest.TestCase):
    """Test the style preset system."""
    
    def setUp(self):
        """Set up test environment."""
        self.preset_manager = StylePresetManager()
        
    def test_preset_initialization(self):
        """Test preset manager initialization."""
        presets = self.preset_manager.list_presets()
        self.assertGreater(len(presets), 0)
        
        # Check for expected presets
        expected_presets = ['edm', 'pop_ballad', 'rock', 'jazz', 'acoustic']
        for preset_name in expected_presets:
            self.assertIn(preset_name, presets)
            
    def test_preset_retrieval(self):
        """Test preset retrieval."""
        edm_preset = self.preset_manager.get_preset('edm')
        self.assertIsNotNone(edm_preset)
        self.assertEqual(edm_preset.name, "Electronic Dance Music")
        
        # Test case insensitivity
        edm_preset2 = self.preset_manager.get_preset('EDM')
        self.assertIsNotNone(edm_preset2)
        
        # Test nonexistent preset
        fake_preset = self.preset_manager.get_preset('nonexistent')
        self.assertIsNone(fake_preset)
        
    def test_preset_content(self):
        """Test preset content structure."""
        jazz_preset = self.preset_manager.get_preset('jazz')
        
        self.assertIn('genre', jazz_preset.default_params)
        self.assertIn('tempo', jazz_preset.default_params)
        self.assertIn('mood', jazz_preset.default_params)
        self.assertIn('instruments', jazz_preset.default_params)
        
        self.assertIn('minimum_overall', jazz_preset.quality_targets)
        self.assertIn('target_lufs', jazz_preset.quality_targets)
        
    def test_preset_info(self):
        """Test preset info dictionary."""
        info = self.preset_manager.get_preset_info('rock')
        self.assertIsNotNone(info)
        self.assertIn('name', info)
        self.assertIn('description', info)
        self.assertIn('default_params', info)

class TestUserInterface(unittest.TestCase):
    """Test the user interface components."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock heavy components
        with patch('pipeline.user_interface.AutomatedMusicPipeline'), \
             patch('pipeline.user_interface.QualityEnhancementSystem'):
            self.ui = UserInterface()
            
    def test_ui_initialization(self):
        """Test UI initialization."""
        self.assertIsNotNone(self.ui.pipeline)
        self.assertIsNotNone(self.ui.quality_system)
        self.assertIsNotNone(self.ui.cache_manager)
        self.assertIsNotNone(self.ui.preset_manager)
        
    def test_session_stats_initialization(self):
        """Test session statistics initialization."""
        stats = self.ui.session_stats
        self.assertEqual(stats['total_generations'], 0)
        self.assertEqual(stats['cache_hits'], 0)
        self.assertEqual(stats['total_processing_time'], 0.0)
        self.assertEqual(stats['successful_generations'], 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_import_all_modules(self):
        """Test that all modules can be imported without errors."""
        try:
            from pipeline.main_controller import AutomatedMusicPipeline
            from pipeline.quality_enhancement import QualityEnhancementSystem
            from pipeline.user_interface import UserInterface, CacheManager, StylePresetManager
            self.assertTrue(True)  # If we get here, imports succeeded
        except ImportError as e:
            self.fail(f"Import failed: {e}")
            
    def test_component_integration(self):
        """Test that components can work together."""
        # Test preset manager with cache manager
        preset_manager = StylePresetManager()
        cache_manager = CacheManager(self.temp_dir / "test_cache")
        
        # Get a preset
        preset = preset_manager.get_preset('edm')
        self.assertIsNotNone(preset)
        
        # Test cache key generation with preset params
        prompt = "test electronic music"
        cache_key = cache_manager._generate_cache_key(prompt, preset.default_params)
        self.assertIsInstance(cache_key, str)
        self.assertGreater(len(cache_key), 0)

def run_phase3_tests():
    """Run all Phase 3 tests and generate report."""
    print("🧪 Running Phase 3 Integration Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMainController,
        TestQualityEnhancement,
        TestCacheManager,
        TestStylePresetManager,
        TestUserInterface,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestClass(test_class)
        suite.addTests(tests)
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "=" * 50)
    print("🧪 Phase 3 Test Summary")
    print("=" * 50)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("✅ All tests passed! Phase 3 implementation is ready.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[-2]}")
                
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_phase3_tests()
    sys.exit(0 if success else 1)
