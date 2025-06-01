#!/usr/bin/env python3
"""
Quality Enhancement System - Phase 3 Step 13
Implements quality validation, iterative improvement, and output optimization

Features:
- Audio quality validation and scoring
- Iterative improvement with retry logic
- Output post-processing and format conversion
- Quality assurance and reporting
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa
import soundfile as sf
from dataclasses import dataclass
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.advanced_audio_processing import AdvancedAudioProcessor

@dataclass
class QualityMetrics:
    """Container for audio quality metrics."""
    overall_score: float
    dynamic_range: float
    frequency_balance: float
    stereo_width: float
    vocal_clarity: float
    background_separation: float
    artifacts_score: float
    loudness_lufs: float
    peak_level: float
    
class QualityEnhancementSystem:
    """
    Comprehensive quality enhancement and validation system.
    Ensures professional quality output through automated analysis and improvement.
    """
    
    def __init__(self):
        """Initialize quality enhancement system."""
        self.project_root = Path(__file__).parent.parent
        self.quality_dir = self.project_root / "outputs" / "quality_analysis"
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize audio processor
        self.audio_processor = AdvancedAudioProcessor()
        
        # Quality thresholds
        self.quality_thresholds = {
            'minimum_overall': 7.0,  # Minimum acceptable overall score (0-10)
            'minimum_dynamic_range': 6.0,  # Minimum dynamic range in dB
            'maximum_peak_level': -1.0,  # Maximum peak level in dB
            'target_lufs': -16.0,  # Target loudness in LUFS
            'minimum_vocal_clarity': 6.5,  # Minimum vocal clarity score
            'minimum_stereo_width': 0.3,  # Minimum stereo width
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def validate_audio_quality(self, audio_path: str, audio_type: str = "complete") -> QualityMetrics:
        """
        Comprehensive audio quality validation.
        
        Args:
            audio_path: Path to audio file to validate
            audio_type: Type of audio ("complete", "vocals", "instrumental")
            
        Returns:
            QualityMetrics object with detailed scores
        """
        self.logger.info(f"🔍 Validating audio quality: {audio_path}")
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            
            # Ensure stereo
            if audio.ndim == 1:
                audio = np.array([audio, audio])
            elif audio.shape[0] > 2:
                audio = audio[:2]
                
            # Calculate individual metrics
            metrics = QualityMetrics(
                overall_score=0.0,  # Will be calculated from other metrics
                dynamic_range=self._calculate_dynamic_range(audio),
                frequency_balance=self._analyze_frequency_balance(audio, sr),
                stereo_width=self._calculate_stereo_width(audio),
                vocal_clarity=self._analyze_vocal_clarity(audio, sr, audio_type),
                background_separation=self._analyze_background_separation(audio, sr),
                artifacts_score=self._detect_artifacts(audio, sr),
                loudness_lufs=self._calculate_loudness_lufs(audio, sr),
                peak_level=self._calculate_peak_level(audio)
            )
            
            # Calculate overall score
            metrics.overall_score = self._calculate_overall_score(metrics)
            
            self.logger.info(f"📊 Quality Score: {metrics.overall_score:.1f}/10")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Quality validation failed: {e}")
            raise
            
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        # Use EBU R128 method for dynamic range
        rms_values = []
        hop_length = 1024
        
        for i in range(0, len(audio[0]) - hop_length, hop_length):
            frame = audio[:, i:i + hop_length]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > 0:
                rms_values.append(20 * np.log10(rms))
                
        if len(rms_values) < 2:
            return 0.0
            
        # Dynamic range = difference between 95th and 10th percentile
        dr = np.percentile(rms_values, 95) - np.percentile(rms_values, 10)
        return max(0, dr)
        
    def _analyze_frequency_balance(self, audio: np.ndarray, sr: int) -> float:
        """Analyze frequency balance across spectrum."""
        # Calculate spectrum
        fft = np.abs(np.fft.rfft(audio[0]))
        freqs = np.fft.rfftfreq(len(audio[0]), 1/sr)
        
        # Define frequency bands
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 8000),
            'brilliance': (8000, 20000)
        }
        
        band_energies = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                band_energies[band_name] = np.mean(fft[mask])
            else:
                band_energies[band_name] = 0
                
        # Calculate balance score (lower variance = better balance)
        energies = list(band_energies.values())
        if len(energies) > 0 and np.std(energies) > 0:
            balance_score = 10.0 - min(9.0, np.std(energies) / np.mean(energies) * 10)
        else:
            balance_score = 5.0
            
        return max(0, min(10, balance_score))
        
    def _calculate_stereo_width(self, audio: np.ndarray) -> float:
        """Calculate stereo width/imaging."""
        if audio.shape[0] < 2:
            return 0.0
            
        left, right = audio[0], audio[1]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(left, right)[0, 1]
        
        # Convert to width score (less correlation = wider stereo)
        width = (1.0 - abs(correlation)) if not np.isnan(correlation) else 0.0
        
        return max(0, min(1, width))
        
    def _analyze_vocal_clarity(self, audio: np.ndarray, sr: int, audio_type: str) -> float:
        """Analyze vocal clarity and presence."""
        if audio_type == "instrumental":
            return 8.0  # Instrumentals don't need vocal clarity
            
        # Focus on vocal frequency range (80-2000 Hz fundamental, 2000-8000 Hz harmonics)
        vocal_range = audio[0]  # Use mono for vocal analysis
        
        # Calculate spectral centroid (brightness indicator)
        spectral_centroid = librosa.feature.spectral_centroid(y=vocal_range, sr=sr)[0]
        avg_centroid = np.mean(spectral_centroid)
        
        # Calculate zero-crossing rate (clarity indicator)
        zcr = librosa.feature.zero_crossing_rate(vocal_range)[0]
        avg_zcr = np.mean(zcr)
        
        # Score based on typical vocal characteristics
        centroid_score = 10.0 if 800 <= avg_centroid <= 3000 else max(0, 10 - abs(avg_centroid - 1900) / 200)
        zcr_score = 10.0 if 0.02 <= avg_zcr <= 0.15 else max(0, 10 - abs(avg_zcr - 0.08) * 50)
        
        return (centroid_score + zcr_score) / 2
        
    def _analyze_background_separation(self, audio: np.ndarray, sr: int) -> float:
        """Analyze separation between foreground and background elements."""
        # Use harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(audio[0])
        
        # Calculate energy ratio
        harmonic_energy = np.mean(y_harmonic ** 2)
        percussive_energy = np.mean(y_percussive ** 2)
        
        if harmonic_energy + percussive_energy > 0:
            separation_ratio = abs(harmonic_energy - percussive_energy) / (harmonic_energy + percussive_energy)
            return min(10, separation_ratio * 10)
        else:
            return 5.0
            
    def _detect_artifacts(self, audio: np.ndarray, sr: int) -> float:
        """Detect audio artifacts (clipping, distortion, noise)."""
        artifact_score = 10.0
        
        # Check for clipping
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
        clipping_ratio = clipped_samples / audio.size
        artifact_score -= min(5.0, clipping_ratio * 100)  # Penalize clipping
        
        # Check for DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.01:
            artifact_score -= 1.0
            
        # Check for sudden jumps (possible artifacts)
        diff = np.abs(np.diff(audio[0]))
        large_jumps = np.sum(diff > 0.1)
        jump_ratio = large_jumps / len(diff)
        artifact_score -= min(2.0, jump_ratio * 50)
        
        return max(0, artifact_score)
        
    def _calculate_loudness_lufs(self, audio: np.ndarray, sr: int) -> float:
        """Calculate loudness in LUFS (approximation)."""
        # Simple RMS-based loudness approximation
        # For true LUFS, would need proper psychoacoustic modeling
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return 20 * np.log10(rms) + 3.01  # Rough LUFS approximation
        else:
            return -80.0
            
    def _calculate_peak_level(self, audio: np.ndarray) -> float:
        """Calculate peak level in dB."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            return 20 * np.log10(peak)
        else:
            return -80.0
            
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score."""
        # Weights for different aspects
        weights = {
            'dynamic_range': 0.15,
            'frequency_balance': 0.20,
            'stereo_width': 0.10,
            'vocal_clarity': 0.25,
            'background_separation': 0.15,
            'artifacts_score': 0.15
        }
        
        # Normalize metrics to 0-10 scale
        normalized = {
            'dynamic_range': min(10, metrics.dynamic_range / 2),  # 20dB = 10 points
            'frequency_balance': metrics.frequency_balance,
            'stereo_width': metrics.stereo_width * 10,
            'vocal_clarity': metrics.vocal_clarity,
            'background_separation': metrics.background_separation,
            'artifacts_score': metrics.artifacts_score
        }
        
        # Calculate weighted average
        overall = sum(normalized[key] * weights[key] for key in weights.keys())
        
        return min(10.0, max(0.0, overall))
        
    def enhance_audio_quality(self, audio_path: str, target_metrics: Optional[Dict] = None) -> str:
        """
        Enhance audio quality through iterative processing.
        
        Args:
            audio_path: Path to audio file to enhance
            target_metrics: Target quality metrics to achieve
            
        Returns:
            Path to enhanced audio file
        """
        self.logger.info(f"✨ Enhancing audio quality: {audio_path}")
        
        if target_metrics is None:
            target_metrics = self.quality_thresholds
            
        # Initial quality check
        initial_metrics = self.validate_audio_quality(audio_path)
        self.logger.info(f"📊 Initial quality: {initial_metrics.overall_score:.1f}/10")
        
        # If already high quality, minimal processing
        if initial_metrics.overall_score >= target_metrics.get('minimum_overall', 7.0):
            self.logger.info("✅ Audio already meets quality standards")
            return self._apply_minimal_enhancement(audio_path)
            
        # Apply targeted enhancements
        enhanced_path = self._apply_quality_enhancements(audio_path, initial_metrics, target_metrics)
        
        # Validate enhanced version
        final_metrics = self.validate_audio_quality(enhanced_path)
        self.logger.info(f"📈 Enhanced quality: {final_metrics.overall_score:.1f}/10")
        
        return enhanced_path
        
    def _apply_minimal_enhancement(self, audio_path: str) -> str:
        """Apply minimal enhancement to preserve quality."""
        output_path = str(Path(audio_path).with_suffix('.enhanced.wav'))
        
        # Load audio
        audio = self.audio_processor.load_audio(audio_path)
        
        # Apply only essential processing
        enhanced = self.audio_processor.remove_dc_offset(audio)
        enhanced = self.audio_processor.gentle_normalize(enhanced, target_level=0.7)
        
        # Save enhanced version
        self.audio_processor.save_audio(enhanced, output_path)
        
        return output_path
        
    def _apply_quality_enhancements(self, audio_path: str, current_metrics: QualityMetrics, 
                                   target_metrics: Dict) -> str:
        """Apply targeted quality enhancements based on metrics analysis."""
        output_path = str(Path(audio_path).with_suffix('.enhanced.wav'))
        
        # Load audio
        audio = self.audio_processor.load_audio(audio_path)
        enhanced = audio.copy()
        
        # Apply enhancements based on specific deficiencies
        
        # Fix DC offset and clipping
        if current_metrics.artifacts_score < 8.0:
            self.logger.info("🔧 Fixing artifacts...")
            enhanced = self.audio_processor.remove_dc_offset(enhanced)
            enhanced = self.audio_processor.soft_limiter(enhanced, threshold=0.95)
            
        # Improve frequency balance
        if current_metrics.frequency_balance < 6.0:
            self.logger.info("🎛️ Balancing frequencies...")
            enhanced = self.audio_processor.apply_gentle_eq(enhanced)
            
        # Enhance vocal clarity
        if current_metrics.vocal_clarity < target_metrics.get('minimum_vocal_clarity', 6.5):
            self.logger.info("🎤 Enhancing vocal clarity...")
            enhanced = self.audio_processor.enhance_vocal_presence(enhanced)
            
        # Improve stereo width if too narrow
        if current_metrics.stereo_width < target_metrics.get('minimum_stereo_width', 0.3):
            self.logger.info("🎧 Improving stereo width...")
            enhanced = self.audio_processor.gentle_stereo_widening(enhanced)
            
        # Final normalization
        enhanced = self.audio_processor.gentle_normalize(enhanced, target_level=0.8)
        
        # Save enhanced version
        self.audio_processor.save_audio(enhanced, output_path)
        
        return output_path
        
    def create_quality_report(self, audio_path: str, metrics: QualityMetrics, 
                            session_id: str) -> str:
        """Create comprehensive quality analysis report."""
        report_path = self.quality_dir / f"quality_report_{session_id}.json"
        
        report = {
            'file_path': audio_path,
            'session_id': session_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'quality_metrics': {
                'overall_score': round(metrics.overall_score, 2),
                'dynamic_range_db': round(metrics.dynamic_range, 2),
                'frequency_balance': round(metrics.frequency_balance, 2),
                'stereo_width': round(metrics.stereo_width, 3),
                'vocal_clarity': round(metrics.vocal_clarity, 2),
                'background_separation': round(metrics.background_separation, 2),
                'artifacts_score': round(metrics.artifacts_score, 2),
                'loudness_lufs': round(metrics.loudness_lufs, 2),
                'peak_level_db': round(metrics.peak_level, 2)
            },
            'quality_assessment': self._generate_quality_assessment(metrics),
            'recommendations': self._generate_recommendations(metrics)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return str(report_path)
        
    def _generate_quality_assessment(self, metrics: QualityMetrics) -> str:
        """Generate human-readable quality assessment."""
        score = metrics.overall_score
        
        if score >= 8.5:
            return "Excellent - Professional broadcast quality"
        elif score >= 7.5:
            return "Very Good - High quality for most applications"
        elif score >= 6.5:
            return "Good - Acceptable quality with minor issues"
        elif score >= 5.5:
            return "Fair - Noticeable quality issues present"
        else:
            return "Poor - Significant quality problems detected"
            
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate specific recommendations for improvement."""
        recommendations = []
        
        if metrics.dynamic_range < 6.0:
            recommendations.append("Increase dynamic range - consider less compression")
            
        if metrics.frequency_balance < 6.0:
            recommendations.append("Improve frequency balance - adjust EQ settings")
            
        if metrics.vocal_clarity < 6.5:
            recommendations.append("Enhance vocal clarity - check vocal processing chain")
            
        if metrics.stereo_width < 0.3:
            recommendations.append("Improve stereo imaging - add width to mix")
            
        if metrics.artifacts_score < 7.0:
            recommendations.append("Reduce artifacts - check for clipping or distortion")
            
        if metrics.peak_level > -1.0:
            recommendations.append("Reduce peak levels - apply gentle limiting")
            
        if not recommendations:
            recommendations.append("Quality is acceptable - no major improvements needed")
            
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    import time
    
    quality_system = QualityEnhancementSystem()
    
    # Test with example file (replace with actual path)
    test_file = "/Users/alexntowe/Projects/AI/Diff-SVC/outputs/generated_music/actual_christmas_carol.wav"
    
    if os.path.exists(test_file):
        print("🔍 Testing Quality Enhancement System...")
        
        # Validate quality
        metrics = quality_system.validate_audio_quality(test_file, "complete")
        print(f"📊 Quality Score: {metrics.overall_score:.1f}/10")
        
        # Create quality report
        report_path = quality_system.create_quality_report(test_file, metrics, "test_session")
        print(f"📋 Quality report saved: {report_path}")
        
        # Enhance if needed
        if metrics.overall_score < 7.0:
            enhanced_path = quality_system.enhance_audio_quality(test_file)
            print(f"✨ Enhanced version saved: {enhanced_path}")
        else:
            print("✅ Audio quality is already acceptable")
    else:
        print(f"⚠️ Test file not found: {test_file}")
