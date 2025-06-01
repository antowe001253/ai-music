#!/usr/bin/env python3
"""
User Interface & Optimization System - Phase 3 Step 14
Command-line interface, batch processing, caching, and configuration presets

Features:
- Interactive command-line interface
- Batch processing capabilities
- Smart caching system
- Style presets and configurations
- Performance optimization
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.main_controller import AutomatedMusicPipeline
from pipeline.quality_enhancement import QualityEnhancementSystem

@dataclass
class StylePreset:
    """Configuration preset for different music styles."""
    name: str
    description: str
    default_params: Dict[str, Any]
    quality_targets: Dict[str, float]
    processing_hints: Dict[str, Any]

class CacheManager:
    """Smart caching system for generated content."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager."""
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict:
        """Load cache index from disk."""
        if self.cache_index_path.exists():
            with open(self.cache_index_path, 'r') as f:
                return json.load(f)
        return {"entries": {}, "stats": {"hits": 0, "misses": 0}}
        
    def _save_cache_index(self):
        """Save cache index to disk."""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
            
    def _generate_cache_key(self, prompt: str, params: Dict) -> str:
        """Generate unique cache key for prompt and parameters."""
        # Create deterministic hash from prompt and relevant parameters
        cache_data = {
            'prompt': prompt.lower().strip(),
            'genre': params.get('genre', ''),
            'tempo': params.get('tempo', ''),
            'mood': params.get('mood', ''),
            'key': params.get('key', ''),
            'duration': params.get('duration', 30)
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
        
    def get_cached_result(self, prompt: str, params: Dict) -> Optional[Dict]:
        """Check if result exists in cache."""
        cache_key = self._generate_cache_key(prompt, params)
        
        if cache_key in self.cache_index["entries"]:
            entry = self.cache_index["entries"][cache_key]
            cache_path = Path(entry["path"])
            
            # Check if cached files still exist
            if cache_path.exists():
                # Update access time
                entry["last_accessed"] = datetime.now().isoformat()
                entry["access_count"] = entry.get("access_count", 0) + 1
                
                self.cache_index["stats"]["hits"] += 1
                self._save_cache_index()
                
                return {
                    "cached": True,
                    "path": str(cache_path),
                    "metadata": entry["metadata"]
                }
                
        self.cache_index["stats"]["misses"] += 1
        return None
        
    def cache_result(self, prompt: str, params: Dict, result_path: str, metadata: Dict):
        """Cache generation result."""
        cache_key = self._generate_cache_key(prompt, params)
        
        entry = {
            "prompt": prompt,
            "params": params,
            "path": result_path,
            "metadata": metadata,
            "created": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "access_count": 1
        }
        
        self.cache_index["entries"][cache_key] = entry
        self._save_cache_index()
        
    def cleanup_old_cache(self, max_age_days: int = 30):
        """Remove old cache entries."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        keys_to_remove = []
        
        for key, entry in self.cache_index["entries"].items():
            last_accessed = datetime.fromisoformat(entry["last_accessed"])
            if last_accessed < cutoff_date:
                # Remove files
                cache_path = Path(entry["path"])
                if cache_path.exists():
                    cache_path.unlink()
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.cache_index["entries"][key]
            
        self._save_cache_index()
        return len(keys_to_remove)

class StylePresetManager:
    """Manager for music style presets and configurations."""
    
    def __init__(self):
        """Initialize preset manager with built-in presets."""
        self.presets = self._create_default_presets()
        
    def _create_default_presets(self) -> Dict[str, StylePreset]:
        """Create default style presets."""
        presets = {}
        
        # Christmas Carol
        presets["christmas"] = StylePreset(
            name="Christmas Carol",
            description="Traditional Christmas music with gentle melodies",
            default_params={
                "genre": "folk",
                "tempo": 80,
                "mood": "peaceful",
                "key": "C",
                "duration": 30,
                "instruments": ["piano", "bells", "strings"]
            },
            quality_targets={
                "minimum_overall": 8.0,
                "minimum_vocal_clarity": 8.5,
                "target_lufs": -18.0
            },
            processing_hints={
                "reverb_type": "hall",
                "vocal_style": "gentle",
                "warmth": True
            }
        )
        
        # Electronic Dance Music
        presets["edm"] = StylePreset(
            name="Electronic Dance Music",
            description="Upbeat electronic music with strong rhythm and synthetic sounds",
            default_params={
                "genre": "electronic",
                "tempo": 128,
                "mood": "energetic",
                "key": "auto",
                "duration": 30,
                "instruments": ["synthesizer", "drum machine", "bass"]
            },
            quality_targets={
                "minimum_overall": 7.5,
                "minimum_dynamic_range": 8.0,
                "target_lufs": -14.0
            },
            processing_hints={
                "stereo_width": "wide",
                "bass_emphasis": True,
                "vocal_style": "electronic"
            }
        )
        
        return presets
        
    def get_preset(self, name: str) -> Optional[StylePreset]:
        """Get preset by name."""
        return self.presets.get(name.lower())
        
    def list_presets(self) -> List[str]:
        """List all available presets."""
        return list(self.presets.keys())

class UserInterface:
    """Command-line user interface for the automated music pipeline."""
    
    def __init__(self):
        """Initialize user interface."""
        self.pipeline = AutomatedMusicPipeline()
        self.quality_system = QualityEnhancementSystem()
        self.cache_manager = CacheManager()
        self.preset_manager = StylePresetManager()

def main():
    """Main entry point for user interface."""
    if len(sys.argv) < 2:
        print("🎵 Automated Music Pipeline - Phase 3")
        print("\nUsage:")
        print('  python pipeline/user_interface.py "your music prompt here"')
        print("\nExamples:")
        print('  python pipeline/user_interface.py "Christmas carol song"')
        print('  python pipeline/user_interface.py "upbeat electronic dance track"')
        return
        
    prompt = sys.argv[1]
    
    try:
        # Initialize pipeline
        ui = UserInterface()
        
        # Generate song
        print(f"🎵 Generating: {prompt}")
        results = ui.pipeline.generate_complete_song(prompt)
        
        # Display results
        print(f"\n✨ Song generation completed!")
        print(f"📁 Output directory: {results['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
