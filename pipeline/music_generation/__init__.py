# Music Generation Module
# Contains MusicGen and melody generation systems

from .melody_generation_system import MelodyGenerationSystem

try:
    from .musicgen_pipeline import MusicGenPipeline
except ImportError:
    pass  # MusicGenPipeline might be incomplete

__all__ = ['MelodyGenerationSystem', 'MusicGenPipeline']
