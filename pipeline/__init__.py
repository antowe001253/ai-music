# Pipeline Package
# Automated Music Generation and Voice Synthesis Pipeline

__version__ = "2.0.0"
__author__ = "AI Music Pipeline"

# Import Phase 1 components (Steps 6-8)
from .music_generation import *
from .audio_processing import *
from .voice_synthesis import *

# Import Phase 2 components (Steps 9-11)
try:
    from .prompt_intelligence import PromptIntelligence, MusicalParameters
    from .orchestration_engine import OrchestrationEngine, TrackInfo, OrchestrationPlan
    from .advanced_audio_processing import AdvancedAudioProcessor
    
    __all__ = [
        # Phase 1
        'MelodyGenerationSystem', 'AudioProcessingSuite',
        # Phase 2
        'PromptIntelligence', 'MusicalParameters',
        'OrchestrationEngine', 'TrackInfo', 'OrchestrationPlan',
        'AdvancedAudioProcessor'
    ]
except ImportError:
    # Phase 2 components not available
    __all__ = ['MelodyGenerationSystem', 'AudioProcessingSuite']
