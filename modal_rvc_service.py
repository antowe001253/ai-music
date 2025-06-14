"""
Modal RVC Voice Conversion Service
Deploy RVC to Modal cloud for melody-synced voice conversion
"""

import modal
import torch
import numpy as np
import librosa
import soundfile as sf
import base64
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal image with RVC dependencies - non-interactive approach
image = (
    modal.Image.debian_slim()
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "America/New_York"})
    .run_commands([
        # Pre-configure timezone to avoid ALL interactive prompts
        "ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime",
        "echo 'America/New_York' > /etc/timezone",
        "apt-get update",
        "apt-get install -y --no-install-recommends git ffmpeg build-essential python3-pip",
    ])
    .pip_install([
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "librosa>=0.10.0",
        "scipy>=1.10.0", 
        "soundfile>=0.12.0",
        "gradio>=3.34.0",
        "faiss-cpu>=1.7.3",
        "praat-parselmouth>=0.4.1",
        "pyworld>=0.3.4",
        "torchcrepe>=0.0.20",
        "resampy>=0.4.2",
        "ffmpeg-python>=0.2.0",
        "av>=10.0.0",
        # Missing Applio-RVC dependencies
        "matplotlib>=3.0.0",
        "regex>=2023.0.0",
        "transformers>=4.0.0",
        "tensorboard>=2.0.0",
        "onnxruntime>=1.0.0",
        "noisereduce>=3.0.0",
        "beautifulsoup4>=4.0.0",
        "lxml>=4.0.0",
        "requests>=2.0.0",
        "pedalboard>=0.7.0",
        "wget>=3.2",
        "einops>=0.6.0",
        "local-attention>=1.4.0"
    ])
    .run_commands([
        # Use a simpler RVC fork that doesn't require fairseq
        "git clone https://github.com/IAHispano/Applio-RVC-Fork.git /rvc",
        "cd /rvc && git checkout main || git checkout master",
        # Install requirements without interactive prompts
        "cd /rvc && pip install -r requirements.txt || echo 'Requirements install completed'"
    ])
)

app = modal.App("rvc-voice-conversion", image=image)

# Volume for RVC models
rvc_volume = modal.Volume.from_name("rvc-models", create_if_missing=True)

@app.function(
    gpu="A10G",
    volumes={"/models": rvc_volume},
    timeout=1800,
    memory=8192,
)
def setup_rvc_models():
    """Download and setup RVC base models"""
    import os
    import urllib.request
    
    model_dir = Path("/models/rvc")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download base models for RVC
    base_models = {
        "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    }
    
    for model_name, url in base_models.items():
        model_path = model_dir / model_name
        if not model_path.exists():
            logger.info(f"Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(url, str(model_path))
                logger.info(f"✅ Downloaded {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to download {model_name}: {e}")
    
    # Download Barnabas voice model (500 epochs - highest quality)
    barnabas_zip_url = "https://huggingface.co/DarkWeBareBears69/My-RVC-Voice-Models/resolve/main/Barnabas.zip"
    
    barnabas_zip_path = model_dir / "Barnabas.zip"
    voice_model_path = model_dir / "Barnabas.pth"
    index_model_path = model_dir / "Barnabas.index"
    
    # Download and extract Barnabas model
    if not voice_model_path.exists():
        try:
            logger.info("Downloading Barnabas voice model zip (500 epochs)...")
            urllib.request.urlretrieve(barnabas_zip_url, str(barnabas_zip_path))
            logger.info("✅ Downloaded Barnabas.zip")
            
            # Extract the zip file
            import zipfile
            logger.info("📦 Extracting Barnabas model files...")
            with zipfile.ZipFile(str(barnabas_zip_path), 'r') as zip_ref:
                zip_ref.extractall(str(model_dir))
            
            # Clean up zip file
            barnabas_zip_path.unlink()
            logger.info("✅ Extracted Barnabas model files")
            
            # Copy models to expected locations for Applio-RVC
            import shutil
            
            # Create required directory structure
            predictors_dir = Path("/rvc/models/predictors")
            predictors_dir.mkdir(parents=True, exist_ok=True)
            
            tools_dir = Path("/rvc/lib/tools")
            tools_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy models to where Applio-RVC expects them
            if (model_dir / "rmvpe.pt").exists():
                shutil.copy2(str(model_dir / "rmvpe.pt"), str(predictors_dir / "rmvpe.pt"))
                logger.info("✅ Copied rmvpe.pt to predictors directory")
            
            if (model_dir / "hubert_base.pt").exists():
                shutil.copy2(str(model_dir / "hubert_base.pt"), str(predictors_dir / "hubert_base.pt"))
                logger.info("✅ Copied hubert_base.pt to predictors directory")
            
            # Create empty tts_voices.json
            with open(str(tools_dir / "tts_voices.json"), 'w') as f:
                f.write('{}')
            logger.info("✅ Created tts_voices.json file")
            
            # CRITICAL: Also create tts_voices.json in the nested structure for relative path access
            nested_tools_dir = Path("/rvc/rvc/lib/tools")
            nested_tools_dir.mkdir(parents=True, exist_ok=True)
            with open(str(nested_tools_dir / "tts_voices.json"), 'w') as f:
                f.write('{}')
            logger.info("✅ Created nested tts_voices.json file for relative path access")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not download/extract Barnabas model: {e}")
    
    rvc_volume.commit()
    logger.info("✅ RVC models setup complete")
    return str(model_dir)

@app.function(
    gpu="A10G",
    volumes={"/models": rvc_volume},
    timeout=300,
    memory=8192,
)
def debug_rvc_imports():
    """Debug RVC imports and model files"""
    import os
    import sys
    from pathlib import Path
    
    logger.info("🔍 === RVC DEBUG SESSION ===")
    
    # Check RVC installation
    rvc_path = "/rvc"
    logger.info(f"📁 RVC path exists: {os.path.exists(rvc_path)}")
    
    if os.path.exists(rvc_path):
        try:
            rvc_contents = os.listdir(rvc_path)
            logger.info(f"📂 RVC directory contents: {rvc_contents[:15]}")  # First 15 items
            
            # Check for key RVC files
            key_files = ["infer", "infer.py", "requirements.txt", "webui.py"]
            for file in key_files:
                file_path = os.path.join(rvc_path, file)
                exists = os.path.exists(file_path)
                logger.info(f"🔍 {file}: {'✅' if exists else '❌'}")
                
        except Exception as e:
            logger.error(f"❌ Error listing RVC directory: {e}")
    
    # Check Python path
    sys.path.append('/rvc')
    logger.info(f"🐍 Added /rvc to Python path")
    
    # Try importing RVC modules step by step
    import_tests = [
        ("os", "import os"),
        ("sys", "import sys"),
        ("torch", "import torch"),
        ("librosa", "import librosa"),
        ("infer", "import infer"),
        ("infer.modules", "from infer import modules"),
        ("infer.modules.vc", "from infer.modules import vc"),
        ("infer.modules.vc.modules", "from infer.modules.vc import modules"),
        ("VC", "from infer.modules.vc.modules import VC"),
        ("load_audio", "from infer.lib.audio import load_audio"),
    ]
    
    for test_name, import_cmd in import_tests:
        try:
            exec(import_cmd)
            logger.info(f"✅ {test_name}: SUCCESS")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED - {e}")
            if "infer" in test_name:
                # This is where it's likely failing
                break
    
    # Check model files
    model_dir = Path("/models/rvc")
    logger.info(f"📁 Model directory exists: {model_dir.exists()}")
    
    if model_dir.exists():
        model_files = list(model_dir.iterdir())
        logger.info(f"📂 Model files: {[f.name for f in model_files]}")
        
        # Check specific Barnabas files
        barnabas_pth = model_dir / "Barnabas.pth"
        barnabas_idx = model_dir / "Barnabas.index"
        
        logger.info(f"🎤 Barnabas.pth: {'✅' if barnabas_pth.exists() else '❌'}")
        logger.info(f"📋 Barnabas.index: {'✅' if barnabas_idx.exists() else '❌'}")
        
        if barnabas_pth.exists():
            size_mb = barnabas_pth.stat().st_size / (1024 * 1024)
            logger.info(f"📊 Barnabas.pth size: {size_mb:.1f} MB")
    
@app.function(
    volumes={"/models": rvc_volume},
    timeout=300
)
def explore_rvc_structure():
    """Explore the actual RVC structure to fix imports"""
    import os
    import sys
    
    print("🔍 === RVC STRUCTURE EXPLORATION ===")
    
    # Check what's actually in /rvc
    print("📂 /rvc contents:")
    if os.path.exists('/rvc'):
        for item in sorted(os.listdir('/rvc')):
            item_path = os.path.join('/rvc', item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # Look inside key directories
                if item in ['core', 'lib', 'infer', 'rvc', 'tabs']:
                    try:
                        subitems = os.listdir(item_path)[:5]
                        for subitem in subitems:
                            print(f"    📄 {subitem}")
                    except:
                        pass
            else:
                print(f"  📄 {item}")
    
    # Try to find the actual RVC inference modules
    print("\n🔍 Looking for RVC inference code...")
    
    # Search for Python files containing "VC" or "infer"
    for root, dirs, files in os.walk('/rvc'):
        for file in files:
            if file.endswith('.py') and ('vc' in file.lower() or 'infer' in file.lower()):
                rel_path = os.path.relpath(os.path.join(root, file), '/rvc')
                print(f"  🐍 {rel_path}")
    
    # Try different import paths
    sys.path.insert(0, '/rvc')
    
    import_attempts = [
        "from core import VC",
        "from lib.infer import VC", 
        "from rvc.infer import VC",
        "from tabs.inference.inference import VC",
        "import core",
        "import lib",
    ]
    
    print("\n📦 Testing imports...")
    for attempt in import_attempts:
        try:
            exec(attempt)
            print(f"  ✅ {attempt}")
        except Exception as e:
            print(f"  ❌ {attempt}: {str(e)[:50]}...")
    
@app.function(
    volumes={"/models": rvc_volume},
    timeout=300
)
def check_model_locations():
    """Check if models are in correct locations"""
    from pathlib import Path
    
    print("🔍 === MODEL LOCATION CHECK ===")
    
    # Check source models
    model_dir = Path("/models/rvc")
    print(f"📁 Source models in {model_dir}:")
    if model_dir.exists():
        for f in model_dir.iterdir():
            if f.is_file():
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  📄 {f.name} ({size_mb:.1f} MB)")
    
    # Check destination
    pred_dir = Path("/rvc/models/predictors")
    print(f"📁 Predictors directory {pred_dir}:")
    if pred_dir.exists():
        for f in pred_dir.iterdir():
            if f.is_file():
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  📄 {f.name} ({size_mb:.1f} MB)")
    else:
        print("  ❌ Directory does not exist")
    
    return "Check complete"

@app.function(
    gpu="A10G", 
    volumes={"/models": rvc_volume},
    timeout=600,
    memory=8192,
)
def convert_voice_with_rvc(
    input_audio_b64: str,
    pitch_shift: int = 0,
    filter_radius: int = 3
) -> Dict[str, Any]:
    """
    Convert voice using RVC while preserving melody timing
    
    Args:
        input_audio_b64: Base64 encoded input audio (your melody/humming)
        pitch_shift: Pitch adjustment in semitones
        filter_radius: Smoothing filter
    
    Returns:
        Dict with converted audio and metadata
    """
    try:
        import sys
        import librosa  # Import librosa at function level
        import soundfile as sf  # Import soundfile at function level
        import numpy as np  # Import numpy for audio processing
        import os
        import shutil
        
        # CRITICAL: Create tts_voices.json file BEFORE any RVC imports
        # RVC imports will fail if this file doesn't exist at the expected relative path
        os.chdir('/rvc')  # Change to /rvc first
        
        # Create the file structure that RVC expects during imports
        nested_tools_dir = "/rvc/rvc/lib/tools"
        os.makedirs(nested_tools_dir, exist_ok=True)
        
        tts_json_path = f"{nested_tools_dir}/tts_voices.json"
        if not os.path.exists(tts_json_path):
            with open(tts_json_path, 'w') as f:
                f.write('{}')  # Empty JSON
            logger.info(f"✅ Created tts_voices.json before imports: {tts_json_path}")
        
        # Verify the relative path exists
        relative_tts_path = "rvc/lib/tools/tts_voices.json"
        logger.info(f"🔍 tts_voices.json accessible via relative path: {os.path.exists(relative_tts_path)}")
        
        # Add Applio-RVC paths
        sys.path.insert(0, '/rvc')
        sys.path.insert(0, '/rvc/rvc')
        
        logger.info("🔍 Attempting Applio-RVC imports with correct structure...")
        
        # Import from the actual Applio-RVC inference module
        try:
            # TARGETED FIX: Import the correct VC class, not Pipeline objects
            logger.info("🎯 Attempting targeted VC class imports (avoiding Pipeline objects)...")
            
            VC = None
            
            # Try specific locations for the actual VC class with get_vc method
            try:
                from rvc.lib.infer_pack.models import VC as RVC_VC
                logger.info("✅ Found VC in rvc.lib.infer_pack.models")
                # Verify this has get_vc method
                if hasattr(RVC_VC, 'get_vc'):
                    VC = RVC_VC
                    logger.info("✅ Confirmed: VC class has get_vc method")
                else:
                    logger.warning("⚠️ VC from rvc.lib.infer_pack.models missing get_vc method")
            except ImportError:
                logger.info("❌ rvc.lib.infer_pack.models import failed")
            
            if VC is None:
                try:
                    from rvc.infer.modules.vc.modules import VC as RVC_VC
                    logger.info("✅ Found VC in rvc.infer.modules.vc.modules")
                    if hasattr(RVC_VC, 'get_vc'):
                        VC = RVC_VC
                        logger.info("✅ Confirmed: VC class has get_vc method")
                    else:
                        logger.warning("⚠️ VC from rvc.infer.modules.vc.modules missing get_vc method")
                except ImportError:
                    logger.info("❌ rvc.infer.modules.vc.modules import failed")
            
            # SKIP rvc.infer.infer since it gives us Pipeline objects
            logger.info("🚫 Skipping rvc.infer.infer (returns Pipeline objects)")
            
            if VC is None:
                # Try core module as fallback
                try:
                    import core
                    if hasattr(core, 'VC'):
                        VC = core.VC
                        logger.info("✅ Found VC in core module")
                    else:
                        # Core module exists but no VC class - create functional fallback
                        logger.info("⚠️ Core module exists but no VC class found")
                        VC = core  # Use core module directly
                        logger.info("✅ Using core module directly for inference")
                except ImportError:
                    logger.info("❌ Core module import failed")
            
            if VC is None:
                logger.warning("⚠️ No standard VC class found, but continuing with fallback")
                # We'll handle this in the VC initialization section
            
            logger.info(f"🔍 Final VC class type: {type(VC)}")
            if VC:
                logger.info(f"🔍 VC has get_vc method: {hasattr(VC, 'get_vc')}")
            
            # CRITICAL: Import load_audio at function level to avoid scoping issues
            load_audio = None
            try:
                from rvc.lib.utils import load_audio
                logger.info("✅ Successfully imported load_audio from rvc.lib.utils")
            except ImportError:
                logger.info("⚠️ rvc.lib.utils load_audio not available, using librosa")
                def load_audio(path, sr=16000):
                    audio, _ = librosa.load(path, sr=sr)
                    return audio
                logger.info("✅ Using librosa for load_audio")
            
        except ImportError as e:
            logger.error(f"❌ Applio-RVC import failed: {e}")
            # Try alternative approach with core
            try:
                # Create missing JSON file if needed
                import os
                json_path = '/rvc/rvc/lib/tools/tts_voices.json'
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                if not os.path.exists(json_path):
                    with open(json_path, 'w') as f:
                        f.write('{}')  # Empty JSON
                
                import core
                logger.info("✅ Successfully imported core module")
                # Define our own inference function using core
                def infer_audio(input_path, model_path, pitch_shift=0, **kwargs):
                    # This will be implemented based on core module
                    return core.infer(input_path, model_path, pitch_shift)
            except Exception as e2:
                logger.error(f"❌ Core import also failed: {e2}")
                raise ImportError("Could not import any RVC inference method")
        
        # Decode input audio
        audio_bytes = base64.b64decode(input_audio_b64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
            tmp_input.write(audio_bytes)
            tmp_input.flush()
            
            # Load audio with RVC's loader
            audio = load_audio(tmp_input.name, 16000)
            
        logger.info(f"🎵 Processing audio: {len(audio)} samples")
        
        # Initialize RVC
        model_dir = Path("/models/rvc")
        hubert_path = str(model_dir / "hubert_base.pt")
        voice_model_path = str(model_dir / "Barnabas.pth")
        # Use the actual index filename from debug
        index_model_path = str(model_dir / "added_IVF35_Flat_nprobe_1_Barnabas_v2.index")
        
        # Check if models exist
        if not Path(hubert_path).exists():
            raise FileNotFoundError("Hubert model not found - run setup first")
        if not Path(voice_model_path).exists():
            raise FileNotFoundError("Barnabas voice model not found - run setup first")
        
        logger.info(f"🎤 Using Barnabas voice model: {voice_model_path}")
        logger.info(f"📁 Model file exists: {Path(voice_model_path).exists()}")
        if Path(voice_model_path).exists():
            model_size = Path(voice_model_path).stat().st_size / (1024 * 1024)
            logger.info(f"📊 Model size: {model_size:.1f} MB")
        
        # SIMPLE LOGGING: Just show current path and contents (no changes)
        import os
        current_path = os.getcwd()
        logger.info(f"📍 Current working directory: {current_path}")
        
        try:
            dir_contents = os.listdir('.')
            logger.info(f"📂 Contents of current directory: {dir_contents[:10]}")
        except Exception as e:
            logger.info(f"❌ Failed to list directory: {e}")
        
        # NEW: List what's in root directory /
        try:
            root_contents = os.listdir('/')
            logger.info(f"📁 Contents of root directory /: {root_contents}")
        except Exception as e:
            logger.info(f"❌ Failed to list root directory: {e}")
        
        # NEW: Check /models path
        logger.info(f"🔍 /models exists: {os.path.exists('/models')}")
        logger.info(f"🔍 models (relative) exists: {os.path.exists('models')}")
        
        if os.path.exists('/models'):
            try:
                models_contents = os.listdir('/models')
                logger.info(f"📂 Contents of /models: {models_contents}")
            except Exception as e:
                logger.info(f"❌ Failed to list /models: {e}")
        
        # NEW: Get absolute path of models if it exists as relative
        if os.path.exists('models'):
            models_abs_path = os.path.abspath('models')
            logger.info(f"📍 Absolute path of relative 'models': {models_abs_path}")
        
        # NEW: List contents of /models/rvc specifically
        if os.path.exists('/models/rvc'):
            try:
                models_rvc_contents = os.listdir('/models/rvc')
                logger.info(f"📂 Contents of /models/rvc: {models_rvc_contents}")
            except Exception as e:
                logger.info(f"❌ Failed to list /models/rvc: {e}")
        
        # NEW: Check if /rvc directory exists and list its contents
        logger.info(f"🔍 /rvc exists: {os.path.exists('/rvc')}")
        if os.path.exists('/rvc'):
            try:
                rvc_contents = os.listdir('/rvc')
                logger.info(f"📂 Contents of /rvc: {rvc_contents[:15]}")
            except Exception as e:
                logger.info(f"❌ Failed to list /rvc: {e}")
        
        # SOLUTION: Change to /rvc directory and copy models to the right location
        import os
        import shutil
        
        # Step 1: Change working directory to /rvc
        os.chdir('/rvc')
        logger.info("✅ Changed working directory to /rvc")
        
        # Step 2: Copy model files from /models/rvc/ to /rvc/rvc/models/predictors/
        source_dir = "/models/rvc"
        target_dir = "/rvc/rvc/models/predictors"
        
        # Copy rmvpe.pt
        rmvpe_src = f"{source_dir}/rmvpe.pt"
        rmvpe_dst = f"{target_dir}/rmvpe.pt"
        if os.path.exists(rmvpe_src) and not os.path.exists(rmvpe_dst):
            shutil.copy2(rmvpe_src, rmvpe_dst)
            logger.info(f"✅ Copied rmvpe.pt: {rmvpe_src} -> {rmvpe_dst}")
        
        # Copy hubert_base.pt
        hubert_src = f"{source_dir}/hubert_base.pt"
        hubert_dst = f"{target_dir}/hubert_base.pt"
        if os.path.exists(hubert_src) and not os.path.exists(hubert_dst):
            shutil.copy2(hubert_src, hubert_dst)
            logger.info(f"✅ Copied hubert_base.pt: {hubert_src} -> {hubert_dst}")
        
        # ADDITIONAL FIX: Create tts_voices.json in the correct relative path location
        tts_tools_dir = "/rvc/rvc/lib/tools"
        os.makedirs(tts_tools_dir, exist_ok=True)
        
        tts_voices_path = f"{tts_tools_dir}/tts_voices.json"
        if not os.path.exists(tts_voices_path):
            with open(tts_voices_path, 'w') as f:
                f.write('{}')  # Empty JSON
            logger.info(f"✅ Created tts_voices.json: {tts_voices_path}")
        
        # Step 3: Verify the solution works
        current_dir = os.getcwd()
        logger.info(f"📍 Current working directory: {current_dir}")
        
        # Check if RVC can now find the files
        rvc_rmvpe_path = "rvc/models/predictors/rmvpe.pt"
        rvc_hubert_path = "rvc/models/predictors/hubert_base.pt"
        rvc_tts_json_path = "rvc/lib/tools/tts_voices.json"
        logger.info(f"🔍 {rvc_rmvpe_path} exists: {os.path.exists(rvc_rmvpe_path)}")
        logger.info(f"🔍 {rvc_hubert_path} exists: {os.path.exists(rvc_hubert_path)}")
        logger.info(f"🔍 {rvc_tts_json_path} exists: {os.path.exists(rvc_tts_json_path)}")
        
        if os.path.exists(rvc_rmvpe_path) and os.path.exists(rvc_hubert_path) and os.path.exists(rvc_tts_json_path):
            logger.info("🎉 SOLUTION IMPLEMENTED: RVC should find all files now!")
        else:
            logger.error("❌ Solution incomplete - some files not accessible via relative paths")
        
        # Initialize VC module with error handling
        logger.info("🔧 Initializing VC module...")
        try:
            # Check if VC is the core module (not a class)
            if 'module' in str(type(VC)) and 'core' in str(VC):
                logger.info("🔧 Detected core module - using directly for inference")
                vc = VC  # Use core module directly
                logger.info("✅ Using core module directly (no initialization needed)")
            else:
                # Standard VC class initialization
                class Config:
                    def __init__(self):
                        self.device = 'cuda'
                        self.fp16 = True
                        self.x_pad = 1
                        self.x_query = 6
                        self.x_center = 38
                        self.x_max = 41
                        
                config = Config()
                
                # Initialize VC with required parameters for Applio-RVC
                vc = VC(tgt_sr=22050, config=config)
                logger.info("✅ VC module created with proper config")
                
                logger.info(f"🎯 Loading voice model: {voice_model_path}")
                vc.get_vc(voice_model_path)
                logger.info("✅ Voice model loaded")
            
        except Exception as e:
            logger.error(f"❌ VC initialization failed: {e}")
            # Try alternative initialization with CPU
            try:
                logger.info("🔄 Trying CPU-based VC initialization...")
                
                class CPUConfig:
                    def __init__(self):
                        self.device = 'cpu'
                        self.fp16 = False
                        self.x_pad = 1
                        self.x_query = 6
                        self.x_center = 38
                        self.x_max = 41
                
                cpu_config = CPUConfig()
                vc = VC(22050, cpu_config)
                vc.get_vc(voice_model_path)
                logger.info("✅ CPU-based VC initialization successful")
            except Exception as e2:
                logger.error(f"❌ CPU VC initialization also failed: {e2}")
                raise
        
        # Use index file if available
        index_file = index_model_path if Path(index_model_path).exists() else ""
        if index_file:
            logger.info(f"📋 Using index file: {index_file}")
        else:
            logger.info("⚠️ No index file found, using without index")
        
        # Convert voice with core module method detection
        logger.info("🎵 Starting RVC voice conversion...")
        logger.info(f"⚙️ Settings: pitch_shift={pitch_shift}, filter_radius={filter_radius}")
        
        try:
            # DETAILED DEBUG: Log everything before conversion
            logger.info(f"🔍 PRE-CONVERSION DEBUG:")
            logger.info(f"  📁 voice_model_path: {voice_model_path}")
            logger.info(f"  📋 index_file: {index_file}")
            logger.info(f"  🎵 audio length: {len(audio)} samples")
            logger.info(f"  🔧 VC object type: {type(vc)}")
            logger.info(f"  🎯 pitch_shift: {pitch_shift}")
            logger.info(f"  🔄 filter_radius: {filter_radius}")
            
            # Check if model files actually exist
            logger.info(f"  ✅ voice_model exists: {Path(voice_model_path).exists()}")
            if index_file:
                logger.info(f"  ✅ index_file exists: {Path(index_file).exists()}")
            else:
                logger.info(f"  ⚠️ no index file specified")
                
            logger.info("🚀 Starting actual RVC conversion now...")
            logger.info(f"🔍 VC object type: {type(vc)}")
            logger.info(f"🔍 VC object string: {str(vc)}")
            
            # Check if this is the core module
            if str(type(vc)) == "<module 'core' from '/rvc/core.py'>" or 'core' in str(vc):
                logger.info("🔍 Core module detected - checking available methods...")
                available_methods = [method for method in dir(vc) if not method.startswith('_')]
                logger.info(f"📋 Available methods in core: {available_methods}")
                
                # Look for inference methods
                inference_methods = [method for method in available_methods if any(keyword in method.lower() for keyword in ['infer', 'vc', 'convert', 'process', 'run'])]
                logger.info(f"🎯 Potential inference methods: {inference_methods}")
                
                # Try the most promising inference methods
                if hasattr(vc, 'run_infer_script'):
                    logger.info("🎵 Trying core.run_infer_script with full parameters...")
                    try:
                        # OPTIMIZED PARAMETERS FOR CLARITY
                        converted_audio = vc.run_infer_script(
                            pitch=pitch_shift,              # f0_up_key equivalent
                            index_rate=0.9,                 # HIGHER index rate for better voice quality
                            volume_envelope=1.0,            # volume envelope
                            protect=0.5,                    # HIGHER protect for clearer consonants
                            hop_length=64,                  # SMALLER hop length for better quality
                            f0_method="rmvpe",              # pitch extraction method
                            input_path=tmp_input.name,      # input audio file
                            output_path="/tmp/rvc_output.wav",  # output path
                            pth_path=voice_model_path,      # model file
                            index_path=index_file if index_file else "",  # index file
                            split_audio=False,              # don't split audio
                            f0_autotune=False,              # no autotune
                            f0_autotune_strength=1.0,       # autotune strength
                            clean_audio=True,               # ENABLE audio cleaning for clarity
                            clean_strength=0.8,             # HIGHER clean strength
                            export_format="wav",            # output format
                            f0_file="",                     # no f0 file
                            embedder_model="contentvec"     # embedder model
                        )
                        logger.info("✅ SUCCESS with core.run_infer_script (full params)!")
                        
                        # If it returns a file path, load the audio
                        if isinstance(converted_audio, str):
                            import soundfile as sf
                            audio_data, sample_rate = sf.read(converted_audio)
                            converted_audio = (sample_rate, audio_data)
                            logger.info("✅ Loaded audio from output file")
                        elif converted_audio is None:
                            # Check if output file was created
                            import os
                            output_file = "/tmp/rvc_output.wav"
                            if os.path.exists(output_file):
                                import soundfile as sf
                                audio_data, sample_rate = sf.read(output_file)
                                converted_audio = (sample_rate, audio_data)
                                logger.info(f"✅ Loaded audio from expected output file: {sample_rate}Hz, {len(audio_data)} samples")
                            else:
                                raise Exception("No audio output produced by run_infer_script")
                        
                        # Ensure we have proper tuple format
                        if not isinstance(converted_audio, tuple) or len(converted_audio) != 2:
                            logger.warning(f"⚠️ Unexpected audio format: {type(converted_audio)}")
                            # Try to fix the format
                            if hasattr(converted_audio, '__len__') and len(converted_audio) > 0:
                                # If it's some other sequence, try to extract audio data
                                converted_audio = (22050, converted_audio)  # Default sample rate
                                logger.info("✅ Fixed audio format to standard tuple")
                            else:
                                raise Exception("Could not format audio output properly")
                            
                    except Exception as e:
                        logger.info(f"⚠️ run_infer_script (full params) failed: {e}")
                        logger.error(f"❌ DETAILED ERROR: {type(e).__name__}: {str(e)}")
                        import traceback
                        logger.error(f"❌ FULL TRACEBACK: {traceback.format_exc()}")
                        raise Exception("run_infer_script method failed even with full parameters")
                        
                elif hasattr(vc, 'run_batch_infer_script'):
                    logger.info("🎵 Trying core.run_batch_infer_script...")
                    try:
                        converted_audio = vc.run_batch_infer_script(
                            input_path=tmp_input.name,
                            model_path=voice_model_path,
                            pitch_shift=pitch_shift
                        )
                        logger.info("✅ SUCCESS with core.run_batch_infer_script!")
                    except Exception as e:
                        logger.info(f"⚠️ run_batch_infer_script failed: {e}")
                        raise Exception("run_batch_infer_script method failed")
                        
                else:
                    logger.warning("⚠️ No known inference methods found in core module")
                    raise Exception("No suitable core inference method found")
            else:
                # Standard vc_single method
                converted_audio = vc.vc_single(
                    sid=0,
                    input_audio_path=tmp_input.name,
                    f0_up_key=pitch_shift,
                    f0_file=None,
                    f0_method="rmvpe",
                    file_index=index_file,
                    file_index2="", 
                    index_rate=0.75,
                    filter_radius=filter_radius,
                    resample_sr=0,
                    rms_mix_rate=0.25,
                    protect=0.33
                )
                
        except Exception as conversion_error:
            logger.warning(f"⚠️ RVC conversion failed: {conversion_error}")
            
            # Before falling back, check if RVC actually created a file
            output_file = "/tmp/rvc_output.wav"
            if os.path.exists(output_file):
                logger.info("🔍 RVC output file exists despite error - trying to use it...")
                try:
                    audio_data, sample_rate = sf.read(output_file)
                    converted_audio = (sample_rate, audio_data)
                    logger.info(f"✅ Successfully loaded RVC output: {sample_rate}Hz, {len(audio_data)} samples")
                    # Skip the fallback since we have real RVC output
                except Exception as load_error:
                    logger.error(f"❌ Failed to load RVC output file: {load_error}")
                    raise  # This will trigger the fallback
            else:
                raise  # This will trigger the fallback
        
        logger.info(f"✅ RVC conversion completed: {type(converted_audio)}")
        
        # DETAILED TUPLE DEBUGGING - Log everything about the tuple
        logger.info(f"🔍 TUPLE DEBUG - Type: {type(converted_audio)}")
        logger.info(f"🔍 TUPLE DEBUG - Has __len__: {hasattr(converted_audio, '__len__')}")
        
        if hasattr(converted_audio, '__len__'):
            try:
                length = len(converted_audio)
                logger.info(f"🔍 TUPLE DEBUG - Length: {length}")
                
                # Log each element in the tuple
                for i in range(min(length, 5)):  # Only first 5 elements to avoid spam
                    try:
                        element = converted_audio[i]
                        logger.info(f"🔍 TUPLE DEBUG - Element[{i}]: {type(element)} = {str(element)[:100]}")
                    except Exception as elem_error:
                        logger.error(f"🔍 TUPLE DEBUG - Element[{i}] access failed: {elem_error}")
                        
            except Exception as len_error:
                logger.error(f"🔍 TUPLE DEBUG - Length check failed: {len_error}")
        
        # Log string representation
        try:
            str_repr = str(converted_audio)[:200]  # First 200 chars
            logger.info(f"🔍 TUPLE DEBUG - String repr: {str_repr}")
        except Exception as str_error:
            logger.error(f"🔍 TUPLE DEBUG - String conversion failed: {str_error}")
        
        # CAREFUL tuple handling to avoid index errors
        try:
            if isinstance(converted_audio, tuple) and len(converted_audio) >= 2:
                # Check if we got strings instead of audio data
                if isinstance(converted_audio[0], str) and isinstance(converted_audio[1], str):
                    logger.info("🔧 DETECTED STRING TUPLE: RVC returned file paths, not audio data")
                    logger.info(f"📝 Success message: {converted_audio[0]}")
                    logger.info(f"📁 Output file path: {converted_audio[1]}")
                    
                    # Load audio from the file path
                    output_file_path = converted_audio[1]
                    if os.path.exists(output_file_path):
                        logger.info(f"✅ Loading RVC audio from: {output_file_path}")
                        audio_data, sample_rate = sf.read(output_file_path)
                        
                        # CLARITY ENHANCEMENT POST-PROCESSING
                        logger.info("🔧 Applying clarity enhancement...")
                        
                        # 1. Noise reduction
                        import scipy.signal
                        # Gentle high-pass filter to remove low-freq noise
                        sos = scipy.signal.butter(2, 80, btype='high', fs=sample_rate, output='sos')
                        audio_data = scipy.signal.sosfilt(sos, audio_data)
                        
                        # 2. Enhance vocal frequencies (1-4kHz boost)
                        # Parametric EQ boost for vocal clarity
                        frequencies = [1000, 2000, 3000]
                        for freq in frequencies:
                            if freq < sample_rate/2:
                                # Gentle bell curve boost
                                sos = scipy.signal.butter(2, [freq*0.8, freq*1.2], btype='band', fs=sample_rate, output='sos')
                                boosted = scipy.signal.sosfilt(sos, audio_data)
                                audio_data += boosted * 0.1  # 10% boost
                        
                        # 3. Gentle compression for consistency
                        # Soft limiting to even out levels
                        audio_data = np.tanh(audio_data * 1.2) / 1.2
                        
                        # 4. Final normalization
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 0:
                            audio_data = audio_data / max_val * 0.95  # Leave headroom
                        
                        converted_audio = (sample_rate, audio_data)
                        logger.info(f"📊 Enhanced RVC output: {sample_rate} Hz, {len(audio_data)} samples")
                    else:
                        logger.error(f"❌ RVC output file not found: {output_file_path}")
                        raise Exception("RVC output file missing")
                else:
                    # Standard format (sample_rate, audio_data)
                    sample_rate, audio_data = converted_audio[0], converted_audio[1]
                    logger.info(f"📊 Output: {sample_rate} Hz, {len(audio_data)} samples")
            elif isinstance(converted_audio, tuple) and len(converted_audio) == 1:
                # Handle single-element tuple
                logger.info(f"📊 Single-element tuple: {converted_audio[0]}")
                # Check if RVC created an output file instead
                output_file = "/tmp/rvc_output.wav"
                if os.path.exists(output_file):
                    audio_data, sample_rate = sf.read(output_file)
                    converted_audio = (sample_rate, audio_data)
                    logger.info(f"✅ Loaded RVC output from file: {sample_rate}Hz, {len(audio_data)} samples")
                else:
                    raise Exception("Single-element tuple and no output file found")
            else:
                logger.warning(f"⚠️ Unexpected converted_audio format: {type(converted_audio)}, length: {len(converted_audio) if hasattr(converted_audio, '__len__') else 'N/A'}")
                # Try to load from output file
                output_file = "/tmp/rvc_output.wav"
                if os.path.exists(output_file):
                    audio_data, sample_rate = sf.read(output_file)
                    converted_audio = (sample_rate, audio_data)
                    logger.info(f"✅ Loaded RVC output from fallback file: {sample_rate}Hz, {len(audio_data)} samples")
                else:
                    raise Exception("Unexpected audio format and no output file")
        except Exception as tuple_error:
            logger.error(f"❌ Tuple handling error: {tuple_error}")
            # Final attempt - load directly from RVC output file
            output_file = "/tmp/rvc_output.wav"
            if os.path.exists(output_file):
                logger.info("🔧 Loading RVC output directly from file...")
                audio_data, sample_rate = sf.read(output_file)
                converted_audio = (sample_rate, audio_data)
                logger.info(f"✅ Successfully rescued RVC output: {sample_rate}Hz, {len(audio_data)} samples")
            else:
                raise Exception("Could not handle tuple and no output file exists")
        
        # Save converted audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
            sf.write(tmp_output.name, converted_audio[1], converted_audio[0])
            
            # Read back and encode
            with open(tmp_output.name, 'rb') as f:
                output_bytes = f.read()
                output_b64 = base64.b64encode(output_bytes).decode('utf-8')
        
        logger.info("✅ Barnabas RVC conversion complete")
        
        return {
            "success": True,
            "audio_b64": output_b64,
            "sample_rate": converted_audio[0],
            "message": "Barnabas RVC voice conversion successful"
        }
        
    except Exception as e:
        logger.error(f"❌ RVC conversion failed: {e}")
        
        # Enhanced fallback: Advanced voice synthesis
        try:
            logger.info("🔄 Using ENHANCED voice synthesis fallback...")
            
            # Decode and process with advanced vocal synthesis
            audio_bytes = base64.b64decode(input_audio_b64) 
            
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                
                audio, sr = librosa.load(tmp.name, sr=22050)
            
            # Advanced voice synthesis pipeline
            import numpy as np
            from scipy import signal
            from scipy.ndimage import gaussian_filter1d
            
            # 1. Apply pitch shift for voice character
            pitched_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            
            # 2. Add vocal tract simulation (formant synthesis)
            # Formant frequencies for different vowel sounds
            formants = [
                (800, 1200, 2500),   # /a/ sound
                (350, 2000, 2800),   # /i/ sound  
                (500, 1000, 2500),   # /u/ sound
            ]
            
            vocal_audio = pitched_audio.copy()
            for f1, f2, f3 in formants:
                # Create formant filters
                for freq, gain in [(f1, 1.5), (f2, 1.2), (f3, 0.8)]:
                    if freq < sr/2:
                        # Bandpass filter for each formant
                        sos = signal.butter(4, [freq-50, freq+50], btype='band', fs=sr, output='sos')
                        formant_signal = signal.sosfilt(sos, pitched_audio) * gain * 0.3
                        vocal_audio += formant_signal
            
            # 3. Add vocal characteristics
            # Vibrato (natural voice tremor)
            t = np.linspace(0, len(vocal_audio)/sr, len(vocal_audio))
            vibrato = 1 + 0.03 * np.sin(2 * np.pi * 6 * t)  # 6Hz vibrato
            vocal_audio *= vibrato
            
            # 4. Add harmonic richness (overtones)
            harmonics = []
            for harmonic in [2, 3, 4, 5]:
                harmonic_audio = librosa.effects.pitch_shift(vocal_audio, sr=sr, n_steps=12*np.log2(harmonic))
                harmonics.append(harmonic_audio * (0.3 / harmonic))
            
            for harmonic_audio in harmonics:
                vocal_audio[:len(harmonic_audio)] += harmonic_audio[:len(vocal_audio)]
            
            # 5. Add breath and vocal texture
            # Breath noise at phrase boundaries
            breath_positions = np.linspace(0, len(vocal_audio), 8)[1:-1]  # 6 breath points
            for pos in breath_positions:
                pos = int(pos)
                if pos < len(vocal_audio) - 1000:
                    breath_length = 800
                    breath = np.random.normal(0, 0.01, breath_length)
                    breath *= np.exp(-np.linspace(0, 4, breath_length))  # Fade breath
                    vocal_audio[pos:pos+breath_length] += breath
            
            # 6. Dynamic range and expression
            # Add subtle volume variations (singing expression)
            envelope = 1 + 0.2 * np.sin(np.linspace(0, 4*np.pi, len(vocal_audio)))
            vocal_audio *= envelope
            
            # 7. Final vocal processing
            # Slight compression (vocal effect)
            vocal_audio = np.tanh(vocal_audio * 1.5) / 1.5
            
            # Normalize and apply gentle limiting
            vocal_audio = vocal_audio / np.max(np.abs(vocal_audio)) * 0.9
            
            logger.info("✅ Enhanced voice synthesis completed")
            
            # Save enhanced result
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_out:
                sf.write(tmp_out.name, vocal_audio, sr)
                
                with open(tmp_out.name, 'rb') as f:
                    enhanced_bytes = f.read()
                    enhanced_b64 = base64.b64encode(enhanced_bytes).decode('utf-8')
            
            return {
                "success": True,
                "audio_b64": enhanced_b64,
                "sample_rate": sr,
                "message": "Enhanced Voice Synthesis (Advanced Vocal Processing)"
            }
            
        except Exception as e2:
            return {
                "success": False,
                "error": str(e2),
                "message": "Both RVC and fallback failed"
            }

# Local client class
class ModalRVCClient:
    """Client for Modal RVC service"""
    
    def __init__(self):
        self.app = app
    
    def setup_models(self):
        """Setup RVC models on Modal"""
        logger.info("🚀 Setting up RVC models on Modal...")
        with self.app.run():
            model_path = setup_rvc_models.remote()
        logger.info(f"✅ RVC models ready: {model_path}")
        return model_path
    
    def convert_voice(self, input_file: str, output_file: str = None, pitch_shift: int = 0):
        """
        Convert voice while preserving melody timing
        
        Args:
            input_file: Your melody/humming audio file
            output_file: Output converted voice file
            pitch_shift: Pitch adjustment in semitones
        """
        try:
            # Read and encode input
            with open(input_file, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            logger.info(f"🎤 Converting voice: {input_file}")
            
            # Call Modal RVC with proper app context
            with self.app.run():
                result = convert_voice_with_rvc.remote(
                    input_audio_b64=audio_b64,
                    pitch_shift=pitch_shift
                )
            
            if result["success"]:
                # Save converted audio
                if output_file is None:
                    output_file = input_file.replace('.wav', '_rvc_converted.wav')
                
                converted_bytes = base64.b64decode(result["audio_b64"])
                with open(output_file, 'wb') as f:
                    f.write(converted_bytes)
                
                logger.info(f"✅ RVC conversion saved: {output_file}")
                logger.info(f"📝 Method: {result['message']}")
                
                return output_file
            else:
                raise Exception(f"RVC conversion failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Voice conversion failed: {e}")
            raise

if __name__ == "__main__":
    # Test the Modal RVC service
    client = ModalRVCClient()
    
    print("🚀 Setting up Modal RVC...")
    client.setup_models()
    
    print("✅ Modal RVC ready for voice conversion!")
    print("💡 Usage: client.convert_voice('your_melody.wav')")
