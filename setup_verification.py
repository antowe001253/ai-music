#!/usr/bin/env python3
"""
Phase 3 Setup Verification Script
Checks that all Phase 3 components are properly installed and configured
"""

import os
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_required_packages():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'numpy', 'scipy', 'librosa', 'soundfile', 'pydub',
        'transformers', 'torch', 'torchaudio', 'mir_eval'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_project_structure():
    """Check project file structure."""
    print("\n📁 Checking project structure...")
    
    project_root = Path(__file__).parent
    required_files = [
        'pipeline/main_controller.py',
        'pipeline/quality_enhancement.py', 
        'pipeline/user_interface.py',
        'pipeline/prompt_intelligence.py',
        'pipeline/orchestration_engine.py',
        'pipeline/advanced_audio_processing.py',
        'tests/test_phase3.py',
        'config.yaml',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {len(missing_files)}")
        return False
    
    return True

def check_pipeline_imports():
    """Check if pipeline components can be imported."""
    print("\n🔄 Checking pipeline imports...")
    
    pipeline_modules = [
        ('pipeline.prompt_intelligence', 'PromptIntelligence'),
        ('pipeline.orchestration_engine', 'OrchestrationEngine'),
        ('pipeline.advanced_audio_processing', 'AdvancedAudioProcessor'),
        ('pipeline.quality_enhancement', 'QualityEnhancementSystem'),
        ('pipeline.user_interface', 'UserInterface')
    ]
    
    import_errors = []
    
    for module_name, class_name in pipeline_modules:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - {str(e)[:50]}...")
            import_errors.append(module_name)
    
    if import_errors:
        print(f"\n⚠️  Import errors in: {', '.join(import_errors)}")
        return False
    
    return True

def check_diffsvc_config():
    """Check Diff-SVC configuration."""
    print("\n🎤 Checking Diff-SVC configuration...")
    
    project_root = Path(__file__).parent
    config_path = project_root / 'config.yaml'
    
    if not config_path.exists():
        print("❌ config.yaml not found")
        return False
    
    # Check for key configuration items
    try:
        with open(config_path, 'r') as f:
            config_content = f.read()
            
        required_configs = ['load_ckpt', 'hubert_path', 'vocoder_ckpt']
        missing_configs = []
        
        for config_key in required_configs:
            if config_key in config_content:
                print(f"✅ {config_key} configured")
            else:
                print(f"❌ {config_key} missing")
                missing_configs.append(config_key)
        
        if missing_configs:
            print(f"\n⚠️  Missing configurations: {', '.join(missing_configs)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def check_output_directories():
    """Check/create output directories."""
    print("\n📂 Checking output directories...")
    
    project_root = Path(__file__).parent
    required_dirs = [
        'outputs',
        'outputs/automated_pipeline',
        'outputs/quality_analysis',
        'cache',
        'temp'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {dir_path}")
        except Exception as e:
            print(f"❌ {dir_path} - Cannot create: {e}")
            return False
    
    return True

def run_quick_test():
    """Run a quick component test."""
    print("\n🧪 Running quick component test...")
    
    try:
        # Test style presets
        from pipeline.user_interface import StylePresetManager
        preset_manager = StylePresetManager()
        presets = preset_manager.list_presets()
        print(f"✅ Style presets: {len(presets)} available")
        
        # Test cache manager  
        from pipeline.user_interface import CacheManager
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        cache_manager = CacheManager(temp_dir / "test_cache")
        print("✅ Cache manager: Initialized")
        
        # Test quality system
        from pipeline.quality_enhancement import QualityEnhancementSystem
        quality_system = QualityEnhancementSystem()
        print("✅ Quality system: Initialized")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    """Main setup verification."""
    print("🔧 PHASE 3 SETUP VERIFICATION")
    print("=" * 35)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Pipeline Imports", check_pipeline_imports),
        ("Diff-SVC Config", check_diffsvc_config),
        ("Output Directories", check_output_directories),
        ("Component Test", run_quick_test)
    ]
    
    results = []
    
    for check_name, check_function in checks:
        print(f"\n--- {check_name} ---")
        try:
            result = check_function()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 35)
    print("📊 SETUP VERIFICATION SUMMARY")
    print("=" * 35)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ Phase 3 setup is complete and ready")
        print("\nNext steps:")
        print("1. Run demo: python demo_phase3.py")
        print("2. Run tests: python tests/test_phase3.py")
        print('3. Generate music: python pipeline/user_interface.py "your prompt here"')
    else:
        print(f"\n⚠️  {total - passed} checks failed")
        print("❌ Please fix the issues above before proceeding")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check file paths and permissions")
        print("- Verify Diff-SVC model checkpoints are available")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
