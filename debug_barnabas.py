#!/usr/bin/env python3
"""
Debug Barnabas RVC model files
"""

from modal_rvc_service import app
import modal

@app.function(
    volumes={"/models": modal.Volume.from_name("rvc-models")},
    timeout=300
)
def debug_model_files():
    """Check what files were extracted from Barnabas.zip"""
    from pathlib import Path
    import os
    
    model_dir = Path("/models/rvc")
    
    print("📁 Contents of /models/rvc:")
    if model_dir.exists():
        for item in sorted(model_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  📄 {item.name} ({size_mb:.1f} MB)")
            elif item.is_dir():
                print(f"  📁 {item.name}/")
                # List subdirectory contents
                for subitem in sorted(item.iterdir()):
                    if subitem.is_file():
                        size_mb = subitem.stat().st_size / (1024 * 1024) 
                        print(f"    📄 {subitem.name} ({size_mb:.1f} MB)")
    else:
        print("❌ /models/rvc directory doesn't exist")
    
    # Check for specific Barnabas files
    barnabas_files = ["Barnabas.pth", "Barnabas.index", "barnabas.pth", "barnabas.index"]
    print("\n🔍 Looking for Barnabas files:")
    for filename in barnabas_files:
        filepath = model_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✅ Found: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ Missing: {filename}")

if __name__ == "__main__":
    debug_model_files.remote()
