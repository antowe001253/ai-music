#!/usr/bin/env python3
"""
Alternative RVC approach - try to work around fairseq issues
"""

# Let's try using the current setup but with a manual fairseq workaround
import sys
sys.path.append('/rvc')

# Try to import without fairseq first
try:
    print("Testing RVC imports without fairseq...")
    
    # Mock fairseq if it's missing
    import types
    fairseq_mock = types.ModuleType('fairseq')
    sys.modules['fairseq'] = fairseq_mock
    
    from infer.modules.vc.modules import VC
    print("✅ VC import successful with mock fairseq")
    
except Exception as e:
    print(f"❌ Even with mock fairseq, import failed: {e}")

print("Done testing RVC workaround")
