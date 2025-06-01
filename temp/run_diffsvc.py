
import sys
import os
sys.path.append('.')

# Set the parameters for inference
project_name = "base_model"
model_path = "./checkpoints/base_model/model_ckpt_steps_100000.ckpt"  
config_path = "./config.yaml"

# Add the current directory to Python path
if '.' not in sys.path:
    sys.path.insert(0, '.')

from infer_tools.infer_tool import Svc
from infer import run_clip

# Load model
try:
    print("Loading Diff-SVC model...")
    model = Svc(project_name, config_path, hubert_gpu=True, model_path=model_path)
    print("Model loaded successfully!")
    
    # Run inference
    print("Running Diff-SVC inference...")
    f0_tst, f0_pred, audio = run_clip(
        svc_model=model,
        key=0,  # No pitch adjustment
        acc=20,  # Acceleration 
        use_pe=False,  # Fixed: PE not enabled in config
        use_crepe=True,
        thre=0.05,
        use_gt_mel=False,
        add_noise_step=500,
        file_path="temp/melody_1748800893.wav",
        out_path="temp/vocals_1748800893.wav",
        project_name=project_name,
        format='wav'
    )
    print("Diff-SVC inference completed!")
    
except Exception as e:
    print(f"Error in Diff-SVC: {e}")
    import traceback
    traceback.print_exc()
