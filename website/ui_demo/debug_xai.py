
import sys
import os
import torch

# Point to correct MSR path
# ui_demo/debug_xai.py -> ../../ml-research
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml-research"))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)

from src.xai.Explainable import ClearViewExplainer

ckpt_path = "../../ml-research/outputs/gold_msr_4class/best_model.pt"

print(f"Loading model from {ckpt_path}...")
try:
    ex = ClearViewExplainer(ckpt_path, msr_strength=0.3)
    print("Model loaded.")
    print("Running IG Conflict...")
    res = ex.explain_ig_conflict("The screen is amazing but battery is bad.", enable_msr=True)
    print("IG Conflict Success!")
    print(res)
except Exception as e:
    print(f"CRASHED: {e}")
    import traceback
    traceback.print_exc()
