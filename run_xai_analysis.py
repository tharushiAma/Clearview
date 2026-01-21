import torch
import pandas as pd
import argparse
import os
import shutil
from src.models.msr_resolver_roberta_focal_loss import MultiAspectRobertaGCN, resolve_row
from src.models.xai import XAIAnalyzer
from src.visualization.xai_viz import plot_msr_resolution, plot_shap_text, plot_lime_html
from transformers import RobertaTokenizerFast

def load_trained_model(model_path, device):
    print(f"Loading model from {model_path}...")
    # Initialize with same architecture as training
    model = MultiAspectRobertaGCN(num_aspects=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", default=".", help="Root project directory")
    parser.add_argument("--model_path", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--sample_text", type=str, default=None, help="Single text to analyze")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- STRICT GPU ENFORCEMENT ---
    if device.type != "cuda":
        raise RuntimeError(
            "CRITICAL: GPU NOT FOUND!\n"
            "This script is configured to run ONLY on a GPU.\n"
            "Please ensure you have an NVIDIA GPU and PyTorch with CUDA support installed.\n"
            "Install command: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    # ------------------------------

    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Model
    model = load_trained_model(args.model_path, device)
    
    # 2. Setup Analyzer
    analyzer = XAIAnalyzer(model, args.project_dir, device)
    
    # 3. Define Samples to Explain
    if args.sample_text:
        samples = [args.sample_text]
    else:
        # Default mixed sentiment examples if none provided
        samples = [
            "The color is absolutely beautiful and vibrant, but the price is way too high for such a small bottle.",
            "I love the smell, it is so fresh. However, the packing was terrible, it arrived broken.",
            "Shipping was fast and the texture is great. Highly recommended!"
        ]

    output_dir = os.path.join(args.project_dir, "outputs/xai_visualizations")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Clean start
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating explanations for {len(samples)} samples...")
    print(f"Results will be saved to: {output_dir}")

    # 4. Run Analysis Loop
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    inv_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    for i, text in enumerate(samples):
        print(f"\n--- Analyzing Sample {i+1}: '{text[:50]}...' ---")
        sample_id = f"sample_{i+1}"
        sample_dir = os.path.join(output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        
        # --- FULL INFERENCE FOR MSR ---
        row_data = {"text": text}
        print(" > Running MSR Inference...")
        for aspect in analyzer.aspects:
            # Set wrapper target to this aspect
            aspect_idx = analyzer.aspects.index(aspect)
            analyzer.wrapper.target_aspect_idx = aspect_idx
            
            # wrapper.predict_proba returns [Batch, 3]
            p = analyzer.wrapper.predict_proba([text])[0] # Get prob array for this text
            
            conf_idx = p.argmax()
            conf_score = p.max()
            
            row_data[f"{aspect}_pred"] = inv_map[conf_idx]
            row_data[f"{aspect}_conf"] = conf_score
            
        resolution, conflict = resolve_row(row_data, analyzer.aspects)
        row_data["msr_resolution"] = resolution
        row_data["conflict_score"] = conflict
        
        print(f" > Resolution: {resolution} (Conflict: {conflict:.2f})")
        
        # B. EXPLAIN MSR (The Tug-of-War)
        print(" > Generating Resolution Chart...")
        msr_img_buf = plot_msr_resolution(row_data, analyzer.aspects)
        with open(f"{sample_dir}/resolution_logic.png", "wb") as f:
            f.write(msr_img_buf.getvalue())

        # C. EXPLAIN ASPECTS (SHAP/LIME)
        # We only explain aspects that were detected as Positive or Negative (Active aspects)
        active_aspects = [a for a in analyzer.aspects if row_data[f"{a}_pred"] in ["positive", "negative"]]
        
        if not active_aspects:
            print(" > No active aspects to explain.")
            continue
            
        for aspect in active_aspects:
            pred_label = row_data[f"{aspect}_pred"]
            print(f" > Explaining aspect '{aspect}' ({pred_label})...")
            
            try:
                # LIME
                lime_exp = analyzer.explain_lime(text, aspect)
                plot_lime_html(lime_exp, f"{sample_dir}/lime_{aspect}.html")
                
                # SHAP
                # Note: SHAP is slow. We do it for the first sample only or if user asked for it specifically.
                # For this demo script, we will run it.
                shap_values = analyzer.explain_shap(text, aspect)
                plot_shap_text(shap_values, f"{sample_dir}/shap_{aspect}.html")
                
            except Exception as e:
                print(f"   ! Error explaining {aspect}: {e}")

    print("\n\nDone! Check the 'outputs/xai_visualizations' folder.")

if __name__ == "__main__":
    main()
