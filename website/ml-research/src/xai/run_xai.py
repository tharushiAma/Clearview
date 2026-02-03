# src/xai/run_xai.py
import argparse
import os
import json
from src.xai.Explainable import ClearViewExplainer, ASPECTS

def main():
    parser = argparse.ArgumentParser(description="Run XAI Suite for ClearView")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text to explain")
    parser.add_argument("--aspect", type=str, default="price", choices=ASPECTS, help="Aspect to explain")
    parser.add_argument("--out_dir", type=str, default="outputs/xai", help="Output directory")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top tokens to show")
    parser.add_argument("--msr", action="store_true", help="Enable MSR for explanation")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Initializing explainer with: {args.ckpt}")
    ex = ClearViewExplainer(args.ckpt)
    
    print(f"Explaining aspect: {args.aspect}")
    ig_aspect = ex.explain_ig_aspect(args.text, args.aspect, enable_msr=args.msr, top_k=args.top_k)
    
    print(f"Explaining conflict detection...")
    ig_conflict = ex.explain_ig_conflict(args.text, enable_msr=args.msr, top_k=args.top_k)
    
    print(f"Generating MSR Delta analysis...")
    msr_delta = ex.explain_msr_delta(args.text, args.aspect, top_k=args.top_k)
    
    report = {
        "input": {
            "text": args.text,
            "aspect": args.aspect,
            "ckpt": args.ckpt,
            "msr_enabled": args.msr
        },
        "results": {
            "ig_aspect": ig_aspect,
            "ig_conflict": ig_conflict,
            "msr_delta": msr_delta
        }
    }
    
    report_path = os.path.join(args.out_dir, f"xai_report_{args.aspect}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    # Save token attributions as CSV if needed
    # (Simplified for now)
    
    print(f"XAI Report saved to {report_path}")

if __name__ == "__main__":
    main()
