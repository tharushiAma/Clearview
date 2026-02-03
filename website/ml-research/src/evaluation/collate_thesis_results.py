# src/evaluation/collate_thesis_results.py
import os
import json
import pandas as pd
from typing import List

ABL_DIR = "outputs/ablations_4class"
ASPECTS = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']

def get_report_data(full_tag: str):
    path = os.path.join(ABL_DIR, f"eval_{full_tag}", "report.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():
    tags = ["1_base", "2_sampler", "3_synth", "4_full"]
    msr_types = ["base", "msr"]
    
    rows = []
    
    for tag in tags:
        for m_type in msr_types:
            full_tag = f"{tag}_{m_type}"
            data = get_report_data(full_tag)
            
            if not data:
                rows.append({
                    "Config": full_tag,
                    "Status": "PENDING"
                })
                continue
                
            absa = data["absa"]
            conf = data["conflict"]
            msr_red = data["msr_error_reduction"]
            
            row = {
                "Config": full_tag,
                "Sentiment-F1": f"{absa['overall_macro_f1_sentiment']:.4f}",
                "4Class-F1": f"{absa['overall_macro_f1_4class']:.4f}",
                "Bal-Acc": f"{absa['overall_balanced_accuracy']:.4f}",
                "Hamming": f"{absa['hamming_loss']:.4f}",
                "Conf-AUC": f"{conf.get('roc_auc', 0):.4f}",
                "Conf-Brier": f"{conf.get('brier_score', 0):.4f}",
                "MSR-Red": msr_red["total_reduction"],
                "p-value": f"{msr_red.get('wilcoxon_p_value', 1):.4f}"
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    out_path = os.path.join(ABL_DIR, "total_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Thesis Results Collation Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
