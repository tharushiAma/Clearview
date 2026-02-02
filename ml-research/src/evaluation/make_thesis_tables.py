import json
import os
import pandas as pd

def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def make_thesis_tables():
    base_path = "outputs/eval/baseline_fixed/report.json"
    msr_path = "outputs/eval/msr/report.json"
    
    if not os.path.exists(base_path) or not os.path.exists(msr_path):
        print("Required report.json files missing.")
        return
        
    base = load_report(base_path)
    msr = load_report(msr_path)
    
    # 1. Main Table
    main_data = [
        {
            "Model": "RoBERTa Baseline",
            "Overall Macro F1": base["absa"]["overall_macro_f1"],
            "Conflict Macro F1": base["conflict"]["conf_f1_macro"],
            "Separation Score": base["conflict"]["separation"],
            "MSR Reduction": 0
        },
        {
            "Model": "RoBERTa + MSR (Eagle v3)",
            "Overall Macro F1": msr["absa"]["overall_macro_f1"],
            "Conflict Macro F1": msr["conflict"]["conf_f1_macro"],
            "Separation Score": msr["conflict"]["separation"],
            "MSR Reduction": msr["msr_error_reduction"]["total_reduction"]
        }
    ]
    pd.DataFrame(main_data).to_csv("outputs/eval/thesis_table_main.csv", index=False)
    
    # 2. Per-Aspect Table
    aspects = list(base["absa"]["per_aspect"].keys())
    aspect_data = []
    for asp in aspects:
        aspect_data.append({
            "Aspect": asp,
            "Baseline F1": base["absa"]["per_aspect"][asp]["f1_macro"],
            "MSR F1": msr["absa"]["per_aspect"][asp]["f1_macro"],
            "Delta": msr["absa"]["per_aspect"][asp]["f1_macro"] - base["absa"]["per_aspect"][asp]["f1_macro"]
        })
    pd.DataFrame(aspect_data).to_csv("outputs/eval/thesis_table_per_aspect.csv", index=False)
    
    # 3. MSR Effect Table
    msr_effect = []
    for asp in aspects:
        eff = msr["msr_error_reduction"]["per_aspect"].get(asp, {"before": 0, "after": 0, "reduction": 0})
        msr_effect.append({
            "Aspect": asp,
            "Errors Before MSR": eff["before"],
            "Errors After MSR": eff["after"],
            "Reduction Count": eff["reduction"]
        })
    pd.DataFrame(msr_effect).to_csv("outputs/eval/thesis_table_msr_effect.csv", index=False)
    
    print("Thesis tables generated in outputs/eval/")

if __name__ == "__main__":
    make_thesis_tables()
