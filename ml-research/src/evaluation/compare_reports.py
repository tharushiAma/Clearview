import os
import json
import argparse
import pandas as pd

ASPECTS = ["stayingpower", "texture", "smell", "price", "colour", "shipping", "packing"]

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main(args):
    base = load_json(args.baseline_report)
    msr  = load_json(args.msr_report)

    rows = []

    # Overall
    rows.append({
        "metric": "ABSA_overall_macro_f1",
        "baseline": base["absa"]["overall_macro_f1"],
        "msr": msr["absa"]["overall_macro_f1"],
        "delta": msr["absa"]["overall_macro_f1"] - base["absa"]["overall_macro_f1"],
    })
    rows.append({
        "metric": "Conflict_macro_f1",
        "baseline": base["conflict"]["conf_f1_macro"],
        "msr": msr["conflict"]["conf_f1_macro"],
        "delta": msr["conflict"]["conf_f1_macro"] - base["conflict"]["conf_f1_macro"],
    })
    rows.append({
        "metric": "MIXED_F1",
        "baseline": base["conflict"]["mixed_f1"],
        "msr": msr["conflict"]["mixed_f1"],
        "delta": msr["conflict"]["mixed_f1"] - base["conflict"]["mixed_f1"],
    })
    rows.append({
        "metric": "Separation",
        "baseline": base["conflict"]["separation"],
        "msr": msr["conflict"]["separation"],
        "delta": msr["conflict"]["separation"] - base["conflict"]["separation"],
    })
    rows.append({
        "metric": "MSR_total_error_reduction",
        "baseline": base["msr_error_reduction"]["total_reduction"],
        "msr": msr["msr_error_reduction"]["total_reduction"],
        "delta": msr["msr_error_reduction"]["total_reduction"] - base["msr_error_reduction"]["total_reduction"],
    })

    # Per-aspect F1 macro
    for asp in ASPECTS:
        b = base["absa"]["per_aspect"].get(asp, {}).get("f1_macro", None)
        m = msr["absa"]["per_aspect"].get(asp, {}).get("f1_macro", None)
        if b is None or m is None:
            continue
        rows.append({
            "metric": f"{asp}_f1_macro",
            "baseline": b,
            "msr": m,
            "delta": m - b
        })

    df = pd.DataFrame(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "comparison.csv")
    md_path  = os.path.join(args.out_dir, "comparison.md")

    df.to_csv(csv_path, index=False)

    # Markdown table
    md = df.copy()
    md["baseline"] = md["baseline"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    md["msr"] = md["msr"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    md["delta"] = md["delta"].apply(lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md.to_markdown(index=False))

    print(f"✅ Saved: {csv_path}")
    print(f"✅ Saved: {md_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_report", type=str, required=True)
    ap.add_argument("--msr_report", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/eval")
    args = ap.parse_args()
    main(args)
