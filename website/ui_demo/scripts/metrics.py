#!/usr/bin/env python3
import sys
import os
import json

# Use hardcoded absolute path
ML_RESEARCH_PATH = r"c:\Users\lucif\Desktop\Clearview\ml-research"

# Look for outputs/*/report.json
eval_dir = os.path.join(ML_RESEARCH_PATH, "outputs")
latest_report = None
latest_time = 0

for root, dirs, files in os.walk(eval_dir):
    if "report.json" in files:
        path = os.path.join(root, "report.json")
        mtime = os.path.getmtime(path)
        if mtime > latest_time:
            latest_time = mtime
            latest_report = path

if latest_report:
    with open(latest_report, "r") as f:
        data = json.load(f)
    print(json.dumps(data))
else:
    # Return fallback
    fallback = {
        "overall_macro_f1_4class": 0.0,
        "overall_macro_f1_sentiment": 0.0,
        "conflict": {
            "conf_f1_macro": 0.0,
            "roc_auc": 0.0,
            "brier_score": 0.0
        },
        "msr_error_reduction": {
            "total_reduction": 0
        }
    }
    print(json.dumps(fallback))
