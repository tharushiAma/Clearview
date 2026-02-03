#!/usr/bin/env python3
import sys
import os
import json

# Use hardcoded absolute path
ML_RESEARCH_PATH = r"c:\Users\lucif\Desktop\Clearview\ml-research"

log_path = os.path.join(ML_RESEARCH_PATH, "outputs", "ui_logs.jsonl")

logs = []
if os.path.exists(log_path):
    try:
        with open(log_path, "r") as f:
            for line in f:
                logs.append(json.loads(line))
    except:
        pass

# Return last 100
print(json.dumps(logs[-100:]))
