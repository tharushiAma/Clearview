#!/usr/bin/env python3
"""
Diagnostic script to inspect raw 3-class model predictions per aspect.
Uses the adapter's built-in raw_3class output field.
"""
import sys
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import importlib.util
adapter_path = os.path.join(project_root, "website", "ml_models", "trained_model_adapter.py")
spec = importlib.util.spec_from_file_location("trained_model_adapter", adapter_path)
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)

TrainedModelAdapter = adapter_module.TrainedModelAdapter

CKPT_PATH = os.path.join(project_root, "results", "cosmetic_sentiment_v1", "best_model.pt")

tests = [
    ("Strongly positive", "This foundation is amazing! Great color, stays all day, smells wonderful, texture is silky."),
    ("Strongly negative", "Worst lipstick ever. Color fades fast, chemical smell terrible, packaging broke, price rip-off."),
    ("Mixed MSR test",    "Love the color and texture. But the smell is horrible and it falls off your face quickly."),
    ("Neutral",           "It is an okay product. Nothing special but does what it says. Price is reasonable."),
]

adapter = TrainedModelAdapter(CKPT_PATH)
ASPECT_NAMES = adapter.aspect_names

all_results = {}
label_counts = {"negative": 0, "neutral": 0, "positive": 0}
total = 0

for name, text in tests:
    print()
    print("=" * 70)
    print("Test: {}".format(name))
    print("Text: \"{}\"".format(text[:70]))
    print("{:<16} {:>8} {:>8} {:>8}   -> 3cls   4cls".format("aspect", "neg", "neu", "pos"))
    print("-" * 70)

    result = adapter.predict(text)
    aspect_preds = {}

    for asp in result["aspects"]:
        r = asp["raw_3class"]
        print("{:<16} {:>8.3f} {:>8.3f} {:>8.3f}   -> {:<8} {}".format(
            asp["name"], r["negative"], r["neutral"], r["positive"],
            r["pred"], asp["label"]))
        aspect_preds[asp["name"]] = r
        label_counts[r["pred"]] += 1
        total += 1

    all_results[name] = aspect_preds

print()
print("=" * 70)
print("BIAS ANALYSIS (3-class raw predictions):")
print("=" * 70)
for label, count in label_counts.items():
    bar = "#" * int(count / total * 50)
    pct = count / total * 100
    print("  {:10s}: {:2d}/{:2d} ({:4.1f}%) {}".format(label, count, total, pct, bar))

out_path = os.path.join(project_root, "tests", "diagnostic_output.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print()
print("Saved to: {}".format(out_path))
