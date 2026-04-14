#!/usr/bin/env python3
"""
trained_model_adapter.py
------------------------
Bridges the trained model to the website backend's /predict endpoint.

This adapter is the translation layer between the raw model output and the
structured JSON response that the frontend expects. It handles:

  - Running predictions across all 7 cosmetic aspects in one call
  - Skipping aspects that aren't mentioned in the review (saves compute +
    avoids misleading the user with irrelevant predictions)
  - Computing a mixed-sentiment conflict score to flag reviews that are
    simultaneously positive about one thing and negative about another
    (e.g. "loved the colour but hated the smell")

3-class output: negative / neutral / positive
Conflict detection: geometric mean of max positive & max negative confidence
"""

import sys
import os
import numpy as np
from typing import Dict, List

# ── Path setup ──────────────────────────────────────────────────────────────
# Both adapter and inference.py live in the same folder (inference_bridge/).
# Also need ml-research/src/ so that inference.py can import models/model.py.
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

inference_dir = os.path.join(project_root, "ml-research", "inference_bridge")
ml_src_dir    = os.path.join(project_root, "ml-research", "src")

if inference_dir not in sys.path:
    sys.path.insert(0, inference_dir)
if ml_src_dir not in sys.path:
    sys.path.insert(0, ml_src_dir)

from inference import SentimentPredictor, ASPECT_KEYWORDS, is_mentioned

# Sentiment class labels — must match the order used during model training
LABEL_NAMES = ['negative', 'neutral', 'positive']



def _compute_conflict_score(aspects_result: List[dict]) -> float:
    """
    Measure how 'conflicted' a review is — i.e. how much it praises some
    aspects while criticising others at the same time.

    Algorithm:
      - Only considers aspects actually mentioned in the review.
      - Extracts all confidently-positive (≥ 0.45) and confidently-negative
        predictions.
      - If both sides exist, returns the geometric mean of the highest
        positive confidence and highest negative confidence.
        (Geometric mean is strict: both sides must be high for a high score.)
      - If only one side has low-confidence predictions, returns a soft
        score of 0.30 to indicate mild uncertainty.
      - Returns 0.0 if there is no detectable conflict.

    Returns a float in [0.0, 1.0].
    """
    CONF_THRESHOLD = 0.45   # minimum confidence to count a prediction as "confident"

    # Only look at aspects the reviewer actually talked about
    mentioned = [a for a in aspects_result if a["label"] != "not_mentioned"]

    pos_confs = [
        a["confidence"] for a in mentioned
        if a["label"] == "positive" and a["confidence"] >= CONF_THRESHOLD
    ]
    neg_confs = [
        a["confidence"] for a in mentioned
        if a["label"] == "negative" and a["confidence"] >= CONF_THRESHOLD
    ]

    if not pos_confs or not neg_confs:
        # Neither side is confident enough — check if labels at least differ
        labels = [a["label"] for a in mentioned]
        if "positive" in labels and "negative" in labels:
            return 0.30   # Mild conflict (low-confidence mix — worth flagging softly)
        return 0.0

    # Strong conflict: both sides have confident predictions
    max_pos = max(pos_confs)
    max_neg = max(neg_confs)
    # Geometric mean gives a high score only when BOTH sides are confidently predicted
    conflict = float(np.sqrt(max_pos * max_neg))
    return min(conflict, 1.0)


class TrainedModelAdapter:
    """
    Adapter that wraps SentimentPredictor to provide a website-compatible
    prediction interface.

    The website backend creates one instance of this class at startup and
    calls predict() for every review analysis request.
    """

    def __init__(self, checkpoint_path: str, temperature: float = 0.5):
        """
        Load the trained model and prepare it for inference.

        Args:
            checkpoint_path: Absolute path to the best_model.pt checkpoint file.
            temperature:      Softmax temperature for calibration. Values < 1.0
                              sharpen the output distribution (make predictions
                              more decisive). 0.5 works well for this model.
        """
        self.predictor    = SentimentPredictor(checkpoint_path, temperature=temperature)
        self.aspect_names = self.predictor.aspect_names
        self.temperature  = temperature

        print("[OK] Trained model adapter ready (3-class, T={}).".format(temperature))
        print("     Aspects: {}".format(", ".join(self.aspect_names)))

    @property
    def device(self):
        """Expose the underlying device (CPU or GPU) for logging/debugging."""
        return self.predictor.device

    def predict(self, text: str, enable_msr: bool = True) -> Dict:
        """
        Run sentiment prediction across all 7 aspects for a given review.

        For each aspect:
          - If the review doesn't mention it → skip inference, return 'not_mentioned'
          - If it is mentioned → run the model and return a 3-class prediction

        After all aspects are predicted, compute a conflict score to detect
        reviews with mixed sentiment (positive in one area, negative in another).

        Args:
            text:       The raw review text to analyse.
            enable_msr: Reserved for future use (MSR is natively integrated in the GCN).

        Returns a dict in the shape the /predict endpoint expects:
            {
                "aspects": [
                    {
                        "name":          str,   # e.g. "colour"
                        "label":         str,   # "negative" | "neutral" | "positive" | "not_mentioned"
                        "confidence":    float, # highest class probability after temperature scaling
                        "probs":         list,  # [neg_prob, neu_prob, pos_prob]
                        "before":        dict,  # same as label/confidence (MSR not separate here)
                        "after":         dict,  # same as before
                        "changed_by_msr": bool, # always False — MSR is native to the GCN
                        "top_tokens":    list,  # words the model paid attention to
                        "mentioned":     bool,  # whether the aspect appeared in the review
                    },
                    ...
                ],
                "conflict_prob": float,  # mixed-sentiment score [0.0, 1.0]
                "timings":       {"total_ms": float}
            }
        """
        import time
        t0 = time.time()

        aspects_result = []
        for asp_name in self.aspect_names:
            mentioned = is_mentioned(text, asp_name)

            if not mentioned:
                # Save compute and avoid misleading the user on irrelevant aspects
                aspects_result.append({
                    "name":           asp_name,
                    "label":          "not_mentioned",
                    "confidence":     0.0,
                    "probs":          [0.0, 0.0, 0.0],
                    "before":         {"label": "not_mentioned", "confidence": 0.0},
                    "after":          {"label": "not_mentioned", "confidence": 0.0},
                    "changed_by_msr": False,
                    "top_tokens":     [],
                    "mentioned":      False,
                })
                continue

            # Run inference with attention to extract top tokens for the UI
            raw        = self.predictor.predict(text, asp_name, return_attention=True)
            neg        = raw["probabilities"]["negative"]
            neu        = raw["probabilities"]["neutral"]
            pos        = raw["probabilities"]["positive"]
            label      = raw["sentiment"]           # "negative" | "neutral" | "positive"
            conf       = raw["confidence"]          # highest probability after temperature scaling
            probs      = [neg, neu, pos]            # 3-element list for the frontend chart
            top_tokens = raw.get("top_tokens", [])

            aspects_result.append({
                "name":           asp_name,
                "label":          label,
                "confidence":     conf,
                "probs":          probs,
                "before":         {"label": label, "confidence": conf},
                "after":          {"label": label, "confidence": conf},
                "changed_by_msr": False,
                "top_tokens":     top_tokens,
                "mentioned":      True,
            })

        # Compute how much the review conflicts with itself
        conflict_prob = _compute_conflict_score(aspects_result)
        elapsed_ms    = (time.time() - t0) * 1000

        return {
            "aspects":       aspects_result,
            "conflict_prob": conflict_prob,
            "timings":       {"total_ms": elapsed_ms},
        }


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test — run this file directly to smoke-test the adapter without
# needing the full website: python trained_model_adapter.py [path/to/checkpoint]
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = os.path.join(project_root, "ml-research", "outputs", "cosmetic_sentiment_v1", "best_model.pt")
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]

    print("Testing adapter with checkpoint: {}".format(ckpt))
    adapter = TrainedModelAdapter(ckpt)

    # A small set of reviews with known expected sentiments for sanity checking
    reviews = [
        ("Strongly Positive",
         "This foundation is amazing! Great colour, stays all day, smells wonderful and the price is perfect."),
        ("Strongly Negative",
         "Worst lipstick ever. Awful colour, fades fast, chemical smell, packaging broke, way overpriced."),
        ("Mixed (colour+/smell-)",
         "Love the colour and texture, but the smell is absolutely horrible and it falls off after 2 hours."),
        ("Neutral",
         "It is an okay product. Nothing special but does what it says."),
    ]

    # Expected labels for specific aspects (used to check accuracy)
    EXPECTED = {
        "Strongly Positive":      {"colour": "positive", "smell": "positive"},
        "Strongly Negative":      {"colour": "negative", "smell": "negative"},
        "Mixed (colour+/smell-)": {"colour": "positive", "smell": "negative"},
    }

    all_correct  = 0
    all_expected = 0

    for label, text in reviews:
        print()
        print("[{}]".format(label))
        print("  {}".format(text[:90]))
        result  = adapter.predict(text)
        exp_map = EXPECTED.get(label, {})

        for asp in result["aspects"]:
            exp    = exp_map.get(asp["name"])
            marker = ""
            if exp:
                all_expected += 1
                if asp["label"] == exp:
                    all_correct += 1
                    marker = " <- OK"
                else:
                    marker = " <- WRONG (expected {})".format(exp)
            print("  {:14s}: {:8s}  conf={:.3f}  probs=[neg={:.2f}, neu={:.2f}, pos={:.2f}]{}".format(
                asp["name"], asp["label"], asp["confidence"],
                asp["probs"][0], asp["probs"][1], asp["probs"][2],
                marker))
        print("  conflict_prob = {:.3f}".format(result["conflict_prob"]))

    print()
    print("Accuracy on labeled aspects: {}/{} ({:.1f}%)".format(
        all_correct, all_expected,
        100 * all_correct / max(all_expected, 1)))
    print()
    print("[OK] Adapter test complete!")
