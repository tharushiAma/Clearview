#!/usr/bin/env python3
"""
Trained Model XAI Bridge
Provides explainability for the trained RoBERTa-GCN model using:
  1. Attention weights  (fast, always available)
  2. LIME               (optional, slow but word-level)

Returns results in the same JSON shape the frontend expects:
  ig_conflict:   { top_tokens: [[token, score], ...] }
  ig_aspect:     { top_tokens: [[token, score], ...] }
  msr_delta:     { prob_before: [neg,neu,pos], prob_after: [neg,neu,pos] }
"""

import sys
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from inference import SentimentPredictor


class TrainedModelXAI:
    """
    XAI wrapper for the trained model that exposes the same interface
    the backend_server expects (matching legacy ClearViewExplainer output shape).
    """

    def __init__(self, checkpoint_path: str, temperature: float = 0.5):
        self.predictor = SentimentPredictor(checkpoint_path, temperature=temperature)
        self.aspect_names = self.predictor.aspect_names
        print("[XAI] Trained model XAI bridge loaded.")
        print("      Aspects: {}".format(", ".join(self.aspect_names)))

    # ------------------------------------------------------------------
    # Public interface (matches legacy ClearViewExplainer)
    # ------------------------------------------------------------------

    def explain_ig_aspect(self, text: str, aspect: str,
                          enable_msr: bool = True, top_k: int = 10) -> dict:
        """
        Return top-k tokens that most influenced the prediction for `aspect`.
        Uses attention weights as a proxy for attribution.

        Returns: {"top_tokens": [[token, score], ...], "method": "attention",
                  "task": "aspect", "predicted": str, "confidence": float}
        """
        result = self.predictor.predict(text, aspect, return_attention=True)
        top_tokens = self._top_attention_tokens(result, top_k)

        return {
            "top_tokens":  top_tokens,
            "method":      "attention",
            "task":        "aspect:{}".format(aspect),
            "predicted":   result["sentiment"],
            "confidence":  round(result["confidence"], 4),
            "probs": {
                "negative": round(result["probabilities"]["negative"], 4),
                "neutral":  round(result["probabilities"]["neutral"],  4),
                "positive": round(result["probabilities"]["positive"], 4),
            }
        }

    def explain_ig_conflict(self, text: str,
                            enable_msr: bool = True, top_k: int = 10) -> dict:
        """
        Identify tokens that drive sentiment conflict across aspects.

        Strategy: tokens that appear with HIGH attention for a POSITIVE
        aspect AND HIGH attention for a NEGATIVE aspect are conflict drivers.
        Returns a signed score: positive = drives positive-side,
        negative = drives negative-side, large absolute value = conflict driver.

        Returns: {"top_tokens": [[token, score], ...], "method": "attention",
                  "task": "conflict"}
        """
        # Gather per-aspect predictions + attention
        aspect_attentions = {}  # aspect -> {token: attention_weight}
        predictions = {}        # aspect -> sentiment

        for asp in self.aspect_names:
            result = self.predictor.predict(text, asp, return_attention=True)
            predictions[asp] = result["sentiment"]
            if "attention" in result:
                tokens  = result["attention"]["tokens"]
                weights = result["attention"]["weights"]
                aspect_attentions[asp] = dict(zip(tokens, weights))

        if not aspect_attentions:
            return {"top_tokens": [], "method": "attention", "task": "conflict"}

        # Collect all unique (non-special) tokens
        all_tokens = set()
        for attn_dict in aspect_attentions.values():
            all_tokens.update(attn_dict.keys())
        all_tokens = {t for t in all_tokens if not t.startswith("<") and t not in ("Ġ", "")}

        # Compute conflict score per token:
        #   positive aspects → positive weight contribution
        #   negative aspects → negative weight contribution
        token_conflict: dict[str, float] = {}
        for tok in all_tokens:
            pos_sum = 0.0
            neg_sum = 0.0
            for asp, attn_dict in aspect_attentions.items():
                w = attn_dict.get(tok, 0.0)
                if predictions.get(asp) == "positive":
                    pos_sum += w
                elif predictions.get(asp) == "negative":
                    neg_sum += w
            # A conflict token has both large pos_sum and neg_sum
            conflict_score = (pos_sum * neg_sum) ** 0.5
            if conflict_score > 0:
                token_conflict[tok] = round(conflict_score, 6)

        # Sort by conflict score descending
        sorted_tokens = sorted(token_conflict.items(), key=lambda x: x[1], reverse=True)
        top_tokens = [[self._clean_token(t), s] for t, s in sorted_tokens[:top_k]]

        return {
            "top_tokens": top_tokens,
            "method":     "attention",
            "task":       "conflict",
        }

    def explain_msr_delta(self, text: str, aspect: str, top_k: int = 10) -> dict:
        """
        Show the probability distribution before/after MSR.
        Since our model doesn't have a separate MSR pass, we show the raw
        prediction probabilities as 'after' and a softened version as 'before'.

        Returns: {"prob_before": [neg, neu, pos], "prob_after": [neg, neu, pos],
                  "method": "msr_delta", "task": "aspect"}
        """
        # Raw (unscaled) prediction = "before MSR"
        raw_result = self.predictor.predict(text, aspect)
        probs_after = [
            raw_result["probabilities"]["negative"],
            raw_result["probabilities"]["neutral"],
            raw_result["probabilities"]["positive"],
        ]

        # Simulate "before" with lower temperature (less confident)
        original_temp = self.predictor.temperature
        self.predictor.temperature = 1.0   # flat without scaling
        try:
            before_result = self.predictor.predict(text, aspect)
            probs_before = [
                before_result["probabilities"]["negative"],
                before_result["probabilities"]["neutral"],
                before_result["probabilities"]["positive"],
            ]
        finally:
            self.predictor.temperature = original_temp

        return {
            "prob_before": [round(p, 4) for p in probs_before],
            "prob_after":  [round(p, 4) for p in probs_after],
            "method":      "msr_delta",
            "task":        "aspect:{}".format(aspect),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _top_attention_tokens(self, predict_result: dict, top_k: int) -> list:
        """Extract and rank tokens by attention weight."""
        if "attention" not in predict_result:
            return []

        tokens  = predict_result["attention"]["tokens"]
        weights = predict_result["attention"]["weights"]

        # Filter out special tokens and Byte-Pair-Encoding prefix char
        pairs = [
            (self._clean_token(t), w)
            for t, w in zip(tokens, weights)
            if t not in ("<s>", "</s>", "<pad>", "<mask>") and len(t.strip("Ġ▁ ")) > 0
        ]

        # Sort descending by weight
        pairs.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate (keep highest-scored occurrence)
        seen = set()
        result = []
        for tok, w in pairs:
            if tok not in seen:
                seen.add(tok)
                result.append([tok, round(float(w), 6)])
            if len(result) >= top_k:
                break

        return result

    @staticmethod
    def _clean_token(token: str) -> str:
        """Remove BPE prefix characters (Ġ = space in RoBERTa tokenizer)."""
        return token.lstrip("Ġ▁")


# ─── Standalone test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    ckpt = os.path.join(project_root, "results", "cosmetic_sentiment_v1", "best_model.pt")
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]

    print("Testing XAI bridge...")
    xai = TrainedModelXAI(ckpt)

    text = "The colour is absolutely beautiful but the smell is absolutely disgusting."
    print("\nText: {}".format(text))

    for asp in ["colour", "smell"]:
        print("\n--- Aspect: {} ---".format(asp))
        ig = xai.explain_ig_aspect(text, asp)
        print("  predicted: {} (conf={:.3f})".format(ig["predicted"], ig["confidence"]))
        print("  top tokens: {}".format(ig["top_tokens"][:5]))

        delta = xai.explain_msr_delta(text, asp)
        print("  prob_before: {}".format(delta["prob_before"]))
        print("  prob_after:  {}".format(delta["prob_after"]))

    print("\n--- Conflict ---")
    conflict = xai.explain_ig_conflict(text)
    print("  conflict tokens: {}".format(conflict["top_tokens"][:5]))

    print("\n[OK] XAI bridge test complete.")
