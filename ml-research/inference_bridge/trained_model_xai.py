#!/usr/bin/env python3
"""
trained_model_xai.py
--------------------
Explainability (XAI) Bridge for the trained RoBERTa-GCN model.

When the model predicts that a review is "negative" about the "smell", the
frontend website can ask "why?". This file provides the answers.

It wraps the core SentimentPredictor and exposes four explanation strategies:
  1. Integrated Gradients (Aspect)   - Which words drove the prediction for a specific aspect?
  2. Integrated Gradients (Conflict) - Which words pushed the review in opposing sentiment directions?
  3. LIME                            - Perturbs text to find locally important words.
  4. SHAP                            - Uses game theory (Shapley values) for robust word attribution.

The methods here format their output into structured JSON so the backend
API can return it directly to the frontend visualizer.
"""

import sys
import os

# ── Path setup ──────────────────────────────────────────────────────────────
# Ensure import inference.py from this directory, and models/model.py
# from the ml-research/src/ directory.
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

inference_dir = os.path.join(project_root, "ml-research", "inference_bridge")
ml_src_dir    = os.path.join(project_root, "ml-research", "src")

if inference_dir not in sys.path:
    sys.path.insert(0, inference_dir)
if ml_src_dir not in sys.path:
    sys.path.insert(0, ml_src_dir)

from inference import SentimentPredictor, is_mentioned


class TrainedModelXAI:
    """
    XAI wrapper for the trained RoBERTa-GCN model.
    Provides methods to explain predictions at the token (word) level.
    """

    def __init__(self, checkpoint_path: str, temperature: float = 0.5):
        """
        Load the underlying predictor.
        
        Args:
            checkpoint_path: Path to the best_model.pt file.
            temperature: Calibration temperature for sharper predictions.
        """
        self.predictor    = SentimentPredictor(checkpoint_path, temperature=temperature)
        self.aspect_names = self.predictor.aspect_names
        
        print("[XAI] Trained model XAI bridge loaded.")
        print("      Available Aspects: {}".format(", ".join(self.aspect_names)))

    # ────────────────────────────────────────────────────────────────────────
    # 1. Integrated Gradients (Aspect-level)
    # ────────────────────────────────────────────────────────────────────────
    def explain_ig_aspect(self, text: str, aspect: str,
                          enable_msr: bool = True, top_k: int = 10) -> dict:
        """
        Explain why the model made its prediction for a specific aspect using
        Integrated Gradients (IG).
        
        IG computes attributions by integrating the gradient of the prediction
        score along a straight path from a neutral baseline (all-padding) to
        the actual input.
        
        Args:
            text: The raw review text.
            aspect: The aspect to explain (e.g., 'colour', 'smell').
            enable_msr: Ignored (MSR is natively integrated in GCN).
            top_k: Number of most influential words to return.
            
        Returns:
            Dictionary matching the frontend's expected XAI shape, containing
            top tokens, prediction details, and raw probabilities.
        """
        # n_steps=25 gives a good balance between accuracy (satisfying the
        # completeness axiom) and execution speed for web requests.
        ig_res = self.predictor.explain_with_integrated_gradients(
            text=text,
            aspect=aspect,
            n_steps=25,
            top_k=top_k,
            save_path=None,
            silent=True
        )
        
        # Format tokens into [[token_string, score], ...] format
        top_tokens = []
        for token, attr in zip(ig_res['tokens'], ig_res['attributions']):
            # Filter out special tokens (<s>, <pad>, etc.) and whitespace tokens
            if token not in ("<s>", "</s>", "<pad>", "<mask>") and len(token.strip("Ġ▁ ")) > 0:
                clean_tok = self._clean_token(token)
                top_tokens.append([clean_tok, round(float(attr), 6)])
                
        # Deduplicate while preserving the highest magnitude score
        top_tokens.sort(key=lambda x: abs(x[1]), reverse=True)
        seen = set()
        deduped_tokens = []
        for t, v in top_tokens:
            if t.lower() not in seen:
                seen.add(t.lower())
                deduped_tokens.append([t, v])
            if len(deduped_tokens) >= top_k:
                break

        # Re-fetch standard prediction to get the baseline probabilities
        pred = self.predictor.predict(text, aspect)

        return {
            "top_tokens":  deduped_tokens,
            "method":      "ig",
            "task":        "aspect:{}".format(aspect),
            "predicted":   ig_res['target_label'],
            "confidence":  round(ig_res['confidence'], 4),
            "probs": {
                "negative": round(pred["probabilities"]["negative"], 4),
                "neutral":  round(pred["probabilities"]["neutral"],  4),
                "positive": round(pred["probabilities"]["positive"], 4),
            }
        }

    # ────────────────────────────────────────────────────────────────────────
    # 2. Integrated Gradients (Conflict Detection)
    # ────────────────────────────────────────────────────────────────────────
    def explain_ig_conflict(self, text: str,
                            enable_msr: bool = True, top_k: int = 10) -> dict:
        """
        Identify the "conflict drivers" in a review.
        
        If a user says "Beautiful colour but terrible smell", this method finds
        the words that are highly active in BOTH a positive aspect prediction
        AND a negative aspect prediction (like "but").
        
        Args:
            text: The raw review text.
            enable_msr: Ignored.
            top_k: Number of most conflicting words to return.
            
        Returns:
            Dictionary containing the top conflict tokens.
        """
        # Gather predictions and IG attributions across all aspects
        aspect_attentions = {}  # {aspect: {token: ig_attribution}}
        predictions       = {}  # {aspect: sentiment}

        for asp in self.aspect_names:
            # Skip aspects not mentioned to avoid calculating conflict between model hallucinations
            if not is_mentioned(text, asp):
                continue
                
            # First, get the prediction to know the sentiment
            pred = self.predictor.predict(text, asp)
            predictions[asp] = pred["sentiment"]
            
            # Now, get the actual Integrated Gradients attributions for this aspect
            # (using a smaller n_steps=10 to keep multi-aspect conflict calculation fast)
            ig_res = self.predictor.explain_with_integrated_gradients(
                text=text, aspect=asp, n_steps=10, top_k=1000, save_path=None, silent=True
            )
            
            if "tokens" in ig_res and "attributions" in ig_res:
                tokens = ig_res["tokens"]
                # Use the absolute magnitude of the IG attribution as the importance weight
                weights = [abs(float(w)) for w in ig_res["attributions"]]
                aspect_attentions[asp] = dict(zip(tokens, weights))

        if not aspect_attentions:
            return {"top_tokens": [], "method": "ig", "task": "conflict"}

        # Extract a unique set of all meaningful words in the review
        all_tokens = set()
        for attn_dict in aspect_attentions.values():
            all_tokens.update(attn_dict.keys())
        all_tokens = {t for t in all_tokens if not t.startswith("<") and t not in ("Ġ", "")}

        # Calculate a conflict score for each token using geometric mean.
        # High score means it heavily influenced BOTH positive and negative aspects.
        token_conflict: dict[str, float] = {}
        for tok in all_tokens:
            pos_sum = 0.0
            neg_sum = 0.0
            for asp, attn_dict in aspect_attentions.items():
                weight = attn_dict.get(tok, 0.0)
                if predictions.get(asp) == "positive":
                    pos_sum += weight
                elif predictions.get(asp) == "negative":
                    neg_sum += weight
            
            # Geometric mean ensures the score is only high if BOTH sums are high
            conflict_score = (pos_sum * neg_sum) ** 0.5
            if conflict_score > 0:
                token_conflict[tok] = round(conflict_score, 6)

        # Sort and format the results
        sorted_tokens = sorted(token_conflict.items(), key=lambda x: x[1], reverse=True)
        top_tokens    = [[self._clean_token(t), s] for t, s in sorted_tokens[:top_k]]

        return {
            "top_tokens": top_tokens,
            "method":     "ig",
            "task":       "conflict",
        }

    # ────────────────────────────────────────────────────────────────────────
    # 3. LIME (Local Interpretable Model-agnostic Explanations)
    # ────────────────────────────────────────────────────────────────────────
    def explain_lime_aspect(self, text: str, aspect: str, num_samples: int = 40, top_k: int = 10) -> dict:
        """
        Explain the prediction using LIME.
        
        LIME creates many slightly altered versions of the text (by randomly 
        removing words). It feeds all these variants to the model, observes how 
        the prediction changes, and builds a simple linear model to figure out 
        which words matter most.
        
        Note: Slower than IG because it runs the model `num_samples` times.
        """
        from lime.lime_text import LimeTextExplainer
        import numpy as np

        # Base prediction bounds our context
        result         = self.predictor.predict(text, aspect)
        pred_sentiment = result["sentiment"]
        conf           = result["confidence"]
        
        labels_map = {"negative": 0, "neutral": 1, "positive": 2}
        target_idx = labels_map.get(pred_sentiment, 2)

        # LIME needs a function that takes a list of strings and returns 
        # a numpy array of probabilities for each class.
        def predictor_fn(texts):
            probs_list = []
            for t in texts:
                res = self.predictor.predict(t, aspect)
                probs = [
                    res["probabilities"]["negative"],
                    res["probabilities"]["neutral"],
                    res["probabilities"]["positive"]
                ]
                probs_list.append(probs)
            return np.array(probs_list)

        explainer = LimeTextExplainer(class_names=["negative", "neutral", "positive"])
        exp       = explainer.explain_instance(
            text, 
            predictor_fn, 
            labels=(target_idx,), 
            num_features=top_k, 
            num_samples=num_samples
        )

        # Extract the features (words) that most influenced the target class
        lime_features = exp.as_list(label=target_idx)
        
        top_tokens = [[word, round(float(score), 6)] for word, score in lime_features]

        return {
            "top_tokens":  top_tokens,
            "method":      "lime",
            "task":        "aspect:{}".format(aspect),
            "predicted":   pred_sentiment,
            "confidence":  round(conf, 4),
            "probs": {
                "negative": round(result["probabilities"]["negative"], 4),
                "neutral":  round(result["probabilities"]["neutral"],  4),
                "positive": round(result["probabilities"]["positive"], 4),
            }
        }

    # ────────────────────────────────────────────────────────────────────────
    # 4. SHAP (SHapley Additive exPlanations)
    # ────────────────────────────────────────────────────────────────────────
    def explain_shap_aspect(self, text: str, aspect: str, max_evals: int = 40, top_k: int = 10) -> dict:
        """
        Explain the prediction using SHAP.
        
        Calculates Shapley values by dividing the text into tokens and evaluating 
        the model's output across different combinations (coalitions) of these 
        tokens. This is theoretically robust and satisfies fairness axioms.
        
        Note: Slower than IG.
        """
        import shap
        import numpy as np
        
        # Base prediction
        result         = self.predictor.predict(text, aspect)
        pred_sentiment = result["sentiment"]
        conf           = result["confidence"]
        
        labels_map = {"negative": 0, "neutral": 1, "positive": 2}
        target_idx = labels_map.get(pred_sentiment, 2)

        def predictor_fn(texts):
            probs_list = []
            for t in texts:
                res = self.predictor.predict(str(t), aspect)
                probs = [
                    res["probabilities"]["negative"],
                    res["probabilities"]["neutral"],
                    res["probabilities"]["positive"]
                ]
                probs_list.append(probs)
            return np.array(probs_list)

        # Use a simple whitespace masker (\W) to rapidly split into words
        masker    = shap.maskers.Text(r"\W")
        explainer = shap.Explainer(predictor_fn, masker, output_names=["negative", "neutral", "positive"])
        
        try:
            # Generate the SHAP Explanation object
            shap_values = explainer([text], max_evals=max_evals)
            
            # Extract values specific to the predicted class
            tokens = shap_values.data[0]
            values = shap_values.values[0, :, target_idx]
            
            # Filter empty whitespace tokens
            pairs = [[t.strip(), round(float(v), 6)] for t, v in zip(tokens, values) if t.strip()]
            
            # Sort by highest impact (positive or negative)
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Deduplicate
            seen = set()
            top_tokens = []
            for t, v in pairs:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    top_tokens.append([t, v])
                if len(top_tokens) >= top_k:
                    break
        except Exception as e:
            print(f"[SHAP] Error generating SHAP explanation: {e}")
            top_tokens = []

        return {
            "top_tokens":  top_tokens,
            "method":      "shap",
            "task":        "aspect:{}".format(aspect),
            "predicted":   pred_sentiment,
            "confidence":  round(conf, 4),
            "probs": {
                "negative": round(result["probabilities"]["negative"], 4),
                "neutral":  round(result["probabilities"]["neutral"],  4),
                "positive": round(result["probabilities"]["positive"], 4),
            }
        }

    # ────────────────────────────────────────────────────────────────────────
    # 5. Attention Weights
    # ────────────────────────────────────────────────────────────────────────
    def explain_attention_aspect(self, text: str, aspect: str, top_k: int = 10) -> dict:
        """
        Explain the prediction using raw attention weights.
        
        This relies on the model's self/cross-attention mechanism to see what 
        words the model 'looked at' most when predicting the aspect sentiment.
        Note: Attention weights are always positive magnitudes, so they indicate
        importance rather than directional impact like IG or SHAP.
        """
        # Get prediction with attention
        result = self.predictor.predict(text, aspect, return_attention=True)
        pred_sentiment = result["sentiment"]
        conf = result["confidence"]
        
        top_tokens = []
        if "attention" in result:
            tokens = result["attention"]["tokens"]
            weights = result["attention"]["weights"]
            
            # Filter special tokens
            pairs = []
            for t, w in zip(tokens, weights):
                if t not in ("<s>", "</s>", "<pad>", "<mask\>") and len(t.strip("Ġ▁ ")) > 0:
                    clean_tok = self._clean_token(t)
                    pairs.append([clean_tok, round(float(w), 6)])
            
            # Sort by highest attention
            pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Deduplicate
            seen = set()
            for t, w in pairs:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    top_tokens.append([t, w])
                if len(top_tokens) >= top_k:
                    break

        return {
            "top_tokens":  top_tokens,
            "method":      "attention",
            "task":        "aspect:{}".format(aspect),
            "predicted":   pred_sentiment,
            "confidence":  round(conf, 4),
            "probs": {
                "negative": round(result["probabilities"]["negative"], 4),
                "neutral":  round(result["probabilities"]["neutral"],  4),
                "positive": round(result["probabilities"]["positive"], 4),
            }
        }

    # ────────────────────────────────────────────────────────────────────────
    # Helper Utilities
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_token(token: str) -> str:
        """
        Remove tokenizer-specific prefix characters.
        'Ġ' is a space prefix in RoBERTa BPE tokenizers.
        """
        return token.lstrip("Ġ▁")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Standalone test -Verify the script works directly without booting the entire website.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ckpt = os.path.join(project_root, "ml-research", "outputs", "cosmetic_sentiment_v1", "best_model.pt")
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]

    print("Testing XAI bridge...")
    xai = TrainedModelXAI(ckpt)

    text = "The colour is absolutely beautiful but the smell is absolutely disgusting."
    print("\nText: {}".format(text))

    print("\n--- 1. Testing Integrated Gradients (Aspect) ---")
    for asp in ["colour", "smell"]:
        ig = xai.explain_ig_aspect(text, asp)
        print("  Aspect: {:8s} | Predicted: {:8s} (conf={:.3f})".format(asp, ig["predicted"], ig["confidence"]))
        print("  Top Tokens: {}".format(ig["top_tokens"][:5]))

    print("\n--- 2. Testing Integrated Gradients (Conflict Drivers) ---")
    conflict = xai.explain_ig_conflict(text)
    print("  Top Conflict Tokens: {}".format(conflict["top_tokens"][:5]))

    print("\n[OK] XAI bridge basic test complete.")
    print("     (Note: LIME/SHAP not executed here to save time; they are tested in the notebooks)")
