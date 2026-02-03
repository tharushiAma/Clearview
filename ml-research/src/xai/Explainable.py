# src/xai/Explainable.py
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast

from captum.attr import LayerIntegratedGradients

# If available in your env (you said deps OK)
from lime.lime_text import LimeTextExplainer
import shap

from src.models.roberta_hierarchical_improved import ImprovedRoBERTaHierarchical
from src.data_layer._common import LABEL_MAP, INV_LABEL, ASPECTS

ID2LAB = INV_LABEL
LAB2ID = LABEL_MAP


class ClearViewExplainer:
    def __init__(self, ckpt_path: str, msr_strength: float = 0.3, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.model = ImprovedRoBERTaHierarchical(
            num_aspects=len(ASPECTS),
            num_classes=4,
            aspect_names=ASPECTS,
            hidden_dropout=0.3,
            msr_strength=msr_strength,
            roberta_name="roberta-base"
        ).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # Captum IG works best on embedding layer
        self.emb_layer = self.model.roberta.embeddings.word_embeddings
        
        # Simple cache for attributions to avoid redundant work
        self._ig_cache = {}

    def encode(self, text: str, max_len: int = 256):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def _get_cache_key(self, text, task, aspect, enable_msr):
        return (text, task, aspect, enable_msr)

    # --------------------------
    # Forward wrappers for Captum
    # --------------------------
    def _forward_aspect_logit(self, input_ids, attention_mask, aspect_idx: int, class_idx: int, enable_msr: bool):
        out = self.model(input_ids, attention_mask, enable_msr=enable_msr)
        aspect_logits = out["aspect_logits"] # List of tensors [B, 3]
        return aspect_logits[aspect_idx][:, class_idx]

    def _forward_conflict_logit(self, input_ids, attention_mask, enable_msr: bool):
        out = self.model(input_ids, attention_mask, enable_msr=enable_msr)
        # conflict_score is already sigmoided. For IG, we might want the logit,
        # but since we only have the score, we use the score's raw contribution.
        conflict_score = out["conflict_score"] # Shape [B, 1]
        return conflict_score.squeeze(-1)

    # --------------------------
    # Integrated Gradients
    # --------------------------
    def explain_ig_aspect(self, text: str, aspect: str, target_label: str = None, enable_msr: bool = True, top_k: int = 12):
        key = self._get_cache_key(text, "aspect", aspect, enable_msr)
        if key in self._ig_cache:
            return self._ig_cache[key]

        aspect_idx = ASPECTS.index(aspect)
        input_ids, attn = self.encode(text)

        # pick target label = predicted if not provided
        preds, probs, conf = self.model.predict(input_ids, attn, enable_msr=enable_msr)
        pred_id = int(preds[0, aspect_idx])
        class_idx = LAB2ID[target_label] if target_label else pred_id

        lig = LayerIntegratedGradients(
            lambda ids, mask: self._forward_aspect_logit(ids, mask, aspect_idx, class_idx, enable_msr),
            self.emb_layer
        )

        attributions, _ = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attn,),
            return_convergence_delta=True
        )

        # token-level scores
        token_ids = input_ids[0].detach().cpu().numpy().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()  # [seq]
        # mask pads
        valid = (attn[0].detach().cpu().numpy() == 1)
        toks = [(t, float(s)) for t, s, v in zip(tokens, scores, valid) if v]

        toks_sorted = sorted(toks, key=lambda x: abs(x[1]), reverse=True)[:top_k]

        res = {
            "method": "integrated_gradients",
            "task": "aspect_sentiment",
            "aspect": aspect,
            "enable_msr": enable_msr,
            "predicted": ID2LAB[pred_id],
            "target_explained": ID2LAB[class_idx],
            "probs": probs[0, aspect_idx].detach().cpu().numpy().tolist(), # [neg, neu, pos]
            "top_tokens": toks_sorted,
            "conflict_score": float(conf[0].detach().cpu().item())
        }
        self._ig_cache[key] = res
        return res

    def explain_ig_conflict(self, text: str, enable_msr: bool = True, top_k: int = 12):
        key = self._get_cache_key(text, "conflict", None, enable_msr)
        if key in self._ig_cache:
            return self._ig_cache[key]

        input_ids, attn = self.encode(text)

        lig = LayerIntegratedGradients(
            lambda ids, mask: self._forward_conflict_logit(ids, mask, enable_msr),
            self.emb_layer
        )

        attributions, _ = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attn,),
            return_convergence_delta=True
        )

        token_ids = input_ids[0].detach().cpu().numpy().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()
        valid = (attn[0].detach().cpu().numpy() == 1)

        toks = [(t, float(s)) for t, s, v in zip(tokens, scores, valid) if v]
        toks_sorted = sorted(toks, key=lambda x: abs(x[1]), reverse=True)[:top_k]

        # conflict prob (score is already sigmoided in model)
        with torch.no_grad():
            prob = self._forward_conflict_logit(input_ids, attn, enable_msr)[0].item()

        res = {
            "method": "integrated_gradients",
            "task": "conflict_detection",
            "enable_msr": enable_msr,
            "conflict_prob": float(prob),
            "top_tokens": toks_sorted
        }
        self._ig_cache[key] = res
        return res

    # --------------------------
    # LIME
    # --------------------------
    def explain_lime_aspect(self, text: str, aspect: str, enable_msr: bool = True, num_features: int = 10, num_samples: int = 200):
        aspect_idx = ASPECTS.index(aspect)

        def predict_proba(texts):
            # return [N, 3] probs for this aspect
            all_probs = []
            for t in texts:
                ids, mask = self.encode(t)
                preds, probs, conf = self.model.predict(ids, mask, enable_msr=enable_msr)
                p = probs[0, aspect_idx].detach().cpu().numpy()  # [3]
                all_probs.append(p)
            return np.array(all_probs)

        explainer = LimeTextExplainer(class_names=["negative", "neutral", "positive", "null"])
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        # list of (token, weight)
        return {
            "method": "lime",
            "task": "aspect_sentiment",
            "aspect": aspect,
            "enable_msr": enable_msr,
            "weights": exp.as_list()
        }

    # --------------------------
    # SHAP (Partition explainer)
    # --------------------------
    def explain_shap_aspect(self, text: str, aspect: str, enable_msr: bool = True, max_evals: int = 200):
        aspect_idx = ASPECTS.index(aspect)

        def f(texts):
            # SHAP expects [N, C]
            all_probs = []
            for t in texts:
                ids, mask = self.encode(t)
                _, probs, _ = self.model.predict(ids, mask, enable_msr=enable_msr)
                all_probs.append(probs[0, aspect_idx].detach().cpu().numpy())
            return np.array(all_probs)

        masker = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(f, masker, output_names=["negative", "neutral", "positive", "null"])
        shap_values = explainer([text], max_evals=max_evals)

        # return token contributions for each class
        return {
            "method": "shap",
            "task": "aspect_sentiment",
            "aspect": aspect,
            "enable_msr": enable_msr,
            "tokens": shap_values.data[0],
            "values": shap_values.values[0].tolist(),  # shape [tokens, 3]
        }

    # --------------------------
    # MSR explanation (before vs after)
    # --------------------------
    def explain_msr_delta(self, text: str, aspect: str, top_k: int = 12):
        # explain how tokens push conflict / change decision
        ig_conf_before = self.explain_ig_conflict(text, enable_msr=False, top_k=top_k)
        ig_conf_after  = self.explain_ig_conflict(text, enable_msr=True,  top_k=top_k)

        ig_asp_before = self.explain_ig_aspect(text, aspect, enable_msr=False, top_k=top_k)
        ig_asp_after  = self.explain_ig_aspect(text, aspect, enable_msr=True,  top_k=top_k)

        return {
            "method": "msr_delta_bundle",
            "aspect": aspect,
            "prob_before": ig_asp_before["probs"],
            "prob_after": ig_asp_after["probs"],
            "delta": (np.array(ig_asp_after["probs"]) - np.array(ig_asp_before["probs"])).tolist(),
            "conflict_before": ig_conf_before,
            "conflict_after": ig_conf_after,
            "aspect_before": ig_asp_before,
            "aspect_after": ig_asp_after
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/xai/explanation.json")
    ap.add_argument("--aspect", type=str, default="price") # Removed choices to allow 'all'
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--run_lime", action="store_true")
    ap.add_argument("--run_shap", action="store_true")
    ap.add_argument("--msr_strength", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ex = ClearViewExplainer(args.ckpt, msr_strength=args.msr_strength)

    if args.aspect == "all":
        aspect_list = ASPECTS
    else:
        if args.aspect not in ASPECTS:
            print(f"Error: Invalid aspect '{args.aspect}'. Choose from {ASPECTS} or 'all'.")
            return
        aspect_list = [args.aspect]

    bundle = {
        "text": args.text,
        "requested_aspect": args.aspect,
        "ig_conflict": ex.explain_ig_conflict(args.text, enable_msr=True, top_k=args.top_k),
        "aspects": {}
    }

    for asp in aspect_list:
        asp_data = {
            "ig_aspect": ex.explain_ig_aspect(args.text, asp, enable_msr=True, top_k=args.top_k),
            "msr_delta": ex.explain_msr_delta(args.text, asp, top_k=args.top_k)
        }
        if args.run_lime:
            asp_data["lime"] = ex.explain_lime_aspect(args.text, asp, enable_msr=True)
        if args.run_shap:
            asp_data["shap"] = ex.explain_shap_aspect(args.text, asp, enable_msr=True)
        
        bundle["aspects"][asp] = asp_data

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved XAI bundle to: {args.out}")


if __name__ == "__main__":
    main()
