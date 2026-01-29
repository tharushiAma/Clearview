# Explainable.py
# Explainability utilities matching ImprovedRoBERTaHierarchical outputs

import numpy as np
import torch


class ClearViewExplainer:
    """
    Works with ImprovedRoBERTaHierarchical:
      - model.forward(..., return_attention=True) returns:
          aspect_logits: list of [B,3]
          conflict_score: [B,1]
          aspect_attention: list of [B,T]
          cross_attention: [B, A, A] (from nn.MultiheadAttention)
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def explain_with_attention(self, text: str, aspect_index: int = 0, max_len: int = 256, top_k: int = 20):
        """
        Fast attention-based token importance (good baseline explainability).
        """
        self.model.eval()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len
        ).to(self.device)

        out = self.model(enc["input_ids"], enc["attention_mask"], return_attention=True)
        attn = out["aspect_attention"][aspect_index][0].detach().cpu().numpy()  # [T]

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].detach().cpu().tolist())

        # mask pads and normalize
        mask = enc["attention_mask"][0].detach().cpu().numpy()
        attn = attn * mask
        attn = attn / (attn.sum() + 1e-9)

        ranked = sorted(list(zip(tokens, attn.tolist())), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def explain_with_ig(
        self,
        text: str,
        aspect_index: int = 0,
        target_class: int = None,
        max_len: int = 256,
        n_steps: int = 50,
        top_k: int = 20
    ):
        """
        Integrated Gradients using Captum on the embedding layer.
        Requires: pip install captum
        """
        from captum.attr import IntegratedGradients

        self.model.eval()

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        emb_layer = self.model.roberta.embeddings.word_embeddings
        embeds = emb_layer(input_ids)  # [1,T,E]
        baseline = torch.zeros_like(embeds)

        def forward_embeds(embeds_):
            # run roberta with inputs_embeds
            ro_out = self.model.roberta(inputs_embeds=embeds_, attention_mask=attention_mask)
            seq = ro_out.last_hidden_state  # [1,T,H]

            aspect_feats, _ = self.model.aspect_attention(seq, attention_mask)
            refined_feats, _ = self.model.cross_aspect(aspect_feats)

            base_logits = [self.model.classifiers[i](refined_feats[i]) for i in range(self.model.num_aspects)]
            conflict_score = self.model.conflict_detector(base_logits)

            # MSR refinement (must match model)
            final_logits = []
            for i in range(self.model.num_aspects):
                x = torch.cat([refined_feats[i], conflict_score], dim=1)
                delta = self.model.msr_refiners[i](x)
                final_logits.append(base_logits[i] + self.model.msr_strength * delta)

            logits = final_logits[aspect_index]  # [1,3]
            if target_class is None:
                tc = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)  # [1]
            else:
                tc = torch.tensor([target_class], device=logits.device)

            # return scalar logit of target class
            return logits.gather(1, tc.view(-1, 1)).squeeze(1)

        ig = IntegratedGradients(forward_embeds)
        attributions = ig.attribute(embeds, baselines=baseline, n_steps=n_steps)  # [1,T,E]

        scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()  # [T]
        mask = attention_mask.squeeze(0).detach().cpu().numpy()
        scores = scores * mask

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu().tolist())

        denom = np.sum(np.abs(scores)) + 1e-9
        scores = scores / denom

        ranked = sorted(list(zip(tokens, scores.tolist())), key=lambda x: abs(x[1]), reverse=True)
        return ranked[:top_k]

    def explain_with_lime(self, text: str, aspect_index: int = 0, max_len: int = 256, num_features: int = 10):
        """
        LIME explanation (optional).
        Requires: pip install lime
        """
        from lime.lime_text import LimeTextExplainer

        class_names = ["negative", "neutral", "positive"]

        def predict_proba(texts):
            self.model.eval()
            out_probs = []
            for t in texts:
                enc = self.tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=max_len
                ).to(self.device)
                with torch.no_grad():
                    out = self.model(enc["input_ids"], enc["attention_mask"])
                    probs = torch.softmax(out["aspect_logits"][aspect_index], dim=-1).squeeze(0).cpu().numpy()
                out_probs.append(probs)
            return np.vstack(out_probs)

        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(text, predict_proba, num_features=num_features)
        return exp.as_list()

    def explain_with_shap(self, text: str, aspect_index: int = 0, max_len: int = 256):
        """
        SHAP explanation (optional).
        Requires: pip install shap
        """
        import shap

        def f(texts):
            self.model.eval()
            out_probs = []
            for t in texts:
                enc = self.tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=max_len
                ).to(self.device)
                with torch.no_grad():
                    out = self.model(enc["input_ids"], enc["attention_mask"])
                    probs = torch.softmax(out["aspect_logits"][aspect_index], dim=-1).squeeze(0).cpu().numpy()
                out_probs.append(probs)
            return np.vstack(out_probs)

        masker = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(f, masker)
        return explainer([text])
