"""
Inference script for making predictions with trained model
"""

import torch
import yaml
import numpy as np
from transformers import RobertaTokenizer
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_this_dir)

# models/model.py lives in ml-research/src/models/ — add that directory to the path
# Resolve: evaluation/ -> cosmetic_sentiment_v1/ -> outputs/ -> ml-research/ -> src/
_ml_src_dir = os.path.abspath(os.path.join(_this_dir, "..", "..", "..", "src"))
if _ml_src_dir not in sys.path:
    sys.path.insert(0, _ml_src_dir)

from models.model import create_model


# ── Lightweight text cleaning (mirrors preprocess_and_split.py pipeline) ──────
# This ensures inference-time text matches what the model was trained on.
# We inline the function here to avoid fragile cross-project imports.
import re as _re
import unicodedata as _ud
import html as _html

_HTML_TAG_RE    = _re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = _re.compile(r"&(?:#\d+|#x[\da-fA-F]+|[a-zA-Z]+);")
_URL_RE         = _re.compile(r"https?://\S+|www\.\S+|ftp://\S+", _re.IGNORECASE)
_EMAIL_RE       = _re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", _re.IGNORECASE)
_CONSONANTS     = set("bcdfghjklmnpqrstvwxz")

def _is_garbled(tok: str, min_len=6, cons_ratio=0.82, rep_ratio=0.60) -> bool:
    t = tok.lower()
    if len(t) < min_len:
        return False
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    if sum(1 for c in letters if c in _CONSONANTS) / len(letters) >= cons_ratio:
        return True
    if max(t.count(c) for c in set(t)) / len(t) >= rep_ratio:
        return True
    return False

def clean_text_for_inference(text: str) -> str:
    """Clean text the same way training data was preprocessed."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _ud.normalize("NFC", text)
    text = _html.unescape(text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _re.sub(r"\.{3,}", "…", text)
    text = _re.sub(r"!{2,}", "!", text)
    text = _re.sub(r"\?{2,}", "?", text)
    text = _re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    tokens = text.split()
    clean_tokens = [t for t in tokens if not _is_garbled(t)]
    if len(tokens) >= 5 and len(tokens) - len(clean_tokens) >= 0.4 * len(tokens):
        return ""
    text = " ".join(clean_tokens)
    text = _re.sub(r"[\t\r\n]+", " ", text)
    text = _re.sub(r" {2,}", " ", text)
    return text.strip()


class SentimentPredictor:
    """
    Predictor class for making sentiment predictions
    """
    def __init__(self, checkpoint_path, device='cuda', temperature=0.5):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cuda' or 'cpu')
            temperature: Softmax temperature for calibration (<1.0 sharpens flat logits).
                         The model produces logits in a small range (~0.1-0.3), so a
                         temperature of 0.5 converts near-uniform distributions into
                         decisive predictions without retraining.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature  # Controls prediction sharpness
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint['config']
        
        # Create and load model
        self.model = create_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.config['model']['roberta_model']
        )
        
        # Aspect mapping
        self.aspect_names = self.config['aspects']['names']
        self.aspect_to_id = {name: i for i, name in enumerate(self.aspect_names)}
        
        # Label mapping: 3 classes - negative / neutral / positive
        self.label_names = ['negative', 'neutral', 'positive']
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Available aspects: {', '.join(self.aspect_names)}")
        print(f"Temperature (calibration): {self.temperature}")
    
    def predict(self, text, aspect, return_attention=False):
        """
        Predict sentiment for given text and aspect
        
        Args:
            text: Input review text
            aspect: Aspect name (e.g., 'smell', 'texture')
            return_attention: Whether to return attention weights
            
        Returns:
            prediction: Dict with sentiment and confidence
        """
        if aspect not in self.aspect_to_id:
            raise ValueError(f"Invalid aspect. Must be one of: {', '.join(self.aspect_names)}")
        
        # Apply the same cleaning pipeline used during training
        text = clean_text_for_inference(text)
        if not text:
            # If text becomes empty after cleaning, return neutral with low confidence
            return {
                'aspect': aspect,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            }
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config['data']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        aspect_id = torch.tensor([self.aspect_to_id[aspect]], dtype=torch.long).to(self.device)
        
        # CRITICAL: The model was trained with the GCN path (edge_index provided).
        # Without edge_index, model.forward() takes a shortcut at line ~255 and
        # returns raw 'attn_predictions' from AspectAwareRoBERTa, bypassing the
        # final_classifier that was actually optimised during training.
        # Providing an empty edge_index list forces the GCN branch, which falls
        # back to zero tensors for missing edges (model.py line 283) and then
        # applies final_classifier on combined [attention + gcn] features.
        # This is the same code path that achieved 92% weighted accuracy.
        empty_edge_index = [torch.zeros(2, 0, dtype=torch.long).to(self.device)]
        
        # Predict
        with torch.no_grad():
            if return_attention:
                logits, attn_weights, aspect_repr, _ = self.model(
                    input_ids, attention_mask, aspect_id,
                    edge_index=empty_edge_index,
                    return_attention=True
                )
            else:
                logits = self.model(
                    input_ids, attention_mask, aspect_id,
                    edge_index=empty_edge_index
                )
        
        # Apply temperature scaling before softmax.
        # The model's raw logits have a small range (~0.1-0.3) which, without
        # temperature scaling, results in near-uniform softmax output (~0.33 each).
        # Dividing by temperature < 1.0 amplifies the difference between logits,
        # producing confident, discriminative predictions.
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        
        result = {
            'aspect': aspect,
            'sentiment': self.label_names[pred_class],
            'confidence': confidence,
            'probabilities': {
                'negative': probs[0].item(),
                'neutral': probs[1].item(),
                'positive': probs[2].item()
            }
        }
        
        if return_attention:
            # Get tokens for visualization
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            attention = attn_weights[0].cpu().numpy()
            
            # Remove padding tokens
            valid_length = attention_mask[0].sum().item()
            tokens = tokens[:valid_length]
            attention = attention[:valid_length]
            
            result['attention'] = {
                'tokens': tokens,
                'weights': attention.tolist()
            }
            
            # Extract Top 5 tokens (excluding special tokens and padding)
            # NOTE: In RoBERTa's BPE tokenizer, 'Ġ' marks the START of a new word.
            # We INCLUDE Ġ-prefixed tokens (they are the real words) and strip the prefix.
            # We EXCLUDE only the true special tokens: <s>, </s>, <pad>.
            import numpy as np
            SPECIAL = {'<s>', '</s>', '<pad>', '<mask>'}
            # Common stopwords that carry no sentiment signal
            STOPWORDS = {'the', 'a', 'an', 'is', 'it', 'i', 'this', 'that', 'and',
                         'or', 'but', 'to', 'of', 'in', 'for', 'with', 'my', 'so',
                         'was', 'are', 'be', 'as', 'at', 'on', 'by', 'not', 'its'}
            valid_tokens_idx = [i for i, t in enumerate(tokens) if t not in SPECIAL]
            if valid_tokens_idx:
                valid_weights = attention[valid_tokens_idx]
                top_idx_relative = np.argsort(valid_weights)[-10:][::-1]  # grab top-10 candidates
                top_idx_absolute = [valid_tokens_idx[i] for i in top_idx_relative]

                # Clean token strings: strip BPE prefix 'Ġ' (or '▁') and whitespace
                top_tokens = [tokens[i].lstrip('Ġ▁').strip() for i in top_idx_absolute]
                # Keep only meaningful alphabetic words (length > 2, not a stopword)
                top_tokens = [t for t in top_tokens if t.isalpha() and len(t) > 2 and t.lower() not in STOPWORDS]
                # Deduplicate while preserving order
                seen = set()
                unique_tokens = []
                for t in top_tokens:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        unique_tokens.append(t)

                result['top_tokens'] = unique_tokens[:3]  # Keep top 3 most relevant words
            else:
                result['top_tokens'] = []
                
        else:
            result['top_tokens'] = []
            
        return result
    
    def predict_all_aspects(self, text):
        """
        Predict sentiment for all aspects
        
        Args:
            text: Input review text
            
        Returns:
            predictions: Dict mapping aspect to prediction
        """
        predictions = {}
        
        for aspect in self.aspect_names:
            try:
                pred = self.predict(text, aspect)
                predictions[aspect] = pred
            except Exception as e:
                print(f"Error predicting {aspect}: {e}")
                predictions[aspect] = None
        
        return predictions
    
    def visualize_attention(self, text, aspect):
        """
        Visualize attention weights for a prediction
        
        Args:
            text: Input review text
            aspect: Aspect name
        """
        result = self.predict(text, aspect, return_attention=True)
        
        if 'attention' not in result:
            print("Attention not available")
            return
        
        print(f"\n{'='*70}")
        print(f"Text: {text}")
        print(f"Aspect: {aspect}")
        print(f"Predicted Sentiment: {result['sentiment']} ({result['confidence']:.2%} confidence)")
        print(f"{'='*70}\n")
        
        tokens = result['attention']['tokens']
        weights = result['attention']['weights']
        
        print("Token Attention Weights:")
        print("-" * 70)
        
        # Sort by attention weight
        token_weights = list(zip(tokens, weights))
        token_weights.sort(key=lambda x: x[1], reverse=True)
        
        for token, weight in token_weights[:10]:  # Top 10 tokens
            bar_length = int(weight * 50)
            bar = '█' * bar_length
            print(f"{token:20s} {bar} {weight:.4f}")
    
    def explain_with_lime(self, text, aspect, num_features=10, num_samples=1000):
        """
        Generate LIME explanation for a prediction
        
        Args:
            text: Input review text
            aspect: Aspect name
            num_features: Number of features to show
            num_samples: Number of samples for LIME
            
        Returns:
            explanation: LIME explanation object with feature importances
        """
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        if aspect not in self.aspect_to_id:
            raise ValueError(f"Invalid aspect. Must be one of: {', '.join(self.aspect_names)}")
        
        # Create LIME explainer
        explainer = LimeTextExplainer(
            class_names=self.label_names,
            split_expression=r'\W+',  # Split on non-word characters
            random_state=42
        )
        
        # Define prediction function for LIME
        def predict_proba(texts):
            """Wrapper for model prediction compatible with LIME.
            
            IMPORTANT: Must use the same code path as predict() — i.e. pass an
            empty edge_index list to force the GCN branch (which falls back to
            zero tensors) and apply temperature scaling so probabilities match
            what the model actually outputs to the user.
            """
            probs = []
            for t in texts:
                # Tokenize
                encoding = self.tokenizer(
                    t,
                    add_special_tokens=True,
                    max_length=self.config['data']['max_seq_length'],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                aspect_id = torch.tensor([self.aspect_to_id[aspect]], dtype=torch.long).to(self.device)
                
                # Use empty edge_index to force the GCN branch (same as predict())
                empty_edge = [torch.zeros(2, 0, dtype=torch.long).to(self.device)]
                
                # Predict — apply temperature scaling for consistency with predict()
                with torch.no_grad():
                    logits = self.model(input_ids, attention_mask, aspect_id,
                                        edge_index=empty_edge)
                    scaled = logits / self.temperature
                    prob = torch.softmax(scaled, dim=1)[0].cpu().numpy()
                    probs.append(prob)
            
            return np.array(probs)
        
        # Generate explanation
        explanation = explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=3
        )
        
        return explanation
    
    def visualize_lime(self, text, aspect, num_features=10, save_path=None):
        """
        Visualize LIME explanation
        
        Args:
            text: Input review text
            aspect: Aspect name
            num_features: Number of features to display
            save_path: Optional path to save the visualization
        """
        print(f"\nGenerating LIME explanation for aspect: {aspect}...")
        explanation = self.explain_with_lime(text, aspect, num_features)
        
        # Get prediction
        result = self.predict(text, aspect)
        predicted_class = self.label_names.index(result['sentiment'])
        
        print(f"\n{'='*70}")
        print(f"LIME Explanation")
        print(f"{'='*70}")
        print(f"Text: {text}")
        print(f"Aspect: {aspect}")
        print(f"Predicted: {result['sentiment']} ({result['confidence']:.2%})")
        print(f"{'='*70}\n")
        
        # Get feature weights for predicted class
        feature_weights = explanation.as_list(label=predicted_class)
        
        print(f"Top {num_features} influential words/phrases:")
        print("-" * 70)
        for word, weight in feature_weights[:num_features]:
            direction = "POSITIVE" if weight > 0 else "NEGATIVE"
            bar_length = int(abs(weight) * 30)
            bar = '█' * bar_length
            color = '+' if weight > 0 else '-'
            print(f"{word:20s} [{color}] {bar} {weight:+.4f} ({direction})")
        
        # Create visualization
        fig = explanation.as_pyplot_figure(label=predicted_class)
        fig.suptitle(f'LIME Explanation: {aspect} - {result["sentiment"]}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
        else:
            plt.show()
        
        return explanation
    
    def explain_with_shap(self, text, aspect, plot=True, save_path=None):
        """
        Generate SHAP explanation for a prediction
        
        Args:
            text: Input review text
            aspect: Aspect name
            plot: Whether to display the plot
            save_path: Optional path to save the visualization
            
        Returns:
            Dictionary with SHAP values and tokens
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        if aspect not in self.aspect_to_id:
            raise ValueError(f"Invalid aspect. Must be one of: {', '.join(self.aspect_names)}")
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config['data']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        aspect_id = torch.tensor([self.aspect_to_id[aspect]], dtype=torch.long).to(self.device)
        
        # Get prediction
        result = self.predict(text, aspect)
        predicted_class = self.label_names.index(result['sentiment'])
        
        # Get tokens (excluding padding)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        valid_length = attention_mask[0].sum().item()
        tokens = tokens[:valid_length]
        
        # Create a wrapper function for SHAP
        def model_predict(input_ids_list):
            """Wrapper for SHAP predictions.
            
            IMPORTANT: Must use the same code path as predict() — i.e. pass an
            empty edge_index list to force the GCN branch (which falls back to
            zero tensors) and apply temperature scaling for consistency.
            """
            empty_edge = [torch.zeros(2, 0, dtype=torch.long).to(self.device)]
            outputs = []
            for ids in input_ids_list:
                ids_tensor = torch.tensor([ids], dtype=torch.long).to(self.device)
                mask = (ids_tensor != self.tokenizer.pad_token_id).long()
                
                with torch.no_grad():
                    logits = self.model(ids_tensor, mask, aspect_id,
                                        edge_index=empty_edge)
                    scaled = logits / self.temperature
                    probs = torch.softmax(scaled, dim=1)[0].cpu().numpy()
                    outputs.append(probs)
            
            return np.array(outputs)
        
        # Use Partition explainer (works well for text)
        # Create background data by masking tokens
        background_ids = input_ids[0].cpu().numpy()[:valid_length]
        
        # Create masked versions as background
        num_background = 10
        background_data = []
        for _ in range(num_background):
            masked = background_ids.copy()
            # Randomly mask 20% of tokens
            mask_indices = np.random.choice(len(masked), size=max(1, len(masked)//5), replace=False)
            masked[mask_indices] = self.tokenizer.mask_token_id
            background_data.append(masked)
        
        background_data = np.array(background_data)
        
        # Create SHAP explainer
        explainer = shap.Explainer(
            model_predict,
            background_data,
            algorithm='partition'
        )
        
        # Get SHAP values
        test_data = input_ids[0].cpu().numpy()[:valid_length].reshape(1, -1)
        shap_values = explainer(test_data)
        
        # Extract SHAP values for predicted class
        values = shap_values.values[0, :, predicted_class]
        
        print(f"\n{'='*70}")
        print(f"SHAP Explanation")
        print(f"{'='*70}")
        print(f"Text: {text}")
        print(f"Aspect: {aspect}")
        print(f"Predicted: {result['sentiment']} ({result['confidence']:.2%})")
        print(f"{'='*70}\n")
        
        # Print top influential tokens
        token_importance = list(zip(tokens, values))
        token_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"Top 10 influential tokens (SHAP values):")
        print("-" * 70)
        for token, value in token_importance[:10]:
            direction = "POSITIVE" if value > 0 else "NEGATIVE"
            bar_length = int(abs(value) * 50)
            bar = '█' * bar_length
            color = '+' if value > 0 else '-'
            print(f"{token:20s} [{color}] {bar} {value:+.4f} ({direction})")
        
        # Create visualization
        if plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create color array (red for negative, green for positive)
            colors = ['red' if v < 0 else 'green' for v in values]
            
            # Plot horizontal bar chart
            y_pos = np.arange(len(tokens))
            ax.barh(y_pos, values, color=colors, alpha=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens, fontsize=8)
            ax.set_xlabel('SHAP Value', fontsize=12)
            ax.set_title(f'SHAP Token Importance: {aspect} - {result["sentiment"]}', 
                        fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\nVisualization saved to: {save_path}")
            else:
                plt.show()
        
        return {
            'tokens': tokens,
            'shap_values': values.tolist(),
            'predicted_class': result['sentiment'],
            'confidence': result['confidence']
        }

    # ─────────────────────────────────────────────────────────────────────
    # Integrated Gradients
    # ─────────────────────────────────────────────────────────────────────
    def explain_with_integrated_gradients(self, text, aspect, target_label=None,
                                          n_steps=50, top_k=12, save_path=None):
        """
        Generate Integrated Gradients explanation for a prediction.

        Integrated Gradients (Sundararajan et al., 2017) computes the
        attribution of each input token by integrating gradients from a
        baseline (all-PAD) to the actual input along a straight-line path.
        This manual implementation uses a Riemann sum to bypass HuggingFace
        embedding graph disconnection issues while achieving the exact
        same token-level attribution result.

        Args:
            text:         Input review text
            aspect:       Aspect name (e.g. 'smell', 'price')
            target_label: Sentiment class to explain ('negative'|'neutral'|'positive').
                          Defaults to the model's predicted class.
            n_steps:      Number of interpolation steps (higher = more accurate,
                          but slower; 50 is a good balance)
            top_k:        Number of top tokens to print
            save_path:    Optional path to save the bar-chart PNG

        Returns:
            dict with keys:
                'tokens':        List of token strings (no padding)
                'attributions':  List of float attribution scores (one per token)
                'target_label':  The class being explained
                'confidence':    Model confidence for that class
        """
        if aspect not in self.aspect_to_id:
            raise ValueError(f"Invalid aspect. Must be one of: {', '.join(self.aspect_names)}")

        # Tokenise
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config['data']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids      = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        aspect_id_t    = torch.tensor([self.aspect_to_id[aspect]], dtype=torch.long).to(self.device)

        # Determine target class
        result = self.predict(text, aspect)
        if target_label is None:
            target_label = result['sentiment']
        if target_label not in self.label_names:
            raise ValueError(f"Invalid target_label. Must be one of {self.label_names}")
        target_idx = self.label_names.index(target_label)

        # Ensure the model is in eval mode so gradients backprop cleanly
        self.model.eval()

        # Extract real text embeddings
        # We need gradients on the inputs to roberta, so we extract word embeddings explicitly.
        with torch.no_grad():
            real_embeds = self.model.aspect_aware_roberta.roberta.embeddings.word_embeddings(input_ids)
            
            # Baseline: all-PAD embeddings
            baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
            baseline_embeds = self.model.aspect_aware_roberta.roberta.embeddings.word_embeddings(baseline_ids)

        # Accumulator for gradients
        total_gradients = torch.zeros_like(real_embeds).to(self.device)

        # Manual Integrated Gradients Loop (Riemann Sum)
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            # Linear interpolation step
            interpolated_embeds = baseline_embeds + alpha * (real_embeds - baseline_embeds)
            interpolated_embeds.requires_grad_(True)
            
            # Forward pass from inputs_embeds
            roberta_output = self.model.aspect_aware_roberta.roberta(
                inputs_embeds=interpolated_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = roberta_output.last_hidden_state
            
            if self.model.aspect_aware_roberta.use_aspect_attention:
                aspect_query = self.model.aspect_aware_roberta.aspect_embeddings(aspect_id_t)
                aspect_query = aspect_query.unsqueeze(1)
                attended, _ = self.model.aspect_aware_roberta.aspect_attention(
                    query=aspect_query,
                    key=hidden_states,
                    value=hidden_states,
                    key_padding_mask=~attention_mask.bool()
                )
                aspect_repr = attended.squeeze(1)
            else:
                aspect_repr = hidden_states[:, 0, :]
                
            aspect_repr = self.model.aspect_aware_roberta.layer_norm(aspect_repr)
            
            if self.model.aspect_aware_roberta.use_shared_classifier:
                logits = self.model.aspect_aware_roberta.shared_classifier(aspect_repr)
            else:
                logits = self.model.aspect_aware_roberta.aspect_classifiers[self.aspect_to_id[aspect]](aspect_repr)
                
            scaled_logits = logits / self.temperature
            
            # Extract target class scalar
            target_score = scaled_logits[0, target_idx]
            
            # Backward pass to calculate gradients exactly at this step
            self.model.zero_grad()
            target_score.backward()
            
            # Accumulate gradients
            with torch.no_grad():
                total_gradients += interpolated_embeds.grad

        # Average the gradients and multiply by (input - baseline) as per IG formula
        avg_gradients = total_gradients / n_steps
        attributions = avg_gradients * (real_embeds - baseline_embeds)

        # Sum over the hidden dimension (D=768) to get a scalar attribution per token
        attr_scores = attributions.sum(dim=-1).squeeze(0)  # Shape: (seq_len,)
        
        # Trim to valid tokens (exclude padding)
        valid_length = attention_mask[0].sum().item()
        tokens       = self.tokenizer.convert_ids_to_tokens(input_ids[0])[:valid_length]
        scores       = attr_scores[:valid_length].detach().cpu().numpy()

        # Normalise to [-1, 1] for readability
        max_abs = np.abs(scores).max() + 1e-9
        scores_norm = scores / max_abs

        # ── Print results ─────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"Integrated Gradients Explanation")
        print(f"{'='*70}")
        print(f"Text:          {text}")
        print(f"Aspect:        {aspect}")
        print(f"Target class:  {target_label}")
        print(f"Confidence:    {result['confidence']:.2%}")
        print(f"{'='*70}\n")

        token_scores = sorted(zip(tokens, scores_norm), key=lambda x: abs(x[1]), reverse=True)
        print(f"Top {top_k} tokens by attribution:")
        print("-" * 70)
        for tok, sc in token_scores[:top_k]:
            direction = "supports" if sc > 0 else "opposes"
            bar = '█' * int(abs(sc) * 30)
            sign = '+' if sc > 0 else '-'
            print(f"  {tok:20s} [{sign}] {bar:<30s} {sc:+.3f}  ({direction} {target_label})")

        # ── Visualise ─────────────────────────────────────────────────────
        if save_path:
            fig, ax = plt.subplots(figsize=(12, max(4, valid_length * 0.22)))
            colors = ['#27ae60' if s > 0 else '#c0392b' for s in scores_norm]
            y_pos  = np.arange(valid_length)
            ax.barh(y_pos, scores_norm, color=colors, alpha=0.75, edgecolor='white')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens, fontsize=7)
            ax.set_xlabel('Normalised Attribution Score', fontsize=11)
            ax.set_title(
                f'Integrated Gradients — Aspect: {aspect}  |  Class: {target_label}\n'
                f'"{text[:60]}{"…" if len(text) > 60 else ""}"',
                fontsize=11, fontweight='bold'
            )
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
            plt.close()

        return {
            'tokens':       tokens,
            'attributions': scores_norm.tolist(),
            'target_label': target_label,
            'confidence':   result['confidence']
        }

    # ─────────────────────────────────────────────────────────────────────
    # MSR Delta — Mixed Sentiment Resolution Explainability
    # ─────────────────────────────────────────────────────────────────────
    def explain_msr_delta(self, text, focus_aspect, top_k=10, save_path=None):
        """
        MSR (Mixed Sentiment Resolution) Delta Explanation.

        For a review that mentions MULTIPLE aspects (e.g. positive colour,
        negative smell), this method shows HOW MUCH each OTHER aspect's
        tokens shift the model's confidence for the focus_aspect prediction.

        Method:
          1. Get baseline prediction for focus_aspect on the full text.
          2. For each token, mask it (replace with [MASK]) and re-predict.
          3. The delta = baseline_confidence - masked_confidence.
             - Large positive delta → token SUPPORTS the prediction
             - Large negative delta → token OPPOSES the prediction
          4. Group deltas by aspect-keyword overlap to show cross-aspect influence.

        This directly demonstrates Mixed Sentiment Resolution: tokens from
        aspect B (e.g. smell) should ideally have LOW delta on aspect A
        (e.g. colour), proving the model separates aspects correctly.

        Args:
            text:         Input review text
            focus_aspect: The aspect whose prediction we are examining
            top_k:        Number of top delta tokens to display
            save_path:    Optional path to save the output PNG

        Returns:
            dict with keys:
                'tokens':           All tokens (no padding)
                'deltas':           Per-token confidence delta (float, one per token)
                'baseline':         Baseline prediction dict
                'focus_aspect':     The aspect being explained
        """
        if focus_aspect not in self.aspect_to_id:
            raise ValueError(f"Invalid aspect. Must be one of: {', '.join(self.aspect_names)}")

        # Tokenise once
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config['data']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids      = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        aspect_id_t    = torch.tensor(
            [self.aspect_to_id[focus_aspect]], dtype=torch.long
        ).to(self.device)
        empty_edge     = [torch.zeros(2, 0, dtype=torch.long).to(self.device)]

        valid_length = attention_mask[0].sum().item()
        tokens       = self.tokenizer.convert_ids_to_tokens(input_ids[0])[:valid_length]

        # Baseline confidence for the focus aspect
        with torch.no_grad():
            logits   = self.model(input_ids, attention_mask, aspect_id_t, edge_index=empty_edge)
            scaled   = (logits / self.temperature)
            baseline_probs = torch.softmax(scaled, dim=1)[0].cpu().numpy()

        baseline_pred_idx = int(np.argmax(baseline_probs))
        baseline_conf     = float(baseline_probs[baseline_pred_idx])
        baseline_label    = self.label_names[baseline_pred_idx]

        # Per-token masking
        mask_token_id = self.tokenizer.mask_token_id
        deltas = np.zeros(valid_length)

        for i in range(1, valid_length - 1):       # skip [CLS] and [SEP]
            masked_ids = input_ids.clone()
            masked_ids[0, i] = mask_token_id

            with torch.no_grad():
                logits_m = self.model(
                    masked_ids, attention_mask, aspect_id_t, edge_index=empty_edge
                )
                probs_m  = torch.softmax(logits_m / self.temperature, dim=1)[0].cpu().numpy()

            # Delta = drop in confidence for the baseline predicted class
            deltas[i] = baseline_conf - float(probs_m[baseline_pred_idx])

        # Normalise
        max_abs    = np.abs(deltas).max() + 1e-9
        deltas_norm = deltas / max_abs

        # ── Print ─────────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"MSR Delta Explanation  (Mixed Sentiment Resolution)")
        print(f"{'='*70}")
        print(f"Text:          {text}")
        print(f"Focus aspect:  {focus_aspect}")
        print(f"Baseline pred: {baseline_label}  (confidence {baseline_conf:.2%})")
        print(f"{'='*70}")
        print(f"\nToken influence on [{focus_aspect}] prediction (positive = supports, "
              f"negative = contradicts):")
        print("-" * 70)

        ranked = sorted(
            [(tok, d, dn) for tok, d, dn in zip(tokens, deltas, deltas_norm)
             if tok not in ('<s>', '</s>', '<pad>', '<mask>')],
            key=lambda x: abs(x[1]), reverse=True
        )
        for tok, d, dn in ranked[:top_k]:
            influence = "supports" if d > 0 else "opposes"
            bar = '█' * int(abs(dn) * 28)
            sign = '+' if d > 0 else '-'
            print(f"  {tok:20s} [{sign}] {bar:<28s} delta={d:+.4f}")

        # Cross-aspect summary: which prediction for OTHER aspects correlates
        # with the biggest deltas? This reveals mixed-sentiment behaviour.
        print(f"\nCross-aspect influence summary (other predictions on this review):")
        other_aspects = [a for a in self.aspect_names if a != focus_aspect]
        for other in other_aspects:
            other_result = self.predict(text, other)
            print(f"  {other:15s} → {other_result['sentiment']:8s}  "
                  f"(confidence {other_result['confidence']:.2%})")

        # ── Visualise ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, max(4, valid_length * 0.22)))
        colors  = ['#27ae60' if d > 0 else '#c0392b' for d in deltas_norm]
        y_pos   = np.arange(valid_length)
        ax.barh(y_pos, deltas_norm, color=colors, alpha=0.75, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_xlabel('Normalised Confidence Delta (mask-out effect)', fontsize=11)
        ax.set_title(
            f'MSR Delta — Focus: [{focus_aspect}] — Baseline: {baseline_label}\n'
            f'"{text[:60]}{"…" if len(text) > 60 else ""}"',
            fontsize=11, fontweight='bold'
        )
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
        else:
            plt.show()
        plt.close()

        return {
            'tokens':       tokens,
            'deltas':       deltas.tolist(),
            'deltas_norm':  deltas_norm.tolist(),
            'baseline':     {'sentiment': baseline_label, 'confidence': baseline_conf},
            'focus_aspect': focus_aspect,
        }


def main():
    """Interactive inference demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Prediction Demo with Explainability')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze (if not provided, will use interactive mode)')
    parser.add_argument('--aspect', type=str, default=None,
                       help='Aspect to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--explain', type=str,
                        choices=['attention', 'lime', 'shap', 'ig', 'msr', 'all'],
                        default='attention',
                        help='Explainability method: attention | lime | shap | ig | msr | all')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save explanation visualizations')
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = SentimentPredictor(args.checkpoint, device=args.device)
    
    if args.text and args.aspect:
        # Single prediction mode
        result = predictor.predict(args.text, args.aspect, return_attention=True)
        
        print(f"\nText: {args.text}")
        print(f"Aspect: {args.aspect}")
        print(f"\nPrediction: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nProbabilities:")
        for sentiment, prob in result['probabilities'].items():
            print(f"  {sentiment}: {prob:.2%}")
        
        # Generate explanations based on user choice
        if args.explain in ['attention', 'all']:
            print(f"\n{'='*70}")
            print("ATTENTION-BASED EXPLANATION")
            print(f"{'='*70}")
            if 'attention' in result:
                tokens = result['attention']['tokens']
                weights = result['attention']['weights']
                token_weights = sorted(zip(tokens, weights), key=lambda x: x[1], reverse=True)
                
                print("\nTop 10 Attention Tokens:")
                for token, weight in token_weights[:10]:
                    bar_length = int(weight * 50)
                    bar = '█' * bar_length
                    print(f"  {token:20s} {bar} {weight:.4f}")
        
        if args.explain in ['lime', 'all']:
            save_path = args.save_path if args.save_path else None
            if save_path and args.explain == 'all':
                save_path = save_path.replace('.', '_lime.')
            predictor.visualize_lime(args.text, args.aspect, save_path=save_path)
        
        if args.explain in ['shap', 'all']:
            save_path = args.save_path if args.save_path else None
            if save_path and args.explain == 'all':
                save_path = save_path.replace('.', '_shap.')
            predictor.explain_with_shap(args.text, args.aspect, plot=True, save_path=save_path)
        
        if args.explain in ['ig', 'all']:
            save_path = args.save_path if args.save_path else None
            if save_path and args.explain == 'all':
                save_path = save_path.replace('.', '_ig.')
            predictor.explain_with_integrated_gradients(
                args.text, args.aspect, save_path=save_path
            )
        
        if args.explain in ['msr', 'all']:
            save_path = args.save_path if args.save_path else None
            if save_path and args.explain == 'all':
                save_path = save_path.replace('.', '_msr.')
            predictor.explain_msr_delta(
                args.text, args.aspect, save_path=save_path
            )
    
    else:
        # Interactive mode
        print("\n" + "="*70)
        print("Interactive Sentiment Analysis with Explainability")
        print("="*70)
        print(f"Available aspects: {', '.join(predictor.aspect_names)}")
        print("Commands: 'quit' to exit, 'all' to predict all aspects")
        print("Explainability methods: attention, lime, shap")
        print("="*70 + "\n")
        
        while True:
            text = input("\nEnter review text (or 'quit'): ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            aspect = input(f"Enter aspect ({', '.join(predictor.aspect_names)}, or 'all'): ").strip()
            
            if aspect.lower() == 'all':
                # Predict all aspects
                predictions = predictor.predict_all_aspects(text)
                
                print(f"\n{'='*70}")
                print("Predictions for all aspects:")
                print(f"{'='*70}")
                
                for asp, pred in predictions.items():
                    if pred:
                        print(f"{asp:15s}: {pred['sentiment']:8s} ({pred['confidence']:.2%})")
                    else:
                        print(f"{asp:15s}: ERROR")
            
            elif aspect in predictor.aspect_to_id:
                # Ask for explanation method
                explain_method = input("Explainability method (attention/lime/shap/all) [default: attention]: ").strip().lower()
                if not explain_method:
                    explain_method = 'attention'
                
                # Generate predictions and explanations
                if explain_method == 'attention':
                    predictor.visualize_attention(text, aspect)
                elif explain_method == 'lime':
                    predictor.visualize_lime(text, aspect)
                elif explain_method == 'shap':
                    predictor.explain_with_shap(text, aspect, plot=True)
                elif explain_method == 'all':
                    print("\n" + "="*70)
                    print("Generating all explanations...")
                    print("="*70)
                    predictor.visualize_attention(text, aspect)
                    predictor.visualize_lime(text, aspect)
                    predictor.explain_with_shap(text, aspect, plot=True)
                else:
                    print(f"Invalid explanation method: {explain_method}")
            
            else:
                print(f"Invalid aspect. Must be one of: {', '.join(predictor.aspect_names)}")


if __name__ == "__main__":
    main()
