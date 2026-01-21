import torch
import torch.nn.functional as F
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients
from transformers import RobertaTokenizerFast
import spacy
from tqdm import tqdm

# Load SpaCy model for dependency parsing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    # We might need to handle this more gracefully or assume it's there based on requirements

# -------------------------------------------------------------------
# HELPER: ADJACENCY MATRIX GENERATION
# -------------------------------------------------------------------
# This function is copied/adapted from the main model file to ensure
# we can generate the graph structure on the fly for new texts.
def dependency_adj_matrix(text, tokenizer, max_len=256):
    """
    Generates the dependency adjacency matrix for a single text input.
    This connects related words (like 'expensive' -> 'price') based on grammar.
    """
    doc = nlp(text)
    encoding = tokenizer(
        text, 
        padding="max_length", 
        truncation=True, 
        max_length=max_len, 
        return_offsets_mapping=True
    )
    offset_mapping = encoding["offset_mapping"] 
    seq_len = len(encoding["input_ids"])
    
    adj = np.zeros((seq_len, seq_len), dtype=np.float32)
    char_to_token_idx = {}
    for i, (start, end) in enumerate(offset_mapping):
        for char_pos in range(start, end):
            char_to_token_idx[char_pos] = i
            
    for token in doc:
        if token.idx in char_to_token_idx:
            current_idx = char_to_token_idx[token.idx]
            if token.head.idx in char_to_token_idx:
                head_idx = char_to_token_idx[token.head.idx]
                adj[current_idx][head_idx] = 1.0
                adj[head_idx][current_idx] = 1.0
                
    for i in range(seq_len):
        adj[i][i] = 1.0
        
    # Normalize
    row_sum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = np.matmul(np.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    return adj_normalized

# -------------------------------------------------------------------
# MODEL WRAPPER
# -------------------------------------------------------------------
class ModelWrapper:
    """
    A bridge between the raw text and the complex GCN model.
    Tools like SHAP and LIME expect a function that takes [List of Strings] 
    and returns [Probabilities]. This wrapper provides that.
    """
    def __init__(self, model, tokenizer, device, max_len=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.model.eval()

    def predict_proba(self, texts):
        """
        The key function for LIME/SHAP.
        Input: List of raw strings (e.g., ["I love it", "Terrible price"])
        Output: Numpy array of probabilities for the target aspect.
        
        NOTE: Since our model has MULTIPLE outputs (Price, Packing, etc.),
        we can't return all of them at once for standard explainers.
        We will set a 'target_aspect_idx' before calling this.
        """
        # Ensure input is a list
        if isinstance(texts, str) or isinstance(texts, np.str_):
            texts = [str(texts)]
            
        # Optimization: Pre-allocate lists
        input_ids_list = []
        attention_mask_list = []
        adj_list = []

        for text in texts:
            # FIX: Ensure strict python string
            text_str = str(text)
            
            # 1. Tokenize
            enc = self.tokenizer(
                text_str,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            input_ids_list.append(enc["input_ids"])
            attention_mask_list.append(enc["attention_mask"])
            
            # 2. Build Graph (Expensive step!)
            # We must rebuild graph because LIME changes the text structure
            adj = dependency_adj_matrix(text_str, self.tokenizer, self.max_len)
            adj_list.append(torch.tensor(adj, dtype=torch.float))

        # Stack into tensors
        input_ids = torch.cat(input_ids_list, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask_list, dim=0).to(self.device)
        adj_matrix = torch.stack(adj_list).to(self.device)

        # 3. Model Inference
        with torch.no_grad():
            logits_list = self.model(input_ids, attention_mask, adj_matrix)
            
            # Select the output for the SPECIFIC aspect we want to explain
            # self.target_aspect_idx must be set externally before calling this!
            target_logits = logits_list[self.target_aspect_idx]
            
            # Convert to probabilities
            probs = F.softmax(target_logits, dim=1)
            
        return probs.cpu().numpy()

# -------------------------------------------------------------------
# XAI ANALYZER (The Orchestrator)
# -------------------------------------------------------------------
class XAIAnalyzer:
    def __init__(self, model, project_dir, device):
        self.model = model
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.device = device
        self.aspects = ["stayingpower","texture","smell","price","colour","shipping","packing"]
        
        # Create the wrapper
        self.wrapper = ModelWrapper(model, self.tokenizer, device)

    def explain_lime(self, text, aspect, num_features=10, num_samples=200):
        """
        Uses LIME (Local Interpretable Model-agnostic Explanations).
        Concept: It tweaks the sentence slightly (removes words) and sees how the prediction changes.
        
        OPTIMIZATION: Default num_samples reduced to 200 (from 5000) for speed.
        Increase this if explanations are unstable.
        """
        if aspect not in self.aspects:
            raise ValueError(f"Unknown aspect: {aspect}")
            
        aspect_idx = self.aspects.index(aspect)
        self.wrapper.target_aspect_idx = aspect_idx # Tell wrapper which output to focus on
        
        explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])
        
        # Run LIME
        # We pass self.wrapper.predict_proba as the "black box" function
        exp = explainer.explain_instance(
            str(text), # Ensure string
            self.wrapper.predict_proba, 
            num_features=num_features,
            num_samples=num_samples, # Reduced for performance
            labels=[0, 2] # Focus on Negative (0) and Positive (2)
        )
        return exp

    def explain_shap(self, text, aspect):
        """
        Uses SHAP (SHapley Additive exPlanations).
        Concept: Game theory approach. assigns each word a "contribution score" to the final probability.
        """
        if aspect not in self.aspects:
            raise ValueError(f"Unknown aspect: {aspect}")
            
        aspect_idx = self.aspects.index(aspect)
        self.wrapper.target_aspect_idx = aspect_idx
        
        # SHAP for text is computationally heavy. We use a partition explainer.
        # We use a masker to "hide" parts of the text (replace with [MASK] token logic)
        masker = shap.maskers.Text(self.tokenizer)
        
        explainer = shap.Explainer(self.wrapper.predict_proba, masker)
        
        # Run SHAP with limited evaluations if possible.
        # max_evals controls speed. Default is often high.
        shap_values = explainer([str(text)], max_evals=100) 
        return shap_values

    def explain_integrated_gradients(self, text, aspect):
        """
        Uses Integrated Gradients (Captum).
        Concept: Advanced calculus. Calculates the gradient (slope) of the output 
        relative to the input word embeddings. High slope = Word matters a lot.
        """
        # NOTE: IG requires direct access to embeddings, which is tricky with our custom GCN wrapper.
        # For simplicity and robustness with the custom Graph layer, 
        # SHAP and LIME are often safer bets for "Black Box" explanation.
        # However, we can implement a specific forward function for Captum if needed.
        # For this phase, we will focus on SHAP/LIME as they are more "User Friendly" visual-wise.
        pass
