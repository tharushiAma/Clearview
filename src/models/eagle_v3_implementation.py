# EAGLE V3 - Pure Adaptive Loss Implementation
# Integrates V2 architectural improvements with Adaptive Focal Loss
# STRICTLY NO SYNTHETIC DATA - Handles imbalance via dynamic weighting

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import RobertaModel
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# PART 1: ADAPTIVE FOCAL LOSS (Dynamic Weighting)
# ==============================================================================

class AdaptiveFocalLoss(nn.Module):
    """
    Dynamically adjusts class weights based on per-class performance.
    
    Key Innovation: Weights update after each epoch based on which classes
    the model struggles with most.
    """
    def __init__(self, num_classes=3, gamma=3.0, initial_alpha=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.device = device
        
        # Initialize alpha (class weights)
        if initial_alpha is not None:
            self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32), requires_grad=False)
        else:
            self.alpha = nn.Parameter(torch.ones(num_classes), requires_grad=False)
        
        # Track per-class performance
        self.class_counts = torch.zeros(num_classes, device=device)
        self.class_correct = torch.zeros(num_classes, device=device)
        self.reset_stats()
    
    def reset_stats(self):
        """Reset statistics at the start of each epoch"""
        self.class_counts = torch.zeros(self.num_classes, device=self.device)
        self.class_correct = torch.zeros(self.num_classes, device=self.device)
    
    def update_stats(self, predictions, targets):
        """
        Update per-class statistics during training.
        Call this in training loop BEFORE loss computation.
        """
        # Filter out ignore_index
        valid_mask = (targets != -100)
        if not valid_mask.any():
            return
        
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        for c in range(self.num_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                self.class_counts[c] += mask.sum()
                self.class_correct[c] += ((predictions == c) & mask).sum()
    
    def update_weights(self):
        """
        Update alpha weights based on accumulated statistics.
        Call this at the END of each epoch.
        """
        # Compute per-class accuracy
        eps = 1e-8
        acc = self.class_correct / (self.class_counts + eps)
        
        # Avoid division by zero
        acc = torch.clamp(acc, min=eps, max=1-eps)
        
        # Inverse accuracy weighting (worse classes get higher weight)
        # Add 1.0 to prevent extreme weights
        new_alpha = (1.0 - acc) + 1.0
        
        # Normalize to sum to num_classes
        new_alpha = new_alpha * (self.num_classes / new_alpha.sum())
        
        # Smooth update (moving average) to prevent oscillation
        self.alpha.data = 0.7 * self.alpha.data + 0.3 * new_alpha
        
        print(f"[AdaptiveFocalLoss] Updated alpha: {self.alpha.data.cpu().numpy()}")
        print(f"  Class accuracies: {acc.cpu().numpy()}")
    
    def forward(self, inputs, targets):
        """
        Compute focal loss with current alpha weights.
        """
        # Compute standard cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=-100)
        
        # Get probability of correct class
        pt = torch.exp(-ce_loss)
        
        # Get alpha for each sample
        # Use a safe version of targets for indexing (replace -100 with 0 temporarily)
        safe_targets = torch.where(targets == -100, torch.zeros_like(targets), targets)
        alpha_t = self.alpha[safe_targets]
        alpha_t = torch.where(targets == -100, torch.tensor(0.0, device=self.device), alpha_t)
        
        # Focal loss: alpha * (1-pt)^gamma * CE
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        
        # Only average over valid samples
        valid_mask = (targets != -100)
        if valid_mask.sum() > 0:
            return focal_loss.sum() / valid_mask.sum()
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)


# ==============================================================================
# PART 2: UNCERTAINTY-AWARE PREDICTION HEAD (From V2)
# ==============================================================================

class UncertaintyHead(nn.Module):
    """
    Predicts uncertainty alongside logits.
    High uncertainty -> push toward neutral class
    """
    def __init__(self, hidden_dim, num_classes=3, dropout=0.3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Uncertainty estimator (predicts epistemic uncertainty)
        self.uncertainty_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Uncertainty in [0, 1]
        )
    
    def forward(self, features):
        logits = self.classifier(features)
        uncertainty = self.uncertainty_net(features).squeeze(-1)
        return logits, uncertainty
    
    def calibrate_neutral(self, logits, uncertainty, neutral_idx=1, threshold=0.5):
        """
        Boost neutral class probability for uncertain predictions.
        """
        # For high uncertainty samples, boost neutral class
        uncertain_mask = (uncertainty > threshold).unsqueeze(-1)
        
        # Create neutral boost (add to neutral logit)
        boost = torch.zeros_like(logits)
        boost[:, neutral_idx] = uncertainty * 2.0  # Scale factor tunable
        
        # Apply boost only to uncertain samples
        calibrated_logits = torch.where(uncertain_mask, logits + boost, logits)
        
        return calibrated_logits


# ==============================================================================
# PART 3: ASPECT-SPECIFIC FEATURE ROUTER (From V2)
# ==============================================================================

class AspectFeatureRouter(nn.Module):
    """
    Learn aspect-specific routing between transformer and GNN features.
    """
    def __init__(self, hidden_dim, num_aspects):
        super().__init__()
        
        # Per-aspect routing networks
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # [transformer_weight, gnn_weight]
                nn.Softmax(dim=-1)
            )
            for _ in range(num_aspects)
        ])
    
    def forward(self, transformer_features, gnn_features, aspect_idx):
        # Concatenate features
        combined = torch.cat([transformer_features, gnn_features], dim=-1)
        
        # Get routing weights
        weights = self.routers[aspect_idx](combined)  # [B, 2]
        
        # Weighted fusion
        fused = weights[:, 0:1] * transformer_features + weights[:, 1:2] * gnn_features
        
        return fused, weights


# ==============================================================================
# PART 4: POSITIONAL GRAPH ATTENTION (From V2)
# ==============================================================================

class PositionAwareGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len=256, dropout=0.3):
        super().__init__()
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(size=(1, out_features)))
        self.a_dst = nn.Parameter(torch.empty(size=(1, out_features)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        
        self.position_embedding = nn.Embedding(max_seq_len, out_features)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = dropout  # For regularization
    
    def forward(self, h, adj, positions):
        B, N, D = h.shape
        
        # Linear transformation + position encoding
        Wh = self.W(h)
        pos_emb = self.position_embedding(positions)
        Wh = Wh + pos_emb
        
        # Attention coefficients
        f_src = torch.matmul(Wh, self.a_src.transpose(0, 1))
        f_dst = torch.matmul(Wh, self.a_dst.transpose(0, 1))
        e = self.leakyrelu(f_src + f_dst.transpose(1, 2))
        
        # Mask with adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        
        # Edge dropout during training
        if self.training:
            edge_mask = (torch.rand_like(attention) > self.edge_dropout).float()
            attention = attention * edge_mask
            # Renormalize
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        attention = self.dropout(attention)
        h_prime = torch.bmm(attention, Wh)
        
        return h_prime, attention


class ResidualGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, max_seq_len=256, dropout=0.3):
        super().__init__()
        self.gat = PositionAwareGraphAttention(in_dim, out_dim, max_seq_len, dropout)
        self.transform = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, h, adj, positions):
        h_prime, attn = self.gat(h, adj, positions)
        h_out = h_prime + self.transform(h)
        h_out = self.layer_norm(h_out)
        return h_out, attn


class DualChannelGCN(nn.Module):
    """
    Two-stream GCN: Syntactic + Semantic
    """
    def __init__(self, hidden_size=768, gcn_dim=300, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Syntactic stream
        self.syntactic_layers = nn.ModuleList([
            ResidualGATLayer(
                in_dim=hidden_size if i == 0 else gcn_dim,
                out_dim=gcn_dim,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Semantic stream
        self.semantic_layers = nn.ModuleList([
            ResidualGATLayer(
                in_dim=hidden_size if i == 0 else gcn_dim,
                out_dim=gcn_dim,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(2 * gcn_dim, gcn_dim)
        self.dropout = nn.Dropout(dropout)
    
    def build_semantic_graph(self, h, aspect_mask):
        B, N, D = h.shape
        
        # Compute cosine similarity
        h_norm = F.normalize(h, p=2, dim=-1)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))
        
        # Top-k connections
        k = min(10, N)
        topk_vals, topk_indices = torch.topk(sim, k=k, dim=-1)
        
        # Create adjacency
        batch_idx = torch.arange(B, device=h.device).view(-1, 1, 1).expand(B, N, k)
        row_idx = torch.arange(N, device=h.device).view(1, -1, 1).expand(B, N, k)
        
        semantic_adj = torch.zeros(B, N, N, device=h.device)
        flat_idx = (batch_idx * N * N + row_idx * N + topk_indices).view(-1)
        semantic_adj.view(-1).scatter_(0, flat_idx, topk_vals.view(-1))
        
        # Symmetrize
        semantic_adj = (semantic_adj + semantic_adj.transpose(1, 2)) / 2
        
        # Add self-loops
        eye = torch.eye(N, device=h.device).unsqueeze(0)
        semantic_adj = semantic_adj + eye
        
        # Normalize
        row_sum = semantic_adj.sum(dim=-1, keepdim=True)
        semantic_adj = semantic_adj / (row_sum + 1e-8)
        
        return semantic_adj
    
    def forward(self, h, syntactic_adj, aspect_mask, positions):
        semantic_adj = self.build_semantic_graph(h, aspect_mask)
        
        # Syntactic stream
        h_syn = h
        for layer in self.syntactic_layers:
            h_syn, _ = layer(h_syn, syntactic_adj, positions)
            h_syn = F.relu(h_syn)
            h_syn = self.dropout(h_syn)
        
        # Semantic stream
        h_sem = h
        for layer in self.semantic_layers:
            h_sem, _ = layer(h_sem, semantic_adj, positions)
            h_sem = F.relu(h_sem)
            h_sem = self.dropout(h_sem)
        
        # Fuse
        h_fused = torch.cat([h_syn, h_sem], dim=-1)
        h_out = self.fusion(h_fused)
        h_out = F.relu(h_out)
        
        return h_out


# ==============================================================================
# PART 5: HIERARCHICAL MSR MODULE (From V2)
# ==============================================================================

class HierarchicalMSR(nn.Module):
    """
    Enhanced MSR with conflict detection and overall sentiment prediction.
    """
    def __init__(self, num_aspects=7, num_classes=3, hidden_dim=128):
        super().__init__()
        
        self.aspect_attention = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.conflict_detector = nn.Sequential(
            nn.Linear(num_aspects * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.sentiment_predictor = nn.Sequential(
            nn.Linear(num_aspects * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, aspect_logits_list):
        aspect_probs = [F.softmax(logits, dim=-1) for logits in aspect_logits_list]
        aspect_probs = torch.stack(aspect_probs, dim=1)
        
        B, num_aspects, num_classes = aspect_probs.shape
        
        # Aspect importance
        aspect_importance = []
        for i in range(num_aspects):
            importance = self.aspect_attention(aspect_probs[:, i, :])
            aspect_importance.append(importance)
        
        aspect_importance = torch.cat(aspect_importance, dim=-1)
        aspect_importance = F.softmax(aspect_importance, dim=-1)
        
        # Flatten for prediction
        flat_probs = aspect_probs.view(B, -1)
        
        # Conflict detection
        conflict_logits = self.conflict_detector(flat_probs)
        conflict_prob = F.softmax(conflict_logits, dim=-1)[:, 1]
        
        # Overall sentiment
        overall_logits = self.sentiment_predictor(flat_probs)
        
        return {
            'overall_sentiment': overall_logits,
            'conflict_score': conflict_prob,
            'aspect_importance': aspect_importance
        }


# ==============================================================================
# PART 6: EAGLE V3 MODEL
# ==============================================================================

class EAGLE_V3(nn.Module):
    """
    EAGLE V3: Pure Adaptive Loss Implementation
    
    Features:
    1. Base: RoBERTa + DualChannelGCN + HierarchicalMSR (from V2)
    2. Imbalance Handling: AdaptiveFocalLoss ONLY (no synthetic data)
    3. Explainability: Feature Routing + Uncertainty Heads
    """
    def __init__(
        self,
        num_aspects=7,
        num_classes=3,
        gcn_dim=300,
        gcn_layers=2,
        aspect_names=None,
        use_uncertainty=True,
        use_feature_routing=True
    ):
        super().__init__()
        
        self.num_aspects = num_aspects
        self.num_classes = num_classes
        self.aspect_names = aspect_names or [
            'stayingpower', 'texture', 'smell', 'price',
            'colour', 'shipping', 'packing'
        ]
        self.use_uncertainty = use_uncertainty
        self.use_feature_routing = use_feature_routing
        
        # 1. Base encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base', attn_implementation="eager")
        hidden_size = self.roberta.config.hidden_size
        
        # 2. Dual-channel GCN
        self.dual_gcn = DualChannelGCN(
            hidden_size=hidden_size,
            gcn_dim=gcn_dim,
            num_layers=gcn_layers,
            dropout=0.3
        )
        
        # 3. Feature router
        if use_feature_routing:
            self.feature_router = AspectFeatureRouter(
                hidden_dim=gcn_dim,
                num_aspects=num_aspects
            )
        
        # 4. Aspect-specific classifiers with uncertainty
        if use_uncertainty:
            self.classifiers = nn.ModuleList([
                UncertaintyHead(gcn_dim, num_classes)
                for _ in range(num_aspects)
            ])
        else:
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(gcn_dim, num_classes)
                )
                for _ in range(num_aspects)
            ])
        
        # 5. MSR module
        self.msr_module = HierarchicalMSR(
            num_aspects=num_aspects,
            num_classes=num_classes,
            hidden_dim=128
        )
        
        # 6. Adaptive Focal Loss (Dynamic)
        # We set initial alphas to 1.0, they will adapt after 1st epoch
        # High gammas for generally harder-to-classify aspects
        dataset_initial_gammas = {
            'price': 4.0, 'packing': 3.5,
            'stayingpower': 2.5, 'texture': 2.5, 'smell': 3.0,
            'colour': 2.0, 'shipping': 2.5
        }
        
        self.aspect_losses = nn.ModuleDict({
            name: AdaptiveFocalLoss(
                num_classes=num_classes,
                gamma=dataset_initial_gammas.get(name, 2.5),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            for name in self.aspect_names
        })
        
    def forward(
        self,
        input_ids,
        attention_mask,
        syntactic_adj,
        aspect_masks,
        positions,
        return_routing_weights=False
    ):
        # 1. BERT encoding
        bert_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        h = bert_output.last_hidden_state
        
        # 2. Process each aspect
        aspect_logits_list = []
        uncertainties_list = []
        routing_weights_list = []
        
        for i in range(self.num_aspects):
            # Get aspect-specific features from GCN
            h_gcn = self.dual_gcn(
                h=h,
                syntactic_adj=syntactic_adj,
                aspect_mask=aspect_masks[:, i],
                positions=positions
            )
            
            # Pool features
            aspect_mask = aspect_masks[:, i].unsqueeze(-1)
            
            # Transformer features (from BERT)
            h_transformer = (h * aspect_mask).sum(dim=1) / (aspect_mask.sum(dim=1) + 1e-8)
            
            # GCN features
            h_gcn_pooled = (h_gcn * aspect_mask).sum(dim=1) / (aspect_mask.sum(dim=1) + 1e-8)
            
            # Feature routing
            if self.use_feature_routing:
                # Ensure dimensions match
                if h_transformer.size(-1) != h_gcn_pooled.size(-1):
                    if not hasattr(self, 'transformer_projection'):
                        self.transformer_projection = nn.Linear(
                            h_transformer.size(-1),
                            h_gcn_pooled.size(-1)
                        ).to(h_transformer.device)
                    h_transformer = self.transformer_projection(h_transformer)
                
                h_aspect, routing_weights = self.feature_router(
                    h_transformer, h_gcn_pooled, i
                )
                routing_weights_list.append(routing_weights)
            else:
                h_aspect = h_gcn_pooled
            
            # Classification with uncertainty
            if self.use_uncertainty:
                logits, uncertainty = self.classifiers[i](h_aspect)
                
                # Calibrate for neutral class
                logits = self.classifiers[i].calibrate_neutral(
                    logits, uncertainty, neutral_idx=1, threshold=0.5
                )
                
                uncertainties_list.append(uncertainty)
            else:
                logits = self.classifiers[i](h_aspect)
            
            aspect_logits_list.append(logits)
        
        # 3. MSR module
        msr_output = self.msr_module(aspect_logits_list)
        
        # Prepare return values
        result = {
            'aspect_logits': aspect_logits_list,
            'msr_output': msr_output
        }
        
        if self.use_uncertainty:
            result['uncertainties'] = uncertainties_list
        
        if return_routing_weights and self.use_feature_routing:
            result['routing_weights'] = torch.stack(routing_weights_list, dim=1)
        
        return result
    
    def compute_loss(
        self,
        forward_output,
        labels,
        overall_labels=None,
        loss_weights={'aspect': 1.0, 'msr_sentiment': 0.3, 'msr_conflict': 0.2}
    ):
        device = forward_output['aspect_logits'][0].device
        aspect_logits_list = forward_output['aspect_logits']
        msr_output = forward_output['msr_output']
        
        # 1. Aspect-level losses
        aspect_loss = 0
        valid_aspects = 0
        aspect_loss_breakdown = {}
        
        for i, aspect_name in enumerate(self.aspect_names):
            aspect_labels = labels[:, i]
            
            if (aspect_labels != -100).sum() == 0:
                continue
            
            # Get adaptive focal loss
            loss_fn = self.aspect_losses[aspect_name]
            
            # Update stats (for dynamic weighting)
            # We must do this inside compute_loss to capture batch stats
            with torch.no_grad():
                preds = torch.argmax(aspect_logits_list[i], dim=-1)
                loss_fn.update_stats(preds, aspect_labels)
            
            # Compute loss
            loss = loss_fn(aspect_logits_list[i], aspect_labels)
            
            aspect_loss += loss
            valid_aspects += 1
            aspect_loss_breakdown[aspect_name] = loss.item()
        
        aspect_loss = aspect_loss / max(valid_aspects, 1)
        
        # 2. MSR losses
        msr_sentiment_loss = torch.tensor(0.0, device=device)
        msr_conflict_loss = torch.tensor(0.0, device=device)
        
        if overall_labels is not None:
            if 'sentiment' in overall_labels:
                msr_sentiment_loss = F.cross_entropy(
                    msr_output['overall_sentiment'],
                    overall_labels['sentiment']
                )
            
            if 'conflict' in overall_labels:
                msr_conflict_loss = F.binary_cross_entropy(
                    msr_output['conflict_score'],
                    overall_labels['conflict'].float()
                )
        
        # 3. Total loss
        total_loss = (
            loss_weights['aspect'] * aspect_loss +
            loss_weights['msr_sentiment'] * msr_sentiment_loss +
            loss_weights['msr_conflict'] * msr_conflict_loss
        )
        
        loss_dict = {
            'aspect_loss': aspect_loss.item(),
            'msr_sentiment_loss': msr_sentiment_loss.item(),
            'msr_conflict_loss': msr_conflict_loss.item(),
            'total_loss': total_loss.item(),
            **aspect_loss_breakdown
        }
        
        return total_loss, loss_dict
    
    def update_focal_loss_weights(self):
        """
        Update all adaptive focal loss weights.
        Call this at the END of each epoch.
        """
        print("\n[EAGLE V3] Updating Adaptive Focal Loss weights based on epoch performance:")
        for aspect_name, loss_fn in self.aspect_losses.items():
            print(f"  Aspect: {aspect_name}")
            loss_fn.update_weights()
            loss_fn.reset_stats() 
