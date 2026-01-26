# EAGLE V2 - Enhanced Implementation with Critical Improvements
# Addresses: Price detection, Neutral class handling, Packing negative recall
# Research Innovation: Aspect-specific feature routing + Uncertainty-aware predictions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import RobertaModel
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# PART 1: ENHANCED FOCAL LOSS WITH LABEL SMOOTHING
# ==============================================================================

class EnhancedFocalLoss(nn.Module):
    """
    Focal Loss with dynamic class weighting and label smoothing for neutral detection.
    
    NEW FEATURES vs original:
    1. Label smoothing (ε=0.1) - prevents over-confident predictions
    2. Aspect-specific gamma values - higher for imbalanced classes
    3. Extreme class weighting for minority classes (price negative: 50x)
    """
    def __init__(
        self, 
        num_classes=3, 
        gamma=3.0, 
        class_weights=None, 
        label_smoothing=0.1,
        device='cuda'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.device = device
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        else:
            self.class_weights = torch.ones(num_classes, device=device)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices (-100 for ignore)
        Returns:
            scalar loss
        """
        # Filter ignore index
        valid_mask = (targets != -100)
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            # Smooth targets
            confidence = 1.0 - self.label_smoothing
            smooth_label = torch.zeros(targets.size(0), self.num_classes, device=self.device)
            smooth_label.fill_(self.label_smoothing / (self.num_classes - 1))
            smooth_label.scatter_(1, targets.unsqueeze(1), confidence)
            
            # Cross-entropy with smooth labels
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_label * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss modulation
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply class weights
        weights = self.class_weights[targets]
        weighted_loss = weights * focal_loss
        
        return weighted_loss.mean()


# ==============================================================================
# PART 2: UNCERTAINTY-AWARE PREDICTION HEAD
# ==============================================================================

class UncertaintyHead(nn.Module):
    """
    Predicts uncertainty alongside logits.
    High uncertainty → push toward neutral class
    
    RESEARCH VALUE:
    - Addresses neutral class under-detection
    - Interpretable: can visualize which predictions are uncertain
    - Expected improvement: Neutral F1 +0.10-0.15
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
            nn.Sigmoid()  # Uncertainty ∈ [0, 1]
        )
    
    def forward(self, features):
        """
        Args:
            features: [B, D] hidden features
        Returns:
            logits: [B, num_classes]
            uncertainty: [B] uncertainty scores [0, 1]
        """
        logits = self.classifier(features)
        uncertainty = self.uncertainty_net(features).squeeze(-1)
        
        return logits, uncertainty
    
    def calibrate_neutral(self, logits, uncertainty, neutral_idx=1, threshold=0.5):
        """
        Boost neutral class probability for uncertain predictions.
        
        Args:
            logits: [B, num_classes]
            uncertainty: [B]
            neutral_idx: index of neutral class (default: 1)
            threshold: uncertainty threshold for calibration
        Returns:
            calibrated_logits: [B, num_classes]
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
# PART 3: ASPECT-SPECIFIC FEATURE ROUTER
# ==============================================================================

class AspectFeatureRouter(nn.Module):
    """
    Learn aspect-specific routing between transformer and GNN features.
    
    KEY INNOVATION:
    - Some aspects benefit more from syntactic structure (GNN)
    - Others benefit from semantic context (transformer)
    - Let the model learn the optimal mix per aspect
    
    EXPECTED BEHAVIOR:
    - Price: Higher transformer weight (semantic: "expensive", "cheap")
    - Texture: Higher GNN weight (syntax: "feels smooth")
    - Shipping: Balanced (both matter)
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
        """
        Args:
            transformer_features: [B, D]
            gnn_features: [B, D]
            aspect_idx: int
        Returns:
            fused_features: [B, D]
            routing_weights: [B, 2] (for interpretability)
        """
        # Concatenate features
        combined = torch.cat([transformer_features, gnn_features], dim=-1)
        
        # Get routing weights
        weights = self.routers[aspect_idx](combined)  # [B, 2]
        
        # Weighted fusion
        fused = weights[:, 0:1] * transformer_features + weights[:, 1:2] * gnn_features
        
        return fused, weights


# ==============================================================================
# PART 4: POSITION-AWARE GRAPH ATTENTION (Enhanced)
# ==============================================================================

class PositionAwareGraphAttention(nn.Module):
    """
    Enhanced GAT with position encoding and edge dropout.
    """
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
        """
        Args:
            h: [B, N, D] node features
            adj: [B, N, N] adjacency matrix
            positions: [B, N] position indices
        Returns:
            h_prime: [B, N, D']
            attention: [B, N, N]
        """
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
    """GAT with residual connection and layer norm."""
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


# ==============================================================================
# PART 5: DUAL-CHANNEL GCN (Enhanced)
# ==============================================================================

class DualChannelGCN(nn.Module):
    """
    Two-stream GCN: Syntactic + Semantic
    Enhanced with better semantic graph construction
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
        """
        Build aspect-aware semantic graph.
        
        Args:
            h: [B, N, D]
            aspect_mask: [B, N]
        Returns:
            semantic_adj: [B, N, N]
        """
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
        """
        Args:
            h: [B, N, D] BERT features
            syntactic_adj: [B, N, N]
            aspect_mask: [B, N]
            positions: [B, N]
        Returns:
            h_out: [B, N, D']
        """
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
# PART 6: HIERARCHICAL MSR MODULE (Enhanced)
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
        """
        Args:
            aspect_logits_list: list of [B, num_classes]
        Returns:
            dict with MSR outputs
        """
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
# PART 7: COMPLETE EAGLE V2 MODEL
# ==============================================================================

class EAGLE_V2(nn.Module):
    """
    Enhanced EAGLE with critical improvements for price, neutral, and packing detection.
    
    NEW FEATURES:
    1. Aspect-specific feature routing (transformer vs GNN)
    2. Uncertainty-aware prediction heads with neutral calibration
    3. Enhanced focal loss with label smoothing
    4. Extreme class weighting for minority classes
    5. Progressive training support
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
        
        # 3. Feature router (NEW)
        if use_feature_routing:
            self.feature_router = AspectFeatureRouter(
                hidden_dim=gcn_dim,
                num_aspects=num_aspects
            )
        
        # 4. Aspect-specific classifiers with uncertainty (NEW)
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
        
        # 6. Enhanced focal loss with aspect-specific configurations
        self.aspect_loss_configs = self._get_aspect_loss_configs()
        self.aspect_losses = nn.ModuleDict({
            name: EnhancedFocalLoss(
                num_classes=num_classes,
                gamma=config['gamma'],
                class_weights=config['weights'],
                label_smoothing=config['label_smoothing'],
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            for name, config in self.aspect_loss_configs.items()
        })
    
    def _get_aspect_loss_configs(self):
        """
        Aspect-specific loss configurations based on data analysis.
        
        CRITICAL TUNING based on metrics:
        - Price: Extreme weights (50:20:1) + high gamma (5.0)
        - Packing: High weights for negative (10:5:1) + gamma (4.0)
        - StayingPower, Texture: Moderate neutral boost
        - Smell: High gamma for neutral
        """
        return {
            'price': {
                'gamma': 5.0,
                'weights': [50.0, 20.0, 1.0],  # [neg, neu, pos]
                'label_smoothing': 0.15  # Higher smoothing for extreme imbalance
            },
            'packing': {
                'gamma': 4.0,
                'weights': [10.0, 5.0, 1.0],
                'label_smoothing': 0.1
            },
            'stayingpower': {
                'gamma': 2.5,
                'weights': [3.0, 5.0, 1.0],  # Boost neutral
                'label_smoothing': 0.1
            },
            'texture': {
                'gamma': 2.5,
                'weights': [2.5, 4.0, 1.0],  # Boost neutral
                'label_smoothing': 0.1
            },
            'smell': {
                'gamma': 3.0,
                'weights': [2.0, 8.0, 1.0],  # High neutral boost
                'label_smoothing': 0.1
            },
            'colour': {
                'gamma': 2.0,
                'weights': [2.0, 3.0, 1.0],
                'label_smoothing': 0.1
            },
            'shipping': {
                'gamma': 2.5,
                'weights': [1.5, 4.0, 1.0],
                'label_smoothing': 0.1
            }
        }
    
    def forward(
        self,
        input_ids,
        attention_mask,
        syntactic_adj,
        aspect_masks,
        positions,
        return_routing_weights=False
    ):
        """
        Enhanced forward pass with optional feature routing.
        
        Args:
            input_ids: [B, N]
            attention_mask: [B, N]
            syntactic_adj: [B, N, N]
            aspect_masks: [B, num_aspects, N]
            positions: [B, N]
            return_routing_weights: bool, for interpretability
        
        Returns:
            aspect_logits_list: list of [B, num_classes]
            uncertainties: list of [B] (if use_uncertainty)
            msr_output: dict
            routing_weights: [B, num_aspects, 2] (if return_routing_weights)
        """
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
            
            # Feature routing (NEW)
            if self.use_feature_routing:
                # Ensure dimensions match
                if h_transformer.size(-1) != h_gcn_pooled.size(-1):
                    # Project transformer features to gcn_dim
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
            
            # Classification with uncertainty (NEW)
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
        """
        Enhanced loss computation with aspect-specific focal loss.
        
        Args:
            forward_output: dict from forward()
            labels: [B, num_aspects]
            overall_labels: dict (optional)
            loss_weights: dict
        
        Returns:
            total_loss: scalar
            loss_dict: dict with breakdown
        """
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
            
            # Get enhanced focal loss
            loss_fn = self.aspect_losses[aspect_name]
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


# ==============================================================================
# PART 8: USAGE EXAMPLE
# ==============================================================================

def example_usage():
    """Example of EAGLE V2 usage."""
    model = EAGLE_V2(
        num_aspects=7,
        num_classes=3,
        gcn_dim=300,
        gcn_layers=2,
        use_uncertainty=True,
        use_feature_routing=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Dummy batch
    B, N, num_aspects = 2, 256, 7
    input_ids = torch.randint(0, 50000, (B, N)).to(device)
    attention_mask = torch.ones(B, N).to(device)
    syntactic_adj = (torch.rand(B, N, N) > 0.9).float().to(device)
    aspect_masks = torch.zeros(B, num_aspects, N).to(device)
    aspect_masks[:, :, :10] = 1
    positions = torch.arange(N).unsqueeze(0).expand(B, -1).to(device)
    labels = torch.randint(0, 3, (B, num_aspects)).to(device)
    
    # Forward
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        syntactic_adj=syntactic_adj,
        aspect_masks=aspect_masks,
        positions=positions,
        return_routing_weights=True
    )
    
    # Compute loss
    loss, loss_dict = model.compute_loss(output, labels)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Loss Breakdown: {loss_dict}")
    
    if 'uncertainties' in output:
        print(f"\\nUncertainties shape: {[u.shape for u in output['uncertainties']]}")
    
    if 'routing_weights' in output:
        print(f"Routing weights shape: {output['routing_weights'].shape}")
        print(f"Routing weights (aspect 0): {output['routing_weights'][0, 0]}")


if __name__ == '__main__':
    example_usage()
