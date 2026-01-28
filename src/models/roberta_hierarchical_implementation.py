# RoBERTa Hierarchical Model
# Simplified Architecture: Shared RoBERTa + Multi-Head Linear Classifiers
# Keeps: Hierarchical MSR + Adaptive Focal Loss
# Removes: GCN/Graph components

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from typing import List, Dict, Optional

# Import reusable components from eagle_v3
from eagle_v3_implementation import AdaptiveFocalLoss, HierarchicalMSR


class RoBERTaHierarchicalModel(nn.Module):
    """
    Simplified Multi-Aspect Sentiment Model
    
    Architecture:
    - Shared RoBERTa Encoder
    - Multi-Head Linear Classifiers (one per aspect)
    - Hierarchical Neural MSR for conflict resolution
    - Adaptive Focal Loss for class imbalance
    
    Key Differences from EAGLE:
    - NO Graph Convolutional Networks
    - NO Dependency parsing
    - Direct linear classification on pooled RoBERTa features
    """
    
    def __init__(
        self,
        num_aspects=7,
        num_classes=3,
        aspect_names=None,
        hidden_dropout=0.3,
        output_attentions=False
    ):
        super().__init__()
        
        self.num_aspects = num_aspects
        self.num_classes = num_classes
        self.aspect_names = aspect_names or [
            'stayingpower', 'texture', 'smell', 'price',
            'colour', 'shipping', 'packing'
        ]
        self.output_attentions = output_attentions
        
        # 1. Shared RoBERTa Encoder
        self.roberta = RobertaModel.from_pretrained(
            'roberta-base',
            attn_implementation="eager",
            output_attentions=output_attentions
        )
        hidden_size = self.roberta.config.hidden_size  # 768
        
        # 2. Multi-Head Linear Classifiers (one per aspect)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
            for _ in range(num_aspects)
        ])
        
        # 3. Hierarchical MSR Module
        self.msr_module = HierarchicalMSR(
            num_aspects=num_aspects,
            num_classes=num_classes,
            hidden_dim=128
        )
        
        # 4. Adaptive Focal Loss (per aspect)
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
        return_attentions=False
    ):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_attentions: whether to return RoBERTa self-attention weights
        
        Returns:
            dict with:
                - aspect_logits: list of [batch_size, num_classes] per aspect
                - msr_output: dict with overall_sentiment, conflict_score, aspect_importance
                - attentions (optional): RoBERTa self-attention weights
        """
        # 1. RoBERTa encoding
        bert_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions or self.output_attentions
        )
        
        # Get [CLS] token representation (pooled output)
        # Shape: [batch_size, hidden_size]
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        
        # 2. Aspect-specific classification
        aspect_logits_list = []
        for i in range(self.num_aspects):
            logits = self.classifiers[i](pooled_output)
            aspect_logits_list.append(logits)
        
        # 3. MSR module (conflict resolution + overall sentiment)
        msr_output = self.msr_module(aspect_logits_list)
        
        # Prepare output
        result = {
            'aspect_logits': aspect_logits_list,
            'msr_output': msr_output
        }
        
        # Add attention weights if requested
        if return_attentions or self.output_attentions:
            result['attentions'] = bert_output.attentions
        
        return result
    
    def compute_loss(
        self,
        forward_output,
        labels,
        overall_labels=None,
        loss_weights={'aspect': 1.0, 'msr_sentiment': 0.3, 'msr_conflict': 0.2}
    ):
        """
        Compute total loss
        
        Args:
            forward_output: output from forward()
            labels: [batch_size, num_aspects] with class indices
            overall_labels: optional dict with 'sentiment' and 'conflict' keys
            loss_weights: weights for different loss components
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dict with breakdown of losses
        """
        device = forward_output['aspect_logits'][0].device
        aspect_logits_list = forward_output['aspect_logits']
        msr_output = forward_output['msr_output']
        
        # 1. Aspect-level losses (Adaptive Focal Loss)
        aspect_loss = 0
        valid_aspects = 0
        aspect_loss_breakdown = {}
        
        for i, aspect_name in enumerate(self.aspect_names):
            aspect_labels = labels[:, i]
            
            # Skip if no valid labels
            if (aspect_labels != -100).sum() == 0:
                continue
            
            # Get adaptive focal loss
            loss_fn = self.aspect_losses[aspect_name]
            
            # Update stats (for dynamic weighting)
            with torch.no_grad():
                preds = torch.argmax(aspect_logits_list[i], dim=-1)
                loss_fn.update_stats(preds, aspect_labels)
            
            # Compute loss
            loss = loss_fn(aspect_logits_list[i], aspect_labels)
            
            aspect_loss += loss
            valid_aspects += 1
            aspect_loss_breakdown[aspect_name] = loss.item()
        
        aspect_loss = aspect_loss / max(valid_aspects, 1)
        
        # 2. MSR losses (if overall labels provided)
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
        print("\n[RoBERTa Hierarchical] Updating Adaptive Focal Loss weights:")
        for aspect_name, loss_fn in self.aspect_losses.items():
            print(f"  Aspect: {aspect_name}")
            loss_fn.update_weights()
            loss_fn.reset_stats()
    
    def get_aspect_importance(self, forward_output):
        """
        Extract aspect importance scores from MSR output.
        
        Returns:
            aspect_importance: [batch_size, num_aspects]
        """
        return forward_output['msr_output']['aspect_importance']
    
    def get_conflict_scores(self, forward_output):
        """
        Extract conflict detection scores.
        
        Returns:
            conflict_score: [batch_size]
        """
        return forward_output['msr_output']['conflict_score']
