"""
Multi-Aspect Sentiment Analysis Model with RoBERTa and Dependency GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig


class AspectAwareRoBERTa(nn.Module):
    """
    RoBERTa with aspect-specific attention mechanism
    Improvement over MAFESA's LDA + GloVe approach
    """
    def __init__(self, roberta_model='roberta-base', num_aspects=7,
                 num_classes=3, hidden_dim=768, dropout=0.1,
                 use_aspect_attention=True, use_shared_classifier=False):
        super(AspectAwareRoBERTa, self).__init__()
        
        self.use_aspect_attention  = use_aspect_attention
        self.use_shared_classifier = use_shared_classifier

        # Load pre-trained RoBERTa
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        
        # Aspect embeddings (learnable) — always needed for GCN gating & loss routing
        self.aspect_embeddings = nn.Embedding(num_aspects, hidden_dim)
        nn.init.xavier_uniform_(self.aspect_embeddings.weight)
        
        if use_aspect_attention:
            # Multi-head attention for aspect-text interaction
            self.aspect_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Ablation 5: shared vs aspect-specific classifiers
        if use_shared_classifier:
            # Single shared classification head for all aspects
            self.shared_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            # Aspect-specific classifiers (separate head for each aspect)
            self.aspect_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_classes)
                ) for _ in range(num_aspects)
            ])
        
        self.num_aspects = num_aspects
        self.hidden_dim  = hidden_dim
        
    def forward(self, input_ids, attention_mask, aspect_id, return_token_embeddings=False):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            aspect_id: (batch_size,) - which aspect to analyze
            return_token_embeddings: bool, whether to return token embeddings for GCN
            
        Returns:
            predictions: (batch_size, num_classes) sentiment logits
            attention_weights: (batch_size, seq_len) aspect-specific attention
            aspect_representation: (batch_size, hidden_dim) for explainability
            token_embeddings: (batch_size, seq_len, hidden_dim) if return_token_embeddings=True
        """
        # Get RoBERTa contextual embeddings
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = roberta_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        if self.use_aspect_attention:
            # Aspect-guided MHA attention (standard path)
            aspect_query = self.aspect_embeddings(aspect_id)   # (batch_size, hidden_dim)
            aspect_query = aspect_query.unsqueeze(1)            # (batch_size, 1, hidden_dim)

            attended_output, attention_weights = self.aspect_attention(
                query=aspect_query,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=~attention_mask.bool()
            )
            aspect_representation = attended_output.squeeze(1)  # (batch_size, hidden_dim)
        else:
            # Ablation 2: CLS pooling (no aspect awareness)
            aspect_representation = hidden_states[:, 0, :]      # (batch_size, hidden_dim)
            # Fake uniform attention weights for interface compatibility
            seq_len = hidden_states.size(1)
            attention_weights = torch.ones(
                hidden_states.size(0), 1, seq_len,
                device=hidden_states.device
            ) / seq_len

        aspect_representation = self.layer_norm(aspect_representation)
        aspect_representation = self.dropout(aspect_representation)
        
        # Classification
        batch_size = input_ids.size(0)
        predictions = []
        
        for i in range(batch_size):
            if self.use_shared_classifier:
                pred = self.shared_classifier(aspect_representation[i])
            else:
                asp_id = aspect_id[i].item()
                pred   = self.aspect_classifiers[asp_id](aspect_representation[i])
            predictions.append(pred)
        
        predictions = torch.stack(predictions)  # (batch_size, num_classes)
        
        if return_token_embeddings:
            return predictions, attention_weights.squeeze(1), aspect_representation, hidden_states
        
        return predictions, attention_weights.squeeze(1), aspect_representation


class AspectOrientedDepGCN(nn.Module):
    """
    Dependency GCN with aspect-oriented gating
    Captures syntactic relationships for mixed sentiment resolution
    """
    def __init__(self, hidden_dim=768, num_layers=2, dropout=0.1):
        super(AspectOrientedDepGCN, self).__init__()
        
        # GCN layers - simple message passing
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Aspect gating mechanism
        self.aspect_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        
    def forward(self, token_embeddings, edge_index, aspect_embedding):
        """
        Args:
            token_embeddings: (num_tokens, hidden_dim)
            edge_index: (2, num_edges) dependency edges
            aspect_embedding: (hidden_dim,) aspect query
            
        Returns:
            node_features: (num_tokens, hidden_dim) enhanced representations
        """
        x = token_embeddings
        num_nodes = x.size(0)
        
        for i in range(self.num_layers):
            # Message passing
            if edge_index.size(1) > 0:
                src, dst = edge_index[0], edge_index[1]
                messages = x[src]  # (num_edges, hidden_dim)
                aggregated = torch.zeros_like(x)
                aggregated = aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
                
                # Apply linear transformation
                x_gcn = self.gcn_layers[i](aggregated)
                x_gcn = F.relu(x_gcn)
            else:
                # No edges, just apply transformation
                x_gcn = self.gcn_layers[i](x)
                x_gcn = F.relu(x_gcn)
            
            # Aspect-oriented gating
            aspect_expanded = aspect_embedding.unsqueeze(0).expand(num_nodes, -1)
            gate_input = torch.cat([x_gcn, aspect_expanded], dim=-1)
            gate = self.aspect_gate(gate_input)
            
            # Gated residual connection
            x = gate * x_gcn + (1 - gate) * x
            x = self.layer_norms[i](x)
            x = self.dropout(x)
        
        return x


class MultiAspectSentimentModel(nn.Module):
    """
    Complete model combining:
    - RoBERTa encoder
    - Aspect-aware attention
    - Dependency GCN (optional)
    - Multi-aspect classification
    """
    def __init__(self, config):
        super(MultiAspectSentimentModel, self).__init__()
        
        self.config = config
        model_config = config['model']
        
        self.use_gcn = model_config.get('use_dependency_gcn', True)
        
        # Ablation flags
        use_aspect_attention  = model_config.get('use_aspect_attention', True)
        use_shared_classifier = model_config.get('use_shared_classifier', False)

        # Main components
        self.aspect_aware_roberta = AspectAwareRoBERTa(
            roberta_model=model_config['roberta_model'],
            num_aspects=model_config['num_aspects'],
            num_classes=model_config['num_classes'],
            hidden_dim=model_config['hidden_dim'],
            dropout=model_config['dropout'],
            use_aspect_attention=use_aspect_attention,
            use_shared_classifier=use_shared_classifier,
        )
        
        if self.use_gcn:
            self.dep_gcn = AspectOrientedDepGCN(
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['gcn_layers'],
                dropout=model_config['dropout']
            )
            
            # Final classifier with both attention and GCN features
            self.final_classifier = nn.Sequential(
                nn.Linear(model_config['hidden_dim'] * 2, model_config['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(model_config['dropout']),
                nn.Linear(model_config['hidden_dim'], model_config['num_classes'])
            )
        
    def forward(self, input_ids, attention_mask, aspect_id, edge_index=None, 
                token_mask=None, return_attention=False):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            aspect_id: (batch_size,)
            edge_index: List of (2, num_edges) tensors for each sample
            token_mask: (batch_size, seq_len) mask for valid tokens (excluding padding)
            return_attention: bool, whether to return attention weights
            
        Returns:
            predictions: (batch_size, num_classes) sentiment logits
            Optional: attention_weights, aspect_repr, gcn_output for explainability
        """
        # Determine if we need token embeddings for GCN upfront
        need_token_embeddings = self.use_gcn and edge_index is not None
        
        # Single forward pass with appropriate parameters based on GCN requirement
        if need_token_embeddings:
            attn_predictions, attention_weights, aspect_repr, token_embeddings = self.aspect_aware_roberta(
                input_ids, attention_mask, aspect_id, return_token_embeddings=True
            )
        else:
            attn_predictions, attention_weights, aspect_repr = self.aspect_aware_roberta(
                input_ids, attention_mask, aspect_id
            )
        
        # If dependency parsing is not used, return attention-based predictions
        if not need_token_embeddings:
            if return_attention:
                return attn_predictions, attention_weights, aspect_repr, None
            return attn_predictions
        
        # Apply Dependency GCN using the token embeddings from above
        # Apply GCN to each sample in batch
        gcn_outputs = []
        for i in range(input_ids.size(0)):
            if edge_index is not None and i < len(edge_index) and edge_index[i].size(1) > 0:
                # Get aspect embedding for this sample
                aspect_emb = self.aspect_aware_roberta.aspect_embeddings(aspect_id[i])
                
                # Apply GCN
                gcn_out = self.dep_gcn(
                    token_embeddings[i],
                    edge_index[i],
                    aspect_emb
                )
                
                # Pool GCN output (mean pooling over tokens)
                mask_expanded = attention_mask[i].unsqueeze(-1).float()
                # Ensure division by non-zero sum, handle cases where mask_expanded.sum(0) might be 0
                sum_mask = mask_expanded.sum(0)
                gcn_pooled = (gcn_out * mask_expanded).sum(0) / (sum_mask + 1e-9) # Add epsilon for stability
                gcn_outputs.append(gcn_pooled)
            else:
                # No dependency tree, use zero tensor or aspect repr
                gcn_outputs.append(torch.zeros_like(aspect_repr[i]))
        
        gcn_output = torch.stack(gcn_outputs)  # (batch_size, hidden_dim)
        
        # Combine attention and GCN features
        combined = torch.cat([aspect_repr, gcn_output], dim=-1)  # (batch_size, hidden_dim*2)
        
        # Final prediction with combined features
        final_predictions = self.final_classifier(combined)
        
        if return_attention:
            return final_predictions, attention_weights, aspect_repr, gcn_output
        
        return final_predictions
    
    def get_num_parameters(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Factory function to create model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: MultiAspectSentimentModel
    """
    model = MultiAspectSentimentModel(config)
    
    num_params = model.get_num_parameters()
    print(f"Created model with {num_params:,} trainable parameters")
    
    return model


if __name__ == "__main__":
    # Test the model
    import yaml
    
    print("Testing model architecture...")
    
    # Load config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    aspect_id = torch.tensor([0, 1, 2, 3])
    
    # Without GCN
    predictions = model(input_ids, attention_mask, aspect_id)
    print(f"Predictions shape (no GCN): {predictions.shape}")
    
    # With GCN (create dummy edge indices)
    edge_index = [
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[], []], dtype=torch.long)
    ]
    
    predictions, attn, aspect_repr, gcn_out = model(
        input_ids, attention_mask, aspect_id, edge_index, 
        return_attention=True
    )
    
    print(f"Predictions shape (with GCN): {predictions.shape}")
    print(f"Attention weights shape: {attn.shape}")
    print(f"Aspect representation shape: {aspect_repr.shape}")
    print(f"GCN output shape: {gcn_out.shape}")
    
    print("\nModel test passed!")
