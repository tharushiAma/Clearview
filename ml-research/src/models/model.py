"""
Multi-Aspect Sentiment Analysis Model with RoBERTa and Dependency GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig


class AspectAwareRoBERTa(nn.Module):
    """
    RoBERTa with aspect-specific attention. Improvement over generic [CLS] pooling.
    Each aspect has a learnable embedding used as the query vector, so attention
    focuses on tokens relevant to *that* aspect (e.g. 'smells great' for smell).
    """
    def __init__(self, roberta_model='roberta-base', num_aspects=7,
                 num_classes=3, hidden_dim=768, dropout=0.1,
                 use_aspect_attention=True, use_shared_classifier=False):
        super(AspectAwareRoBERTa, self).__init__()
        
        self.use_aspect_attention  = use_aspect_attention # Ablation 2 flag
        self.use_shared_classifier = use_shared_classifier # Ablation 5 flag

        # Load pre-trained RoBERTa weights
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        
        # Learnable aspect query vectors (one per aspect).
        # Used both for MHA attention queries and for GCN aspect-gating.
        # Xavier uniform init keeps initial gradient magnitudes stable.
        self.aspect_embeddings = nn.Embedding(num_aspects, hidden_dim)
        nn.init.xavier_uniform_(self.aspect_embeddings.weight)
        
        if use_aspect_attention:
            # 8 heads over 768-dim → each head attends over 96-dim subspace.
            # batch_first=True keeps (batch, seq, features) convention throughout.
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
            # Each head specialises for its aspect's class distribution (e.g.
            # price has extreme imbalance; smell/texture are more balanced).
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
        # Run through pre-trained RoBERTa - Get RoBERTa contextual embeddings
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = roberta_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        if self.use_aspect_attention:
            # Aspect-guided MHA: the aspect embedding acts as the query, the full token
            # sequence is both key and value. This forces attention to focus on tokens
            # most relevant to THIS aspect (e.g., 'smells great' for the smell aspect).
            aspect_query = self.aspect_embeddings(aspect_id)   # (batch_size, hidden_dim)
            aspect_query = aspect_query.unsqueeze(1)            # (batch_size, 1, hidden_dim) — single query token

            attended_output, attention_weights = self.aspect_attention(
                query=aspect_query,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=~attention_mask.bool()  # True marks positions to IGNORE (padding)
            )
            aspect_representation = attended_output.squeeze(1)  # (batch_size, hidden_dim) — remove query token dim
        else:
            # Ablation 2: CLS pooling (no aspect awareness)
            # Falls back to the same strategy used by baseline models (PlainRoBERTa, BERT, DistilBERT)
            aspect_representation = hidden_states[:, 0, :]      # (batch_size, hidden_dim)
            # Fake uniform attention weights for interface compatibility
            seq_len = hidden_states.size(1)
            attention_weights = torch.ones(
                hidden_states.size(0), 1, seq_len,
                device=hidden_states.device
            ) / seq_len

        aspect_representation = self.layer_norm(aspect_representation)
        aspect_representation = self.dropout(aspect_representation)
        
        # Route each sample through its aspect-specific classifier using the sample's aspect_id.
        # Loop is necessary because different samples in the batch may have different aspect_ids.
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
        
        self.gcn_layers  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])   # GCN layers - simple message passing
        self.aspect_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()) # Aspect gating mechanism
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout     = nn.Dropout(dropout)
        self.num_layers  = num_layers
        
    def forward(self, token_embeddings, edge_index, aspect_embedding):
        x = token_embeddings
        num_nodes = x.size(0)
        
        for i in range(self.num_layers):
            # Message passing
            if edge_index.size(1) > 0:
                src, dst = edge_index[0], edge_index[1]
                messages = x[src]  # (num_edges, hidden_dim) - gather source node features
                aggregated = torch.zeros_like(x)
                # scatter_add_ accumulates each message into the destination node row.
                # dst.unsqueeze(1).expand_as(messages) broadcasts scalar node indices
                # to match the hidden_dim columns of messages.
                aggregated = aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
                
                # Apply linear transformation
                x_gcn = self.gcn_layers[i](aggregated)
                x_gcn = F.relu(x_gcn)
            else:
                # No dependency edges (e.g. very short text or parse failure).
                # Apply the linear layer directly to keep parameter flow intact.
                x_gcn = self.gcn_layers[i](x)
                x_gcn = F.relu(x_gcn)
            
            # Aspect-oriented gating: the gate is computed from both the GCN output
            # and the aspect embedding. Tokens relevant to this aspect get gate ≈ 1
            # (keep GCN features) while irrelevant tokens get gate ≈ 0 (keep residual).
            aspect_expanded = aspect_embedding.unsqueeze(0).expand(num_nodes, -1)
            gate_input = torch.cat([x_gcn, aspect_expanded], dim=-1)  # (num_nodes, hidden_dim*2)
            gate = self.aspect_gate(gate_input)  # (num_nodes, hidden_dim) — sigmoid output in [0, 1]
            
            # Gated residual connection: blend GCN features with the previous repr.
            x = gate * x_gcn + (1 - gate) * x
            x = self.layer_norms[i](x)
            x = self.dropout(x)
        
        return x


class MultiAspectSentimentModel(nn.Module):
    """
    Full model: RoBERTa + Aspect-Aware Attention + Dependency GCN.
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
        need_token_embeddings = self.use_gcn and edge_index is not None
        
        # Single forward pass with appropriate parameters based on GCN requirement
        if need_token_embeddings:
            attn_predictions, attention_weights, aspect_repr, token_embeddings = self.aspect_aware_roberta(
                input_ids, attention_mask, aspect_id, return_token_embeddings=True)
        else:
            attn_predictions, attention_weights, aspect_repr = self.aspect_aware_roberta(
                input_ids, attention_mask, aspect_id)
        
        # If dependency parsing is not used, return attention-based predictions
        if not need_token_embeddings:
            if return_attention: return attn_predictions, attention_weights, aspect_repr, None
            return attn_predictions
        
        # Apply Dependency GCN using the token embeddings to each sample in batchfrom above
        gcn_outputs = []
        for i in range(input_ids.size(0)):
            if edge_index is not None and i < len(edge_index) and edge_index[i] is not None and edge_index[i].size(1) > 0:
                # Get aspect embedding for this sample
                aspect_emb = self.aspect_aware_roberta.aspect_embeddings(aspect_id[i])
                
                # Apply GCN
                gcn_out = self.dep_gcn(token_embeddings[i], edge_index[i], aspect_emb)
                
                # Pool GCN output (mean pooling over tokens)
                mask_expanded = attention_mask[i].unsqueeze(-1).float()
                # Mean-pool GCN token features over the non-padding positions.
                # 1e-9 epsilon prevents division by zero for all-padding sequences.
                sum_mask = mask_expanded.sum(0)
                gcn_pooled = (gcn_out * mask_expanded).sum(0) / (sum_mask + 1e-9)
                gcn_outputs.append(gcn_pooled)
            else:
                # No dependency tree: fill with zeros so the final_classifier still receives a valid (hidden_dim,) tensor from the GCN branch.
                gcn_outputs.append(torch.zeros_like(aspect_repr[i]))
        
        gcn_output = torch.stack(gcn_outputs)  
        
        # Concatenate attention-based and GCN-based representations.
        # The final_classifier then learns to weight each branch's contribution.
        combined = torch.cat([aspect_repr, gcn_output], dim=-1)  
        
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
