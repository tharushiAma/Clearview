# Chapter 5: Model Architecture

## 5.1 Architecture Overview

The MultiAspectSentimentModel integrates three core components:

RoBERTa-base Encoder -> Aspect-Aware Attention Module -> Aspect-Oriented Dependency GCN -> Per-Aspect Classifiers

Total trainable parameters: ~132M (with GCN), ~129M (without GCN)

## 5.2 RoBERTa Encoder

- Pre-trained roberta-base (HuggingFace)

- 12 transformer layers, 768 hidden dimensions, 12 attention heads

- 125M parameters, full fine-tuning

- Input: [CLS] review_text [EOS] [PAD] (max 128 tokens)

- Output: last_hidden_state (batch, seq_len, 768)

### Why RoBERTa over BERT?

160GB training data vs 16GB, dynamic masking, no NSP objective, consistently outperforms BERT on sentiment benchmarks.

## 5.3 Aspect-Aware Attention Module

### Core Innovation

Uses a learnable aspect embedding as the query in Multi-Head Attention instead of the [CLS] token. Forces selective retrieval of only aspect-relevant information from the token sequence.

### Architecture

- 7 learnable aspect embeddings (7 x 768), Xavier initialized

- 8-head Multi-Head Attention

- aspect_query (batch, 1, 768) as query, hidden_states (batch, seq, 768) as key/value

- Output: aspect_repr (batch, 768) + attention weights for XAI

### Properties

- Aspect-specific: each embedding independently learned

- Interpretable: attention weights show which tokens inform each aspect

- Parameter efficient: only 7x768=5,376 additional parameters

### Ablation A2

When use_aspect_attention=False, replaced with CLS pooling + aspect ID offset. Measures contribution of aspect-conditioned attention.

## 5.4 Aspect-Oriented Dependency GCN

### Motivation

For mixed-sentiment reviews, transformer attention distributes globally. Dependency GCN enforces syntactic locality: opinion words syntactically adjacent to an aspect noun receive higher weight.

### Aspect Gate Mechanism

gate = sigmoid(W_gate * aspect_embedding). Gates the node features before message aggregation. Controls which syntactic information flows based on current aspect.

Example: for aspect=smell, the gate suppresses colour-related features, preventing propagation to smell nodes.

### 2-Layer GCN

Layer 1: aggregates direct syntactic neighbors. Layer 2: aggregates 2-hop neighbors. More layers cause oversmoothing on short review graphs.

### Residual Connections

H_new = LayerNorm(W_msg(messages) + H_prev). Prevents degradation across GCN layers.

### Ablation A1

use_dependency_gcn=False bypasses GCN, measures its contribution to mixed sentiment resolution.

## 5.5 Fusion and Classifiers

fused = aspect_repr + gcn_output (residual addition, no learned gate needed based on ablation)

7 independent classifier heads: Linear(768->384) -> ReLU -> Dropout(0.1) -> Linear(384->3)

### Why Per-Aspect Heads?

Different aspects have different linguistic patterns. Shared classifier forces same decision boundary for smell vs shipping language. Ablation A5 tests the shared-head hypothesis.

## 5.6 Ablation Flags

| Flag | Default | Ablation |

|------|---------|---------|

| use_dependency_gcn | True | A1 |

| use_aspect_attention | True | A2 |

| use_shared_classifier | False | A5 |
