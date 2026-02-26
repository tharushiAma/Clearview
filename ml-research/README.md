# Multi-Aspect Sentiment Analysis with Explainability for Cosmetic Domain

A research implementation of multi-aspect sentiment analysis combining RoBERTa, Dependency GCN, and comprehensive class imbalance handling techniques.

## 🎯 Research Objectives

1. **Multi-Aspect Sentiment Analysis**: Analyze sentiment for 7 aspects (stayingpower, texture, smell, price, colour, shipping, packing)
2. **Handle Severe Class Imbalance**: Address extreme imbalance using hybrid loss functions and data augmentation
3. **Explainability**: Provide interpretable predictions using attention mechanisms, LIME, and dependency path visualization
4. **Mixed Sentiment Resolution**: Use Dependency GCN to capture syntactic relationships for conflicting sentiments

## 📊 Dataset Statistics

### Training Data (9,268 reviews)
- **Severely Imbalanced Aspects**:
  - Price: pos:neg:neu = 132:1:0.9
  - Packing: pos:neg:neu = 145:1:0.2
- **Moderately Imbalanced**:
  - Smell: pos:neg:neu = 17:3:1
  - Staying Power: pos:neg:neu = 5:3:1
- **Balanced Aspects**:
  - Colour, Texture, Shipping

## 🏗️ Architecture

```
Input Text → RoBERTa Encoder → Aspect-Aware Attention → Dependency GCN → Sentiment Classification
                 ↓
           Explainability Module (Attention + LIME + Integrated Gradients)
```

### Key Components

1. **RoBERTa Base**: Pre-trained contextual embeddings (768-dim)
2. **Aspect-Aware Attention**: Learnable aspect embeddings with multi-head attention
3. **Dependency GCN**: Aspect-oriented graph convolution for syntactic relationships
4. **Hybrid Loss**: Combination of Focal Loss + Class-Balanced Loss + Dice Loss
5. **Multi-Level Explainability**: Attention weights, LIME explanations, dependency paths

## 📁 Project Structure

```
cosmetic_sentiment_project/
├── configs/
│   └── config.yaml              # Main configuration file
├── data/                        # Place your CSV files here
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── model.py                 # Main model architecture
│   └── losses.py                # Loss functions for imbalance
├── utils/
│   ├── data_utils.py           # Data loading and preprocessing
│   ├── metrics.py              # Evaluation metrics
│   └── augmentation.py         # Data augmentation (optional)
├── notebooks/                   # Jupyter notebooks for analysis
├── experiments/                 # Experiment scripts
├── results/                     # Training outputs
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd cosmetic_sentiment_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for dependency parsing
python -m spacy download xx_ent_wiki_sm
```

### 2. Prepare Data

Place your CSV files in the `data/` directory:
- `train.csv`
- `val.csv`
- `test.csv`

Required columns:
- `text_clean`: Review text
- `stayingpower`, `texture`, `smell`, `price`, `colour`, `shipping`, `packing`: Aspect labels (positive/negative/neutral)

### 3. Configure

Edit `configs/config.yaml` to adjust:
- Model architecture (RoBERTa variant, hidden dimensions, GCN layers)
- Training parameters (batch size, learning rate, epochs)
- Loss function parameters (focal gamma, class-balanced beta)
- Explainability methods

### 4. Train Model

```bash
# Basic training
python train.py --config configs/config.yaml

# Resume from checkpoint
python train.py --config configs/config.yaml --resume results/experiment_name/best_model.pt
```

### 5. Evaluate

```bash
# Evaluate best model on test set
python evaluate.py --checkpoint results/experiment_name/best_model.pt --data data/test.csv
```

### 6. Monitor Training (Optional)

```bash
# Enable wandb in config.yaml
use_wandb: true
wandb_project: "cosmetic-sentiment-analysis"

# Or use tensorboard
tensorboard --logdir results/
```

## 📈 Expected Results

Based on the data distribution:

| Aspect | Coverage | Imbalance | Expected Macro-F1 |
|--------|----------|-----------|-------------------|
| Colour | 57% | Low | 0.78-0.85 |
| Texture | 37% | Low | 0.78-0.85 |
| Shipping | 41% | Moderate | 0.75-0.82 |
| Staying Power | 21% | Moderate | 0.68-0.75 |
| Smell | 22% | High | 0.68-0.75 |
| Price | 25% | Extreme | 0.50-0.65 |
| Packing | 23% | Extreme | 0.50-0.65 |

**Note**: For severely imbalanced aspects (Price, Packing), focus on minority class recall rather than overall accuracy.

## 🔧 Configuration Guide

### Model Configuration

```yaml
model:
  roberta_model: "roberta-base"      # or "xlm-roberta-base" for multilingual
  hidden_dim: 768
  num_aspects: 7
  num_classes: 3
  gcn_layers: 2                       # Dependency GCN layers
  dropout: 0.1
  use_dependency_gcn: true            # Enable/disable GCN
  use_aspect_attention: true
```

### Loss Configuration for Severe Imbalance

```yaml
training:
  loss_weights:
    focal: 1.0
    class_balanced: 0.5
    dice: 0.3
  
  # Aspect-specific focal gamma (higher = more focus on hard examples)
  focal_gamma:
    default: 2.0
    price: 3.0      # Extreme imbalance
    packing: 3.0
    smell: 2.5
  
  # Class-balanced beta (higher = stronger reweighting)
  class_balanced_beta:
    default: 0.999
    price: 0.9999   # Extreme imbalance
    packing: 0.9999
```

### Data Augmentation

```yaml
data:
  augmentation:
    enabled: true
    techniques: ["back_translation", "synonym_replacement", "mixup"]
    augment_minority_only: true
    augmentation_ratio:
      negative: 3.0  # Generate 3x samples for negative class
      neutral: 2.0
      positive: 1.0
```

## 📊 Evaluation Metrics

The system reports:

1. **Overall Metrics**:
   - Accuracy
   - Macro F1-Score (equal weight to all classes)
   - Weighted F1-Score (weighted by class frequency)
   - Matthews Correlation Coefficient (MCC)

2. **Per-Class Metrics**:
   - Precision, Recall, F1 for each sentiment class
   - Support (number of samples)

3. **Confusion Matrices**: Visual representation of predictions vs ground truth

4. **Error Analysis**: Detailed breakdown of misclassifications

## 🔬 Explainability Features

### 1. Attention Visualization
```python
# Visualize which words the model focuses on for each aspect
predictions, attention_weights, _, _ = model(
    input_ids, attention_mask, aspect_id, 
    return_attention=True
)
```

### 2. LIME Explanations
```python
from explainability.lime_explainer import LIMEExplainer

explainer = LIMEExplainer(model, tokenizer)
explanation = explainer.explain(text, aspect="smell")
explainer.visualize(explanation)
```

### 3. Dependency Path Analysis
```python
# Show how sentiment flows through syntactic dependencies
from explainability.dependency_viz import visualize_dependency_paths

visualize_dependency_paths(
    text="beautiful color but terrible smell",
    aspect="smell",
    model=model
)
```

## 🎓 Research Contributions

1. **Novel Architecture**: RoBERTa + Aspect-Oriented Dependency GCN
2. **Comprehensive Imbalance Handling**: Hybrid loss + targeted augmentation
3. **Multi-Level Explainability**: Attention + LIME + Dependency paths
4. **Domain-Specific**: Vietnamese cosmetics reviews

## 📝 Key Improvements Over MAFESA

| Component | MAFESA | Our Approach |
|-----------|--------|--------------|
| Base Encoder | GloVe (static) | RoBERTa (contextualized) |
| Aspect Extraction | LDA (unsupervised) | Aspect-aware attention (supervised) |
| Sentiment Model | Hierarchical NN | Dependency GCN |
| Class Imbalance | Not addressed | Hybrid loss + augmentation |
| Explainability | LIME only | Multi-level (Attention + LIME + IG) |

## ⚙️ Advanced Usage

### Custom Loss Functions

```python
# Define custom loss for specific aspects
from models.losses import HybridLoss

custom_loss = HybridLoss(
    samples_per_class=[17, 15, 2244],  # neg, neu, pos
    focal_gamma=3.5,                   # Higher for extreme imbalance
    cb_beta=0.99999,
    weights={'focal': 1.5, 'cb': 0.5, 'dice': 0.3}
)
```

### Ablation Studies

```bash
# Train without GCN
python train.py --config configs/config_no_gcn.yaml

# Train with different loss functions
python train.py --config configs/config_focal_only.yaml

# Train without augmentation
python train.py --config configs/config_no_aug.yaml
```

### Hyperparameter Tuning

```python
# Use Optuna for hyperparameter search
python experiments/hyperparameter_search.py \
    --config configs/config.yaml \
    --n-trials 50 \
    --study-name cosmetic-sentiment-hpo
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024multiaspect,
  title={Class Imbalance Handled Multi-Aspect Mixed Sentiment Resolution with Explainability in Cosmetic Domain},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

```yaml
# Reduce batch size
training:
  batch_size: 8  # or 4

# Or use gradient accumulation
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # effective batch size = 16
```

### Slow Training

```yaml
# Enable mixed precision training
hardware:
  mixed_precision: true

# Freeze lower RoBERTa layers (in model.py)
for param in self.roberta.encoder.layer[:6].parameters():
    param.requires_grad = False
```

### Poor Performance on Minority Classes

```yaml
# Increase focal gamma for that aspect
training:
  focal_gamma:
    aspect_name: 4.0  # Very high focusing

# Increase augmentation for minority class
data:
  augmentation:
    augmentation_ratio:
      negative: 5.0  # Generate 5x samples
```

## 📧 Support

For questions and issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- MAFESA paper for inspiration
- Hugging Face Transformers library
- PyTorch Geometric for GCN implementation
- Vietnamese NLP community for tools and resources

## 🔮 Future Work

1. Integrate larger language models (RoBERTa-large, XLM-RoBERTa)
2. Implement curriculum learning for gradual difficulty increase
3. Add contrastive learning for better representation
4. Extend to other domains (e-commerce, healthcare)
5. Deploy as REST API for real-time inference
6. Build interactive demo with Gradio/Streamlit

---

**Good luck with your research! Remember to report results honestly, acknowledge limitations, and contribute back to the community.** 🚀
