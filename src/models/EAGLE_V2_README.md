# EAGLE V2 - Enhanced Model Documentation

## 🎯 Overview

EAGLE V2 is an enhanced version of the EAGLE (Explainable Adaptive Graph-Enhanced ABSA with Learnable MSR) model, specifically designed to address the critical weaknesses identified in the original EAGLE_FINAL model through comprehensive metric analysis.

### Performance Comparison

| Aspect | EAGLE_FINAL | Target (EAGLE V2) | Improvement Strategy |
|--------|-------------|-------------------|----------------------|
| **Price (Macro F1)** | 0.33 | **0.50+** | Extreme class weights (50:20:1), data augmentation |
| **Packing (Macro F1)** | 0.54 | **0.60+** | Boosted negative class, hard negative mining |
| **Neutral (Avg F1)** | ~0.40 | **0.50+** | Uncertainty calibration, label smoothing |
| **Overall (Macro F1)** | 0.6469 | **0.70+** | All improvements combined |

---

## 🚀 Key Innovations

### 1. **Enhanced Focal Loss with Label Smoothing**

**Problem**: Original model produces over-confident predictions, leading to poor neutral class detection.

**Solution**:
- Label smoothing (ε=0.1) prevents over-confident predictions
- Aspect-specific gamma values (price: 5.0, packing: 4.0)
- Extreme class weighting for minority classes (price negative: 50x)

```python
# Example: Price aspect configuration
'price': {
    'gamma': 5.0,                    # High focus on hard examples
    'weights': [50.0, 20.0, 1.0],   # [negative, neutral, positive]
    'label_smoothing': 0.15          # Higher smoothing for extreme imbalance
}
```

---

### 2. **Uncertainty-Aware Prediction Heads**

**Problem**: Model doesn't know when it's uncertain, leading to incorrect neutral classifications.

**Solution**:
- Dual-head architecture: Classifier + Uncertainty estimator
- High uncertainty → boost neutral class probability
- Interpretable: Can visualize prediction confidence

**Expected Impact**: Neutral F1 +0.10-0.15

```python
class UncertaintyHead(nn.Module):
    def forward(self, features):
        logits = self.classifier(features)          # [B, 3]
        uncertainty = self.uncertainty_net(features) # [B]
        # calibrate_neutral() boosts neutral for uncertain predictions
        calibrated_logits = self.calibrate_neutral(logits, uncertainty)
        return calibrated_logits, uncertainty
```

---

### 3. **Aspect-Specific Feature Routing**

**Problem**: Not all aspects benefit equally from syntactic (GNN) vs semantic (transformer) features.

**Solution**:
- Learnable routing network per aspect
- Dynamically mixes transformer and GNN features
- Model learns optimal fusion weights per aspect

**Expected Behavior**:
- **Price**: Higher transformer weight (semantic: "expensive", "cheap")
- **Texture**: Higher GNN weight (syntax: "feels smooth")
- **Shipping**: Balanced (both matter)

```python
# Learns weights [transformer_weight, gnn_weight] for each aspect
fusion = router_weights[:, 0] * transformer_features + 
         router_weights[:, 1] * gnn_features
```

---

### 4. **Targeted Data Augmentation**

**Problem**: Severe class imbalance in training data:
- Price negative: 2 samples
- Price neutral: 7 samples
- Packing negative: 11 samples

**Solution**:

#### Price Aspect Augmentation
- **Target**: 50+ negative samples, 50+ neutral samples
- **Method**: Template-based generation with domain-specific vocabulary

```python
# Negative templates
"This product is {overpriced/expensive/costly}, not worth the price."
"The price is {highway robbery/too expensive}, disappointed."

# Neutral templates
"The price is {fair/reasonable/okay}, nothing special."
"Price seems {acceptable/decent/moderate}, typical for this category."
```

#### Packing Negative Augmentation
- **Target**: 30+ negative samples
- **Method**: Templates describing damaged/poor packaging

```python
"The packaging was {terrible/awful/damaged}, product arrived broken."
"Packaging is {flimsy/inadequate}, product leaked everywhere."
```

#### Neutral Class Boost (All Aspects)
- **Method**: Paraphrasing existing neutral samples
- **Factor**: 1.5x (50% increase)

```python
# Paraphrasing strategies
"Good quality" → "Okay quality, I guess"
"Terrible" → "Not great, could be better"
+ ", somewhat" / ", nothing special" suffixes
```

---

## 📁 File Structure

```
src/models/
│
├── eagle_v2_implementation.py     # NEW: Enhanced EAGLE V2 model
│   ├── EnhancedFocalLoss          # Label smoothing + extreme weights
│   ├── UncertaintyHead            # Uncertainty-aware predictions
│   ├── AspectFeatureRouter        # Dynamic feature fusion
│   ├── PositionAwareGraphAttention # Enhanced GAT
│   ├── DualChannelGCN             # Syntactic + Semantic GCN
│   ├── HierarchicalMSR            # MSR module
│   └── EAGLE_V2                   # Complete model
│
├── train_eagle_v2.py              # NEW: Training script with augmentation
│   ├── augment_price_aspect()     # Price neg/neu generation
│   ├── augment_packing_negative() # Packing neg generation
│   ├── augment_neutral_class()    # Neutral class boost
│   └── EAGLE_V2_Dataset           # Dataset with all inputs
│
└── EAGLE_V2_README.md             # THIS FILE
```

---

## 🔧 Usage

### Installation

```bash
# Ensure you have required packages
pip install torch transformers spacy pandas numpy scikit-learn
python -m spacy download en_core_web_sm
```

### Training EAGLE V2

```bash
cd c:/Users/lucif/Desktop/Clearview

# Basic training with all enhancements
python src/models/train_eagle_v2.py \
    --project_dir "." \
    --augment \
    --use_uncertainty \
    --use_feature_routing \
    --epochs 10 \
    --batch_size 16 \
    --lr 2e-5

# Without data augmentation (not recommended)
python src/models/train_eagle_v2.py \
    --project_dir "." \
    --epochs 10

# Custom configuration
python src/models/train_eagle_v2.py \
    --gcn_dim 512 \
    --gcn_layers 3 \
    --batch_size 32 \
    --lr 1e-5 \
    --epochs 15
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--project_dir` | Current dir | Project root directory |
| `--gcn_dim` | 300 | GCN hidden dimension |
| `--gcn_layers` | 2 | Number of GCN layers |
| `--max_len` | 256 | Max sequence length |
| `--use_uncertainty` | True | Enable uncertainty-aware heads |
| `--use_feature_routing` | True | Enable aspect-specific routing |
| `--batch_size` | 16 | Training batch size |
| `--lr` | 2e-5 | Learning rate |
| `--epochs` | 10 | Number of training epochs |
| `--eval_every` | 2 | Evaluation frequency (epochs) |
| `--augment` | True | Apply data augmentation |

---

## 📊 Expected Results

### Aspect-Level Improvements

#### Price Aspect
```
Current (EAGLE_FINAL):
               precision    recall  f1-score   support
    negative       0.00      0.00      0.00         2
     neutral       0.00      0.00      0.00         7
    positive       0.97      1.00      0.99       315
    macro avg      0.32      0.33      0.33       324

Expected (EAGLE V2):
               precision    recall  f1-score   support
    negative       0.40      0.50      0.44        52  ← Augmented
     neutral       0.50      0.60      0.55        57  ← Augmented
    positive       0.97      0.95      0.96       315
    macro avg      0.62      0.68      0.65       424  ← +0.32 F1
```

#### Packing Aspect
```
Current (EAGLE_FINAL):
               precision    recall  f1-score   support
    negative       0.75      0.55      0.63        11
     neutral       0.00      0.00      0.00         1
    positive       0.98      0.99      0.99       287
    macro avg      0.58      0.51      0.54       299

Expected (EAGLE V2):
               precision    recall  f1-score   support
    negative       0.80      0.70      0.75        30  ← Augmented
     neutral       0.00      0.00      0.00         1
    positive       0.98      0.99      0.98       287
    macro avg      0.59      0.56      0.58       318  ← +0.04 F1
```

#### Neutral Class (Average across aspects)
```
Current: ~0.40 F1
Expected: 0.50+ F1 (+0.10)

Improvement mechanisms:
1. Label smoothing prevents over-confidence
2. Uncertainty calibration boosts neutral for ambiguous cases
3. 1.5x neutral data augmentation
```

---

## 🔬 Interpretability Features

### 1. Routing Weights Visualization

```python
# During inference
output = model(
    input_ids, attention_mask, syntactic_adj,
    aspect_masks, positions,
    return_routing_weights=True
)

# output['routing_weights']: [B, num_aspects, 2]
# [:, :, 0] = transformer weight
# [:, :, 1] = gnn weight

# Example: Aspect 0 (StayingPower) routing
print(f"Transformer weight: {output['routing_weights'][0, 0, 0]:.3f}")
print(f"GNN weight: {output['routing_weights'][0, 0, 1]:.3f}")
```

### 2. Uncertainty Scores

```python
# output['uncertainties']: list of [B] tensors

for i, aspect_name in enumerate(aspect_names):
    uncertainty = output['uncertainties'][i][0].item()
    print(f"{aspect_name}: Uncertainty = {uncertainty:.3f}")
    # High uncertainty (>0.5) → prediction pushed toward neutral
```

### 3. Aspect Importance (MSR)

```python
# output['msr_output']['aspect_importance']: [B, num_aspects]

importance = output['msr_output']['aspect_importance'][0]
for i, aspect_name in enumerate(aspect_names):
    print(f"{aspect_name}: {importance[i]:.3f}")
# Shows which aspects contribute most to overall sentiment
```

---

## 🛠️ Advanced: Progressive Training

For optimal results, consider progressive training strategy:

```python
# Stage 1: Learn majority classes (5 epochs)
# - Low focal gamma (1.0)
# - Only transformer features
# - Simple cross-entropy

# Stage 2: Introduce GNN (5 epochs)
# - Medium focal gamma (2.0)
# - GNN weight: 0.3
# - Start feature routing

# Stage 3: Focus on minority classes (10 epochs)
# - High focal gamma (4.0)
# - Full GNN weight: 0.5
# - Oversample price_neg, price_neutral, packing_neg

# Stage 4: Fine-tune neutral boundaries (5 epochs)
# - Focal gamma: 3.0
# - Label smoothing: 0.1
# - Neutral loss weight: 2.0
```

Implementation in `train_eagle_v2.py`:

```python
# TODO: Implement progressive training
# For now, use single-stage training with all enhancements
```

---

## 📈 Monitoring Training

### Loss Components

```
Epoch 1 Loss Breakdown:
  total_loss: 1.2345
  aspect_loss: 1.1000
  msr_sentiment_loss: 0.0800
  msr_conflict_loss: 0.0545
  
  stayingpower: 0.1200
  texture: 0.1500
  smell: 0.0900
  price: 0.4500       ← Should decrease as augmentation helps
  colour: 0.1100
  shipping: 0.1300
  packing: 0.2500     ← Should decrease as augmentation helps
```

### Validation Metrics

```
Validation Macro F1: 0.6800

Target milestones:
- Epoch 2:  F1 > 0.68 (baseline)
- Epoch 5:  F1 > 0.70
- Epoch 10: F1 > 0.72 (final target)
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Reduce batch size
python train_eagle_v2.py --batch_size 8

# Or reduce model size
python train_eagle_v2.py --gcn_dim 200 --gcn_layers 1
```

### Issue: Price/Packing metrics still zero

**Diagnosis**: Data augmentation may not be diverse enough.

**Solution**:
1. Check augmented dataset: `outputs/cache/train_adj_v2.pkl`
2. Increase augmentation targets:
   ```python
   # In train_eagle_v2.py, line ~270
   df = augment_price_aspect(df, target_neg=100, target_neu=100)
   df = augment_packing_negative(df, target_neg=50)
   ```
3. Add back-translation augmentation (advanced)

### Issue: Neutral class still under-predicted

**Solution**:
1. Increase label smoothing:
   ```python
   # In eagle_v2_implementation.py, _get_aspect_loss_configs()
   'label_smoothing': 0.2  # Increase from 0.1
   ```
2. Lower uncertainty threshold:
   ```python
   # In eagle_v2_implementation.py, EAGLE_V2.forward()
   calibrated_logits = self.classifiers[i].calibrate_neutral(
       logits, uncertainty, neutral_idx=1, threshold=0.3  # Lower from 0.5
   )
   ```

---

## 📚 References

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
2. **Graph Attention Networks**: Veličković et al., "Graph Attention Networks" (ICLR 2018)
3. **ABSA with GCN**: Zhang et al., "Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks" (EMNLP 2019)
4. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture" (CVPR 2016)
5. **Uncertainty Estimation**: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)

---

## 📝 Citation

If you use EAGLE V2 in your research, please cite:

```bibtex
@software{eagle_v2_2026,
  title={EAGLE V2: Enhanced Adaptive Graph-Enhanced ABSA with Uncertainty-Aware Predictions},
  author={[Your Name]},
  year={2026},
  note={Enhanced version with aspect-specific feature routing and targeted data augmentation}
}
```

---

## ✅ Quick Start Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Download SpaCy model (`python -m spacy download en_core_web_sm`)
- [ ] Verify data files exist:
  - [ ] `data/train.parquet`
  - [ ] `data/val.parquet`
- [ ] Run training: `python src/models/train_eagle_v2.py --augment`
- [ ] Monitor metrics in `outputs/reports/eagle_v2_epoch*_metrics.txt`
- [ ] Compare with EAGLE_FINAL: Check price, packing, neutral F1 scores
- [ ] Best model saved to: `outputs/checkpoints/eagle_v2_best.pt`

---

## 📧 Support

For questions or issues:
1. Check this README
2. Review training logs in `outputs/`
3. Inspect augmented data distribution
4. Ensure CUDA is available for GPU training

---

**Last Updated**: 2026-01-25

**Version**: EAGLE V2.0

**Status**: Ready for Training ✅
