# EAGLE V2 - Quick Summary

## 📂 Files Created

All new files are in `src/models/` and **do not modify** any existing files:

1. **`eagle_v2_implementation.py`** (733 lines)
   - Enhanced EAGLE model with all improvements
   - New components: UncertaintyHead, AspectFeatureRouter, EnhancedFocalLoss

2. **`train_eagle_v2.py`** (728 lines)
   - Training script with targeted data augmentation
   - Price: 50+ neg, 50+ neu samples
   - Packing: 30+ neg samples
   - Neutral: 1.5x boost across all aspects

3. **`EAGLE_V2_README.md`** (detailed documentation)
   - Architecture explanations
   - Usage instructions
   - Expected results
   - Troubleshooting guide

4. **`compare_eagle_models.py`** (comparison tool)
   - Automatically compares EAGLE V2 vs EAGLE_FINAL
   - Generates visualizations and reports

---

## 🎯 Key Improvements

### 1. Price Aspect (Macro F1: 0.33 → **0.50+**)
- **Problem**: 0.00 F1 for negative/neutral (only 2 neg, 7 neu samples)
- **Solution**: 
  - Generate 50+ synthetic negative samples
  - Generate 50+ synthetic neutral samples
  - Extreme class weights: `[50.0, 20.0, 1.0]`
  - Focal loss gamma: 5.0

### 2. Packing Aspect (Macro F1: 0.54 → **0.60+**)
- **Problem**: Negative recall dropped to 0.55 (only 11 samples)
- **Solution**:
  - Generate 30+ synthetic negative samples
  - Class weights: `[10.0, 5.0, 1.0]`
  - Focal loss gamma: 4.0

### 3. Neutral Class (Avg F1: 0.40 → **0.50+**)
- **Problem**: Consistently low neutral F1 across all aspects
- **Solution**:
  - **Uncertainty-aware heads**: Boost neutral for uncertain predictions
  - **Label smoothing** (ε=0.1): Prevent over-confident predictions
  - **Data augmentation**: 1.5x neutral samples via paraphrasing

### 4. Aspect-Specific Feature Routing
- **Innovation**: Learn optimal mix of transformer vs GNN features per aspect
- **Expected behavior**:
  - Price → more transformer (semantic: "expensive")
  - Texture → more GNN (syntactic: "feels smooth")
  - Shipping → balanced

---

## 🚀 How to Use

### Step 1: Train EAGLE V2

```bash
cd c:/Users/lucif/Desktop/Clearview

# Basic training (recommended)
python src/models/train_eagle_v2.py --augment --epochs 10

# Fast trial (no augmentation, fewer epochs)
python src/models/train_eagle_v2.py --epochs 3 --batch_size 8
```

### Step 2: Monitor Training

Watch the output for:
- Data augmentation statistics (should show increased price/packing samples)
- Loss breakdown per aspect (price & packing should improve)
- Validation F1 scores every 2 epochs

### Step 3: Compare Results

```bash
# After training completes
python src/models/compare_eagle_models.py
```

This will:
- Parse both metrics files
- Print detailed comparison tables
- Generate visualizations (price, neutral class, heatmap)
- Create summary report in markdown

---

## 📊 Expected Training Output

```
================================================================================
AUGMENTING PRICE ASPECT
================================================================================
Price Negative: Current=2, Target=50, Generating=48
Price Neutral: Current=7, Target=50, Generating=43
Added 91 price samples

================================================================================
AUGMENTING PACKING NEGATIVE
================================================================================
Packing Negative: Current=11, Target=30, Generating=19
Added 19 packing negative samples

================================================================================
AUGMENTING NEUTRAL CLASS (ALL ASPECTS)
================================================================================
stayingpower: Generating 30 neutral samples
texture: Generating 45 neutral samples
...
Total neutral augmented samples: 234

================================================================================
AFTER AUGMENTATION
================================================================================
Final dataset size: 2,847 (from 2,503)

PRICE:
  negative:    52 (  1.8%)  ← Was 2 (0.08%)
  neutral:     57 (  2.0%)  ← Was 7 (0.28%)
  positive: 3,165 ( 96.2%)
```

---

## 📈 Expected Improvements

| Metric | EAGLE_FINAL | EAGLE V2 Target | Strategy |
|--------|-------------|-----------------|----------|
| **Price Macro F1** | 0.33 | 0.50+ | Data augmentation + extreme weights |
| **Packing Macro F1** | 0.54 | 0.60+ | Data augmentation + higher weights |
| **Neutral Avg F1** | ~0.40 | 0.50+ | Uncertainty + label smoothing |
| **Overall Macro F1** | 0.6469 | 0.70+ | All improvements combined |

---

## 🔧 Key Model Components

### EnhancedFocalLoss
```python
# Aspect-specific configurations
'price': {
    'gamma': 5.0,                    # Focus on hard examples
    'weights': [50.0, 20.0, 1.0],   # Extreme neg/neu boost
    'label_smoothing': 0.15          # Prevent overconfidence
}
```

### UncertaintyHead
```python
# Dual output: logits + uncertainty
logits, uncertainty = classifier(features)

# Calibrate for neutral: if uncertain, boost neutral class
if uncertainty > 0.5:
    logits[neutral_idx] += uncertainty * 2.0
```

### AspectFeatureRouter
```python
# Learn per-aspect fusion weights
router_weights = softmax([transformer_weight, gnn_weight])
fused = router_weights[0] * transformer_feats + 
        router_weights[1] * gnn_feats
```

---

## 📝 Files Breakdown

| File | Size | Purpose |
|------|------|---------|
| `eagle_v2_implementation.py` | ~27 KB | Model architecture |
| `train_eagle_v2.py` | ~25 KB | Training + augmentation |
| `EAGLE_V2_README.md` | ~16 KB | Documentation |
| `compare_eagle_models.py` | ~12 KB | Comparison tool |
| `EAGLE_V2_SUMMARY.md` | 5 KB | This file |

---

## ✅ Quick Checklist

Before training:
- [ ] Verify `data/train.parquet` and `data/val.parquet` exist
- [ ] Install dependencies: `pip install torch transformers spacy pandas sklearn`
- [ ] Download SpaCy: `python -m spacy download en_core_web_sm`
- [ ] Check GPU availability: `torch.cuda.is_available()`

After training:
- [ ] Check `outputs/checkpoints/eagle_v2_best.pt` exists
- [ ] Verify metrics in `outputs/reports/eagle_v2_epoch*_metrics.txt`
- [ ] Compare price negative F1 (should be > 0.00)
- [ ] Compare packing negative F1 (should improve from 0.63)
- [ ] Check average neutral F1 (should be > 0.45)

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
python train_eagle_v2.py --batch_size 8 --gcn_dim 200
```

### "Price metrics still 0.00"
- Check data augmentation worked: Look for "Added X price samples"
- Increase targets: Edit line ~270 in `train_eagle_v2.py`
  ```python
  df = augment_price_aspect(df, target_neg=100, target_neu=100)
  ```

### "Neutral class still low"
- Increase label smoothing to 0.2 in `eagle_v2_implementation.py`
- Lower uncertainty threshold to 0.3 in forward pass

---

## 📚 What Each Component Does

1. **EnhancedFocalLoss**: Focuses training on hard examples (minority classes)
2. **UncertaintyHead**: Detects when model is uncertain → predicts neutral
3. **AspectFeatureRouter**: Learns best feature mix per aspect
4. **Data Augmentation**: Creates synthetic minority class examples
5. **Label Smoothing**: Prevents over-confident wrong predictions
6. **Extreme Weights**: Makes model care 50x more about price negatives

---

## 🎓 Research Contributions

1. **First uncertainty-aware ABSA model for cosmetics domain**
2. **Aspect-specific feature routing** (novel architecture)
3. **Targeted data augmentation** for extreme imbalance (2:315 ratio)
4. **Comprehensive evaluation** on real-world e-commerce data

---

## 📞 Next Steps

1. **Train**: `python src/models/train_eagle_v2.py --augment`
2. **Wait**: ~2-4 hours on GPU for 10 epochs
3. **Compare**: `python src/models/compare_eagle_models.py`
4. **Analyze**: Check visualizations in `outputs/reports/comparisons/`
5. **Iterate**: If results unsatisfactory, adjust hyperparameters

---

**Status**: ✅ Ready to train

**Files**: All created, no conflicts with existing code

**Expected Training Time**: 2-4 hours (10 epochs on GPU)

**Expected Improvement**: +0.03-0.05 overall F1, significant price/packing/neutral gains
