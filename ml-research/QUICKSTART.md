# Quick Start Guide

## Step-by-Step Instructions to Get Started

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- 8GB+ RAM
- Your dataset CSV files (train.csv, val.csv, test.csv)

---

## 1. Initial Setup (5 minutes)

### On Linux/Mac:
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download xx_ent_wiki_sm
```

---

## 2. Prepare Your Data (2 minutes)

### Copy your CSV files to the data directory:
```bash
cp /path/to/your/train.csv data/train.csv
cp /path/to/your/val.csv data/val.csv
cp /path/to/your/test.csv data/test.csv
```

### Verify your data format:
Your CSV files should have these columns:
- `text_clean`: The review text
- `stayingpower`, `texture`, `smell`, `price`, `colour`, `shipping`, `packing`: Aspect sentiment labels

Labels should be one of: `positive`, `negative`, `neutral`, or empty/NaN

---

## 3. Configure Your Experiment (3 minutes)

Edit `configs/config.yaml`:

### Basic Configuration:
```yaml
# Choose your model size
model:
  roberta_model: "roberta-base"  # Fast, good performance
  # roberta_model: "roberta-large"  # Slower, better performance
  # roberta_model: "xlm-roberta-base"  # For Vietnamese text

# Adjust training parameters
training:
  batch_size: 16  # Reduce to 8 or 4 if out of memory
  num_epochs: 30
  learning_rate: 2.0e-5

# Enable/disable features
model:
  use_dependency_gcn: true  # Set to false for faster training
data:
  use_dependency_parsing: true  # Set to false for faster data loading
```

### For Quick Testing (Small Model):
```yaml
model:
  roberta_model: "roberta-base"
  gcn_layers: 1
  use_dependency_gcn: false

training:
  batch_size: 32
  num_epochs: 5
```

### For Best Performance (Full Model):
```yaml
model:
  roberta_model: "roberta-large"
  gcn_layers: 2
  use_dependency_gcn: true

training:
  batch_size: 8  # Smaller due to larger model
  num_epochs: 30
```

---

## 4. Start Training (< 1 minute to start)

### Basic Training:
```bash
python train.py --config configs/config.yaml
```

### Training with Weights & Biases (Recommended):
```bash
# First time: login to wandb
wandb login

# Enable in config
# use_wandb: true
# wandb_project: "cosmetic-sentiment"

# Train
python train.py --config configs/config.yaml
```

### Training Output:
```
Creating dataloaders...
Loaded 9268 samples from data/train.csv
Loaded 1324 samples from data/val.csv

Creating model...
Created model with 125,236,227 trainable parameters

Computing class weights...
Initialized loss for stayingpower:
  Class counts: [647, 220, 1060]
  Imbalance ratio: 4.82
  Focal gamma: 2.5
...

Starting training for 30 epochs
========================================

Epoch 1/30:  15%|████▏                 | 87/580 [02:31<13:29,  1.64s/it, loss=1.234]
```

---

## 5. Monitor Training Progress

### Using Weights & Biases (Recommended):
1. Go to https://wandb.ai
2. Select your project
3. View:
   - Training loss curves
   - Validation metrics
   - Per-aspect performance
   - Confusion matrices

### Using TensorBoard (Alternative):
```bash
# In another terminal
tensorboard --logdir results/
# Open http://localhost:6006
```

### Check Files:
Training saves:
- `results/experiment_name/best_model.pt` - Best model checkpoint
- `results/experiment_name/checkpoint_epoch_*.pt` - Epoch checkpoints
- `results/experiment_name/test_results.json` - Final test results

---

## 6. Evaluate Your Model (2 minutes)

### After Training Completes:
```bash
python evaluate.py \
    --checkpoint results/experiment_name/best_model.pt \
    --data data/test.csv \
    --output-dir results/experiment_name/evaluation
```

### Evaluation Output:
```
======================================================================
Results for SMELL
======================================================================
Accuracy:      0.7823
Macro F1:      0.7156
Weighted F1:   0.7745
MCC:           0.6234

Class        Precision    Recall       F1           Support
----------------------------------------------------------------------
Negative     0.6842       0.7647       0.7222       51
Neutral      0.5833       0.5000       0.5385       12
Positive     0.8456       0.8235       0.8344       119

Overall Accuracy: 0.7823
Overall Macro-F1: 0.7156
```

You'll get:
- Detailed metrics for each aspect
- Confusion matrices (PNG files)
- Error analysis (CSV file)
- LaTeX table for thesis
- All predictions (CSV file)

---

## 7. Common Issues and Solutions

### Issue: Out of Memory (OOM)
**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 8  # or even 4
```

**Solution 2**: Use gradient accumulation
```python
# In train.py, modify training loop
gradient_accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Solution 3**: Use smaller model
```yaml
model:
  roberta_model: "distilroberta-base"  # Smaller, faster
```

---

### Issue: Training is Too Slow
**Solution 1**: Disable dependency parsing
```yaml
data:
  use_dependency_parsing: false
model:
  use_dependency_gcn: false
```

**Solution 2**: Enable mixed precision
```yaml
hardware:
  mixed_precision: true  # 2x faster on modern GPUs
```

**Solution 3**: Reduce number of workers
```yaml
hardware:
  num_workers: 2  # or 0
```

---

### Issue: Poor Performance on Minority Classes
**Solution 1**: Increase focal gamma
```yaml
training:
  focal_gamma:
    smell: 3.5  # Higher = more focus on hard examples
    price: 4.0
    packing: 4.0
```

**Solution 2**: Increase augmentation
```yaml
data:
  augmentation:
    enabled: true
    augmentation_ratio:
      negative: 5.0  # Generate 5x samples
      neutral: 3.0
```

**Solution 3**: Train class-specific models
```python
# Train separate models for each aspect
# Then ensemble predictions
```

---

## 8. Next Steps

### Experiment with Different Configurations:
```bash
# Without GCN
python train.py --config configs/config_no_gcn.yaml

# Different loss functions
python train.py --config configs/config_focal_only.yaml

# Different augmentation strategies
python train.py --config configs/config_heavy_aug.yaml
```

### Analyze Results:
```bash
# Compare experiments
python experiments/compare_runs.py \
    --run1 results/experiment1 \
    --run2 results/experiment2
```

### Generate Thesis Figures:
```python
from utils.visualization import generate_thesis_figures

generate_thesis_figures(
    results_dir='results/best_experiment',
    output_dir='thesis_figures'
)
```

---

## 9. Production Deployment

### Save Model for Inference:
```python
# In your code
import torch

# Load model
model, config = load_model('results/best_model.pt')

# Save for production (smaller file)
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'production_model.pt')
```

### Create Inference API:
```python
# inference.py
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
model = load_model('production_model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    aspect = request.json['aspect']
    
    prediction = model.predict(text, aspect)
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 10. Tips for Best Results

1. **Start Simple**: Train without GCN first, then add it
2. **Monitor Closely**: Use wandb to track all experiments
3. **Save Everything**: Keep all checkpoints for comparison
4. **Test Early**: Run evaluation on val set frequently
5. **Document**: Keep notes on what works and what doesn't
6. **Be Patient**: Good models take time to train (hours, not minutes)
7. **Focus on F1**: Don't rely on accuracy for imbalanced data
8. **Analyze Errors**: Learn from misclassifications

---

## Need Help?

1. Check the main README.md for detailed documentation
2. Look at example notebooks in `notebooks/`
3. Review error messages carefully
4. Open an issue on GitHub
5. Contact: [your-email@example.com]

---

## Estimated Time Requirements

| Task | Time (CPU) | Time (GPU) |
|------|-----------|-----------|
| Setup | 5 min | 5 min |
| Data Prep | 5 min | 5 min |
| First Epoch (no GCN) | 30 min | 3 min |
| First Epoch (with GCN) | 60 min | 6 min |
| Full Training (30 epochs, no GCN) | 15 hours | 1.5 hours |
| Full Training (30 epochs, with GCN) | 30 hours | 3 hours |
| Evaluation | 5 min | 1 min |

**Recommendation**: Use GPU for serious training!

---

Good luck with your research! 🚀
