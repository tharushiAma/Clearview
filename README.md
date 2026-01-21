# Clearview

Class-Balanced Multi-Aspect Mixed Sentiment Analysis with Explainability in Cosmetic Domain

## Project Overview

This project implements aspect-based sentiment analysis for the cosmetic domain using transformer-based models (DistilBERT, RoBERTa) with class balancing techniques and explainability features.

## Setup Instructions

### Prerequisites

- **Python 3.11 or 3.12** (Required - Python 3.14 is NOT compatible with SpaCy)
- Git
- NVIDIA GPU with CUDA support (Recommended for faster training)

> **Note:** Python 3.14 cannot be used for this project due to incompatibility between SpaCy and Pydantic V1. SpaCy's internal dependencies fail to load with Python 3.14+. Please use Python 3.11 or 3.12 for full compatibility.

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd Clearview
```

#### 2. Create a Virtual Environment

It's recommended to use a virtual environment to isolate project dependencies.

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

Install the required packages from the root directory:

```bash
pip install -r requirements.txt
```

This will install:

- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers library
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning utilities
- `numpy` - Numerical computing

#### 4. Install PyTorch with GPU Support (Recommended)

To run models on GPU with Python 3.11 or 3.12, install the CUDA-enabled version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Important:** CUDA-enabled PyTorch is only available for Python 3.11 and 3.12. If using Python 3.14, only the CPU version will be installed, significantly slowing down training.

#### 5. Verify Installation

You can verify that the packages are installed correctly:

```bash
pip list
```

### Project Structure

```
Clearview/
├── .git/                 # Git version control
├── .gitignore           # Git ignore rules
├── .gitattributes       # Git attributes
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── baseline_analysis.md # Analysis documentation
├── walkThroughFixes.md  # Documentation of fixes
│
├── configs/             # Configuration files
│   └── data_config.yaml
│
├── data/                # Dataset files
│   ├── raw/            # Original raw datasets
│   ├── processed/      # Cleaned and preprocessed data
│   └── splits/         # Train/validation/test splits
│
├── notebooks/           # Jupyter notebooks for exploration
│   ├── google_translator.py
│   └── remove_other_and_no_aspect_rows.py
│
├── outputs/             # Model outputs and results
│   ├── checkpoints/    # Saved model checkpoints
│   ├── logs/           # Training logs
│   ├── plots/          # Visualization plots
│   └── reports/        # Analysis reports and predictions
│
└── src/                 # Source code
    ├── data_layer/     # Data processing scripts
    │   ├── 01_validate.py   # Data validation
    │   ├── 02_Clean.py      # Data cleaning
    │   ├── 03_split.py      # Train/val/test splitting
    │   └── _common.py       # Common utilities
    │
    ├── models/         # Model implementations
    │   ├── baseline_distilbert.py
    │   └── baseline_roberta.py
    │
    ├── aspects/        # Aspect-based analysis modules
    ├── eval/           # Evaluation utilities
    ├── msr/            # Multi-aspect sentiment recognition
    └── xai/            # Explainability features
```

### Running the Project

#### Step 1: Data Preparation

1. Place your raw dataset CSV files in the `data/raw/` directory
2. Run the data pipeline scripts in order:

```bash
# Validate the data
python src/data_layer/01_validate.py --project_dir .

# Clean the data
python src/data_layer/02_Clean.py --project_dir .

# Create train/validation/test splits
python src/data_layer/03_split.py --project_dir .
```

#### Step 2: Training Baseline Models

Run the baseline model training scripts:

**DistilBERT Baseline:**

```bash
python src/models/baseline_distilbert.py --project_dir .
```

**RoBERTa Baseline:**

```bash
python src/models/baseline_roberta.py --project_dir .
```

**MSR Resolver (RoBERTa + DepGCN):**

This model uses dependency graphs to resolve mixed sentiments. Not only does it train the model, but it also pre-computes and caches dependency graphs for speed.

First, ensure you have the SpaCy model installed:
```bash
python -m spacy download en_core_web_sm
```

Then run the training script:
```bash
python src/models/msr_resolver_roberta.py --project_dir .
```

#### Step 3: View Results

- Model checkpoints will be saved in `outputs/checkpoints/`
- Training logs will be in `outputs/logs/`
- Predictions and reports will be in `outputs/reports/`

### Troubleshooting

#### Issue: `ModuleNotFoundError`

**Solution:** Make sure your virtual environment is activated and all dependencies are installed:

```bash
pip install -r requirements.txt
```

#### Issue: CUDA/GPU errors

**Solution:** If you don't have a GPU, the models will automatically run on CPU. For GPU support, verify your PyTorch installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it returns `False`, reinstall PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
```

#### Issue: Out of Memory

**Solution:** Reduce the batch size in the training configuration files located in `configs/`.

#### Issue: Data files not found

**Solution:** Ensure your dataset is placed in `data/raw/` and the paths in `configs/data_config.yaml` are correct.

### Deactivating the Virtual Environment

When you're done working on the project:

```bash
deactivate
```

## Project Workflow

1. **Data Pipeline**: Validate → Clean → Split
2. **Model Training**: Run baseline models (DistilBERT/RoBERTa)
3. **Evaluation**: Check outputs in `outputs/reports/`
4. **Analysis**: Review `baseline_analysis.md` for performance metrics

## Next Steps

After successfully training the baseline models:

1.  **Analyze Performance**:
    - Check `outputs/reports/roberta_metrics.txt` for classification reports per aspect.
    - Review `outputs/reports/roberta_predictions.csv` to see model predictions on the validation set.

2.  **Experiment**:
    - Try different hyperparameters in the scripts or config files.
    - Implement additional models or advanced techniques (e.g., handling class imbalance more explicitly).

3.  **Explainability**:
    - Explore the `src/xai/` directory and run explainability scripts (if available) to understand model decisions.

## Contributing

Please ensure all code follows the project structure and includes appropriate documentation.

## License

[Specify your license here]
