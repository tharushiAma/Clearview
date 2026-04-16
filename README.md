# ClearView — Class imbalance handled Multi Aspect mixed sentiment resolution with Explainability in cosmetic domain

My final year project. Classifies sentiment across 7 aspects of cosmetic product reviews (colour, smell, texture, price, stayingpower, packing, shipping) while handling severe class imbalance and capturing aspect-specific conflicting opinions within the same review through aspect-wise prediction and conflict analysis.

The core idea: a review like "I love the colour but the smell is awful" should be classified as `colour: positive, smell: negative` — not just "mixed". That sounds obvious but getting a model to actually do this correctly requires more than just a vanilla BERT fine-tune.

## What's in here

**`ml-research/`** — the core ML pipeline: data preprocessing, RoBERTa + aspect attention + dependency GCN, hybrid loss functions, ablation experiments, inference bridge and XAI methods (SHAP, LIME, Integrated Gradients).

**`website/`** — a Next.js frontend + FastAPI backend that lets you type in a review to see aspect-level sentiment predictions alongside detailed XAI visual explanations.

## Quick numbers

| Metric | Score |
| --- | --- |
| Overall Accuracy | 92.36% |
| Overall Macro-F1 | 0.7856 |
| Weighted F1 | 0.9221 |
| MCC | 0.7838 |

Best gains from the full model (vs plain RoBERTa baseline): price aspect +6.85% F1, packing +27.99% F1. These were the worst-performing aspects due to extreme class imbalance (174:1 and 185:1 positive-to-negative ratio before augmentation).

## Running it locally

**ML research:**

```bash
cd ml-research
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter lab notebooks/
```

**Website (both frontend + backend):**

```bash
cd website

# 1. Install backend dependencies (or ensure your ML venv is active)
pip install -r backend/requirements.txt

# 2. Install frontend dependencies (requires Node.js)
cd frontend
npm install
cd ..

# 3. Start the application
./run_all.ps1
```

Frontend at <http://localhost:3000>, backend at <http://localhost:8000>.

## Stack

ML: PyTorch, HuggingFace Transformers, RoBERTa-base, Captum, SHAP, LIME, spaCy

Website: Next.js, TypeScript, Tailwind CSS, FastAPI

## Structure

```
Clearview/
├── ml-research/                  # ML core pipeline, XAI bridge, and trained models
└── website/
    ├── frontend/                 # Next.js web application
    └── backend/                  # FastAPI inference server serving XAI
```
