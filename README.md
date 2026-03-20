# ClearView — Multi-Aspect Sentiment Analysis for Cosmetic Reviews

My final year project. Classifies sentiment across 7 aspects of cosmetic product reviews (colour, smell, texture, price, stayingpower, packing, shipping) while handling severe class imbalance and resolving conflicting opinions within the same review.

The core idea: a review like "I love the colour but the smell is awful" should be classified as `colour: positive, smell: negative` — not just "mixed". That sounds obvious but getting a model to actually do this correctly requires more than just a vanilla BERT fine-tune.

## What's in here

**`ml-research/`** — the actual ML work: RoBERTa + aspect attention + dependency GCN, hybrid loss functions, ablation experiments, XAI methods. See [`ml-research/README.md`](ml-research/README.md) for setup.

**`website/`** — a Next.js frontend + FastAPI backend that lets you type in a review and see the predictions and explanations. See [`DEPLOYMENT.md`](DEPLOYMENT.md) to host it.

**`thesis/`** — thesis chapters as markdown files.

**`docs/`** — research notes and data analysis reports.

## Quick numbers

| Metric | Score |
| --- | --- |
| Overall Accuracy | 92.14% |
| Overall Macro-F1 | 0.7981 |
| Weighted F1 | 0.9242 |
| MCC | 0.7842 |

Best gains from the full model (vs plain RoBERTa baseline): price aspect +9.57% F1, packing +15.81% F1. These were the worst-performing aspects due to extreme class imbalance (174:1 and 185:1 positive-to-negative ratio before augmentation).

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
./run_all.ps1
```

Frontend at <http://localhost:3000>, backend at <http://localhost:8000>.

## Stack

ML: PyTorch, HuggingFace Transformers, RoBERTa-base, Captum, SHAP, LIME, spaCy

Website: Next.js, TypeScript, Tailwind CSS, FastAPI, Uvicorn

## Structure

```
Clearview/
├── ml-research/      # ML code, notebooks, trained model
├── website/
│   ├── frontend/     # Next.js app
│   └── backend/      # FastAPI server
├── thesis/           # Thesis chapters
├── docs/             # Research notes
└── DEPLOYMENT.md     # Hosting guide
```
