# Clearview: Multi-aspect Sentiment Analysis with MSR

An advanced Aspect-Based Sentiment Analysis (ABSA) system with Multi-aspect Sentiment Resolution (MSR) for cosmetic product reviews.

## Project Overview

Clearview addresses the challenge of analyzing customer reviews with conflicting sentiments across multiple product aspects (e.g., "The texture is good but the price is too high").

**Key Innovation**: MSR mechanism resolves sentiment conflicts across 7 product aspects while maintaining aspect-specific accuracy.

## Project Structure

This repository contains two main components:

### 🔬 ML Research (`ml-research/`)

Python-based machine learning research codebase:
- RoBERTa-based EAGLE architecture
- MSR conflict resolution mechanism
- Comprehensive evaluation suite with XAI
- 8-configuration ablation study

**Quick Start**: See [`ml-research/README.md`](ml-research/README.md)

### 🌐 Website (`website/`)

Next.js website for interactive demonstrations:
- Real-time sentiment analysis
- Multi-aspect visualization
- XAI explanations (Integrated Gradients)

**Quick Start**: See [`website/README.md`](website/README.md)

### 📚 Documentation (`docs/`)

Shared documentation and research logs:
- Experiment logs
- Research notes
- Design decisions

## Performance Highlights

| Metric | Baseline | EAGLE+MSR | Gain |
|:-------|:---------|:----------|:-----|
| Macro-F1 | 0.6953 | 0.7241 | +2.88% |
| MSR Error Reduction | 0 | 50 | +50 fixes |

**Best Improvements**:
- Price aspect: +9.57% F1
- Packing aspect: +15.81% F1

## Key Features

✅ **4-Class ABSA**: Negative/Neutral/Positive/None per aspect
✅ **MSR Mechanism**: Conflict-aware sentiment resolution  
✅ **7 Product Aspects**: Texture, price, smell, colour, shipping, packing, staying power  
✅ **XAI Suite**: Integrated Gradients, LIME, SHAP  
✅ **Production-Ready**: End-to-end pipeline with GPU acceleration  

## Technology Stack

**ML Research**:
- PyTorch 2.x
- Transformers (HuggingFace)
- RoBERTa-base (125M parameters)
- Captum, SHAP, LIME for XAI

**Website**:
- Next.js (React + TypeScript)
- Tailwind CSS
- shadcn/ui components

## Getting Started

### ML Research

```bash
cd ml-research
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run training
python src/models/train_roberta_improved.py --use_synthetic --use_sampler --msr_strength 0.3
```

### Website

```bash
cd website
pnpm install
pnpm dev  # http://localhost:3000
```

## Documentation

- **Technical Methodology**: [`ml-research/METHODOLOGY.md`](ml-research/METHODOLOGY.md)
- **4-Class Migration**: [`ml-research/MIGRATION_NOTE.md`](ml-research/MIGRATION_NOTE.md)
- **ML README**: [`ml-research/README.md`](ml-research/README.md)
- **Website README**: [`website/README.md`](website/README.md)

## Repository Structure

```
Clearview/
├── ml-research/       # ML/AI research code (Python)
│   ├── src/          # Model code, evaluation, XAI
│   ├── outputs/      # Results, checkpoints, reports
│   ├── data/         # Dataset splits
│   └── ...
├── website/          # Frontend (Next.js/TypeScript)
│   ├── app/          # Routes
│   ├── components/   # UI components
│   └── ...
├── docs/             # Shared documentation
└── README.md         # This file
```

## Citation

If you use this code in your research, please cite:

```
[Your thesis citation details]
```

## License

[Specify if applicable]

## Contact

[Your contact information]

---

**Authors**: [Your Name/Team]  
**Institution**: [Your University]  
**Year**: 2026
