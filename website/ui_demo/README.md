# ClearView UI Demo

This folder contains the UI Demo for the ClearView project, featuring a Next.js frontend and a FastAPI backend.

## Prerequisites

- Node.js (v18+)
- Python (3.9+) with the project virtual environment active.
- `pip install fastapi uvicorn pydantic pandas` (if not already installed)

## Setup

1. Copy `.env.example` to `.env.local` in this directory:
   ```bash
   cp .env.example .env.local
   ```

2. Ensure your model checkpoint exists. By default, it looks for:
   `../../outputs/exp_b_msr_fixed_er/best_model.pt`
   You can override this by setting `CKPT_PATH` in the environment variables for the backend.

## Running the Demo

Use the helper script to start both services:

**Windows (PowerShell):**
```powershell
./run_all.ps1
```

**Manual Start:**

Backend (Terminal 1):
```bash
python -m uvicorn backend_api.main:app --reload --port 8000
```

Frontend (Terminal 2):
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Backend API

- `POST /api/predict`: Get predictions with before/after MSR.
- `POST /api/explain`: Get XAI attribution data.
- `GET /api/metrics`: Get evaluation metrics from `outputs/`.
- `GET /api/logs`: Get recent logs.
