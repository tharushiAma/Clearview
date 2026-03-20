# ClearView — Full Deployment Guide

This guide walks through hosting the complete ClearView stack from scratch:

- **Frontend** (Next.js) → Vercel (free)
- **Backend** (FastAPI) → Railway (from ~$5/mo)
- **Model file** (505 MB `.pt`) → HuggingFace Hub (free)

> **Why not just upload the backend?** The backend alone is useless — it needs the 505 MB
> model file and the `ml-research/` Python modules at runtime. The strategy is:
> upload the model once to HuggingFace Hub, then deploy the full repo to Railway and let
> the server download the model on first start.

---

## Architecture Overview

```
Browser
  │
  ▼
Vercel (Next.js frontend)
  │  BACKEND_URL env var
  ▼
Railway (FastAPI backend)   ←── reads model from ──→  HuggingFace Hub
  │  imports at runtime
  ▼
ml-research/inference_bridge/  →  outputs/.../inference.py  →  src/models/model.py
```

---

## Part 1 — Upload the Model to HuggingFace Hub

The model checkpoint is 505 MB — too large for GitHub. HuggingFace Hub hosts it for free.

### 1.1 Create a HuggingFace account

Go to [huggingface.co](https://huggingface.co) → Sign up → Confirm email.

### 1.2 Create a new model repository

1. Click your profile icon → **New Model**
2. Repository name: `clearview-model`
3. Visibility: **Private** (recommended) or Public
4. Click **Create Model**

### 1.3 Get your HuggingFace access token

1. Go to **Settings** → **Access Tokens** → **New token**
2. Name: `clearview-deploy`
3. Role: **Write**
4. Copy the token — it looks like `hf_xxxxxxxxxxxxxxxxxxxxxx`

### 1.4 Upload the model file

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='ml-research/outputs/cosmetic_sentiment_v1/best_model.pt',
    path_in_repo='best_model.pt',
    repo_id='YOUR_USERNAME/clearview-model',
    token='hf_xxxxxxxxxxxxxxxxxxxx',
)
print('Upload complete')
"
```

Replace `YOUR_USERNAME` and `hf_xxxxxxxxxxxxxxxxxxxx` with your actual values.

Upload takes 5–15 minutes depending on connection speed.

---

## Part 2 — Prepare the Backend for Production

### 2.1 Add a model download script

Create `website/backend/download_model.py`:

```python
"""
Download best_model.pt from HuggingFace Hub on first run.
Called automatically by the start command if the model is missing.
"""
import os
import sys

CKPT_PATH = os.environ.get("CKPT_PATH", "/app/model/best_model.pt")
HF_REPO   = os.environ.get("HF_REPO_ID", "")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")

if os.path.exists(CKPT_PATH):
    print(f"[MODEL] Checkpoint already exists: {CKPT_PATH}")
    sys.exit(0)

if not HF_REPO:
    print("[MODEL] HF_REPO_ID not set — skipping download")
    sys.exit(0)

print(f"[MODEL] Downloading from {HF_REPO} → {CKPT_PATH} ...")
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id=HF_REPO,
    filename="best_model.pt",
    token=HF_TOKEN or None,
    local_dir=os.path.dirname(CKPT_PATH),
)
print(f"[MODEL] Downloaded to: {path}")
```

### 2.2 Update backend requirements

Add `huggingface_hub` to `website/backend/requirements.txt`:

```
fastapi
uvicorn
python-multipart
torch
transformers
captum
shap
scikit-learn
pandas
numpy
tqdm
huggingface_hub
spacy
torch-geometric
```

### 2.3 Set the start command

The Railway/Render start command will be:

```bash
python website/backend/download_model.py && uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

This downloads the model if missing, then starts the API.

---

## Part 3 — Deploy Backend to Railway

### 3.1 Create a Railway account

Go to [railway.app](https://railway.app) → Sign in with GitHub.

### 3.2 Create a new project

1. Click **New Project** → **Deploy from GitHub repo**
2. Select your `Clearview` repository
3. Railway will detect it automatically

### 3.3 Configure the service

In Railway dashboard → your service → **Settings**:

| Setting | Value |
| --- | --- |
| Root Directory | `website` |
| Start Command | `python backend/download_model.py && uvicorn backend.main:app --host 0.0.0.0 --port 8000` |
| Watch Paths | `website/backend/**` |

### 3.4 Set environment variables

Go to your service → **Variables** → add each one:

```
# Required
CKPT_PATH         = /app/model/best_model.pt
HF_REPO_ID        = YOUR_USERNAME/clearview-model
HF_TOKEN          = hf_xxxxxxxxxxxxxxxxxxxx

# Set this after your Vercel frontend is deployed (Step 4)
ALLOWED_ORIGINS   = https://your-app.vercel.app

# Optional — only if using legacy model
# LEGACY_CKPT_PATH = /app/model/gold_msr_best.pt
```

> Railway stores env vars securely — never put secrets in code or commit them.

### 3.5 Add a volume for the model (important)

Without a persistent volume, Railway re-downloads the 505 MB model on every deploy.

1. Railway dashboard → **New** → **Volume**
2. Attach it to your service
3. Mount path: `/app/model`

Now the model downloads once and persists across deploys.

### 3.6 Deploy

Click **Deploy**. Watch the logs — you should see:

```
[MODEL] Downloading from YOUR_USERNAME/clearview-model → /app/model/best_model.pt ...
[MODEL] Downloaded to: /app/model/best_model.pt
[LOAD] Loading trained model adapter from: /app/model/best_model.pt
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3.7 Get your backend URL

Railway gives you a public URL like:
```
https://clearview-production-xxxx.up.railway.app
```

Copy this — you need it for the frontend.

---

## Part 4 — Deploy Frontend to Vercel

### 4.1 Create a Vercel account

Go to [vercel.com](https://vercel.com) → Sign in with GitHub.

### 4.2 Import the project

1. Click **Add New** → **Project**
2. Select your `Clearview` repository
3. Set **Root Directory** to `website/frontend`
4. Framework Preset: **Next.js** (auto-detected)

### 4.3 Set environment variables

In Vercel → your project → **Settings** → **Environment Variables**:

```
# The URL of your Railway backend (from Step 3.7)
# Used by Next.js server-side API routes to proxy requests
BACKEND_URL = https://clearview-production-xxxx.up.railway.app

# The same URL, exposed to the browser
# Used if any client-side code calls the backend directly
NEXT_PUBLIC_API_URL = https://clearview-production-xxxx.up.railway.app
```

Set both to **Production**, **Preview**, and **Development** environments.

### 4.4 Deploy

Click **Deploy**. Vercel builds and deploys automatically.

Your frontend will be live at:
```
https://clearview-xxxx.vercel.app
```

---

## Part 5 — Update CORS After Both Are Deployed

Now that you have both URLs, go back to Railway and update `ALLOWED_ORIGINS`:

```
ALLOWED_ORIGINS = https://clearview-xxxx.vercel.app
```

If you have a custom domain, add it too (comma-separated):

```
ALLOWED_ORIGINS = https://clearview-xxxx.vercel.app,https://yourdomain.com
```

Railway automatically redeploys when you change env vars.

---

## Part 6 — Custom Domain (Optional)

### Frontend (Vercel)

1. Vercel → your project → **Settings** → **Domains**
2. Add your domain (e.g. `clearview.yourdomain.com`)
3. Add the CNAME record shown to your DNS provider
4. Vercel issues an SSL certificate automatically

### Backend (Railway)

1. Railway → your service → **Settings** → **Domains**
2. Add your domain (e.g. `api.clearview.yourdomain.com`)
3. Add the CNAME record to your DNS provider
4. Update `BACKEND_URL` on Vercel and `ALLOWED_ORIGINS` on Railway to use the new domain

---

## Environment Variables — Full Reference

### Backend (set in Railway)

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `CKPT_PATH` | Yes | `../ml-research/outputs/.../best_model.pt` | Absolute path to model checkpoint on server |
| `HF_REPO_ID` | Yes (for auto-download) | — | HuggingFace repo, e.g. `username/clearview-model` |
| `HF_TOKEN` | Yes if repo is private | — | HuggingFace access token (`hf_xxx...`) |
| `ALLOWED_ORIGINS` | Yes | `http://localhost:3000` | Comma-separated list of allowed frontend URLs |
| `LEGACY_CKPT_PATH` | No | — | Path to legacy model checkpoint (if applicable) |

### Frontend (set in Vercel)

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `BACKEND_URL` | Yes | `http://localhost:8000` | Backend URL used by Next.js server-side API routes |
| `NEXT_PUBLIC_API_URL` | Yes | `http://localhost:8000` | Backend URL exposed to the browser |

---

## Local Development .env Files

Create these files locally — **never commit them to Git**.

`website/frontend/.env.local`:

```env
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

`website/backend/.env` (optional — backend reads env vars directly):

```env
CKPT_PATH=C:/Users/yourname/Desktop/Clearview/ml-research/outputs/cosmetic_sentiment_v1/best_model.pt
ALLOWED_ORIGINS=http://localhost:3000
```

---

## Verify Everything Works

After deploying, test each layer:

### 1. Backend health check

```bash
curl https://clearview-production-xxxx.up.railway.app/
```

Expected response:

```json
{
  "status": "running",
  "service": "ClearView ML Backend",
  "models_loaded": 1,
  "explainers_loaded": 0
}
```

### 2. Backend prediction

```bash
curl -X POST https://clearview-production-xxxx.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The colour is beautiful but the smell is awful."}'
```

Expected: JSON with `predictions` array and `conflict_prob`.

### 3. Frontend

Open your Vercel URL in a browser. The demo page should load and predictions should work.

---

## Troubleshooting

| Problem | Cause | Fix |
| --- | --- | --- |
| Backend returns 503 | Model not loaded yet | Check Railway logs for download progress |
| CORS error in browser | `ALLOWED_ORIGINS` not set | Add your Vercel URL to Railway `ALLOWED_ORIGINS` |
| `ModuleNotFoundError: inference_bridge` | Wrong root directory | Set Railway root to `website`, not `website/backend` |
| Model downloads on every deploy | No volume mounted | Add a Railway volume mounted at `/app/model` |
| Vercel build fails | Missing env vars | Check `BACKEND_URL` is set in Vercel settings |
| `torch` install timeout on Railway | Slow pip install | Add `torch` to a pre-built Docker image or use Railway's nixpacks |
| Frontend calls fail (404 on `/api/*`) | `BACKEND_URL` wrong | Ensure it points to Railway URL without trailing slash |

---

## Redeployment

### When you update backend code

Push to GitHub → Railway redeploys automatically (model is cached in the volume, no re-download).

### When you update frontend code

Push to GitHub → Vercel redeploys automatically.

### When you retrain the model

1. Upload new `best_model.pt` to HuggingFace Hub (overwrites the old one)
2. Delete the file from the Railway volume: SSH into Railway → `rm /app/model/best_model.pt`
3. Trigger a Railway redeploy → model re-downloads automatically
