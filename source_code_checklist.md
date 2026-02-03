# Essential Project Source Code Files for Sharing

## 1. Machine Learning (Core Logic)
These files contain your custom research logic, model definitions, and XAI implementation.

- **`ml-research/src/model/`**: All files here (Model architecture).
    - `roberta_hierarchical.py` (or similar main model definition)
- **`ml-research/src/xai/`**: All files here (Explainability logic).
    - `Explainable.py` (Core XAI logic for MSR and attributes)
- **`ml-research/src/data/`**: (Optional) Data loading scripts if they contain custom logic.

## 2. Backend (Python/FastAPI)
The bridge between your ML model and the web interface.

- **`website/backend_server.py`**: The API server handling requests and model inference.
- **`website/requirements.txt`** (Create this if missing): List of dependencies (`pip freeze > requirements.txt`).
- **`website/run_backend.ps1`**: Script to start the backend.

## 3. Frontend (Next.js/React)
The user interface code.

- **`website/components/ClearViewDemo.tsx`**: The main UI component handling Predict/Explain interaction.
- **`website/app/page.tsx`**: The main page rendering the component.
- **`website/lib/api.ts`**: API functions calling the backend.
- **`website/package.json`**: Frontend dependencies.
- **`website/run_all.ps1`**: Main startup script.

## 4. Configuration
- **`website/.env.example`**: Environment variables (Rename to `.env` for sharing).

---
**📦 Notes for Sharing:**
- Do **NOT** share `node_modules`, `.next`, `__pycache__`, or huge model weights (`.pt` files) unless specifically asked (they are too large).
- Share the `README.md` if you have one explaining how to run usage.
