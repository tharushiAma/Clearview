# GPU Usage Guide for Clearview

If you are running the training scripts (`train_eagle.py` or `msr_resolver_roberta_focal_loss.py`) and don't see GPU activity in Task Manager or `nvidia-smi`, don't worry! Here is why:

## Execution Phases

### 1. Stage 1: Preprocessing (CPU Only)

- **What's happening**: The script is using SpaCy to perform "Dependency Parsing" on every sentence in your dataset (~10,000 samples).
- **GPU Usage**: **0%**.
- **CPU Usage**: **High** (Multi-core if possible, or single-core if not).
- **Time**: Can take 30-60 minutes depending on your CPU.
- **Why**: This is a pure NLP task that doesn't benefit from GPU acceleration. Once done, it saves a `.pkl` file to `outputs/cache`, so next time it will be instant.

### 2. Stage 2: Training (GPU Accelerated)

- **What's happening**: The actual Neural Network is training.
- **GPU Usage**: **High (20% - 90%+)**.
- **VRAM Usage**: Will increase as the model and data are loaded.
- **Time**: Faster than CPU training.

## How to Verify GPU Usage

1. **Check the logs**: Look for the message `Successfully connected to GPU: NVIDIA GeForce RTX 4060 Laptop GPU`.
2. **Monitor Memory**: Run `nvidia-smi` and look for the memory allocated by the `python.exe` process.
3. **Use --head for Testing**: If you want to verify it works without waiting, run with `--head 100` to only process 100 samples.

```powershell
python src/models/train_eagle.py --project_dir . --epochs 1 --batch_size 2 --head 100
```
