#!/usr/bin/env python3
"""
EAGLE V2 GPU Training Launcher
Automatically uses GPU if available, optimized for full dataset training.
"""

import subprocess
import sys
import os

print("=" * 80)
print("EAGLE V2 - FULL DATASET GPU TRAINING")
print("=" * 80)
print()

# Check CUDA availability
try:
    import torch
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Detected: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  NO GPU DETECTED")
        print("   PyTorch version:", torch.__version__)
        print("   Training will run on CPU (VERY SLOW - not recommended)")
        print()
        print("   To enable GPU:")
        print("   1. Ensure NVIDIA drivers are installed")
        print("   2. Run PowerShell/CMD as Administrator")
        print("   3. Check GPU: nvidia-smi")
        print()
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            sys.exit(0)
except ImportError:
    print("❌ PyTorch not installed!")
    sys.exit(1)

print()
print("Dataset Configuration:")
print("  • Training: train_augmented.parquet (9,422 rows - FULL)")
print("  • Validation: val.parquet (1,324 rows)")
print()
print("Model Configuration:")
print("  • Epochs: 10")
print("  • Batch size: 16 (GPU) / 8 (CPU)")
print("  • Learning rate: 2e-5")
print("  • Uncertainty heads: ENABLED")
print("  • Feature routing: ENABLED")
print()
print("Expected Results:")
print("  • Price F1: 0.33 → 0.50+")
print("  • Packing F1: 0.54 → 0.60+")
print("  • Overall F1: 0.6469 → 0.70+")
print()
print("=" * 80)

# Determine batch size based on device
batch_size = 16 if cuda_available else 8

print(f"\nStarting training on {'GPU' if cuda_available else 'CPU'}...")
print(f"Batch size: {batch_size}")
print()

# Build command
cmd = [
    sys.executable,
    "src/models/train_eagle_v2.py",
    "--use_preaugmented",  # Use full pre-augmented dataset
    "--epochs", "10",
    "--batch_size", str(batch_size),
    "--lr", "2e-5",
    "--use_uncertainty",
    "--use_feature_routing",
    "--eval_every", "2"
]

print("Command:", " ".join(cmd))
print()
print("=" * 80)
print()

# Run training
try:
    subprocess.run(cmd, check=True)
    
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Results saved to:")
    print("  • Metrics: outputs/reports/eagle_v2_epoch*_metrics.txt")
    print("  • Best model: outputs/checkpoints/eagle_v2_best.pt")
    print()
    print("Next steps:")
    print("  1. Compare results: python src/models/compare_eagle_models.py")
    print("  2. Review metrics files in outputs/reports/")
    
except subprocess.CalledProcessError as e:
    print()
    print("=" * 80)
    print("❌ TRAINING FAILED")
    print("=" * 80)
    print(f"Error code: {e.returncode}")
    print()
    print("Check the error messages above for details.")
    sys.exit(1)
except KeyboardInterrupt:
    print()
    print("=" * 80)
    print("⚠️  TRAINING INTERRUPTED")
    print("=" * 80)
    print("Training was stopped by user.")
    sys.exit(1)
