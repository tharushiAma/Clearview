# EAGLE V2 Training Quick Start Script
# Run this to start training with recommended settings

"""
This script will:
1. Train EAGLE V2 for 10 epochs
2. Apply data augmentation (price, packing, neutral)
3. Use uncertainty heads and feature routing
4. Save best model to outputs/checkpoints/eagle_v2_best.pt
5. Generate metrics files for comparison

Estimated time: 2-4 hours on GPU, 8-12 hours on CPU
"""

import subprocess
import sys

print("=" * 80)
print("EAGLE V2 TRAINING - QUICK START")
print("=" * 80)
print()
print("Configuration:")
print("  • Epochs: 10")
print("  • Batch size: 16")
print("  • Learning rate: 2e-5")
print("  • Data augmentation: ENABLED")
print("  • Uncertainty heads: ENABLED")
print("  • Feature routing: ENABLED")
print()
print("Expected improvements:")
print("  • Price F1: 0.33 → 0.50+")
print("  • Packing F1: 0.54 → 0.60+")
print("  • Neutral F1: 0.40 → 0.50+")
print("  • Overall F1: 0.6469 → 0.70+")
print()
print("=" * 80)

response = input("\nStart training? (y/n): ")

if response.lower() == 'y':
    print("\nStarting training...")
    print("This may take 2-4 hours on GPU.\n")
    
    # Run training script
    cmd = [
        sys.executable,  # Python interpreter
        "src/models/train_eagle_v2.py",
        "--augment",
        "--epochs", "10",
        "--batch_size", "16",
        "--lr", "2e-5",
        "--use_uncertainty",
        "--use_feature_routing"
    ]
    
    subprocess.run(cmd)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Compare results: python src/models/compare_eagle_models.py")
    print("2. Check metrics: outputs/reports/eagle_v2_epoch*_metrics.txt")
    print("3. View best model: outputs/checkpoints/eagle_v2_best.pt")
    
else:
    print("\nTraining cancelled.")
    print("\nTo train manually with custom settings:")
    print("  python src/models/train_eagle_v2.py --help")
