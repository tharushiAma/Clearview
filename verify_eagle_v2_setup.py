#!/usr/bin/env python3
"""
EAGLE V2 Setup Verification Script
Checks all dependencies and data files before training.
"""

import sys
import os
from pathlib import Path

def check_imports():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'spacy': 'SpaCy',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm'
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(module)
    
    return missing


def check_spacy_model():
    """Check if SpaCy English model is downloaded."""
    print("\nChecking SpaCy model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("  ✓ en_core_web_sm")
        return True
    except:
        print("  ✗ en_core_web_sm (not downloaded)")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available: {device_name}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("  ⚠ CUDA not available (will use CPU - training will be slow)")
            return False
    except:
        print("  ✗ Cannot check CUDA")
        return False


def check_data_files(project_dir):
    """Check if required data files exist."""
    print("\nChecking data files...")
    
    data_dir = Path(project_dir) / "data" / "splits"
    required_files = ['train.parquet', 'val.parquet']
    
    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {filename} (missing)")
            all_exist = False
    
    return all_exist


def check_model_files(project_dir):
    """Check if EAGLE V2 files exist."""
    print("\nChecking EAGLE V2 files...")
    
    src_dir = Path(project_dir) / "src" / "models"
    required_files = [
        'eagle_v2_implementation.py',
        'train_eagle_v2.py',
        'compare_eagle_models.py',
        'EAGLE_V2_README.md'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = src_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (missing)")
            all_exist = False
    
    return all_exist


def main():
    """Main verification function."""
    print("="*80)
    print("EAGLE V2 Setup Verification")
    print("="*80 + "\n")
    
    project_dir = Path(__file__).parent
    
    # Check all components
    missing_imports = check_imports()
    spacy_ok = check_spacy_model()
    cuda_ok = check_cuda()
    data_ok = check_data_files(project_dir)
    files_ok = check_model_files(project_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_good = True
    
    if missing_imports:
        print(f"✗ Missing packages: {', '.join(missing_imports)}")
        print(f"  Install with: pip install {' '.join(missing_imports)}")
        all_good = False
    else:
        print("✓ All packages installed")
    
    if not spacy_ok:
        print("✗ SpaCy model not downloaded")
        print("  Download with: python -m spacy download en_core_web_sm")
        all_good = False
    else:
        print("✓ SpaCy model ready")
    
    if not cuda_ok:
        print("⚠ CUDA not available (training will be slower on CPU)")
    else:
        print("✓ CUDA ready")
    
    if not data_ok:
        print("✗ Data files missing")
        all_good = False
    else:
        print("✓ Data files ready")
    
    if not files_ok:
        print("✗ EAGLE V2 files missing")
        all_good = False
    else:
        print("✓ EAGLE V2 files ready")
    
    print("\n" + "="*80)
    
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to train!")
        print("\nTo start training:")
        print("  python src/models/train_eagle_v2.py --augment --epochs 10")
        return 0
    else:
        print("❌ SETUP INCOMPLETE - Please fix the issues above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
