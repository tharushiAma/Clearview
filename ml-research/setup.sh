#!/bin/bash

# Setup script for Multi-Aspect Sentiment Analysis Project

echo "=========================================="
echo "Setting up Cosmetic Sentiment Analysis"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "\n1. Creating virtual environment..."
python3 -m venv venv
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created"
else
    echo "✗ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo -e "\n2. Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo -e "\n3. Upgrading pip..."
pip install --upgrade pip
echo "✓ Pip upgraded"

# Install requirements
echo -e "\n4. Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Download spaCy model
echo -e "\n5. Downloading spaCy model for dependency parsing..."
python -m spacy download en_core_web_sm
if [ $? -eq 0 ]; then
    echo "✓ spaCy model downloaded"
else
    echo "⚠ spaCy model download failed (optional for non-GCN mode)"
fi

# Create necessary directories
echo -e "\n6. Creating project directories..."
mkdir -p data
mkdir -p outputs
mkdir -p notebooks
mkdir -p experiments
echo "✓ Directories created"

# Check if data files exist
echo -e "\n7. Checking for data files..."
if [ -f "data/train.csv" ] && [ -f "data/val.csv" ] && [ -f "data/test.csv" ]; then
    echo "✓ Data files found"
else
    echo "⚠ Data files not found. Please copy train.csv, val.csv, test.csv to data/ directory"
    echo "  Expected files:"
    echo "    - data/train.csv"
    echo "    - data/val.csv"
    echo "    - data/test.csv"
fi

# Check CUDA availability
echo -e "\n8. Checking CUDA availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    print("⚠ CUDA not available. Training will use CPU (slower)")
EOF

echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your CSV files to data/ directory"
echo "2. Review and edit configs/config.yaml"
echo "3. Run: python train.py --config configs/config.yaml"
echo ""
echo "For more information, see README.md"
echo ""
echo "To activate the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""
