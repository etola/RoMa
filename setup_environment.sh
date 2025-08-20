#!/bin/bash

# RoMa Environment Setup Script
# This script creates a new conda environment and installs all required packages

set -e  # Exit on any error

# Configuration
ENV_NAME="roma_env"
PYTHON_VERSION="3.9"

echo "🚀 Setting up RoMa environment..."
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "❌ Aborted. Please choose a different environment name or remove the existing one."
        exit 1
    fi
fi

# Create new conda environment
echo "🔨 Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install conda packages first (better compatibility for some packages)
echo "📦 Installing core packages from conda-forge..."
conda install -c conda-forge -y \
    opencv \
    matplotlib \
    h5py \
    tqdm

# Install PyTorch with CUDA 12.8 support
echo "📦 Installing PyTorch with CUDA 12.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining packages with pip
echo "📦 Installing additional packages with pip..."
pip install \
    einops \
    kornia \
    albumentations \
    loguru \
    wandb \
    timm \
    poselib \
    pycolmap \
    open3d

# Install the project in editable mode
echo "🔧 Installing RoMa package in editable mode..."
pip install -e .

# Optional: Install xformers if requested
echo ""
read -p "🤔 Do you want to install xformers for memory-efficient attention? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing xformers with CUDA 12.8 support..."
    pip install xformers --index-url https://download.pytorch.org/whl/cu128
fi

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "📋 To use the environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "📋 To deactivate:"
echo "   conda deactivate"
echo ""
echo "📋 To remove the environment (if needed):"
echo "   conda env remove -n $ENV_NAME"
echo ""
echo "🎉 Happy coding with RoMa!" 