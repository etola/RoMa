# RoMa Environment Setup

This directory contains scripts to automatically set up a conda environment for the RoMa project with all required dependencies.

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) must be installed
- Git (to clone this repository)

## Quick Start

### For Linux/macOS users:

```bash
# Navigate to the RoMa directory
cd /path/to/RoMa

# Run the setup script
./setup_environment.sh
```

### For Windows users:

```cmd
# Navigate to the RoMa directory
cd C:\path\to\RoMa

# Run the setup script
setup_environment.bat
```

## What the scripts do

1. **Create a new conda environment** named `roma_env` with Python 3.9
2. **Install core packages** from conda-forge for better compatibility:
   - OpenCV
   - Matplotlib
   - H5PY
   - TQDM

3. **Install PyTorch with CUDA 12.8 support** from the official PyTorch CUDA index:
   - PyTorch, TorchVision, TorchAudio

4. **Install additional packages** via pip:
   - einops
   - kornia
   - albumentations
   - loguru
   - wandb
   - timm
   - poselib
   - pycolmap (for COLMAP integration)
   - open3d (for 3D point cloud processing)

5. **Install the RoMa package** in editable mode using `pip install -e .`
6. **Optionally install xformers** with CUDA 12.8 support for memory-efficient attention (user choice)

## After Installation

Once the setup is complete, activate the environment:

```bash
conda activate roma_env
```

You can now run any of the demo scripts or experiments:

```bash
# Example: Run a demo
python demo/demo_match.py

# Example: Run experiments
python experiments/eval_roma_outdoor.py
```

## Managing the Environment

### Deactivate the environment:
```bash
conda deactivate
```

### Remove the environment (if needed):
```bash
conda env remove -n roma_env
```

### List all conda environments:
```bash
conda env list
```

## Troubleshooting

- **"conda: command not found"**: Make sure Anaconda/Miniconda is installed and added to your PATH
- **Permission denied (Linux/macOS)**: Run `chmod +x setup_environment.sh` to make the script executable
- **Environment already exists**: The script will prompt you to remove the existing environment
- **Package installation fails**: Try updating conda with `conda update conda` and running the script again

## Customization

You can modify the environment name and Python version by editing the configuration variables at the top of the setup scripts:

- `ENV_NAME`: Change the environment name (default: "roma_env")
- `PYTHON_VERSION`: Change the Python version (default: "3.9")

## GPU Support

The scripts now install **PyTorch with CUDA 12.8 support by default**. This provides GPU acceleration out of the box if you have:

- An NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- NVIDIA drivers supporting CUDA 12.8 or compatible

### For different CUDA versions:

If you need a different CUDA version, modify the `--index-url` in the setup scripts:

- **CUDA 11.8**: `https://download.pytorch.org/whl/cu118`
- **CUDA 12.1**: `https://download.pytorch.org/whl/cu121`  
- **CPU only**: `https://download.pytorch.org/whl/cpu`

### Verify GPU installation:

After setup, test GPU support:
```bash
conda activate roma_env
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
``` 