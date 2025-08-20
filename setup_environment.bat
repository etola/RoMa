@echo off
REM RoMa Environment Setup Script for Windows
REM This script creates a new conda environment and installs all required packages

setlocal enabledelayedexpansion

REM Configuration
set ENV_NAME=roma_env
set PYTHON_VERSION=3.9

echo 🚀 Setting up RoMa environment...
echo Environment name: %ENV_NAME%
echo Python version: %PYTHON_VERSION%
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr /B "%ENV_NAME% " >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo ⚠️  Environment '%ENV_NAME%' already exists
    set /p "choice=Do you want to remove it and create a new one? (y/N): "
    if /i "!choice!"=="y" (
        echo 🗑️  Removing existing environment...
        conda env remove -n %ENV_NAME% -y
    ) else (
        echo ❌ Aborted. Please choose a different environment name or remove the existing one.
        pause
        exit /b 1
    )
)

REM Create new conda environment
echo 🔨 Creating conda environment '%ENV_NAME%' with Python %PYTHON_VERSION%...
conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to create environment
    pause
    exit /b 1
)

REM Activate environment
echo 🔄 Activating environment...
call conda activate %ENV_NAME%

REM Install conda packages first (better compatibility for some packages)
echo 📦 Installing core packages from conda-forge...
conda install -c conda-forge -y opencv matplotlib h5py tqdm
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to install conda packages
    pause
    exit /b 1
)

REM Install PyTorch with CUDA 12.8 support
echo 📦 Installing PyTorch with CUDA 12.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to install PyTorch
    pause
    exit /b 1
)

REM Install remaining packages with pip
echo 📦 Installing additional packages with pip...
pip install einops kornia albumentations loguru wandb timm poselib pycolmap open3d
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to install pip packages
    pause
    exit /b 1
)

REM Install the project in editable mode
echo 🔧 Installing RoMa package in editable mode...
pip install -e .
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to install RoMa package
    pause
    exit /b 1
)

REM Optional: Install xformers if requested
echo.
set /p "choice=🤔 Do you want to install xformers for memory-efficient attention? (y/N): "
if /i "!choice!"=="y" (
    echo 📦 Installing xformers with CUDA 12.8 support...
    pip install xformers --index-url https://download.pytorch.org/whl/cu128
)

echo.
echo ✅ Environment setup complete!
echo.
echo 📋 To use the environment:
echo    conda activate %ENV_NAME%
echo.
echo 📋 To deactivate:
echo    conda deactivate
echo.
echo 📋 To remove the environment (if needed):
echo    conda env remove -n %ENV_NAME%
echo.
echo 🎉 Happy coding with RoMa!
echo.
pause 