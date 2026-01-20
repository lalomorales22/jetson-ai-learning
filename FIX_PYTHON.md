# Python & PyTorch Setup Fix

## Problem 1: "externally-managed-environment"

You have Python 3.14 (from Homebrew) which is too new for PyTorch.
You also have Python 3.10 (system) which PyTorch fully supports!

## Problem 2: "no kernel image is available for execution"

PyTorch from PyPI doesn't support Jetson Orin's GPU architecture (sm_87).
You need NVIDIA's Jetson-specific PyTorch wheels!

## The Solution

### 1. Install Python 3.10 venv package
```bash
sudo apt install python3.10-venv
```

### 2. Run the Jetson-specific installer
```bash
cd /home/lalomorales/Desktop/jetson-ai-learning
./install_pytorch_jetson.sh
```

This uses NVIDIA's official PyTorch builds for Jetson (not PyPI).

That's it! This will:
- Create a virtual environment with Python 3.10
- Install PyTorch with CUDA support
- Install all dependencies
- Verify everything works

## Why Python 3.10?

| Version | Status |
|---------|--------|
| Python 3.14 | Too new - PyTorch not available |
| Python 3.10 | Perfect - Fully supported by PyTorch |

## After Installation

Every time you work on this project:
```bash
cd /home/lalomorales/Desktop/jetson-ai-learning
source setup_env.sh
```

You'll see `(venv)` in your prompt showing the environment is active.

## Quick Test

After installation, verify:
```bash
source setup_env.sh
python --version        # Should show Python 3.10.x
python -c "import torch; print(torch.cuda.is_available())"  # Should show True
```

---

## Common Errors & Fixes

### "no kernel image is available for execution on the device"

**Cause**: Wrong PyTorch build (doesn't support Jetson Orin sm_87)

**Fix**: Use NVIDIA's Jetson PyTorch wheels
```bash
rm -rf venv/
./install_pytorch_jetson.sh
```

### Manual Installation (if script fails)

1. Check your JetPack version:
```bash
sudo apt-cache show nvidia-jetpack | grep Version
```

2. Visit NVIDIA's PyTorch forum:
   https://forums.developer.nvidia.com/t/pytorch-for-jetson/

3. Download the .whl for your JetPack version

4. Install manually:
```bash
source venv/bin/activate
pip install torch-*-cp310-*.whl
pip install torchvision numpy pillow opencv-python
```

## Do This Now:

1. Run: `sudo apt install python3.10-venv`
2. Run: `./install_pytorch_jetson.sh` (uses NVIDIA's wheels)
3. Run: `source setup_env.sh`
4. Test: `cd 02-ml-training && python train_classifier.py`

🚀 Let's get PyTorch running on Jetson!
