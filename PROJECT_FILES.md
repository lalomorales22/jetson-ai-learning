# Project Files Reference

## 📋 Quick Reference

**First time setup:**
```bash
sudo apt install python3.10-venv
./install_pytorch_jetson.sh
```

**Every time you work:**
```bash
source setup_env.sh
```

**Need help?**
Start with **START_HERE.md** - it has everything you need!

---

## Installation & Setup

| File | Purpose |
|------|---------|
| `install_pytorch_jetson.sh` | **ONE-TIME SETUP** - Installs PyTorch for Jetson (NVIDIA build) |
| `setup_env.sh` | **DAILY USE** - Activates CUDA paths + Python venv |

---

## Documentation

| File | Description |
|------|-------------|
| `START_HERE.md` | **📖 Main guide** - Everything in one place:<br>• Quick start (10 min)<br>• Complete walkthrough (90 min)<br>• Explanations and troubleshooting |
| `README.md` | Project overview and quick setup |
| `FIX_PYTHON.md` | Python/PyTorch troubleshooting |
| `KERNEL_IMAGE_ERROR_FIX.md` | Fix "no kernel image" GPU error |
| `NUMPY_FIX.md` | Fix NumPy version compatibility |
| `PROJECT_FILES.md` | This file - quick reference |

---

## Learning Modules

| Directory | Topic | README |
|-----------|-------|--------|
| `01-cuda-basics/` | GPU programming with CUDA | `01-cuda-basics/README.md` |
| `02-ml-training/` | Train neural networks | `02-ml-training/README.md` |
| `03-tensorrt-optimization/` | Optimize with TensorRT | `03-tensorrt-optimization/README.md` |
| `04-vision-llm-integration/` | Vision + Language AI | `04-vision-llm-integration/README.md` |
| `demos/` | Example applications | `demos/README.md` |

---

## Utilities

| File | Purpose |
|------|---------|
| `test_image_download.py` | Download test image for CUDA examples |
| `.gitignore` | Git ignore patterns |

---

## What Got Cleaned Up

We simplified the documentation structure:

### Removed (redundant):
- ❌ `install_with_python310.sh` - Used PyPI PyTorch (doesn't work on Jetson)
- ❌ `QUICKSTART.md` - Merged into START_HERE.md
- ❌ `GET_STARTED.md` - Merged into START_HERE.md
- ❌ Other failed install attempts

### Result:
✅ **One main guide** (START_HERE.md) instead of 3 confusing files
✅ **One installer** (install_pytorch_jetson.sh) that actually works
✅ Clean, simple structure

---

## File Details

### START_HERE.md - Your Main Resource

This file contains:

1. **What Makes Jetson Special** - vs Raspberry Pi, Mac, Cloud
2. **Choose Your Path** - Quick start or complete guide
3. **Quick Start Section** - 10 minutes to get running
4. **Complete Walkthrough** - 90 minute detailed guide
5. **Troubleshooting** - Common errors and fixes
6. **Next-Level Projects** - What to build next
7. **Resources** - Links and learning materials

**Just read this one file** and you'll have everything you need!

### install_pytorch_jetson.sh - The Right Installer

Why this one works:
- ✅ Uses NVIDIA's Jetson PyTorch builds (not PyPI)
- ✅ Supports Jetson Orin sm_87 architecture
- ✅ Detects JetPack 6.1 automatically
- ✅ Installs PyTorch 2.5.0 + all dependencies
- ✅ Verifies GPU computation works

### setup_env.sh - Daily Activation

What it does:
- Adds CUDA to PATH
- Activates Python virtual environment
- Shows status of everything

Run this every time you open a terminal to work on the project.

---

## Your Setup

| Component | Version |
|-----------|---------|
| Hardware | Jetson Orin Nano Dev Kit |
| JetPack | 6.1 (R36.4.7) |
| CUDA | 12.6 |
| TensorRT | 10.3 |
| Python | 3.10 (in venv) |
| PyTorch | 2.5.0 (NVIDIA build) |
| GPU Arch | sm_87 (Ampere) |

---

## Common Tasks

### Run CUDA example
```bash
cd 01-cuda-basics
make
./vector_add
```

### Train a model
```bash
source setup_env.sh
cd 02-ml-training
python train_classifier.py
```

### Optimize with TensorRT
```bash
source setup_env.sh
cd 03-tensorrt-optimization
python convert_to_onnx.py
python convert_to_tensorrt.py
```

### Vision + LLM demo
```bash
source setup_env.sh
cd 04-vision-llm-integration
python vision_llm_demo.py --image test.jpg --mode describe
```

---

## Troubleshooting Quick Links

| Error | Fix |
|-------|-----|
| "No module named torch" | `source setup_env.sh` |
| "no kernel image available" | See `KERNEL_IMAGE_ERROR_FIX.md` |
| "externally-managed-environment" | See `FIX_PYTHON.md` |
| "CUDA: False" | `source setup_env.sh` then check `nvcc --version` |

---

**Remember**: Everything starts with `START_HERE.md` 📖
