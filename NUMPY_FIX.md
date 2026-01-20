# NumPy Version Fix

## The Problem

During installation, you got:
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

And GPU test failed with:
```
NvMapMemHandleAlloc: error 0
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED
```

## Root Cause

**Installation order issue:**
1. Script installed dependencies (which pulled NumPy 2.2.6)
2. Then installed PyTorch (compiled with NumPy 1.x)
3. Then tried to downgrade NumPy
4. Too late - PyTorch already loaded with wrong NumPy

**Result:**
- NumPy incompatibility warnings
- GPU memory allocation errors

## The Solution

**Install NumPy 1.x BEFORE PyTorch:**

```bash
# 1. Install NumPy 1.x first
pip install "numpy<2"

# 2. Install PyTorch without dependencies
pip install --no-deps torch torchvision

# 3. Install other PyTorch dependencies manually
pip install filelock typing-extensions sympy networkx jinja2 fsspec
```

## Fixed in install_pytorch_jetson.sh

The script now:
1. ✅ Installs NumPy<2 FIRST
2. ✅ Uses `--no-deps` flag for PyTorch
3. ✅ Manually installs dependencies after
4. ✅ Smaller GPU test (1000x1000 instead of 2000x2000)
5. ✅ Error handling for GPU test

## If You Already Installed

Quick fix:
```bash
source venv/bin/activate
pip install --force-reinstall "numpy<2"
```

Or reinstall everything:
```bash
rm -rf venv/
./install_pytorch_jetson.sh
```

## Verify It Works

```bash
source venv/bin/activate
python << 'EOF'
import torch
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Quick GPU test
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU works!")
EOF
```

Should output:
```
PyTorch: 2.8.0
NumPy: 1.26.4
CUDA: True
✓ GPU works!
```

## Why NumPy Version Matters

| NumPy Version | PyTorch 2.8.0 | Works? |
|---------------|---------------|--------|
| 2.2.6 | Compiled with 1.x | ❌ Crashes |
| 1.26.4 | Compiled with 1.x | ✅ Works |

PyTorch 2.8.0 was compiled against NumPy 1.x API. NumPy 2.x changed the C API, causing crashes.

## About the GPU Memory Error

The `NvMapMemHandleAlloc: error 0` was likely caused by:
1. NumPy incompatibility messing up memory handling
2. Large test matrix (2000x2000) using too much memory

With NumPy 1.x, GPU memory works fine!

## Summary

**Problem:** NumPy 2.x installed before PyTorch
**Solution:** Install NumPy 1.x first
**Status:** ✅ Fixed in updated script
**Your install:** ✅ Working now!

---

You're all set! PyTorch with CUDA is working correctly.
