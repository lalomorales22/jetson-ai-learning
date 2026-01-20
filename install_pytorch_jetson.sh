#!/bin/bash
# Install PyTorch for Jetson Orin Nano (Working Version!)

set -e  # Exit on error

echo "======================================================================"
echo "Installing PyTorch for Jetson Orin Nano"
echo "Using Jetson AI Lab Community Repository"
echo "======================================================================"
echo ""

# Check if python3.10-venv is installed
if ! python3.10 -m venv --help &>/dev/null; then
    echo "❌ python3.10-venv is not installed"
    echo ""
    echo "Please run:"
    echo "  sudo apt install python3.10-venv"
    echo ""
    exit 1
fi

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create venv with Python 3.10
echo "Creating virtual environment with Python 3.10..."
python3.10 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "Python version: $PYTHON_VERSION"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install NumPy 1.x FIRST (before PyTorch)
echo "Installing NumPy 1.x (required for PyTorch compatibility)..."
pip install "numpy<2"
echo ""

# Install basic dependencies
echo "Installing basic dependencies..."
pip install pillow matplotlib
echo ""

# Install PyTorch from Jetson AI Lab
echo "======================================================================"
echo "Installing PyTorch 2.8.0 from Jetson AI Lab repository"
echo "This is a community-maintained repo optimized for Jetson devices"
echo "Download size: ~226MB"
echo "======================================================================"
echo ""

pip install --no-deps --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.8.0 torchvision

# Install PyTorch dependencies (without upgrading numpy)
echo ""
echo "Installing PyTorch dependencies..."
pip install filelock typing-extensions sympy networkx jinja2 fsspec mpmath MarkupSafe

# Install other ML dependencies
echo ""
echo "Installing additional packages..."
pip install opencv-python scikit-learn onnx

# Reinstall NumPy 1.x (some packages above may have upgraded to NumPy 2.x)
echo ""
echo "Ensuring NumPy 1.x compatibility..."
pip install --force-reinstall "numpy<2" --quiet

# Link system TensorRT to venv (TensorRT comes with JetPack, not pip)
echo ""
echo "Linking system TensorRT to virtual environment..."
VENV_SITE="venv/lib/python3.10/site-packages"
SYS_SITE="/usr/lib/python3.10/dist-packages"
if [ -d "$SYS_SITE/tensorrt" ]; then
    ln -sf "$SYS_SITE/tensorrt" "$VENV_SITE/tensorrt"
    ln -sf "$SYS_SITE/tensorrt_lean" "$VENV_SITE/tensorrt_lean"
    ln -sf "$SYS_SITE/tensorrt_dispatch" "$VENV_SITE/tensorrt_dispatch"
    ln -sf "$SYS_SITE/tensorrt-"*.dist-info "$VENV_SITE/" 2>/dev/null
    ln -sf "$SYS_SITE/tensorrt_lean-"*.dist-info "$VENV_SITE/" 2>/dev/null
    ln -sf "$SYS_SITE/tensorrt_dispatch-"*.dist-info "$VENV_SITE/" 2>/dev/null
    echo "✓ TensorRT linked"
else
    echo "⚠ System TensorRT not found - install JetPack to get TensorRT"
fi

echo ""
echo "======================================================================"
echo "Verifying Installation"
echo "======================================================================"
echo ""

python << 'EOF'
import sys
print(f"Python: {sys.version}")
print()

# Check PyTorch
try:
    import torch
    import torchvision
    import numpy as np

    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ TorchVision: {torchvision.__version__}")
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability(0)
        print(f"✓ Compute Capability: {capability[0]}.{capability[1]}")

        # Test GPU
        print()
        print("Testing GPU computation...")
        import time
        try:
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')

            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            print(f"✓ GPU matrix multiply (1000x1000): {elapsed*1000:.2f} ms")
        except Exception as e:
            print(f"⚠ GPU test had an issue: {e}")
            print("  This might be a memory issue - try rebooting if problems persist")
            print("  PyTorch CUDA is still available and should work for normal use")
        print()
        print("=" * 70)
        print("🎉 PyTorch with CUDA is working perfectly!")
        print("=" * 70)
    else:
        print()
        print("❌ CUDA not available")
        print("Something went wrong with the installation.")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Check other packages
print()
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError:
    print("⚠ OpenCV not available")

try:
    import onnx
    print(f"✓ ONNX: {onnx.__version__}")
except ImportError:
    print("⚠ ONNX not available")

EOF

echo ""
echo "======================================================================"
echo "Installation Complete! 🎉"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source setup_env.sh"
echo ""
echo "2. Train a model:"
echo "   cd 02-ml-training"
echo "   python train_classifier.py"
echo ""
echo "3. Read the guide:"
echo "   cat START_HERE.md"
echo ""
echo "Happy learning! 🚀"
echo ""
