#!/bin/bash
# Environment setup for Jetson AI Learning

PROJECT_DIR="/home/lalomorales/Desktop/jetson-ai-learning"
VENV_DIR="$PROJECT_DIR/venv"

# Add CUDA to PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Activating Python virtual environment..."
    source "$VENV_DIR/bin/activate"
    echo "✓ Virtual environment activated (venv)"
else
    echo "⚠ Virtual environment not found. Run: ./setup_venv.sh"
fi

# Verify CUDA
echo -e "\nCUDA Version:"
nvcc --version | grep "release"

# Check GPU
echo -e "\nGPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check Python
if command -v python &> /dev/null; then
    echo -e "\nPython: $(python --version)"
    if python -c "import torch" 2>/dev/null; then
        echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
        echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    fi
fi

echo -e "\n✓ Environment ready!"
