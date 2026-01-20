# 🚀 Jetson AI Learning - Complete Guide

**Welcome! You have a complete learning environment for CUDA, ML, and Edge AI on your Jetson Orin Nano.**

## What You Have

A structured path from GPU programming basics to production-ready AI systems:

```
📚 Complete Learning Modules
├── 01-cuda-basics/          ✓ GPU Programming fundamentals
├── 02-ml-training/          ✓ Train neural networks
├── 03-tensorrt-optimization/ ✓ Production optimization
└── 04-vision-llm-integration/ ✓ Multimodal AI systems
```

## What Makes Jetson Special

### vs Raspberry Pi:
- ✅ **20-30x faster** ML inference
- ✅ **Hardware AI acceleration** (Tensor Cores)
- ✅ **CUDA programming** capabilities
- ✅ **Production-grade** deployment

### vs Mac (even M-series):
- ✅ **CUDA ecosystem** (not Metal)
- ✅ **TensorRT optimization** (5-10x speedup)
- ✅ **Edge AI deployment** tools
- ✅ **Industry-standard** GPU programming

### vs Cloud AI:
- ✅ **Privacy**: All processing on-device
- ✅ **No latency**: No network round-trips
- ✅ **No cost**: No API fees
- ✅ **Offline**: Works without internet

---

# Choose Your Path

## 🏃 Path 1: Quick Start (10 minutes)
Jump in and see results immediately. [Go to Quick Start ↓](#quick-start-10-minutes)

## 🎓 Path 2: Complete Walkthrough (90 minutes)
Learn everything step-by-step with explanations. [Go to Complete Guide ↓](#complete-walkthrough-90-minutes)

## 🔧 Path 3: Jump to Modules
Know what you want? Go directly to a module:
- CUDA programming → `01-cuda-basics/README.md`
- Train models → `02-ml-training/README.md`
- Optimize for speed → `03-tensorrt-optimization/README.md`
- Vision + LLM → `04-vision-llm-integration/README.md`

---

# Quick Start (10 minutes)

Get up and running fast!

## Step 1: One-Time Setup (5 minutes)

### Install Python 3.10 venv
```bash
sudo apt install python3.10-venv
```

### Run automated installer
```bash
cd /home/lalomorales/Desktop/jetson-ai-learning
./install_pytorch_jetson.sh
```

**This installs PyTorch (Jetson build), CUDA support, and all dependencies. You only do this once!**

---

## Step 2: Test CUDA (Already Done! ✓)

You already compiled and ran your first CUDA program!

```bash
cd /home/lalomorales/Desktop/jetson-ai-learning
source setup_env.sh  # Activates CUDA + Python venv
```

You'll see `(venv)` in your prompt.

**Try it again:**
```bash
cd 01-cuda-basics
./vector_add
```

Output: `✓ Result verified correctly! GPU Time: ~3ms`

---

## Step 3: Image Processing with CUDA

```bash
cd 01-cuda-basics

# Download a test image
python ../test_image_download.py

# Compile and run
make image_grayscale
./image_grayscale test_image.jpg output_gray.jpg
```

RGB to grayscale conversion on the GPU!

---

## Step 4: Train Your First Neural Network (20-30 minutes)

```bash
source setup_env.sh  # Activate environment
cd 02-ml-training
python train_classifier.py
```

Watch as you train a CNN on CIFAR-10:
- 50,000 training images
- ~70% accuracy after 10 epochs
- See GPU acceleration in action!

**Monitor GPU in another terminal:**
```bash
watch -n 1 nvidia-smi
```

---

## Step 5: Optimize with TensorRT (10 minutes)

```bash
source setup_env.sh
cd 03-tensorrt-optimization

# Convert PyTorch to ONNX
python convert_to_onnx.py

# Build TensorRT engines (FP32 and FP16)
python convert_to_tensorrt.py
```

See the **5-10x speedup** from TensorRT optimization!

---

## Step 6: Vision + LLM System (5 minutes)

```bash
source setup_env.sh
cd 04-vision-llm-integration

# Describe an image
python vision_llm_demo.py --image ../01-cuda-basics/test_image.jpg --mode describe

# Ask questions about an image
python vision_llm_demo.py --image ../01-cuda-basics/test_image.jpg --mode qa --question "What is this?"
```

Combines your vision model with Ollama's LLM!

---

## Daily Workflow

Every time you open a new terminal:

```bash
cd /home/lalomorales/Desktop/jetson-ai-learning
source setup_env.sh
```

That's it! Now you can run any Python script or CUDA program.

---

## Quick Troubleshooting

### "No module named torch"
```bash
source setup_env.sh  # Activate the environment
```

### "no kernel image is available for execution"
Wrong PyTorch build. Fix:
```bash
rm -rf venv/
./install_pytorch_jetson.sh
```

See **KERNEL_IMAGE_ERROR_FIX.md** for details.

### "CUDA: False"
```bash
source setup_env.sh
nvcc --version  # Should show CUDA 12.6
```

---

**🎉 Quick Start Complete!** You're now ready to explore. Want more details? Continue to the Complete Walkthrough below.

---
---

# Complete Walkthrough (90 minutes)

A thorough step-by-step guide with explanations.

## Prerequisites

- [x] Jetson Orin Nano Dev Kit
- [x] CUDA 12.6 installed
- [x] TensorRT 10.3 installed
- [x] Ollama with Gemma models
- [ ] Python 3.10 venv (install: `sudo apt install python3.10-venv`)
- [ ] PyTorch for Jetson (install: `./install_pytorch_jetson.sh`)

---

## Phase 1: CUDA Fundamentals (30 minutes)

Learn GPU programming from scratch.

### Setup Environment

```bash
cd /home/lalomorales/Desktop/jetson-ai-learning
source setup_env.sh
```

This sets up:
- CUDA paths (nvcc, libraries)
- Python virtual environment
- All dependencies

### Your First CUDA Program

```bash
cd 01-cuda-basics
make vector_add
./vector_add
```

**What you'll see:**
```
Vector Addition of 1000000 elements
Launching kernel with 3907 blocks of 256 threads
✓ Result verified correctly!
GPU Time: 3.667 ms
Throughput: 3.27 GB/s
```

**What just happened:**
- 1 million parallel additions
- Executed on GPU in ~4ms
- ~3900 thread blocks, each with 256 threads
- Total: ~1 million threads running in parallel!

### Image Processing Example

```bash
# Download test image
python ../test_image_download.py

# Compile and run
make image_grayscale
./image_grayscale test_image.jpg output_gray.jpg
```

**What you learned:**
- Writing CUDA kernels (`__global__` functions)
- Memory management (host ↔ device)
- Thread/block organization
- Performance measurement with CUDA events

**Key Concepts:**
- **Host**: Your CPU and its memory
- **Device**: Your GPU and its memory
- **Kernel**: Function that runs on GPU
- **Thread**: Single execution unit
- **Block**: Group of threads (up to 1024)
- **Grid**: Collection of blocks

---

## Phase 2: Install PyTorch (5-10 minutes)

**Why Python 3.10?**
- Python 3.14 (Homebrew) is too new for PyTorch
- Python 3.10 (system) is fully supported
- Virtual environment keeps everything isolated

### Install

```bash
# Install venv support
sudo apt install python3.10-venv

# Run installer (downloads ~200MB)
cd /home/lalomorales/Desktop/jetson-ai-learning
./install_pytorch_jetson.sh
```

This installs:
- Python 3.10 virtual environment
- **PyTorch 2.5.0 for JetPack 6.1** (from NVIDIA, not PyPI!)
- TorchVision for computer vision
- NumPy, OpenCV, ONNX, and other ML libraries

**Why Jetson-specific PyTorch?**
- Generic PyPI wheels don't support Jetson Orin's GPU (sm_87)
- NVIDIA's builds have kernels compiled for your specific GPU
- This is why you got the "no kernel image" error earlier

### Verify

```bash
source setup_env.sh
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Should output: `CUDA: True`

**Test GPU computation:**
```bash
python << 'EOF'
import torch
x = torch.randn(100, 100).cuda()
y = torch.randn(100, 100).cuda()
z = torch.matmul(x, y)
print("✓ GPU computation successful!")
EOF
```

---

## Phase 3: Train a Neural Network (20-30 minutes)

Train a CNN to classify images!

```bash
source setup_env.sh
cd 02-ml-training
python train_classifier.py
```

**What happens:**

1. **Downloads CIFAR-10 dataset** (60K images, 32x32 color)
   - 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
   - 50K training, 10K test images

2. **Defines CNN architecture**
   - 3 convolutional layers
   - Max pooling
   - 3 fully connected layers
   - ~150K parameters

3. **Trains for 10 epochs**
   - Each epoch: ~30 seconds on GPU
   - Batch size: 64
   - Optimizer: Adam
   - Loss: CrossEntropy

4. **Achieves ~70% accuracy**
   - Saves best model to `best_model.pth`
   - Saves final model to `final_model.pth`

**Monitor GPU usage:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

You'll see:
- GPU utilization: 80-100%
- Memory usage: ~1-2GB
- Power: 5-15W

**What you learned:**
- Loading and augmenting datasets
- Defining neural network architectures
- Training loop (forward pass, loss, backward pass, optimize)
- GPU acceleration (20-30x faster than CPU!)
- Model checkpointing

**Key Concepts:**
- **Epoch**: One pass through entire training set
- **Batch**: Subset of data processed together
- **Forward pass**: Input → model → predictions
- **Loss**: How wrong the predictions are
- **Backward pass**: Calculate gradients
- **Optimizer**: Update weights to reduce loss

---

## Phase 4: TensorRT Optimization (10 minutes)

Convert PyTorch → ONNX → TensorRT for **5-10x faster inference**!

### Step 1: Convert to ONNX

```bash
source setup_env.sh
cd 03-tensorrt-optimization
python convert_to_onnx.py
```

**What happens:**
- Loads your trained PyTorch model
- Exports to ONNX format (intermediate representation)
- Verifies the ONNX model is valid
- Creates `model.onnx` (~600KB)

**What is ONNX?**
- Open Neural Network Exchange format
- Portable between frameworks (PyTorch, TensorFlow, etc.)
- TensorRT can optimize ONNX models

### Step 2: Build TensorRT Engines

```bash
python convert_to_tensorrt.py
```

**What happens:**
- Parses ONNX model
- TensorRT analyzes the network
- Fuses layers (combines operations)
- Optimizes kernel selection
- Builds two engines:
  - `model_fp32.trt` - Full precision
  - `model_fp16.trt` - Half precision (2x faster!)

**This takes a few minutes** as TensorRT explores optimizations.

**What you learned:**
- Model export pipeline
- TensorRT optimization process
- Precision modes (FP32 vs FP16 vs INT8)
- Building inference engines

**Performance comparison:**
- PyTorch FP32: ~50ms inference
- TensorRT FP32: ~15ms (3x faster)
- TensorRT FP16: ~8ms (6x faster)
- TensorRT INT8: ~5ms (10x faster, needs calibration)

**Why TensorRT is faster:**
1. **Layer fusion**: Combines multiple operations
2. **Kernel optimization**: Selects best CUDA kernels
3. **Precision calibration**: Uses lower precision safely
4. **Tensor Cores**: Uses hardware acceleration
5. **Graph optimization**: Removes unnecessary ops

---

## Phase 5: Vision + LLM Integration (5 minutes)

Combine everything into a multimodal AI system!

```bash
source setup_env.sh
cd 04-vision-llm-integration
```

### Describe an Image

```bash
python vision_llm_demo.py --image ../01-cuda-basics/test_image.jpg --mode describe
```

**What happens:**
1. **Vision model** (TensorRT-optimized) classifies image (< 10ms)
2. **Results** sent to Ollama's Gemma LLM
3. **LLM** generates natural language description
4. **Output** combines vision + language understanding

**Example output:**
```
Vision Model Results:
   1. cat          85.23%
   2. dog          8.45%
   3. bird         3.21%

LLM Description:
This is a domestic cat, likely a tabby. Cats are popular pets
known for their independence and hunting skills. An interesting
fact: cats can rotate their ears 180 degrees!
```

### Visual Question Answering

```bash
python vision_llm_demo.py --image test.jpg --mode qa --question "Is this a good pet?"
```

The LLM uses vision model results to answer your question!

**What you learned:**
- Multimodal AI pipelines
- Combining vision + language models
- Edge AI deployment (all on-device!)
- Integration with local LLMs (Ollama)

**Why this is special:**
- **Private**: No data sent to cloud
- **Fast**: No network latency
- **Free**: No API costs
- **Offline**: Works without internet

---

## What You've Accomplished

✅ **CUDA Programming Skills**
- Write GPU kernels
- Optimize memory transfers
- Measure performance
- Understand parallelism

✅ **ML Training Pipeline**
- Load and preprocess datasets
- Define neural network architectures
- Train models with GPU acceleration
- Validate and save models

✅ **Production Optimization**
- Export models to ONNX
- Build TensorRT engines
- Optimize precision (FP16/INT8)
- Benchmark performance improvements

✅ **Multimodal AI System**
- Vision model inference with TensorRT
- LLM integration with Ollama
- Natural language outputs
- End-to-end edge AI pipeline

---

## Next-Level Projects

Now build something cool:

### 1. Real-time Object Detection
- Deploy YOLOv8 with TensorRT
- Process video at 30+ FPS
- Add bounding box visualization
- Build a surveillance system

### 2. Custom Dataset Training
- Collect your own images
- Train a custom classifier
- Optimize with TensorRT
- Deploy as a web service

### 3. Advanced Vision + LLM
- Image captioning system
- Visual question answering
- Scene understanding
- Educational tools

### 4. Edge AI Service
- REST API for inference
- Multi-model support
- Performance monitoring
- Auto-scaling

### 5. LLM Optimization
- Install TensorRT-LLM
- Optimize Llama 3.2 or Phi-3
- Build local AI assistant
- Voice interface integration

---

## Helpful Resources

### Documentation
- Each module has a detailed README.md
- Code is heavily commented
- Step-by-step instructions

### Troubleshooting
- **FIX_PYTHON.md** - Python/PyTorch issues
- **KERNEL_IMAGE_ERROR_FIX.md** - GPU kernel errors
- Module READMEs for specific topics

### Learning More
- [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70)
- [JetsonHacks YouTube](https://www.youtube.com/c/JetsonHacks)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## File Reference

| File | Purpose |
|------|---------|
| `START_HERE.md` | **This file** - Complete guide |
| `README.md` | Project overview |
| `install_pytorch_jetson.sh` | One-time setup |
| `setup_env.sh` | Daily activation |
| `FIX_PYTHON.md` | Troubleshooting |
| `KERNEL_IMAGE_ERROR_FIX.md` | GPU error fixes |

---

## Your Jetson's Specs

- **GPU**: Orin (1024 CUDA cores + Tensor Cores)
- **Architecture**: sm_87 (Ampere)
- **CUDA**: 12.6
- **TensorRT**: 10.3
- **JetPack**: 6.1 (R36.4.7)
- **Memory**: Unified CPU/GPU
- **Power**: 5-15W

---

## Daily Reminder

Every time you work on this project:

```bash
cd /home/lalomorales/Desktop/jetson-ai-learning
source setup_env.sh
```

This activates:
- ✅ CUDA paths
- ✅ Python virtual environment
- ✅ All dependencies

You'll see `(venv)` in your prompt when ready.

---

**🎉 Congratulations!** You now have a complete edge AI development environment and the skills to build sophisticated AI applications!

**Happy building!** 🚀
