# Jetson AI Learning Project

**A comprehensive learning path for CUDA, ML, and Edge AI on Jetson Orin Nano.**

Learn GPU programming, train neural networks, optimize with TensorRT, and build Vision+Language AI systems - all running on your Jetson!

---

## What You'll Learn

| Module | Skills |
|--------|--------|
| **01-cuda-basics** | GPU kernels, parallel computing, CUDA C++ |
| **02-ml-training** | PyTorch, CNN training, GPU acceleration |
| **03-tensorrt-optimization** | ONNX export, TensorRT, 10x speedup |
| **04-vision-llm-integration** | Vision models, Ollama LLM, multimodal AI |
| **05-fun** | Flask chatbot, persistent memory, full-stack AI |

---

## Quick Start

### Step 1: Install Python 3.10 venv support
```bash
sudo apt install python3.10-venv
```

### Step 2: Run the installer (5-10 minutes)
```bash
cd /home/lalomorales/Desktop/jetson-ai-learning
./install_pytorch_jetson.sh
```

**What gets installed:**
- Python 3.10 virtual environment
- **PyTorch 2.8.0** (from Jetson AI Lab, optimized for Jetson Orin)
- TorchVision 0.23.0
- TensorRT 10.3 (linked from system)
- CUDA 12.6 support
- NumPy 1.x, OpenCV, ONNX, Matplotlib

### Step 3: Daily activation
```bash
source setup_env.sh  # Activates CUDA + Python venv
```

You'll see `(venv)` in your prompt when active.

### Step 4: Verify it works
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Should output:** `CUDA: True`

---

## Project Structure

```
jetson-ai-learning/
├── install_pytorch_jetson.sh    # One-time setup
├── setup_env.sh                 # Daily activation
│
├── 01-cuda-basics/              # GPU Programming
│   ├── vector_add.cu            # Your first CUDA kernel
│   ├── image_grayscale.cu       # Image processing on GPU
│   └── README.md
│
├── 02-ml-training/              # Neural Networks
│   ├── train_classifier.py      # Train CNN on CIFAR-10
│   ├── best_model.pth           # Saved model (after training)
│   └── README.md
│
├── 03-tensorrt-optimization/    # Production Speed
│   ├── convert_to_onnx.py       # PyTorch → ONNX
│   ├── convert_to_tensorrt.py   # ONNX → TensorRT
│   ├── benchmark.py             # Compare speeds
│   └── README.md
│
├── 04-vision-llm-integration/   # Multimodal AI
│   ├── describe_image.py        # 1000-class image description
│   ├── visual_qa.py             # Ask questions about images
│   ├── vision_llm_demo.py       # CIFAR-10 + LLM demo
│   ├── benchmark_pipeline.py    # End-to-end latency
│   └── README.md
│
└── 05-fun/                      # JetBrain Chatbot
    ├── app.py                   # Flask application
    ├── templates/index.html     # Chat UI
    ├── .jetbrain                # Persistent memory (JSON)
    └── README.md
```

---

## Learning Path

### Phase 1: CUDA Basics (30 min)
```bash
cd 01-cuda-basics
make
./vector_add
./image_grayscale
```
**Learn:** GPU programming, kernels, memory management

### Phase 2: Train Neural Network (30 min)
```bash
source setup_env.sh
cd 02-ml-training
python train_classifier.py
```
**Learn:** PyTorch, CNNs, GPU-accelerated training

### Phase 3: TensorRT Optimization (10 min)
```bash
cd 03-tensorrt-optimization
python convert_to_onnx.py
python convert_to_tensorrt.py
python benchmark.py
```
**Learn:** Model optimization, 5-10x speedup

**Typical Results:**
| Engine | Time | Speedup |
|--------|------|---------|
| PyTorch FP32 | 2.3 ms | 1.0x |
| TensorRT FP32 | 0.4 ms | 5.1x |
| TensorRT FP16 | 0.2 ms | **9.5x** |

### Phase 4: Vision + LLM (15 min)
```bash
# Start Ollama first
ollama serve &
ollama pull gemma3:1b

cd 04-vision-llm-integration

# Describe any image (1000 classes)
python describe_image.py --image photo.jpg

# Ask questions about images
python visual_qa.py --image photo.jpg --question "What is this?"

# Interactive Q&A
python visual_qa.py --image photo.jpg --interactive
```
**Learn:** Multimodal AI, edge deployment

### Phase 5: JetBrain Chatbot (5 min)
```bash
cd 05-fun
pip install flask
python app.py
# Open http://localhost:5000
```
**Learn:** Full-stack AI apps, Flask, SQLite, persistent memory

**Slash Commands:**
- `/remember fact: I work at NVIDIA` - Save to memory
- `/memory` - View all memories
- `/help` - Show all commands
- `/status` - Check system status

---

## Your Setup

| Component | Version |
|-----------|---------|
| **Hardware** | Jetson Orin Nano Dev Kit |
| **JetPack** | 6.1 (R36.4.7) |
| **CUDA** | 12.6 |
| **TensorRT** | 10.3 |
| **Python** | 3.10 (in venv) |
| **PyTorch** | 2.8.0 (Jetson AI Lab build) |
| **GPU Architecture** | sm_87 (Ampere) |

---

## Command Reference

### 01-cuda-basics
```bash
make                    # Compile all CUDA programs
./vector_add            # Run vector addition
./image_grayscale       # Run image processing
make clean              # Clean build files
```

### 02-ml-training
```bash
python train_classifier.py      # Train for 10 epochs
# Creates: best_model.pth, final_model.pth
```

### 03-tensorrt-optimization
```bash
python convert_to_onnx.py       # PyTorch → ONNX
python convert_to_tensorrt.py   # ONNX → TensorRT (FP32 + FP16)
python benchmark.py             # Compare all engines
```

### 04-vision-llm-integration
```bash
# Image description (1000 classes)
python describe_image.py --image photo.jpg
python describe_image.py --image photo.jpg --top-k 10

# Visual Q&A
python visual_qa.py --image photo.jpg --question "What color is it?"
python visual_qa.py --image photo.jpg --interactive

# CIFAR-10 demo (10 classes, TensorRT)
python vision_llm_demo.py --image cat.jpg --mode describe

# Benchmark
python benchmark_pipeline.py --image photo.jpg
python benchmark_pipeline.py --image photo.jpg --skip-llm
```

### 05-fun (JetBrain)
```bash
pip install flask               # Install Flask
python app.py                   # Start server
# Open http://localhost:5000

# Slash commands in chat:
/help                           # Show all commands
/remember fact: I like Python   # Save to memory
/memory                         # View memories
/status                         # System status
/model llama3.2:1b              # Switch LLM
```

---

## Troubleshooting

### "No module named torch"
```bash
source setup_env.sh  # Activate the environment
```

### "CUDA not available" or "CUDA: False"
```bash
rm -rf venv/
./install_pytorch_jetson.sh
```

### "NumPy 2.x incompatibility"
```bash
source venv/bin/activate
pip install --force-reinstall "numpy<2"
```

### "TensorRT not available via pip"
TensorRT is pre-installed with JetPack. The install script links it automatically.

### "Cannot connect to Ollama"
```bash
ollama serve          # Start Ollama
ollama pull gemma3:1b # Pull a model
```

### "no kernel image is available for execution"
Wrong PyTorch build. Reinstall with the provided script.

---

## What Makes Jetson Special?

### vs Raspberry Pi
- **20-30x faster** ML inference
- **Hardware AI acceleration** (Tensor Cores)
- **CUDA programming** capabilities

### vs Cloud AI
- **Privacy**: All processing on-device
- **No latency**: No network round-trips
- **No cost**: No API fees
- **Offline**: Works without internet

---

## Resources

### Documentation
- [NVIDIA Jetson Developer Site](https://developer.nvidia.com/embedded/jetson)
- [PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

### Community
- [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70)
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [JetsonHacks](https://www.jetsonhacks.com/)

---

## Next-Level Projects

After completing the learning modules:

1. **Real-time Object Detection** - YOLOv8 + TensorRT at 30+ FPS
2. **Custom Dataset Training** - Your own classifier optimized with TensorRT
3. **Image Captioning** - Vision model + LLM descriptions
4. **Edge AI API** - REST service for inference
5. **LLM Optimization** - TensorRT-LLM for faster local AI

---

**Ready to start?** Activate your environment and dive in:
```bash
source setup_env.sh
```

Happy learning! 🚀
