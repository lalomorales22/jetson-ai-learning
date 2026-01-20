# Vision + Language Integration

Combine computer vision with LLMs for multimodal AI on Jetson!

## Overview

This module chains a vision model with a local LLM to:
1. Analyze images with a neural network
2. Send results to an LLM for natural language understanding
3. Generate descriptions, answer questions, and more

## Models Used

| Component | Model | Classes/Capability |
|-----------|-------|-------------------|
| Vision (Basic) | CIFAR-10 CNN | 10 classes (airplane, car, bird, cat, etc.) |
| Vision (Full) | MobileNetV2 | **1000 ImageNet classes** |
| Language | Ollama (gemma3:1b, llama3.2, etc.) | Natural language generation |

## Prerequisites

1. **Ollama installed and running:**
   ```bash
   # Install Ollama (if not already)
   curl -fsSL https://ollama.com/install.sh | sh

   # Start Ollama server
   ollama serve

   # Pull a model (in another terminal)
   ollama pull gemma3:1b
   ```

2. **Virtual environment activated:**
   ```bash
   source ../setup_env.sh
   ```

---

## Scripts

### 1. describe_image.py - Image Description

Uses MobileNetV2 (1000 ImageNet classes) for accurate image recognition.

**Basic usage:**
```bash
python describe_image.py --image photo.jpg
```

**All options:**
```bash
python describe_image.py --image <path> [OPTIONS]

Options:
  --image PATH       Path to input image (required)
  --top-k N          Number of top predictions to show (default: 5)
  --llm-model NAME   Ollama model to use (default: gemma3:1b)
  --quiet            Minimal output
```

**Examples:**
```bash
# Describe a photo
python describe_image.py --image vacation.jpg

# Show top 10 predictions
python describe_image.py --image mystery.png --top-k 10

# Use a different LLM
python describe_image.py --image cat.jpg --llm-model llama3.2:1b
```

---

### 2. visual_qa.py - Visual Question Answering

Ask questions about images and get natural language answers.

**Basic usage:**
```bash
python visual_qa.py --image photo.jpg --question "What is this?"
```

**Interactive mode (multiple questions):**
```bash
python visual_qa.py --image photo.jpg --interactive
```

**All options:**
```bash
python visual_qa.py --image <path> [OPTIONS]

Options:
  --image PATH       Path to input image (required)
  --question TEXT    Question to ask about the image
  --interactive      Interactive mode - ask multiple questions
  --llm-model NAME   Ollama model to use (default: gemma3:1b)
  --quiet            Minimal output
```

**Examples:**
```bash
# Ask a single question
python visual_qa.py --image dog.jpg --question "What breed is this dog?"

# Interactive Q&A session
python visual_qa.py --image scene.jpg --interactive

# Ask about colors
python visual_qa.py --image car.png --question "What color is this?"
```

---

### 3. vision_llm_demo.py - Original CIFAR-10 Demo

Uses your trained CIFAR-10 model (10 classes) with TensorRT optimization.

**Basic usage:**
```bash
python vision_llm_demo.py --image photo.jpg --mode describe
```

**All options:**
```bash
python vision_llm_demo.py --image <path> [OPTIONS]

Options:
  --image PATH       Path to input image (required)
  --mode MODE        Mode: describe or qa (default: describe)
  --question TEXT    Question for QA mode
  --no-tensorrt      Use PyTorch instead of TensorRT
```

**Examples:**
```bash
# Describe an image
python vision_llm_demo.py --image cat.jpg --mode describe

# Ask a question
python vision_llm_demo.py --image plane.jpg --mode qa --question "Can this fly?"

# Use PyTorch (no TensorRT)
python vision_llm_demo.py --image truck.jpg --no-tensorrt
```

**Note:** This uses your custom-trained CIFAR-10 model which only recognizes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

### 4. benchmark_pipeline.py - Performance Testing

Measure latency for each stage of the pipeline.

**Basic usage:**
```bash
python benchmark_pipeline.py --image photo.jpg
```

**All options:**
```bash
python benchmark_pipeline.py --image <path> [OPTIONS]

Options:
  --image PATH       Path to input image (required)
  --llm-model NAME   Ollama model to benchmark (default: gemma3:1b)
  --vision-runs N    Number of vision model iterations (default: 50)
  --llm-runs N       Number of LLM iterations (default: 3)
  --skip-llm         Skip LLM benchmark (vision only)
```

**Examples:**
```bash
# Full pipeline benchmark
python benchmark_pipeline.py --image test.jpg

# Vision model only (faster)
python benchmark_pipeline.py --image test.jpg --skip-llm

# More iterations for accuracy
python benchmark_pipeline.py --image test.jpg --vision-runs 100
```

---

## Example Workflow

```
Image of a golden retriever
         ↓
MobileNetV2 → "golden retriever" (89%), "Labrador" (5%), ...
         ↓
Ollama LLM → "This image shows a golden retriever, a popular
              breed known for their friendly temperament and
              golden-colored coat. They are often used as
              service dogs and family pets."
```

---

## Why Use This on Jetson?

| Feature | Cloud API | Jetson Local |
|---------|-----------|--------------|
| Privacy | Data sent to servers | All on-device |
| Latency | 500ms+ network | <100ms local |
| Cost | Pay per API call | Free |
| Offline | Requires internet | Works offline |

---

## Recommended LLM Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| gemma3:1b | 1B | Fast | Good |
| llama3.2:1b | 1B | Fast | Good |
| gemma3:4b | 4B | Medium | Better |
| llama3.2:3b | 3B | Medium | Better |

For Jetson Orin Nano, start with 1B models for best speed.

---

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Start Ollama server
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# Pull the model first
ollama pull gemma3:1b
```

### Slow LLM responses
- Use a smaller model (1B instead of 7B)
- Close other GPU applications
- Check GPU memory: `tegrastats`

### Poor image recognition
- Use `describe_image.py` (1000 classes) instead of `vision_llm_demo.py` (10 classes)
- Ensure good lighting and image quality
- The model works best with single objects, not complex scenes

---

## What's Next?

After mastering this module:

1. **Add TensorRT to MobileNetV2** - Convert for faster inference
2. **Try CLIP** - Zero-shot classification (any text label)
3. **Explore LLaVA** - True vision-language model (sees the actual image)
4. **Build an API** - REST endpoint for image analysis
