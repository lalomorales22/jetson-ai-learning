# CUDA Basics

Learn CUDA programming step by step.

## What is CUDA?

CUDA lets you write code that runs on NVIDIA GPUs. Unlike CPUs (few powerful cores), GPUs have thousands of small cores for massive parallelism.

## Why Learn CUDA?

1. **Speed**: 10-100x faster for parallel tasks
2. **Control**: Fine-tune performance for your specific needs
3. **ML Foundation**: Understand how PyTorch/TensorFlow work under the hood
4. **Custom Operations**: Build operations not available in frameworks

## Learning Path

1. **vector_add.cu** - Basic parallel vector addition
2. **image_grayscale.cu** - Image processing (RGB to grayscale)
3. **matrix_multiply.cu** - Matrix operations for ML
4. **image_blur.cu** - Convolution operations (used in CNNs)

## Compilation

```bash
# Compile a CUDA program
nvcc -o program_name source_file.cu

# Run
./program_name
```

## CUDA Concepts

- **Host**: Your CPU and its memory
- **Device**: Your GPU and its memory
- **Kernel**: Function that runs on GPU
- **Thread**: Single execution unit on GPU
- **Block**: Group of threads (up to 1024)
- **Grid**: Collection of blocks

Let's start with vector addition!
