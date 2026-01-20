# ML Training on Jetson

Train custom neural networks using PyTorch with GPU acceleration.

## What You'll Learn

1. Load and preprocess image datasets
2. Define a neural network architecture
3. Train on GPU with CUDA
4. Save models for deployment
5. Compare CPU vs GPU training speed

## Dataset

We'll use the CIFAR-10 dataset (60,000 32x32 color images in 10 classes):
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Why CIFAR-10?
- Small enough to train quickly on Jetson
- Complex enough to see GPU benefits
- Standard benchmark dataset

## Scripts

1. **train_classifier.py** - Train a CNN on CIFAR-10
2. **inference_test.py** - Test the trained model
3. **visualize_results.py** - Visualize predictions

## Training Process

1. **Data Loading**: Download and prepare dataset
2. **Model Definition**: Define CNN architecture
3. **Training Loop**: Train with backpropagation
4. **Validation**: Check accuracy on test set
5. **Save Model**: Export for deployment

## Why GPU Matters

- CPU training: ~5-10 minutes per epoch
- GPU training: ~10-30 seconds per epoch
- **20-30x speedup!**

The Jetson's GPU makes experimentation practical!
