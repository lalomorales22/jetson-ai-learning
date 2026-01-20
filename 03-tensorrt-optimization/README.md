# TensorRT Optimization

Convert PyTorch models to TensorRT for **5-10x faster inference**!

## What is TensorRT?

TensorRT is NVIDIA's inference optimizer that:
- Fuses layers (combines operations)
- Optimizes kernel selection
- Supports INT8/FP16 quantization
- Uses Tensor Cores on Jetson

## Why TensorRT?

**PyTorch/ONNX Runtime**: Great for flexibility
**TensorRT**: Optimized for production deployment

On Jetson Orin Nano:
- PyTorch FP32: ~50ms inference
- TensorRT FP32: ~15ms (3x faster)
- TensorRT FP16: ~8ms (6x faster)
- TensorRT INT8: ~5ms (10x faster)

## Optimization Pipeline

```
PyTorch Model (.pth)
    ↓
ONNX Model (.onnx)
    ↓
TensorRT Engine (.trt)
    ↓
Fast Inference!
```

## Scripts

1. **convert_to_onnx.py** - Export PyTorch to ONNX
2. **convert_to_tensorrt.py** - Build TensorRT engine
3. **benchmark.py** - Compare PyTorch vs TensorRT
4. **optimize_int8.py** - INT8 quantization (advanced)

## Key Concepts

- **ONNX**: Open format for ML models
- **Engine**: Optimized TensorRT model for specific GPU
- **FP16**: Half precision (2x faster, minimal accuracy loss)
- **INT8**: 8-bit integers (4x faster, needs calibration)

Let's optimize your model!
