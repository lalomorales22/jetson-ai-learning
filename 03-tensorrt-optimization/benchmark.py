#!/usr/bin/env python3
"""
Benchmark PyTorch vs TensorRT inference performance
Shows the speedup from TensorRT optimization on Jetson
"""

import torch
import tensorrt as trt
import numpy as np
import time
import os
import sys

# Add parent path for model import
sys.path.append('../02-ml-training')
from train_classifier import SimpleCNN

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def benchmark_pytorch(model, input_tensor, num_warmup=10, num_runs=100):
    """Benchmark PyTorch inference"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    return times


def benchmark_tensorrt(engine_path, input_shape, num_warmup=10, num_runs=100):
    """Benchmark TensorRT inference"""

    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers using PyTorch
    d_input = torch.randn(*input_shape, dtype=torch.float32, device='cuda')
    d_output = torch.empty((input_shape[0], 10), dtype=torch.float32, device='cuda')

    stream = torch.cuda.current_stream()

    # Set tensor addresses
    context.set_tensor_address('input', d_input.data_ptr())
    context.set_tensor_address('output', d_output.data_ptr())

    # Warmup
    for _ in range(num_warmup):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    return times


def print_stats(name, times):
    """Print timing statistics"""
    times = np.array(times)
    print(f"\n{name}:")
    print(f"  Mean:   {times.mean():.3f} ms")
    print(f"  Std:    {times.std():.3f} ms")
    print(f"  Min:    {times.min():.3f} ms")
    print(f"  Max:    {times.max():.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    return times.mean()


def main():
    print("=" * 60)
    print("PyTorch vs TensorRT Benchmark")
    print("=" * 60)

    # Check files exist
    pytorch_model_path = '../02-ml-training/best_model.pth'
    trt_fp32_path = 'model_fp32.trt'
    trt_fp16_path = 'model_fp16.trt'

    missing = []
    if not os.path.exists(pytorch_model_path):
        missing.append(pytorch_model_path)
    if not os.path.exists(trt_fp32_path):
        missing.append(trt_fp32_path)
    if not os.path.exists(trt_fp16_path):
        missing.append(trt_fp16_path)

    if missing:
        print("\nMissing files:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun these first:")
        print("  python convert_to_onnx.py")
        print("  python convert_to_tensorrt.py")
        sys.exit(1)

    # Setup
    device = torch.device('cuda')
    input_shape = (1, 3, 32, 32)
    num_warmup = 50
    num_runs = 200

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Input shape: {input_shape}")
    print(f"Warmup runs: {num_warmup}")
    print(f"Benchmark runs: {num_runs}")

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    model = SimpleCNN().to(device)
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    input_tensor = torch.randn(*input_shape, device=device)

    # Benchmark PyTorch
    print("\nBenchmarking PyTorch FP32...")
    pytorch_times = benchmark_pytorch(model, input_tensor, num_warmup, num_runs)
    pytorch_mean = print_stats("PyTorch FP32", pytorch_times)

    # Benchmark TensorRT FP32
    print("\nBenchmarking TensorRT FP32...")
    trt_fp32_times = benchmark_tensorrt(trt_fp32_path, input_shape, num_warmup, num_runs)
    trt_fp32_mean = print_stats("TensorRT FP32", trt_fp32_times)

    # Benchmark TensorRT FP16
    print("\nBenchmarking TensorRT FP16...")
    trt_fp16_times = benchmark_tensorrt(trt_fp16_path, input_shape, num_warmup, num_runs)
    trt_fp16_mean = print_stats("TensorRT FP16", trt_fp16_times)

    # Summary
    print("\n" + "=" * 60)
    print("SPEEDUP SUMMARY")
    print("=" * 60)

    speedup_fp32 = pytorch_mean / trt_fp32_mean
    speedup_fp16 = pytorch_mean / trt_fp16_mean

    print(f"\n{'Engine':<20} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 42)
    print(f"{'PyTorch FP32':<20} {pytorch_mean:<12.3f} {'1.0x (baseline)':<10}")
    print(f"{'TensorRT FP32':<20} {trt_fp32_mean:<12.3f} {speedup_fp32:.1f}x")
    print(f"{'TensorRT FP16':<20} {trt_fp16_mean:<12.3f} {speedup_fp16:.1f}x")

    print("\n" + "=" * 60)
    print(f"TensorRT FP16 is {speedup_fp16:.1f}x faster than PyTorch!")
    print("=" * 60)

    # Throughput
    print("\nThroughput (images/second):")
    print(f"  PyTorch FP32:   {1000/pytorch_mean:.1f} img/s")
    print(f"  TensorRT FP32:  {1000/trt_fp32_mean:.1f} img/s")
    print(f"  TensorRT FP16:  {1000/trt_fp16_mean:.1f} img/s")


if __name__ == '__main__':
    main()
