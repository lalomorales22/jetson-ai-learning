#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine
This is where the real optimization happens!
"""

import tensorrt as trt
import os
import sys

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_file_path, engine_file_path, precision='fp32'):
    """
    Build TensorRT engine from ONNX file

    Args:
        onnx_file_path: Path to ONNX model
        engine_file_path: Path to save TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
    """
    print("=" * 60)
    print(f"Building TensorRT Engine ({precision.upper()})")
    print("=" * 60)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"\nParsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print("✓ ONNX file parsed successfully")

    # Builder configuration
    config = builder.create_builder_config()

    # Set memory pool limit (in bytes) - 2GB for Jetson
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # Enable precision mode
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            print("✓ FP16 mode enabled")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Warning: FP16 not supported on this platform")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            print("✓ INT8 mode enabled")
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 requires calibration data
            print("Note: INT8 requires calibration for best accuracy")
        else:
            print("Warning: INT8 not supported on this platform")

    # Build engine
    print("\nBuilding TensorRT engine...")
    print("This may take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None

    print("✓ Engine built successfully!")

    # Save engine
    print(f"\nSaving engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)

    engine_size = os.path.getsize(engine_file_path) / 1e6
    print(f"✓ Engine saved! Size: {engine_size:.2f} MB")

    print("\n" + "=" * 60)
    print("TensorRT Engine Ready!")
    print("=" * 60)

    return serialized_engine


def test_engine(engine_file_path):
    """Test the TensorRT engine with dummy input using PyTorch for GPU memory"""
    print("\nTesting TensorRT engine...")

    import torch
    import numpy as np

    # Load engine
    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("ERROR: Failed to load engine")
        return

    context = engine.create_execution_context()

    # Allocate buffers using PyTorch (avoids pycuda dependency)
    input_shape = (1, 3, 32, 32)
    output_shape = (1, 10)

    # Create GPU tensors with PyTorch
    d_input = torch.randn(*input_shape, dtype=torch.float32, device='cuda')
    d_output = torch.empty(output_shape, dtype=torch.float32, device='cuda')

    # Run inference using tensor data pointers
    context.set_tensor_address('input', d_input.data_ptr())
    context.set_tensor_address('output', d_output.data_ptr())
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Get results
    h_output = d_output.cpu().numpy()

    print("✓ Inference successful!")
    print(f"  Input shape: {tuple(d_input.shape)}")
    print(f"  Output shape: {h_output.shape}")
    print(f"  Output: {h_output[0][:5]}...")  # Show first 5 values


if __name__ == '__main__':
    onnx_file = 'model.onnx'
    engine_file_fp32 = 'model_fp32.trt'
    engine_file_fp16 = 'model_fp16.trt'

    if not os.path.exists(onnx_file):
        print(f"Error: {onnx_file} not found!")
        print("Please convert to ONNX first:")
        print("  python3 convert_to_onnx.py")
        sys.exit(1)

    # Check TensorRT version
    print(f"TensorRT version: {trt.__version__}\n")

    # Build FP32 engine
    build_engine(onnx_file, engine_file_fp32, precision='fp32')

    print("\n")

    # Build FP16 engine (faster on Jetson)
    build_engine(onnx_file, engine_file_fp16, precision='fp16')

    # Test one of the engines
    print("\n" + "=" * 60)
    try:
        test_engine(engine_file_fp16)
    except Exception as e:
        print(f"Engine test failed: {e}")
        print("You can still use the engine for inference")

    print("\n" + "=" * 60)
    print("Next step: Benchmark PyTorch vs TensorRT")
    print("  python3 benchmark.py")
