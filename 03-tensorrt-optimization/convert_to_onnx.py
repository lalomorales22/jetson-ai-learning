#!/usr/bin/env python3
"""
Convert PyTorch model to ONNX format
ONNX is an intermediate format that TensorRT can optimize
"""

import torch
import sys
import os

# Import model from training module
sys.path.append('../02-ml-training')
from train_classifier import SimpleCNN


def export_to_onnx(pytorch_model_path, onnx_output_path, input_shape=(1, 3, 32, 32)):
    """
    Export PyTorch model to ONNX format

    Args:
        pytorch_model_path: Path to .pth file
        onnx_output_path: Path to save .onnx file
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    print("=" * 60)
    print("PyTorch to ONNX Conversion")
    print("=" * 60)

    # Load PyTorch model
    print(f"\nLoading PyTorch model from {pytorch_model_path}...")
    model = SimpleCNN()
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)
    print(f"Input shape: {input_shape}")

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=17,  # ONNX opset version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
        # Note: Not using dynamic_axes for TensorRT compatibility
        # Fixed batch size of 1 is used for inference
    )

    print(f"✓ ONNX model saved to {onnx_output_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid!")

    # Check file size
    file_size = os.path.getsize(onnx_output_path) / 1e6
    print(f"ONNX model size: {file_size:.2f} MB")

    # Test inference with ONNX Runtime
    try:
        import onnxruntime as ort
        print("\nTesting ONNX Runtime inference...")

        session = ort.InferenceSession(
            onnx_output_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        import numpy as np
        test_input = np.random.randn(*input_shape).astype(np.float32)
        result = session.run([output_name], {input_name: test_input})

        print(f"✓ ONNX Runtime inference successful!")
        print(f"  Output shape: {result[0].shape}")

    except ImportError:
        print("\nONNX Runtime not installed. Install with:")
        print("  pip3 install onnxruntime-gpu")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nNext step: Convert ONNX to TensorRT")
    print("  python3 convert_to_tensorrt.py")


if __name__ == '__main__':
    # Default paths
    pytorch_model = '../02-ml-training/best_model.pth'
    onnx_output = 'model.onnx'

    if not os.path.exists(pytorch_model):
        print(f"Error: {pytorch_model} not found!")
        print("Please train the model first:")
        print("  cd ../02-ml-training")
        print("  python3 train_classifier.py")
        sys.exit(1)

    export_to_onnx(pytorch_model, onnx_output)
