#!/usr/bin/env python3
"""
Vision + LLM Integration Demo
Combines TensorRT vision model with Ollama LLM
"""

import sys
import os
import requests
import json
from PIL import Image
import numpy as np

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"


def preprocess_image(image_path):
    """Load and preprocess image for CIFAR-10 model"""
    img = Image.open(image_path).convert('RGB')

    # Resize to 32x32 (CIFAR-10 input size)
    img = img.resize((32, 32), Image.BILINEAR)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize (same as training)
    img_array = img_array / 255.0
    img_array = (img_array - 0.5) / 0.5

    # Transpose to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Ensure array is contiguous (required for pycuda/TensorRT)
    img_array = np.ascontiguousarray(img_array)

    return img_array


def run_pytorch_inference(image_path):
    """Run inference using PyTorch model (fallback)"""
    try:
        import torch
        sys.path.append('../02-ml-training')
        from train_classifier import SimpleCNN

        # Load model
        model = SimpleCNN()
        checkpoint = torch.load('../02-ml-training/best_model.pth', map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Preprocess
        img_array = preprocess_image(image_path)
        input_tensor = torch.from_numpy(img_array)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)

        results = []
        for i in range(5):
            results.append({
                'class': CLASSES[top5_idx[i]],
                'confidence': float(top5_prob[i])
            })

        return results

    except Exception as e:
        print(f"PyTorch inference failed: {e}")
        return None


def run_tensorrt_inference(image_path, engine_path):
    """Run inference using TensorRT engine"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Preprocess image
        h_input = preprocess_image(image_path)
        h_output = np.empty((1, 10), dtype=np.float32)

        # Allocate device memory
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # Transfer input to GPU
        cuda.memcpy_htod(d_input, h_input)

        # Run inference
        context.set_tensor_address('input', int(d_input))
        context.set_tensor_address('output', int(d_output))
        context.execute_async_v3(stream_handle=0)

        # Get results
        cuda.memcpy_dtoh(h_output, d_output)

        # Softmax
        output = h_output[0]
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / np.sum(exp_output)

        # Get top 5
        top5_idx = np.argsort(probabilities)[::-1][:5]

        results = []
        for idx in top5_idx:
            results.append({
                'class': CLASSES[idx],
                'confidence': float(probabilities[idx])
            })

        return results

    except Exception as e:
        print(f"TensorRT inference failed: {e}")
        return run_pytorch_inference(image_path)


def query_ollama(prompt, model="gemma3:1b"):
    """Query Ollama LLM"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"

    except Exception as e:
        return f"Error querying Ollama: {e}"


def describe_image(image_path, use_tensorrt=True):
    """
    Analyze image and generate description using vision model + LLM
    """
    print("=" * 60)
    print("Vision + LLM Image Analysis")
    print("=" * 60)

    # Run vision model
    print(f"\n1. Analyzing image: {image_path}")

    if use_tensorrt and os.path.exists('../03-tensorrt-optimization/model_fp16.trt'):
        print("   Using TensorRT FP16 model...")
        results = run_tensorrt_inference(
            image_path,
            '../03-tensorrt-optimization/model_fp16.trt'
        )
    else:
        print("   Using PyTorch model...")
        results = run_pytorch_inference(image_path)

    if not results:
        print("Vision model inference failed!")
        return

    # Display predictions
    print("\n2. Vision Model Results:")
    for i, pred in enumerate(results):
        print(f"   {i+1}. {pred['class']:12s} {pred['confidence']*100:5.2f}%")

    # Create prompt for LLM
    top_class = results[0]['class']
    top_conf = results[0]['confidence']

    prompt = f"""The computer vision model identified this image as a '{top_class}' with {top_conf*100:.1f}% confidence.

Please provide:
1. A brief description of what a {top_class} is
2. Key characteristics or features
3. An interesting fact

Keep the response concise (3-4 sentences)."""

    # Query LLM
    print("\n3. Querying LLM for description...")
    print("   (This may take a few seconds...)")

    llm_response = query_ollama(prompt)

    print("\n4. LLM Description:")
    print("-" * 60)
    print(llm_response)
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


def visual_qa(image_path, question, use_tensorrt=True):
    """Answer a question about an image"""
    print("=" * 60)
    print("Visual Question Answering")
    print("=" * 60)

    # Run vision model
    print(f"\nAnalyzing image: {image_path}")

    if use_tensorrt and os.path.exists('../03-tensorrt-optimization/model_fp16.trt'):
        results = run_tensorrt_inference(
            image_path,
            '../03-tensorrt-optimization/model_fp16.trt'
        )
    else:
        results = run_pytorch_inference(image_path)

    if not results:
        print("Vision model inference failed!")
        return

    # Create context for LLM
    predictions = ", ".join([
        f"{r['class']} ({r['confidence']*100:.1f}%)"
        for r in results[:3]
    ])

    prompt = f"""Based on a computer vision analysis, this image most likely contains: {predictions}

User question: {question}

Please answer the question based on this information."""

    # Query LLM
    print(f"\nQuestion: {question}")
    print("\nLLM Answer:")
    print("-" * 60)

    answer = query_ollama(prompt)
    print(answer)
    print("-" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Vision + LLM Demo')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--mode', type=str, default='describe',
                        choices=['describe', 'qa'],
                        help='Mode: describe or qa (question answering)')
    parser.add_argument('--question', type=str, default='',
                        help='Question for QA mode')
    parser.add_argument('--no-tensorrt', action='store_true',
                        help='Use PyTorch instead of TensorRT')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found!")
        return

    use_trt = not args.no_tensorrt

    if args.mode == 'describe':
        describe_image(args.image, use_tensorrt=use_trt)
    elif args.mode == 'qa':
        if not args.question:
            print("Error: --question required for QA mode")
            return
        visual_qa(args.image, args.question, use_tensorrt=use_trt)


if __name__ == '__main__':
    main()
