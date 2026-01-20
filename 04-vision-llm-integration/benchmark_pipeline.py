#!/usr/bin/env python3
"""
Benchmark the Vision + LLM Pipeline
Measures latency for each stage: preprocessing, vision model, LLM
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import time
import argparse
import os
import numpy as np

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"


def benchmark_preprocessing(image_path, num_runs=20):
    """Benchmark image preprocessing"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Warmup
    img = Image.open(image_path).convert('RGB')
    _ = transform(img)

    times = []
    for _ in range(num_runs):
        img = Image.open(image_path).convert('RGB')
        start = time.perf_counter()
        tensor = transform(img).unsqueeze(0)
        times.append((time.perf_counter() - start) * 1000)

    return np.array(times), tensor


def benchmark_vision_model(model, image_tensor, device, num_warmup=10, num_runs=50):
    """Benchmark vision model inference"""
    image_tensor = image_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(image_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            output = model(image_tensor)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append((time.perf_counter() - start) * 1000)

    return np.array(times), output


def benchmark_llm(prompt, model_name="gemma3:1b", num_runs=3):
    """Benchmark LLM response time"""
    times = []

    for i in range(num_runs):
        start = time.perf_counter()
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            elapsed = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                times.append(elapsed)
                result = response.json()['response']
            else:
                print(f"  Run {i+1}: Error {response.status_code}")
                return None, None

        except Exception as e:
            print(f"  Run {i+1}: Error - {e}")
            return None, None

    return np.array(times), result


def print_stats(name, times_ms):
    """Print timing statistics"""
    if times_ms is None or len(times_ms) == 0:
        print(f"  {name}: No data")
        return 0

    print(f"  {name}:")
    print(f"    Mean:   {times_ms.mean():8.2f} ms")
    print(f"    Std:    {times_ms.std():8.2f} ms")
    print(f"    Min:    {times_ms.min():8.2f} ms")
    print(f"    Max:    {times_ms.max():8.2f} ms")
    return times_ms.mean()


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Vision + LLM Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_pipeline.py --image photo.jpg
  python benchmark_pipeline.py --image photo.jpg --llm-model llama3.2:1b
  python benchmark_pipeline.py --image photo.jpg --vision-runs 100
        """
    )

    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--llm-model', type=str, default='gemma3:1b',
                        help='Ollama model to benchmark (default: gemma3:1b)')
    parser.add_argument('--vision-runs', type=int, default=50,
                        help='Number of vision model runs (default: 50)')
    parser.add_argument('--llm-runs', type=int, default=3,
                        help='Number of LLM runs (default: 3)')
    parser.add_argument('--skip-llm', action='store_true',
                        help='Skip LLM benchmark')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Vision + LLM Pipeline Benchmark")
    print("=" * 60)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image: {args.image}")
    print(f"LLM Model: {args.llm_model}")

    # Load vision model
    print("\nLoading MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # 1. Benchmark preprocessing
    print("\n" + "-" * 60)
    print("Stage 1: Image Preprocessing")
    print("-" * 60)
    preprocess_times, image_tensor = benchmark_preprocessing(args.image)
    preprocess_mean = print_stats("Preprocessing", preprocess_times)

    # 2. Benchmark vision model
    print("\n" + "-" * 60)
    print("Stage 2: Vision Model Inference")
    print("-" * 60)
    print(f"  Running {args.vision_runs} iterations...")
    vision_times, output = benchmark_vision_model(
        model, image_tensor, device, num_runs=args.vision_runs
    )
    vision_mean = print_stats("MobileNetV2", vision_times)

    # Get top prediction for LLM prompt
    probs = F.softmax(output[0], dim=0)
    top_prob, top_idx = torch.max(probs, dim=0)

    # 3. Benchmark LLM (optional)
    llm_mean = 0
    if not args.skip_llm:
        print("\n" + "-" * 60)
        print("Stage 3: LLM Response Generation")
        print("-" * 60)

        prompt = f"The image shows a {top_idx.item()} with {top_prob.item()*100:.1f}% confidence. Describe it briefly in 2 sentences."

        print(f"  Running {args.llm_runs} iterations...")
        llm_times, _ = benchmark_llm(prompt, args.llm_model, args.llm_runs)

        if llm_times is not None:
            llm_mean = print_stats(args.llm_model, llm_times)
        else:
            print("  LLM benchmark failed - is Ollama running?")
            print("  Start with: ollama serve")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = preprocess_mean + vision_mean + llm_mean

    print(f"\n{'Stage':<25} {'Time (ms)':<12} {'% of Total':<10}")
    print("-" * 50)
    print(f"{'Preprocessing':<25} {preprocess_mean:<12.2f} {preprocess_mean/total*100:<10.1f}%")
    print(f"{'Vision Model':<25} {vision_mean:<12.2f} {vision_mean/total*100:<10.1f}%")
    if not args.skip_llm and llm_mean > 0:
        print(f"{'LLM ({})'.format(args.llm_model):<25} {llm_mean:<12.2f} {llm_mean/total*100:<10.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total:<12.2f} {'100.0':<10}%")

    print(f"\nEnd-to-end latency: {total:.0f} ms ({1000/total:.1f} queries/sec)")

    # Bottleneck analysis
    print("\n" + "-" * 60)
    print("Analysis")
    print("-" * 60)

    if llm_mean > vision_mean * 10:
        print("Bottleneck: LLM response generation")
        print("Suggestions:")
        print("  - Use a smaller LLM model (e.g., gemma3:1b instead of 7b)")
        print("  - Use shorter prompts")
        print("  - Consider response streaming for perceived speed")
    elif vision_mean > preprocess_mean * 10:
        print("Bottleneck: Vision model inference")
        print("Suggestions:")
        print("  - Convert to TensorRT for 5-10x speedup")
        print("  - Use FP16 precision")
        print("  - Use a lighter model (MobileNet vs ResNet)")
    else:
        print("Pipeline is well-balanced!")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
