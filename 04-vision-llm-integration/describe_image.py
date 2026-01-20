#!/usr/bin/env python3
"""
Image Description with ImageNet Model + LLM
Uses MobileNetV2 (1000 classes) for much better image understanding
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
import argparse
import os

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# ImageNet class labels (will be loaded from file or URL)
IMAGENET_LABELS = None


def load_imagenet_labels():
    """Load ImageNet class labels"""
    global IMAGENET_LABELS

    labels_file = os.path.join(os.path.dirname(__file__), 'imagenet_labels.txt')

    # Try to load from local file first
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            IMAGENET_LABELS = [line.strip() for line in f.readlines()]
        return

    # Download if not present
    print("Downloading ImageNet labels...")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            IMAGENET_LABELS = response.text.strip().split('\n')
            # Save for future use
            with open(labels_file, 'w') as f:
                f.write('\n'.join(IMAGENET_LABELS))
            print(f"Saved {len(IMAGENET_LABELS)} labels to {labels_file}")
        else:
            # Fallback - use indices
            IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]
    except Exception as e:
        print(f"Could not download labels: {e}")
        IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]


def load_model(device):
    """Load MobileNetV2 pretrained on ImageNet"""
    print("Loading MobileNetV2 (ImageNet, 1000 classes)...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess image for ImageNet model"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


def get_predictions(model, image_tensor, device, top_k=5):
    """Run inference and get top-k predictions"""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        label = IMAGENET_LABELS[idx.item()] if IMAGENET_LABELS else f"class_{idx.item()}"
        results.append({
            'class': label,
            'confidence': prob.item()
        })

    return results


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
            timeout=60
        )

        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return f"Error: {e}"


def describe_image(image_path, llm_model="gemma3:1b", top_k=5, verbose=True):
    """
    Analyze image and generate natural language description

    Args:
        image_path: Path to image file
        llm_model: Ollama model to use
        top_k: Number of top predictions to consider
        verbose: Print detailed output

    Returns:
        dict with predictions and description
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 60)
        print("Image Description (ImageNet + LLM)")
        print("=" * 60)
        print(f"\nDevice: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load labels
    load_imagenet_labels()

    # Load model
    model = load_model(device)

    # Preprocess image
    if verbose:
        print(f"\nAnalyzing: {image_path}")
    image_tensor = preprocess_image(image_path)

    # Get predictions
    predictions = get_predictions(model, image_tensor, device, top_k)

    if verbose:
        print(f"\nTop {top_k} Predictions:")
        for i, pred in enumerate(predictions):
            print(f"  {i+1}. {pred['class']:30s} {pred['confidence']*100:5.2f}%")

    # Build LLM prompt
    pred_text = ", ".join([
        f"{p['class']} ({p['confidence']*100:.1f}%)"
        for p in predictions[:3]
    ])

    prompt = f"""An image was analyzed by a computer vision model. The top predictions are: {pred_text}

Based on these predictions, provide:
1. A natural description of what's likely in the image (1-2 sentences)
2. Any interesting details about the main subject
3. Possible context or setting

Keep the response concise and informative."""

    if verbose:
        print("\nGenerating description with LLM...")

    description = query_ollama(prompt, llm_model)

    if verbose:
        print("\nDescription:")
        print("-" * 60)
        print(description)
        print("-" * 60)

    return {
        'predictions': predictions,
        'description': description
    }


def main():
    parser = argparse.ArgumentParser(
        description='Describe an image using ImageNet model + LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python describe_image.py --image photo.jpg
  python describe_image.py --image cat.png --top-k 10
  python describe_image.py --image scene.jpg --llm-model llama3.2:1b
        """
    )

    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions (default: 5)')
    parser.add_argument('--llm-model', type=str, default='gemma3:1b',
                        help='Ollama model for description (default: gemma3:1b)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    describe_image(
        args.image,
        llm_model=args.llm_model,
        top_k=args.top_k,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
