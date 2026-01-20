#!/usr/bin/env python3
"""
Visual Question Answering (VQA)
Ask questions about images using ImageNet model + LLM
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

# ImageNet labels
IMAGENET_LABELS = None


def load_imagenet_labels():
    """Load ImageNet class labels"""
    global IMAGENET_LABELS

    labels_file = os.path.join(os.path.dirname(__file__), 'imagenet_labels.txt')

    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            IMAGENET_LABELS = [line.strip() for line in f.readlines()]
        return

    # Download if not present
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            IMAGENET_LABELS = response.text.strip().split('\n')
            with open(labels_file, 'w') as f:
                f.write('\n'.join(IMAGENET_LABELS))
    except:
        IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]


def load_model(device):
    """Load MobileNetV2 pretrained on ImageNet"""
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
    original_size = img.size
    return transform(img).unsqueeze(0), original_size


def get_predictions(model, image_tensor, device, top_k=10):
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


def visual_qa(image_path, question, llm_model="gemma3:1b", verbose=True):
    """
    Answer a question about an image

    Args:
        image_path: Path to image file
        question: Question to answer about the image
        llm_model: Ollama model to use
        verbose: Print detailed output

    Returns:
        dict with predictions and answer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 60)
        print("Visual Question Answering")
        print("=" * 60)
        print(f"\nImage: {image_path}")
        print(f"Question: {question}")
        print(f"Device: {device}")

    # Load labels and model
    load_imagenet_labels()
    model = load_model(device)

    # Preprocess image
    image_tensor, img_size = preprocess_image(image_path)

    # Get predictions
    predictions = get_predictions(model, image_tensor, device, top_k=10)

    if verbose:
        print(f"\nVision Model Detections:")
        for i, pred in enumerate(predictions[:5]):
            print(f"  {i+1}. {pred['class']:30s} {pred['confidence']*100:5.2f}%")

    # Build context from predictions
    high_conf = [p for p in predictions if p['confidence'] > 0.05]
    if not high_conf:
        high_conf = predictions[:3]

    detected_objects = ", ".join([
        f"{p['class']} ({p['confidence']*100:.1f}%)"
        for p in high_conf[:5]
    ])

    # Build prompt for LLM
    prompt = f"""You are analyzing an image. A computer vision model detected these objects/subjects:
{detected_objects}

The image dimensions are {img_size[0]}x{img_size[1]} pixels.

User's question: {question}

Please answer the question based on what was detected in the image. If the question cannot be fully answered from the detections alone, provide your best inference and explain your reasoning. Be concise but helpful."""

    if verbose:
        print("\nGenerating answer...")

    answer = query_ollama(prompt, llm_model)

    if verbose:
        print("\nAnswer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)

    return {
        'predictions': predictions,
        'question': question,
        'answer': answer
    }


def interactive_mode(image_path, llm_model="gemma3:1b"):
    """Interactive Q&A session about an image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Interactive Visual Q&A")
    print("=" * 60)
    print(f"\nLoading model on {device}...")

    # Load once
    load_imagenet_labels()
    model = load_model(device)

    # Analyze image once
    image_tensor, img_size = preprocess_image(image_path)
    predictions = get_predictions(model, image_tensor, device, top_k=10)

    print(f"\nImage: {image_path} ({img_size[0]}x{img_size[1]})")
    print(f"\nDetected:")
    for i, pred in enumerate(predictions[:5]):
        print(f"  {i+1}. {pred['class']:30s} {pred['confidence']*100:5.2f}%")

    # Build context
    detected_objects = ", ".join([
        f"{p['class']} ({p['confidence']*100:.1f}%)"
        for p in predictions[:5]
    ])

    print("\n" + "=" * 60)
    print("Ask questions about the image (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not question:
                continue

            prompt = f"""You are analyzing an image. A computer vision model detected: {detected_objects}

User's question: {question}

Answer based on the detections. Be concise."""

            answer = query_ollama(prompt, llm_model)
            print(f"\nAnswer: {answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description='Visual Question Answering - Ask questions about images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single question
  python visual_qa.py --image photo.jpg --question "What is this?"

  # Interactive mode - ask multiple questions
  python visual_qa.py --image photo.jpg --interactive

  # Use different LLM
  python visual_qa.py --image cat.jpg --question "What color is it?" --llm-model llama3.2:1b
        """
    )

    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--question', type=str, default='',
                        help='Question to ask about the image')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode - ask multiple questions')
    parser.add_argument('--llm-model', type=str, default='gemma3:1b',
                        help='Ollama model to use (default: gemma3:1b)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    if args.interactive:
        interactive_mode(args.image, args.llm_model)
    elif args.question:
        visual_qa(
            args.image,
            args.question,
            llm_model=args.llm_model,
            verbose=not args.quiet
        )
    else:
        print("Error: Provide --question or use --interactive mode")
        parser.print_help()


if __name__ == '__main__':
    main()
