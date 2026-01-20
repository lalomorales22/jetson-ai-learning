#!/usr/bin/env python3
"""Download a test image for CUDA examples"""

import urllib.request
import os

def download_test_image():
    """Download a sample image from Unsplash"""
    url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4"
    params = "?w=1920&q=80"
    output_path = "01-cuda-basics/test_image.jpg"

    if os.path.exists(output_path):
        print(f"Test image already exists at {output_path}")
        return output_path

    print(f"Downloading test image...")
    try:
        urllib.request.urlretrieve(url + params, output_path)
        print(f"✓ Downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading: {e}")
        print("You can use any .jpg image instead!")
        return None

if __name__ == "__main__":
    download_test_image()
