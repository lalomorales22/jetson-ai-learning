#include <stdio.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// CUDA kernel: Convert RGB to Grayscale
// Formula: Gray = 0.299*R + 0.587*G + 0.114*B (weighted for human perception)
__global__ void rgb2gray(unsigned char *input, unsigned char *output,
                         int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int grayIdx = row * width + col;
        int rgbIdx = grayIdx * 3;

        unsigned char r = input[rgbIdx];
        unsigned char g = input[rgbIdx + 1];
        unsigned char b = input[rgbIdx + 2];

        // Weighted grayscale conversion
        output[grayIdx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <input.jpg> <output.jpg>\n", argv[0]);
        return 1;
    }

    // Load image
    int width, height, channels;
    unsigned char *h_input = stbi_load(argv[1], &width, &height, &channels, 3);

    if (!h_input) {
        printf("Error loading image %s\n", argv[1]);
        return 1;
    }

    printf("Image: %dx%d with %d channels\n", width, height, channels);

    size_t inputSize = width * height * 3;
    size_t outputSize = width * height;

    // Allocate host output
    unsigned char *h_output = (unsigned char*)malloc(outputSize);

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy input to device
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);

    // Launch kernel with 2D grid
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    printf("Launching kernel with grid (%d, %d) and block (%d, %d)\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Time the kernel
    cudaEventRecord(start);
    rgb2gray<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Time: %.3f ms\n", milliseconds);
    printf("Megapixels/sec: %.2f\n", (width * height / 1e6) / (milliseconds / 1e3));

    // Save output
    stbi_write_jpg(argv[2], width, height, 1, h_output, 90);
    printf("Saved to %s\n", argv[2]);

    // Cleanup
    stbi_image_free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
