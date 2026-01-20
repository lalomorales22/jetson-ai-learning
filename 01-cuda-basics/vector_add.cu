#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel: runs on GPU
// __global__ means it's called from CPU but runs on GPU
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Each thread computes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;  // 1 million elements
    size_t size = n * sizeof(float);

    printf("Vector Addition of %d elements\n", n);

    // Allocate host (CPU) memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device (GPU) memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel: 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel with %d blocks of %d threads\n",
           blocksPerGrid, threadsPerBlock);

    // Start timing
    cudaEventRecord(start);

    // Launch the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Verify result
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("✓ Result verified correctly!\n");
    }

    printf("GPU Time: %.3f ms\n", milliseconds);
    printf("Throughput: %.2f GB/s\n", (3.0 * size / 1e9) / (milliseconds / 1e3));

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
