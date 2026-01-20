# Makefile for CUDA examples
NVCC = /usr/local/cuda-12.6/bin/nvcc
CUDA_FLAGS = -arch=sm_87 -O2

# Jetson Orin Nano uses sm_87 architecture
# -O2 enables optimizations

all: vector_add image_grayscale

vector_add: vector_add.cu
	$(NVCC) $(CUDA_FLAGS) vector_add.cu -o vector_add

image_grayscale: image_grayscale.cu stb_image.h stb_image_write.h
	$(NVCC) $(CUDA_FLAGS) image_grayscale.cu -o image_grayscale

clean:
	rm -f vector_add image_grayscale

run_vector: vector_add
	./vector_add

run_image: image_grayscale
	@if [ ! -f test_image.jpg ]; then \
		echo "Please provide a test image: test_image.jpg"; \
		echo "Or run: python3 ../test_image_download.py"; \
		exit 1; \
	fi
	./image_grayscale test_image.jpg test_output.jpg
