# Build GPU RFC vector test (test_k12_all_parallel.cu in ../GPU/)
#
#   make          # build
#   make test     # build and run
#   make clean
#
# Optional: make CUDA_ARCH=sm_61   (GTX 1080 Ti)
#           make CUDA_ARCH=sm_75   (RTX 2080 Ti)

NVCC      ?= nvcc
GPU_DIR   := ../GPU
NVCCFLAGS := -std=c++17 -O2 -I$(GPU_DIR)
ifdef CUDA_ARCH
NVCCFLAGS += -arch=$(CUDA_ARCH)
endif

TARGET := test_k12_all_parallel

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(GPU_DIR)/k12_gpu.cu $(GPU_DIR)/test_k12_all_parallel.cu $(GPU_DIR)/k12_gpu.h
	$(NVCC) $(NVCCFLAGS) $(GPU_DIR)/k12_gpu.cu $(GPU_DIR)/test_k12_all_parallel.cu -o $@

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
