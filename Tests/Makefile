# Tests: GPU RFC vectors + CPU/GPU timing benchmarks
#
#   make test-vectors     # 14 RFC vectors (GPU)
#   make compare          # CPU vs GPU timing (cases 0-5)
#   BENCH_ALL=1 make compare   # include 50 MB x50 case

CXX       ?= g++
NVCC      ?= nvcc
CPU_DIR   := ../CPU
GPU_DIR   := ../GPU

CXXFLAGS  := -std=c++17 -O2 -Wall -I$(CPU_DIR)
NVCCFLAGS := -std=c++17 -O2 -I$(GPU_DIR)
ifdef CUDA_ARCH
NVCCFLAGS += -arch=$(CUDA_ARCH)
endif

.PHONY: all clean compare test-vectors \
        bench_cpu bench_gpu test_k12_all_parallel

all: bench_cpu bench_gpu test_k12_all_parallel

# GPU RFC vector test
test_k12_all_parallel: $(GPU_DIR)/k12_gpu.cu $(GPU_DIR)/keccak_turboshake_gpu.cuh \
		$(GPU_DIR)/test_k12_all_parallel.cu $(GPU_DIR)/k12_gpu.h
	$(NVCC) $(NVCCFLAGS) $(GPU_DIR)/k12_gpu.cu $(GPU_DIR)/test_k12_all_parallel.cu -o $@

test-vectors: test_k12_all_parallel
	./test_k12_all_parallel

# Timing benchmarks
bench_cpu: bench_cpu.cpp bench_common.h
	$(CXX) $(CXXFLAGS) bench_cpu.cpp \
		$(CPU_DIR)/keccak_p.cpp $(CPU_DIR)/turboshake128.cpp $(CPU_DIR)/k12.cpp \
		-o $@

bench_gpu: bench_gpu.cu bench_common.h $(GPU_DIR)/k12_gpu.cu $(GPU_DIR)/keccak_turboshake_gpu.cuh \
		$(GPU_DIR)/k12_gpu.h
	$(NVCC) $(NVCCFLAGS) bench_gpu.cu $(GPU_DIR)/k12_gpu.cu -o $@

compare: bench_cpu bench_gpu
	@./bench_cpu | tee cpu_bench.out
	@./bench_gpu | tee gpu_bench.out
	@python3 compare_metrics.py

clean:
	rm -f bench_cpu bench_gpu test_k12_all_parallel cpu_bench.out gpu_bench.out
