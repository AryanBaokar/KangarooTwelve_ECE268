# Tests

Timing benchmarks (CPU vs GPU) and the GPU RFC vector test.

## RFC correctness (GPU)

```bash
make test_k12_all_parallel
make test-vectors
```

## Timing benchmarks

| Case | Workload |
|------|----------|
| 0 | 1 KB × 1 |
| 1 | 8 KB × 1 |
| 2 | 24 KB × 1 |
| 3 | 48 KB × 1 |
| 4 | 1 MB × 1 |
| 5 | 1 MB × 10 (batch) |
| 6 | 50 MB × 50 (batch, ~2.5 GiB packed) — **only with `BENCH_ALL=1`** |

All cases: empty customization, 32-byte digest, `byte[i] = i % 251`.

```bash
make compare              # cases 0–5 + comparison table
BENCH_ALL=1 make compare  # include case 6 (slow, needs lots of RAM/VRAM)
```

Or separately:

```bash
make bench_cpu && ./bench_cpu
make bench_gpu && ./bench_gpu
```

Optional GPU arch: `make bench_gpu CUDA_ARCH=sm_61` (1080 Ti), `sm_75` (2080 Ti).

## Outputs

Benchmarks print `BENCH_METRIC` lines; `make compare` saves `cpu_bench.out` / `gpu_bench.out` and runs `compare_metrics.py`.

**Cold start:** GPU times the first `k12_gpu_hash_batch()` after upload. CPU times a loop of `kangaroo_twelve()` (batch cases run multiple hashes sequentially).
