#!/usr/bin/env python3
"""
Matrix multiplication benchmark for Apple Silicon using PyTorch MPS backend.

This script measures effective GEMM throughput (which may include overhead).
Larger matrix sizes reduce launch overhead relative to computation time.
"""

import argparse
import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark matrix multiplication on Apple Silicon (MPS)"
    )
    parser.add_argument("--m", type=int, default=4096, help="Matrix A rows (default: 4096)")
    parser.add_argument("--n", type=int, default=4096, help="Matrix B columns (default: 4096)")
    parser.add_argument("--k", type=int, default=4096, help="Matrix A columns / B rows (default: 4096)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Data type: fp16 or fp32 (default: fp16)"
    )
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations to benchmark (default: 100)")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations (default: 20)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("ERROR: MPS backend is not available on this system.")
        print("This script requires Apple Silicon with PyTorch MPS support.")
        sys.exit(1)
    
    if not torch.backends.mps.is_built():
        print("ERROR: PyTorch was not built with MPS support.")
        sys.exit(1)
    
    device = torch.device("mps")
    
    # Map dtype string to torch dtype
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    m, n, k = args.m, args.n, args.k
    
    print(f"Matrix multiplication benchmark (MPS)")
    print(f"M={m}, N={n}, K={k}, dtype={args.dtype}")
    print(f"Warmup iterations: {args.warmup}, Benchmark iterations: {args.iters}")
    print()
    
    # Get initial memory stats
    torch.mps.synchronize()
    mem_before = torch.mps.current_allocated_memory()
    driver_mem_before = torch.mps.driver_allocated_memory()
    
    # Create tensors on device
    A = torch.randn(m, k, dtype=dtype, device=device)
    B = torch.randn(k, n, dtype=dtype, device=device)
    
    # Warmup loop
    print("Warming up...")
    for _ in range(args.warmup):
        C = A @ B
        torch.mps.synchronize()
    
    # Get memory stats after allocation
    torch.mps.synchronize()
    mem_after = torch.mps.current_allocated_memory()
    driver_mem_after = torch.mps.driver_allocated_memory()
    
    # Benchmark loop
    print("Benchmarking...")
    torch.mps.synchronize()
    
    import time
    start_time = time.perf_counter()
    
    for _ in range(args.iters):
        C = A @ B
        torch.mps.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time_seconds = end_time - start_time
    avg_time_ms = (total_time_seconds / args.iters) * 1000.0
    
    # TFLOPs = 2 * m * n * k operations per matmul
    # TFLOPs/s = (2 * m * n * k) / time_seconds / 1e12
    ops_per_matmul = 2 * m * n * k
    tflops = (ops_per_matmul / (total_time_seconds / args.iters)) / 1e12
    
    # Memory stats in MB
    mem_used_mb = (mem_after - mem_before) / (1024 * 1024)
    driver_mem_mb = driver_mem_after / (1024 * 1024)
    
    # Print results
    print()
    print("=" * 60)
    print("Results:")
    print(f"  M={m}, N={n}, K={k}, dtype={args.dtype}")
    print(f"  Average time per matmul: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {tflops:.3f} TFLOPs/s")
    print(f"  MPS memory allocated: {mem_used_mb:.2f} MB")
    print(f"  MPS driver memory: {driver_mem_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()

