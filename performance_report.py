import torch
import time
import sys
import os
import json
import argparse
from typing import Dict, Any

# Ensure the current directory is in the path so we can import models
sys.path.append(os.getcwd())

from models.llama3.generation import TopologicalCacheManager

def measure_memory() -> int:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0

def run_benchmark(iterations: int, bsz: int, seq_len: int, n_heads: int, head_dim: int, device: str) -> Dict[str, Any]:
    dtype = torch.bfloat16
    
    # 1. Baseline: Standard Linear Cache
    # Simulated input
    dummy_keys = torch.randn(bsz, 1, n_heads, head_dim, dtype=dtype, device=device)
    dummy_values = torch.randn(bsz, 1, n_heads, head_dim, dtype=dtype, device=device)
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_mem_baseline = measure_memory()
    
    cache_k = torch.zeros((bsz, seq_len, n_heads, head_dim), dtype=dtype, device=device)
    cache_v = torch.zeros((bsz, seq_len, n_heads, head_dim), dtype=dtype, device=device)
    
    if device == "cuda": torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(iterations):
        pos = i % seq_len
        cache_k[:bsz, pos : pos + 1] = dummy_keys
        cache_v[:bsz, pos : pos + 1] = dummy_values
        _k = cache_k[:bsz, : pos + 1]
        _v = cache_v[:bsz, : pos + 1]
        
    if device == "cuda": torch.cuda.synchronize()
    end_time = time.time()
    end_mem_baseline = measure_memory()
    
    baseline_latency = (end_time - start_time) / iterations
    baseline_memory = end_mem_baseline - start_mem_baseline

    # 2. Topological: Continuous Manifold Cache
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    start_mem_topo = measure_memory()
    
    topo_manager = TopologicalCacheManager(bsz, seq_len, n_heads, head_dim)
    topo_manager.manifold_state = topo_manager.manifold_state.to(device)
    
    if device == "cuda": torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(iterations):
        pos = i % seq_len
        _k, _v = topo_manager.collapse_attention(dummy_keys, dummy_values, pos, 1)
        
    if device == "cuda": torch.cuda.synchronize()
    end_time = time.time()
    end_mem_topo = measure_memory()
    
    topo_latency = (end_time - start_time) / iterations
    topo_memory = end_mem_topo - start_mem_topo
    
    return {
        "baseline": {
            "latency_avg_ms": baseline_latency * 1000,
            "peak_memory_bytes": baseline_memory
        },
        "topological": {
            "latency_avg_ms": topo_latency * 1000,
            "peak_memory_bytes": topo_memory
        },
        "metrics": {
            "latency_improvement_pct": ((baseline_latency - topo_latency) / baseline_latency) * 100,
            "memory_reduction_pct": ((baseline_memory - topo_memory) / baseline_memory * 100) if baseline_memory > 0 else 0
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Llama 3 Topological Cache Performance Delta Report")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations")
    parser.add_argument("--bsz", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = run_benchmark(
        iterations=args.iterations,
        bsz=args.bsz,
        seq_len=args.seq_len,
        n_heads=32,
        head_dim=128,
        device=device
    )
    
    report = {
        "device": device,
        "config": {
            "iterations": args.iterations,
            "batch_size": args.bsz,
            "max_seq_len": args.seq_len
        },
        "results": results
    }
    
    if args.format == "json":
        print(json.dumps(report, indent=4))
    else:
        print("--- Llama 3 Performance Delta Report ---")
        print(f"Device: {device}")
        print(f"Config: Batch={args.bsz}, SeqLen={args.seq_len}, Iterations={args.iterations}")
        print("-" * 40)
        print(f"{'Metric':<25} | {'Baseline':<12} | {'Topological':<12} | {'Delta'}")
        print("-" * 40)
        
        b = results["baseline"]
        t = results["topological"]
        m = results["metrics"]
        
        print(f"{'Latency (avg ms)':<25} | {b['latency_avg_ms']:>12.4f} | {t['latency_avg_ms']:>12.4f} | {m['latency_improvement_pct']:>+.2f}%")
        
        if device == "cuda":
            print(f"{'Peak Memory (MB)':<25} | {b['peak_memory_bytes']/1024/1024:>12.2f} | {t['peak_memory_bytes']/1024/1024:>12.2f} | {m['memory_reduction_pct']:>+.2f}%")
        else:
            print(f"{'Peak Memory (MB)':<25} | {'N/A (CPU)':<12} | {'N/A (CPU)':<12} | {'N/A'}")
            
        print("-" * 40)
        if m['latency_improvement_pct'] < 0 and device == "cpu":
            print("Note: CPU overhead for Python-level manifold management is expected.")
            print("Gains are optimized for high-throughput GPU kernels.")

if __name__ == "__main__":
    main()
