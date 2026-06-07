import torch
import time
import sys
import os
import json
import argparse

# Ensure the current directory is in the path so we can import models
sys.path.append(os.getcwd())

from models.llama3.generation import TopologicalCacheManager

def benchmark_cache(verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"--- Benchmark: Topological vs Standard Cache Performance ---")
        print(f"Executing on device: {device}")
    
    # Parameters for realistic inference load
    bsz = 8
    max_seq_len = 2048
    n_kv_heads = 32
    head_dim = 128
    dtype = torch.bfloat16
    
    # 1. Standard Linear Cache Logic (Replicated from original model.py)
    # Original layout was (B, L, H, D)
    cache_k = torch.zeros((bsz, max_seq_len, n_kv_heads, head_dim), dtype=dtype, device=device)
    cache_v = torch.zeros((bsz, max_seq_len, n_kv_heads, head_dim), dtype=dtype, device=device)
    
    # 2. Topological Cache Logic
    # New layout is (B, H, L, D) for manifold continuity
    topo_manager = TopologicalCacheManager(bsz, max_seq_len, n_kv_heads, head_dim)
    topo_manager.manifold_state = topo_manager.manifold_state.to(device)
    
    # Warmup data
    dummy_keys = torch.randn(bsz, 1, n_kv_heads, head_dim, dtype=dtype, device=device)
    dummy_values = torch.randn(bsz, 1, n_kv_heads, head_dim, dtype=dtype, device=device)
    
    iterations = 200
    
    # Benchmark Standard
    if verbose:
        print(f"Measuring Standard Cache (Linear {iterations} steps)...")
    if device == "cuda": torch.cuda.synchronize()
    start = time.time()
    for i in range(iterations):
        pos = i % max_seq_len
        cache_k[:bsz, pos:pos+1] = dummy_keys
        cache_v[:bsz, pos:pos+1] = dummy_values
        _k = cache_k[:bsz, :pos+1]
        _v = cache_v[:bsz, :pos+1]
    if device == "cuda": torch.cuda.synchronize()
    standard_time = time.time() - start
    
    # Benchmark Topological
    if verbose:
        print(f"Measuring Topological Cache (Manifold {iterations} steps)...")
    if device == "cuda": torch.cuda.synchronize()
    start = time.time()
    for i in range(iterations):
        pos = i % max_seq_len
        _k, _v = topo_manager.collapse_attention(dummy_keys, dummy_values, pos, 1)
    if device == "cuda": torch.cuda.synchronize()
    topo_time = time.time() - start
    
    improvement = (standard_time - topo_time) / standard_time * 100
    
    results = {
        "device": device,
        "iterations": iterations,
        "standard_time_sec": standard_time,
        "topological_time_sec": topo_time,
        "improvement_pct": improvement
    }
    
    if verbose:
        print("-" * 40)
        print(f"Standard Cache Time:    {standard_time:.6f}s")
        print(f"Topological Cache Time: {topo_time:.6f}s")
        
        if improvement > 0:
            print(f"Performance Gain:       {improvement:.2f}%")
        else:
            print(f"Performance Change:     {improvement:.2f}% (overhead exceeding gain on this hardware)")
        
        print("-" * 40)
        print("Note: Performance gains are typically realized on GPU hardware with large sequence horizons.")
        print("On CPU, the Python-level overhead for manifold slicing may dominate.")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()
    
    res = benchmark_cache(verbose=(args.format == "text"))
    if args.format == "json":
        print(json.dumps(res, indent=4))
