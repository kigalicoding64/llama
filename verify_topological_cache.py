import torch
import sys
import os
import json
import argparse

# Ensure the current directory is in the path so we can import models
sys.path.append(os.getcwd())

from models.llama3.generation import TopologicalCacheManager, ResilientExecutionWrapper

def test_topological_cache(verbose=True):
    """
    Verifies the functional integrity of the TopologicalCacheManager.
    Checks shape consistency and value persistence across the 'manifold' projection.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"--- Verification: Topological Cache Functional Test ---")
        print(f"Executing on device: {device}")
    
    # Initialize Manager
    bsz, heads, seq_len, dim = 2, 4, 128, 64
    manager = TopologicalCacheManager(max_batch_size=bsz, max_seq_len=seq_len, n_kv_heads=heads, head_dim=dim)
    
    # Simulated input tensors (B, H, L, D)
    dummy_keys = torch.randn(bsz, heads, 1, dim, dtype=torch.bfloat16, device=device)
    dummy_values = torch.randn(bsz, heads, 1, dim, dtype=torch.bfloat16, device=device)
    
    # Execute Projection via Resilient Wrapper
    if verbose:
        print("Projecting tensors into continuous manifold...")
    k_out, v_out = ResilientExecutionWrapper.execute(
        manager.collapse_attention, dummy_keys, dummy_values, start_pos=0, seqlen=1
    )
    
    # Shape Validation
    if verbose:
        print(f"Output Shapes: K={k_out.shape}, V={v_out.shape}")
    
    results = {
        "status": "PASS",
        "device": device,
        "validations": []
    }

    try:
        assert k_out.shape == (bsz, heads, 1, dim), f"K shape mismatch: {k_out.shape}"
        assert v_out.shape == (bsz, heads, 1, dim), f"V shape mismatch: {v_out.shape}"
        results["validations"].append({"test": "Initial Shape", "result": "PASS"})
        
        # Value Integrity Check
        assert torch.allclose(k_out, dummy_keys), "Key value integrity compromised during manifold projection."
        assert torch.allclose(v_out, dummy_values), "Value integrity compromised during manifold projection."
        results["validations"].append({"test": "Value Integrity", "result": "PASS"})
        
        # Multi-step persistence check
        dummy_keys_2 = torch.randn(bsz, heads, 1, dim, dtype=torch.bfloat16, device=device)
        dummy_values_2 = torch.randn(bsz, heads, 1, dim, dtype=torch.bfloat16, device=device)
        
        k_out_2, v_out_2 = manager.collapse_attention(dummy_keys_2, dummy_values_2, start_pos=1, seqlen=1)
        
        assert k_out_2.shape == (bsz, heads, 2, dim), f"Cumulative K shape mismatch: {k_out_2.shape}"
        assert torch.allclose(k_out_2[:, :, 0:1, :], dummy_keys), "Historical state corrupted by new injection."
        assert torch.allclose(k_out_2[:, :, 1:2, :], dummy_keys_2), "New state injection failed."
        results["validations"].append({"test": "Historical Persistence", "result": "PASS"})
        
        if verbose:
            print("Verification Result: SUCCESS - Topological Memory Projection Verified.")
    except Exception as e:
        results["status"] = "FAIL"
        results["error"] = str(e)
        if verbose:
            print(f"Verification Result: FAILED - {str(e)}")
        raise e
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()
    
    try:
        res = test_topological_cache(verbose=(args.format == "text"))
        if args.format == "json":
            print(json.dumps(res, indent=4))
    except Exception:
        if args.format == "json":
            print(json.dumps({"status": "FAIL"}, indent=4))
        sys.exit(1)
