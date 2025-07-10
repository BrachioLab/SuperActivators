#!/usr/bin/env python3
"""Test script to verify embedding stats JSON functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embedding_stats_utils import EmbeddingStatsManager, load_normalization_stats

def test_stats_loading():
    """Test loading stats from JSON."""
    
    # Test 1: Using the manager class
    print("=== Test 1: Using EmbeddingStatsManager ===")
    manager = EmbeddingStatsManager()
    
    # List available configs
    configs = manager.list_available_configs()
    print(f"\nAvailable configurations:")
    for dataset in configs:
        print(f"  {dataset}: {list(configs[dataset].keys())}")
    
    # Try to load CLEVR/Llama stats if available
    if 'CLEVR' in configs and 'Llama' in configs.get('CLEVR', {}):
        print("\n=== Test 2: Loading CLEVR/Llama stats ===")
        mean_emb, norm = manager.get_normalization_stats(
            'CLEVR', 'Llama', 'patch', 100, device='cpu'
        )
        
        if mean_emb is not None:
            print(f"Successfully loaded stats:")
            print(f"  Mean embedding shape: {mean_emb.shape}")
            print(f"  Train norm: {norm}")
            
            # Get info about the embeddings
            info = manager.get_embedding_info('CLEVR', 'Llama', 'patch', 100)
            print(f"\nEmbedding info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("No stats found for CLEVR/Llama")
    
    # Test 3: Using the convenience function
    print("\n=== Test 3: Using convenience function ===")
    mean_emb2, norm2 = load_normalization_stats(
        'CLEVR', 'Llama', 'patch', 100, device='cpu'
    )
    
    if mean_emb2 is not None:
        print(f"Successfully loaded via convenience function")
        print(f"  Shapes match: {mean_emb.shape == mean_emb2.shape if mean_emb is not None else 'N/A'}")
    
    # Test 4: Check if layer analysis would work
    print("\n=== Test 4: Simulating layer analysis loading ===")
    from utils.layer_analysis_utils import load_final_layer_stats
    
    try:
        mean_emb3, norm3 = load_final_layer_stats('CLEVR', 'llama', device='cpu')
        print("Layer analysis loading successful!")
        print(f"  Mean shape: {mean_emb3.shape}")
        print(f"  Norm: {norm3}")
    except Exception as e:
        print(f"Layer analysis loading failed: {e}")
    
    # Print summary
    manager.print_summary()


if __name__ == "__main__":
    test_stats_loading()