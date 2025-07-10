#!/usr/bin/env python3
"""Test script to understand how final Llama embeddings are 4096-dimensional."""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check what's in the saved embeddings
embeddings_file = 'Embeddings/CLEVR/Llama_patch_embeddings_percentthrumodel_100.pt'
print(f"Loading embeddings from: {embeddings_file}")
data = torch.load(embeddings_file, map_location='cpu')

print("\nKeys in embeddings file:")
for key in data.keys():
    if isinstance(data[key], torch.Tensor):
        print(f"  {key}: shape {data[key].shape}, dtype {data[key].dtype}")
    else:
        print(f"  {key}: type {type(data[key])}")

# Check the actual embeddings
if 'normalized_embeddings' in data:
    emb = data['normalized_embeddings']
    print(f"\nNormalized embeddings shape: {emb.shape}")
    print(f"Dimension: {emb.shape[1]}")
    print(f"This confirms the saved embeddings are {emb.shape[1]}-dimensional")
    
# Check raw embeddings if available
if 'embeddings' in data:
    raw_emb = data['embeddings']
    print(f"\nRaw embeddings shape: {raw_emb.shape}")
    print(f"Raw dimension: {raw_emb.shape[1]}")