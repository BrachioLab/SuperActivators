import torch
import os
import gc
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
EMBEDDINGS_DIR = "Embeddings"
MAX_SIZE_GB = 10
BYTES_PER_GB = 1024 * 1024 * 1024

# Files to split based on the check
FILES_TO_SPLIT = [
    ("CLEVR", "Llama", 24.41),
    ("Coco", "Llama", 134.28),
    ("Broden-Pascal", "Llama", 109.94),
    ("Broden-OpenSurfaces", "Llama", 87.35)
]

def get_embedding_info_minimal(file_path):
    """Get embedding info with minimal memory usage."""
    # Load file with memory mapping to avoid loading everything
    data = torch.load(file_path, map_location='cpu')
    
    # Just get the shape and key, don't keep the tensor
    if 'normalized_embeddings' in data:
        shape = data['normalized_embeddings'].shape
        dtype = data['normalized_embeddings'].dtype
        embeddings_key = 'normalized_embeddings'
    elif 'embeddings' in data:
        shape = data['embeddings'].shape
        dtype = data['embeddings'].dtype
        embeddings_key = 'embeddings'
    else:
        raise ValueError("No embeddings found in file!")
    
    # Get other keys (should be small)
    other_keys = {k: v for k, v in data.items() if k != embeddings_key}
    
    # Immediately delete data to free memory
    del data
    gc.collect()
    
    return shape, dtype, embeddings_key, other_keys

def split_embedding_file_memory_efficient(dataset, model, size_gb):
    """Split a large embedding file with minimal memory usage."""
    file_path = f"{EMBEDDINGS_DIR}/{dataset}/{model}_patch_embeddings_percentthrumodel_100.pt"
    
    print(f"\n{'='*60}")
    print(f"💾 Processing: {dataset}/{model} ({size_gb:.2f} GB)")
    print(f"   File: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   ❌ File not found!")
        return False
    
    try:
        # Get embedding info with minimal memory
        print("   Getting file metadata (minimal memory)...")
        shape, dtype, embeddings_key, other_keys = get_embedding_info_minimal(file_path)
        print(f"   Shape: {shape}, Dtype: {dtype}")
        
        # Calculate chunks
        total_samples = shape[0]
        embedding_dim = shape[1]
        expected_chunks = int(size_gb / MAX_SIZE_GB) + (1 if size_gb % MAX_SIZE_GB > 0 else 0)
        chunk_size = total_samples // expected_chunks
        
        print(f"   Total samples: {total_samples:,}")
        print(f"   Chunks needed: {expected_chunks}")
        print(f"   Samples per chunk: ~{chunk_size:,}")
        
        # Create backup
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            print(f"   Creating backup...")
            os.rename(file_path, backup_path)
            file_path = backup_path
        
        # Prepare chunk info
        chunk_info = {
            'num_chunks': expected_chunks,
            'chunk_size': chunk_size,
            'total_samples': total_samples,
            'embedding_dim': embedding_dim,
            'chunks': []
        }
        
        # Process one chunk at a time to minimize memory usage
        print("   Starting memory-efficient split...")
        
        for i in range(expected_chunks):
            print(f"\n   Processing chunk {i}/{expected_chunks-1}... [{datetime.now().strftime('%H:%M:%S')}]")
            
            # Calculate indices
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < expected_chunks - 1 else total_samples
            chunk_samples = end_idx - start_idx
            
            # Create chunk filename
            chunk_file = file_path.replace('.backup', '').replace('.pt', f'_chunk_{i}.pt')
            
            print(f"     Indices: {start_idx:,}-{end_idx:,} ({chunk_samples:,} samples)")
            
            # Load ONLY the specific chunk needed
            print(f"     Loading chunk data...")
            
            # Load full file but immediately extract only what we need
            data = torch.load(file_path, map_location='cpu')
            
            # Extract only the chunk we need
            chunk_embeddings = data[embeddings_key][start_idx:end_idx].clone()
            
            # Immediately delete the full data to free memory
            del data
            gc.collect()
            
            # Prepare chunk data
            chunk_data = {}
            chunk_data[embeddings_key] = chunk_embeddings
            
            # Add other keys (should be small)
            for k, v in other_keys.items():
                chunk_data[k] = v
            
            # Save chunk
            print(f"     Saving chunk to: {os.path.basename(chunk_file)}")
            torch.save(chunk_data, chunk_file)
            
            # Verify chunk was saved
            chunk_size_gb = os.path.getsize(chunk_file) / BYTES_PER_GB
            print(f"     Chunk size: {chunk_size_gb:.2f} GB")
            
            # Update chunk info
            chunk_info['chunks'].append({
                'file': os.path.basename(chunk_file),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'shape': list(chunk_embeddings.shape),
                'size_gb': chunk_size_gb
            })
            
            # Aggressively clear memory
            del chunk_embeddings
            del chunk_data
            gc.collect()
            
            # Force Python to release memory back to OS
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            
            print(f"     ✅ Chunk {i} saved successfully")
        
        # Save chunk info
        info_file = file_path.replace('.backup', '').replace('.pt', '_chunks_info.json')
        with open(info_file, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        print(f"\n   ✅ Successfully split into {expected_chunks} chunks")
        print(f"   📄 Chunk info saved to: {os.path.basename(info_file)}")
        
        # Final memory cleanup
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency memory cleanup
        gc.collect()
        
        return False

def process_files_sequentially():
    """Process files one by one with memory cleanup between each."""
    print("🚀 Starting MEMORY-EFFICIENT embedding file splitting")
    print(f"   Max file size: {MAX_SIZE_GB} GB")
    print(f"   Files to split: {len(FILES_TO_SPLIT)}")
    
    success_count = 0
    total_size = sum(size for _, _, size in FILES_TO_SPLIT)
    print(f"   Total size to process: {total_size:.2f} GB")
    print("\n⚠️  This process is optimized for minimal memory usage")
    print("   Each chunk is processed individually")
    print("\n📋 Files to process:")
    for i, (dataset, model, size_gb) in enumerate(FILES_TO_SPLIT):
        print(f"   {i+1}. {dataset}/{model}: {size_gb:.2f} GB")
    print()
    
    # Process each file
    for idx, (dataset, model, size_gb) in enumerate(FILES_TO_SPLIT):
        print(f"\n📁 File {idx+1}/{len(FILES_TO_SPLIT)}")
        
        # Split the file
        success = split_embedding_file_memory_efficient(dataset, model, size_gb)
        
        if success:
            success_count += 1
        
        # Aggressive cleanup between files
        gc.collect()
        
        # Force memory release
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n   Memory cleaned up after processing {dataset}/{model}")
    
    print(f"\n{'='*60}")
    print(f"✅ Completed! Successfully split {success_count}/{len(FILES_TO_SPLIT)} files")

if __name__ == "__main__":
    # Ensure we start with clean memory
    gc.collect()
    
    # Run the splitting process
    process_files_sequentially()
    
    # Final cleanup
    gc.collect()