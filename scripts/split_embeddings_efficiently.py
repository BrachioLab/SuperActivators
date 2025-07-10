import torch
import os
import gc
import json
from pathlib import Path

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

def split_embedding_file(dataset, model, size_gb):
    """Split a large embedding file into smaller chunks."""
    file_path = f"{EMBEDDINGS_DIR}/{dataset}/{model}_patch_embeddings_percentthrumodel_100.pt"
    
    print(f"\n{'='*60}")
    print(f"💾 Processing: {dataset}/{model} ({size_gb:.2f} GB)")
    print(f"   File: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   ❌ File not found!")
        return False
    
    try:
        # Calculate expected chunks
        expected_chunks = int(size_gb / MAX_SIZE_GB) + (1 if size_gb % MAX_SIZE_GB > 0 else 0)
        print(f"   Expected chunks: {expected_chunks}")
        
        # Load only the embeddings shape first
        print("   Loading file metadata...")
        data = torch.load(file_path, map_location='cpu')
        
        # Extract embeddings info
        if 'normalized_embeddings' in data:
            embeddings_shape = data['normalized_embeddings'].shape
            embeddings_key = 'normalized_embeddings'
        elif 'embeddings' in data:
            embeddings_shape = data['embeddings'].shape
            embeddings_key = 'embeddings'
        else:
            print(f"   ❌ No embeddings found in file!")
            return False
        
        print(f"   Shape: {embeddings_shape}")
        
        # Calculate chunk size
        total_samples = embeddings_shape[0]
        chunk_size = total_samples // expected_chunks
        
        # Create backup of original file
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            print(f"   Creating backup...")
            os.rename(file_path, backup_path)
            file_path = backup_path
        
        # Split and save chunks
        chunk_info = {
            'num_chunks': expected_chunks,
            'chunk_size': chunk_size,
            'total_samples': total_samples,
            'embedding_dim': embeddings_shape[1],
            'chunks': []
        }
        
        # Reload data for splitting
        print("   Starting split operation...")
        data = torch.load(file_path, map_location='cpu')
        embeddings = data[embeddings_key]
        
        for i in range(expected_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < expected_chunks - 1 else total_samples
            
            # Create chunk filename
            chunk_file = file_path.replace('.backup', '').replace('.pt', f'_chunk_{i}.pt')
            
            print(f"   Chunk {i}/{expected_chunks-1}: indices {start_idx:,}-{end_idx:,} → {os.path.basename(chunk_file)}")
            
            # Extract chunk
            chunk_embeddings = embeddings[start_idx:end_idx]
            
            # Save chunk with same structure as original
            chunk_data = {}
            chunk_data[embeddings_key] = chunk_embeddings
            
            # Copy other keys from original data (if any)
            for key in data:
                if key != embeddings_key:
                    chunk_data[key] = data[key]
            
            torch.save(chunk_data, chunk_file)
            
            chunk_info['chunks'].append({
                'file': os.path.basename(chunk_file),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'shape': list(chunk_embeddings.shape)
            })
            
            # Clear memory
            del chunk_embeddings
            del chunk_data
            gc.collect()
        
        # Save chunk info file
        info_file = file_path.replace('.backup', '').replace('.pt', '_chunks_info.json')
        with open(info_file, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        print(f"   ✅ Successfully split into {expected_chunks} chunks")
        print(f"   📄 Chunk info saved to: {os.path.basename(info_file)}")
        
        # Clear memory
        del embeddings
        del data
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to split all large embedding files."""
    print("🚀 Starting embedding file splitting process")
    print(f"   Max file size: {MAX_SIZE_GB} GB")
    print(f"   Files to split: {len(FILES_TO_SPLIT)}")
    
    success_count = 0
    total_size = sum(size for _, _, size in FILES_TO_SPLIT)
    print(f"   Total size to process: {total_size:.2f} GB\n")
    
    # Split each file
    for dataset, model, size_gb in FILES_TO_SPLIT:
        if split_embedding_file(dataset, model, size_gb):
            success_count += 1
        
        # Force garbage collection between files
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'='*60}")
    print(f"✅ Completed! Successfully split {success_count}/{len(FILES_TO_SPLIT)} files")

if __name__ == "__main__":
    main()