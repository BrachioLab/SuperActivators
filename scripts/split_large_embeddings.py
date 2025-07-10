import torch
import os
import gc
from pathlib import Path

# Configuration
EMBEDDINGS_DIR = "Embeddings"
MAX_SIZE_GB = 10
BYTES_PER_GB = 1024 * 1024 * 1024

# Datasets and models configuration
IMAGE_DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
TEXT_DATASETS = ['Sarcasm', 'iSarcasm']
IMAGE_MODELS = ['CLIP', 'Llama']
TEXT_MODELS = ['Llama', 'Qwen']

def get_file_size_gb(file_path):
    """Get file size in GB."""
    return os.path.getsize(file_path) / BYTES_PER_GB

def check_embedding_files():
    """Check all embedding files and their sizes."""
    print("🔍 Checking embedding file sizes...\n")
    
    files_to_split = []
    
    # Check image datasets
    for dataset in IMAGE_DATASETS:
        for model in IMAGE_MODELS:
            embedding_file = f"{EMBEDDINGS_DIR}/{dataset}/{model}_patch_embeddings_percentthrumodel_100.pt"
            if os.path.exists(embedding_file):
                size_gb = get_file_size_gb(embedding_file)
                print(f"📁 {dataset}/{model}: {size_gb:.2f} GB")
                if size_gb > MAX_SIZE_GB:
                    print(f"   ⚠️  Needs splitting!")
                    files_to_split.append((embedding_file, dataset, model, 'patch'))
            else:
                print(f"❌ Not found: {embedding_file}")
    
    # Check text datasets
    for dataset in TEXT_DATASETS:
        for model in TEXT_MODELS:
            embedding_file = f"{EMBEDDINGS_DIR}/{dataset}/{model}_patch_embeddings_percentthrumodel_100.pt"
            if os.path.exists(embedding_file):
                size_gb = get_file_size_gb(embedding_file)
                print(f"📁 {dataset}/{model}: {size_gb:.2f} GB")
                if size_gb > MAX_SIZE_GB:
                    print(f"   ⚠️  Needs splitting!")
                    files_to_split.append((embedding_file, dataset, model, 'patch'))
            else:
                print(f"❌ Not found: {embedding_file}")
    
    print(f"\n📊 Total files needing split: {len(files_to_split)}")
    return files_to_split

def estimate_chunks_needed(embeddings_shape, dtype_size=4):
    """Estimate number of chunks needed based on tensor shape and dtype."""
    # Estimate size in bytes (assuming float32 = 4 bytes)
    total_elements = embeddings_shape[0] * embeddings_shape[1]
    total_bytes = total_elements * dtype_size
    total_gb = total_bytes / BYTES_PER_GB
    
    if total_gb <= MAX_SIZE_GB:
        return 1
    
    # Calculate chunks needed
    chunks_needed = int(total_gb / MAX_SIZE_GB) + 1
    return chunks_needed

def split_embedding_file(file_path, dataset, model, embedding_type):
    """Split a large embedding file into smaller chunks."""
    print(f"\n💾 Processing: {file_path}")
    
    try:
        # Load the embedding file
        print("   Loading embeddings...")
        data = torch.load(file_path, map_location='cpu')
        
        # Extract embeddings
        if 'normalized_embeddings' in data:
            embeddings = data['normalized_embeddings']
            embeddings_key = 'normalized_embeddings'
        elif 'embeddings' in data:
            embeddings = data['embeddings']
            embeddings_key = 'embeddings'
        else:
            print(f"   ❌ No embeddings found in file!")
            return False
        
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dtype: {embeddings.dtype}")
        
        # Estimate chunks needed
        chunks_needed = estimate_chunks_needed(embeddings.shape)
        print(f"   Chunks needed: {chunks_needed}")
        
        if chunks_needed == 1:
            print(f"   ✅ File is already under {MAX_SIZE_GB}GB, no split needed")
            return True
        
        # Calculate chunk size
        total_samples = embeddings.shape[0]
        chunk_size = total_samples // chunks_needed
        
        # Create backup of original file
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            print(f"   Creating backup at: {backup_path}")
            os.rename(file_path, backup_path)
        
        # Split and save chunks
        chunk_info = {
            'num_chunks': chunks_needed,
            'chunk_size': chunk_size,
            'total_samples': total_samples,
            'embedding_dim': embeddings.shape[1],
            'chunks': []
        }
        
        for i in range(chunks_needed):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < chunks_needed - 1 else total_samples
            
            chunk_embeddings = embeddings[start_idx:end_idx]
            
            # Create chunk filename
            chunk_file = file_path.replace('.pt', f'_chunk_{i}.pt')
            
            print(f"   Saving chunk {i}/{chunks_needed-1}: indices {start_idx}-{end_idx}")
            
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
            gc.collect()
        
        # Save chunk info file
        info_file = file_path.replace('.pt', '_chunks_info.json')
        import json
        with open(info_file, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        print(f"   ✅ Successfully split into {chunks_needed} chunks")
        print(f"   📄 Chunk info saved to: {info_file}")
        
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
    """Main function to check and split all large embedding files."""
    print("🚀 Starting embedding file splitting process")
    print(f"   Max file size: {MAX_SIZE_GB} GB\n")
    
    # Check all files
    files_to_split = check_embedding_files()
    
    if not files_to_split:
        print("\n✅ No files need splitting!")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  About to split {len(files_to_split)} files. Continue? (y/n): ", end='')
    # Auto-confirm for script execution
    confirm = 'y'
    print(confirm)
    
    if confirm.lower() != 'y':
        print("❌ Cancelled")
        return
    
    # Split each file
    success_count = 0
    for file_path, dataset, model, emb_type in files_to_split:
        if split_embedding_file(file_path, dataset, model, emb_type):
            success_count += 1
    
    print(f"\n✅ Completed! Successfully split {success_count}/{len(files_to_split)} files")

if __name__ == "__main__":
    main()