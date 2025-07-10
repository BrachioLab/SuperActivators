import torch
import os
import gc
import json
from datetime import datetime
import sys

def split_single_file(dataset, model, expected_size_gb, max_size_gb=10):
    """Split a single embedding file with detailed progress updates."""
    
    file_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100.pt"
    print(f"\n{'='*70}")
    print(f"📁 Processing: {dataset}/{model}")
    print(f"   Expected size: {expected_size_gb:.2f} GB")
    print(f"   File: {file_path}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(file_path):
        print("   ❌ ERROR: File not found!")
        return False
    
    try:
        # Step 1: Get file info
        print("\n1️⃣ Loading file metadata...")
        start_time = datetime.now()
        
        # First pass - just get shape
        data = torch.load(file_path, map_location='cpu')
        
        if 'normalized_embeddings' in data:
            shape = data['normalized_embeddings'].shape
            dtype = data['normalized_embeddings'].dtype
            embeddings_key = 'normalized_embeddings'
        else:
            shape = data['embeddings'].shape  
            dtype = data['embeddings'].dtype
            embeddings_key = 'embeddings'
        
        other_keys = [k for k in data.keys() if k != embeddings_key]
        print(f"   ✓ Shape: {shape}")
        print(f"   ✓ Dtype: {dtype}")
        print(f"   ✓ Other keys: {other_keys}")
        print(f"   ✓ Time taken: {(datetime.now() - start_time).total_seconds():.1f}s")
        
        # Clear memory immediately
        del data
        gc.collect()
        
        # Step 2: Calculate splits
        print("\n2️⃣ Calculating splits...")
        total_samples = shape[0]
        bytes_per_gb = 1024 * 1024 * 1024
        expected_chunks = int(expected_size_gb / max_size_gb) + (1 if expected_size_gb % max_size_gb > 0 else 0)
        chunk_size = total_samples // expected_chunks
        
        print(f"   ✓ Total samples: {total_samples:,}")
        print(f"   ✓ Chunks needed: {expected_chunks}")
        print(f"   ✓ Samples per chunk: ~{chunk_size:,}")
        
        # Step 3: Create backup
        print("\n3️⃣ Creating backup...")
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            os.rename(file_path, backup_path)
            print(f"   ✓ Backup created: {backup_path}")
        else:
            print(f"   ✓ Backup already exists: {backup_path}")
        
        # Step 4: Split file
        print(f"\n4️⃣ Splitting file into {expected_chunks} chunks...")
        
        chunk_info = {
            'dataset': dataset,
            'model': model,
            'num_chunks': expected_chunks,
            'total_samples': total_samples,
            'embedding_dim': shape[1],
            'chunks': []
        }
        
        for i in range(expected_chunks):
            chunk_start_time = datetime.now()
            
            # Calculate indices
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < expected_chunks - 1 else total_samples
            
            print(f"\n   Chunk {i+1}/{expected_chunks}:")
            print(f"   • Indices: {start_idx:,} - {end_idx:,} ({end_idx-start_idx:,} samples)")
            
            # Load data
            print(f"   • Loading data... ", end='', flush=True)
            data = torch.load(backup_path, map_location='cpu')
            print("✓")
            
            # Extract chunk
            print(f"   • Extracting chunk... ", end='', flush=True)
            chunk_data = {}
            chunk_data[embeddings_key] = data[embeddings_key][start_idx:end_idx].clone()
            
            # Copy other data
            for k in other_keys:
                if k in data:
                    chunk_data[k] = data[k]
            
            print("✓")
            
            # Clear original data immediately
            del data
            gc.collect()
            
            # Save chunk
            chunk_filename = f"{model}_patch_embeddings_percentthrumodel_100_chunk_{i}.pt"
            chunk_path = f"Embeddings/{dataset}/{chunk_filename}"
            
            print(f"   • Saving to: {chunk_filename}... ", end='', flush=True)
            torch.save(chunk_data, chunk_path)
            print("✓")
            
            # Get chunk size
            chunk_size_gb = os.path.getsize(chunk_path) / bytes_per_gb
            print(f"   • Chunk size: {chunk_size_gb:.2f} GB")
            print(f"   • Time: {(datetime.now() - chunk_start_time).total_seconds():.1f}s")
            
            # Update info
            chunk_info['chunks'].append({
                'file': chunk_filename,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size_gb': chunk_size_gb,
                'samples': end_idx - start_idx
            })
            
            # Clear memory
            del chunk_data
            gc.collect()
            
            # Force memory release
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except:
                pass
        
        # Save info file
        info_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100_chunks_info.json"
        with open(info_path, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        print(f"\n✅ SUCCESS! Split into {expected_chunks} chunks")
        print(f"   Info saved to: {info_path}")
        print(f"   Total time: {(datetime.now() - start_time).total_seconds():.1f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure cleanup
        gc.collect()

if __name__ == "__main__":
    # Process one file at a time
    files = [
        ("CLEVR", "Llama", 24.41),
        ("Coco", "Llama", 134.28),
        ("Broden-Pascal", "Llama", 109.94),
        ("Broden-OpenSurfaces", "Llama", 87.35)
    ]
    
    print("🚀 EMBEDDING FILE SPLITTER")
    print(f"   Max chunk size: 10 GB")
    print(f"   Files to process: {len(files)}")
    
    for i, (dataset, model, size_gb) in enumerate(files):
        print(f"\n\n📋 FILE {i+1}/{len(files)}")
        success = split_single_file(dataset, model, size_gb)
        
        if not success:
            print("\n⚠️  Failed! Continue with next file? (y/n): ", end='')
            if input().lower() != 'y':
                break
        
        # Cleanup between files
        gc.collect()
        print("\n" + "="*70)