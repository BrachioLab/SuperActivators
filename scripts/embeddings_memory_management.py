import torch
import os
import gc
import json
from datetime import datetime

def split_single_file(dataset, model, expected_size_gb, max_size_gb=10):
    """Split a single embedding file without renaming the original."""
    
    file_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100.pt"
    print(f"\n{'='*70}")
    print(f"📁 Processing: {dataset}/{model}")
    print(f"   Expected size: {expected_size_gb:.2f} GB")
    print(f"   File: {file_path}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for backup file first (from previous run)
    backup_path = file_path + '.backup'
    if os.path.exists(backup_path):
        print(f"   Using backup file: {backup_path}")
        source_file = backup_path
    elif os.path.exists(file_path):
        print(f"   Using original file: {file_path}")
        source_file = file_path
    else:
        print("   ❌ ERROR: File not found!")
        return False
    
    try:
        # Step 1: Check if chunks already exist
        chunk_0_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100_chunk_0.pt"
        if os.path.exists(chunk_0_path):
            print("\n⚠️  Chunks already exist! Checking completion...")
            
            # Try to load existing info
            info_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100_chunks_info.json"
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    chunk_info = json.load(f)
                print(f"   Found {chunk_info['num_chunks']} chunks in info file")
                
                # Check if all chunks exist
                all_exist = True
                for i in range(chunk_info['num_chunks']):
                    chunk_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100_chunk_{i}.pt"
                    if not os.path.exists(chunk_path):
                        print(f"   Missing chunk {i}")
                        all_exist = False
                        break
                
                if all_exist:
                    print("   ✅ All chunks already exist! Skipping.")
                    return True
        
        # Step 2: Check which chunks need to be created first
        print("\n1️⃣ Checking which chunks need to be created...")
        
        # Pre-calculate expected chunks based on size (minimum 1 chunk)
        bytes_per_gb = 1024 * 1024 * 1024
        expected_chunks = max(1, int(expected_size_gb / max_size_gb) + (1 if expected_size_gb % max_size_gb > 0 else 0))
        
        chunks_to_create = []
        chunks_exist = []
        
        for i in range(expected_chunks):
            chunk_filename = f"{model}_patch_embeddings_percentthrumodel_100_chunk_{i}.pt"
            chunk_path = f"Embeddings/{dataset}/{chunk_filename}"
            
            if os.path.exists(chunk_path):
                print(f"   ✓ Chunk {i+1}/{expected_chunks}: Already exists")
                chunks_exist.append(i)
            else:
                print(f"   ⏳ Chunk {i+1}/{expected_chunks}: Needs to be created")
                chunks_to_create.append(i)
        
        if not chunks_to_create:
            print(f"\n✅ All {expected_chunks} chunks already exist! Skipping file loading entirely.")
            
            # Still need to create info file if it doesn't exist
            info_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100_chunks_info.json"
            if not os.path.exists(info_path):
                print("   Creating info file from existing chunks...")
                # Get metadata from the first chunk instead of loading full file
                first_chunk_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100_chunk_0.pt"
                chunk_data = torch.load(first_chunk_path, map_location='cpu')
                
                if 'normalized_embeddings' in chunk_data:
                    embedding_dim = chunk_data['normalized_embeddings'].shape[1]
                else:
                    embedding_dim = chunk_data['embeddings'].shape[1]
                del chunk_data
                gc.collect()
                
                # Calculate total samples from all chunks
                total_samples = 0
                chunk_info = {
                    'dataset': dataset,
                    'model': model,
                    'num_chunks': expected_chunks,
                    'total_samples': 0,  # Will update below
                    'embedding_dim': embedding_dim,
                    'chunks': []
                }
                
                for i in range(expected_chunks):
                    chunk_filename = f"{model}_patch_embeddings_percentthrumodel_100_chunk_{i}.pt"
                    chunk_path = f"Embeddings/{dataset}/{chunk_filename}"
                    
                    # Load chunk to get actual sample count
                    chunk_data = torch.load(chunk_path, map_location='cpu')
                    if 'normalized_embeddings' in chunk_data:
                        num_samples = chunk_data['normalized_embeddings'].shape[0]
                    else:
                        num_samples = chunk_data['embeddings'].shape[0]
                    del chunk_data
                    gc.collect()
                    
                    start_idx = total_samples
                    end_idx = total_samples + num_samples
                    total_samples += num_samples
                    
                    chunk_size_gb = os.path.getsize(chunk_path) / (1024 * 1024 * 1024)
                    
                    chunk_info['chunks'].append({
                        'file': chunk_filename,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'size_gb': chunk_size_gb,
                        'samples': num_samples
                    })
                
                chunk_info['total_samples'] = total_samples
                
                with open(info_path, 'w') as f:
                    json.dump(chunk_info, f, indent=2)
                print(f"   ✓ Info file created: {os.path.basename(info_path)}")
            
            return True
        
        # Step 3: Load embeddings ONCE for all chunk creation
        print(f"\n2️⃣ Loading embeddings once to create {len(chunks_to_create)} chunks...")
        start_time = datetime.now()
        load_start_time = datetime.now()
        print(f"   • Loading full embeddings file... ", end='', flush=True)
        data = torch.load(source_file, map_location='cpu')
        
        if 'normalized_embeddings' in data:
            shape = data['normalized_embeddings'].shape
            dtype = data['normalized_embeddings'].dtype
            embeddings_key = 'normalized_embeddings'
        else:
            shape = data['embeddings'].shape  
            dtype = data['embeddings'].dtype
            embeddings_key = 'embeddings'
        
        embeddings = data[embeddings_key]
        other_keys = [k for k in data.keys() if k != embeddings_key]
        
        print(f"✓ ({(datetime.now() - load_start_time).total_seconds():.1f}s)")
        print(f"   ✓ Shape: {shape}")
        print(f"   ✓ Dtype: {dtype}")
        print(f"   ✓ Other keys: {other_keys}")
        
        # Step 4: Calculate splits and create chunks
        print(f"\n3️⃣ Creating chunks from loaded data...")
        total_samples = shape[0]
        chunk_size = total_samples // expected_chunks
        
        print(f"   ✓ Total samples: {total_samples:,}")
        print(f"   ✓ Chunks needed: {expected_chunks}")
        print(f"   ✓ Samples per chunk: ~{chunk_size:,}")
        
        chunk_info = {
            'dataset': dataset,
            'model': model,
            'num_chunks': expected_chunks,
            'total_samples': total_samples,
            'embedding_dim': shape[1],
            'chunks': []
        }
        
        # First, add existing chunks to info
        for i in chunks_exist:
            chunk_filename = f"{model}_patch_embeddings_percentthrumodel_100_chunk_{i}.pt"
            chunk_path = f"Embeddings/{dataset}/{chunk_filename}"
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < expected_chunks - 1 else total_samples
            chunk_size_gb = os.path.getsize(chunk_path) / bytes_per_gb
            
            chunk_info['chunks'].append({
                'file': chunk_filename,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size_gb': chunk_size_gb,
                'samples': end_idx - start_idx
            })
        
        # Now create the missing chunks from the loaded data
        for i in chunks_to_create:
            chunk_start_time = datetime.now()
            chunk_filename = f"{model}_patch_embeddings_percentthrumodel_100_chunk_{i}.pt"
            chunk_path = f"Embeddings/{dataset}/{chunk_filename}"
            
            # Calculate indices
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < expected_chunks - 1 else total_samples
            
            print(f"\n   Creating chunk {i+1}/{expected_chunks}:")
            print(f"   • Indices: {start_idx:,} - {end_idx:,} ({end_idx-start_idx:,} samples)")
            
            # Extract chunk from already-loaded embeddings
            print(f"   • Extracting chunk from memory... ", end='', flush=True)
            chunk_data = {}
            chunk_data[embeddings_key] = embeddings[start_idx:end_idx].clone()
            
            # Copy other data (metadata that doesn't change)
            for k in other_keys:
                if k in data:
                    chunk_data[k] = data[k]
            
            print("✓")
            
            # Save chunk
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
            
            # Clear chunk data to save memory
            del chunk_data
            gc.collect()
        
        # Clear the full embeddings data
        del embeddings, data
        gc.collect()
        
        # Force memory release
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass
        
        # Sort chunks by start_idx to ensure correct order in info file
        chunk_info['chunks'].sort(key=lambda x: x['start_idx'])
        
        # Save info file
        info_path = f"Embeddings/{dataset}/{model}_patch_embeddings_percentthrumodel_100_chunks_info.json"
        with open(info_path, 'w') as f:
            json.dump(chunk_info, f, indent=2)
        
        print(f"\n✅ SUCCESS! Split into {expected_chunks} chunks")
        print(f"   Info saved to: {os.path.basename(info_path)}")
        print(f"   Total time: {(datetime.now() - start_time).total_seconds():.1f}s")
        print(f"   Original file unchanged: {file_path}")
        
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
    # Process all dataset/model combinations
    files = [
        # Vision datasets - CLIP and Llama
        ("CLEVR", "CLIP", 8.2),
        ("CLEVR", "Llama", 24.41),
        ("Coco", "CLIP", 45.1),
        ("Coco", "Llama", 134.28),
        ("Broden-Pascal", "CLIP", 37.0),
        ("Broden-Pascal", "Llama", 109.94),
        ("Broden-OpenSurfaces", "CLIP", 29.4),
        ("Broden-OpenSurfaces", "Llama", 87.35),
        
        # Text datasets - Llama and Qwen
        ("Sarcasm", "Llama", 12.5),
        ("Sarcasm", "Qwen", 11.8),
        ("iSarcasm", "Llama", 15.2),
        ("iSarcasm", "Qwen", 14.6)
    ]
    
    print("🚀 EMBEDDING FILE SPLITTER (NO RENAME)")
    print(f"   Max chunk size: 10 GB")
    print(f"   Files to process: {len(files)}")
    print(f"   Original files will NOT be renamed")
    
    success_count = 0
    for i, (dataset, model, size_gb) in enumerate(files):
        print(f"\n\n📋 FILE {i+1}/{len(files)}")
        success = split_single_file(dataset, model, size_gb)
        
        if success:
            success_count += 1
        
        # Cleanup between files
        gc.collect()
        print("\n" + "="*70)
    
    print(f"\n🎯 FINAL RESULTS: {success_count}/{len(files)} files processed successfully")