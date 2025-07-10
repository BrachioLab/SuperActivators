import pandas as pd
import os
import gc
import json
from datetime import datetime

def split_similarity_distance_file(dataset, model, sample_type, concept_type, file_type):
    """
    Split cosine similarity or distance files using the same chunking as embeddings.
    
    Args:
        dataset: Dataset name (e.g., 'CLEVR', 'Coco')
        model: Model name (e.g., 'CLIP', 'Llama')
        sample_type: 'cls' or 'patch'
        concept_type: Type of concepts (e.g., 'avg_concepts', 'kmeans_1000_concepts', 'linsep_concepts_BD_True_BN_False')
        file_type: 'cosine_similarities' or 'dists'
    """
    
    # Build file paths
    if file_type == 'cosine_similarities':
        folder = 'Cosine_Similarities'
        prefix = 'cosine_similarities'
    else:
        folder = 'Distances'
        prefix = 'dists'
    
    file_name = f"{prefix}_{concept_type}_{model}_{sample_type}_embeddings_percentthrumodel_100.csv"
    file_path = f"{folder}/{dataset}/{file_name}"
    
    print(f"\n{'='*70}")
    print(f"📁 Processing: {dataset}/{model}/{sample_type}/{concept_type}")
    print(f"   File type: {file_type}")
    print(f"   File: {file_path}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(file_path):
        print("   ❌ ERROR: File not found!")
        return False
    
    # Get file size
    file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
    print(f"   File size: {file_size_gb:.2f} GB")
    
    try:
        # Load embedding chunk info to get the splitting indices
        embedding_info_path = f"Embeddings/{dataset}/{model}_{sample_type}_embeddings_percentthrumodel_100_chunks_info.json"
        
        if not os.path.exists(embedding_info_path):
            print(f"   ❌ ERROR: Embedding chunk info not found: {embedding_info_path}")
            print("   Please run embeddings_memory_management.py first!")
            return False
        
        with open(embedding_info_path, 'r') as f:
            embedding_info = json.load(f)
        
        num_chunks = embedding_info['num_chunks']
        chunks_data = embedding_info['chunks']
        
        print(f"\n   Using embedding chunk info:")
        print(f"   • Number of chunks: {num_chunks}")
        
        # Check if chunks already exist
        all_chunks_exist = True
        for i in range(num_chunks):
            chunk_filename = f"{prefix}_{concept_type}_{model}_{sample_type}_embeddings_percentthrumodel_100_chunk_{i}.csv"
            chunk_path = f"{folder}/{dataset}/{chunk_filename}"
            
            if not os.path.exists(chunk_path):
                all_chunks_exist = False
                break
        
        if all_chunks_exist:
            print(f"\n✅ All {num_chunks} chunks already exist! Skipping.")
            return True
        
        # Load the full CSV file
        print(f"\n📊 Loading CSV file...")
        start_time = datetime.now()
        
        # Try to read with different encodings if needed
        try:
            df = pd.read_csv(file_path, index_col=0)
        except UnicodeDecodeError:
            print("   Trying latin-1 encoding...")
            df = pd.read_csv(file_path, index_col=0, encoding='latin-1')
        
        print(f"   ✓ Loaded in {(datetime.now() - start_time).total_seconds():.1f}s")
        print(f"   ✓ Shape: {df.shape}")
        
        # Create chunks using the exact indices from embedding chunks
        print(f"\n✂️  Creating {num_chunks} chunks...")
        
        for i, chunk_info in enumerate(chunks_data):
            start_idx = chunk_info['start_idx']
            end_idx = chunk_info['end_idx']
            expected_samples = chunk_info['samples']
            
            print(f"\n   Creating chunk {i}:")
            print(f"   • Using indices from embedding chunk: {start_idx:,} - {end_idx:,}")
            print(f"   • Expected samples: {expected_samples:,}")
            
            # Extract chunk
            chunk_df = df.iloc[start_idx:end_idx]
            
            if chunk_df.shape[0] != expected_samples:
                print(f"   ⚠️  WARNING: Sample count mismatch! Expected {expected_samples}, got {chunk_df.shape[0]}")
            
            # Save chunk
            chunk_filename = f"{prefix}_{concept_type}_{model}_{sample_type}_embeddings_percentthrumodel_100_chunk_{i}.csv"
            chunk_path = f"{folder}/{dataset}/{chunk_filename}"
            
            print(f"   • Saving to: {chunk_filename}... ", end='', flush=True)
            chunk_df.to_csv(chunk_path)
            print("✓")
            
            # Get chunk size
            chunk_size_gb = os.path.getsize(chunk_path) / (1024 * 1024 * 1024)
            print(f"   • Chunk size: {chunk_size_gb:.2f} GB")
            print(f"   • Actual samples: {chunk_df.shape[0]:,}")
            
            # Clear chunk data
            del chunk_df
            gc.collect()
        
        # Clear the full dataframe
        del df
        gc.collect()
        
        print(f"\n✅ SUCCESS! Created {num_chunks} chunks")
        print(f"   Original file preserved: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        gc.collect()

def process_all_files():
    """Process all cosine similarity and distance files that need chunking."""
    
    # Define which files need processing based on the sizes you showed
    large_files = [
        # CLEVR
        ("CLEVR", "CLIP", "patch", "avg_concepts", "cosine_similarities"),  # 663M
        ("CLEVR", "Llama", "patch", "avg_concepts", "cosine_similarities"),  # 4.1G
        ("CLEVR", "CLIP", "patch", "kmeans_1000_concepts", "cosine_similarities"),  # 18G
        ("CLEVR", "Llama", "patch", "kmeans_1000_concepts", "cosine_similarities"),  # 112G
        ("CLEVR", "CLIP", "patch", "linsep_concepts_BD_True_BN_False", "dists"),  # 643M
        ("CLEVR", "Llama", "patch", "linsep_concepts_BD_True_BN_False", "dists"),  # 4.0G
        ("CLEVR", "CLIP", "patch", "kmeans_1000_linsep_concepts", "dists"),  # 18G
        ("CLEVR", "Llama", "patch", "kmeans_1000_linsep_concepts", "dists"),  # 108G
        
        # COCO
        ("Coco", "CLIP", "patch", "avg_concepts", "cosine_similarities"),  # 2.1G
        ("Coco", "Llama", "patch", "avg_concepts", "cosine_similarities"),  # 14G
        ("Coco", "CLIP", "patch", "kmeans_1000_concepts", "cosine_similarities"),  # 23G
        ("Coco", "Llama", "patch", "kmeans_1000_concepts", "cosine_similarities"),  # 141G
        ("Coco", "CLIP", "patch", "linsep_concepts_BD_True_BN_False", "dists"),  # 2.1G
        ("Coco", "CLIP", "patch", "kmeans_1000_linsep_concepts", "dists"),  # 22G
        ("Coco", "Llama", "patch", "kmeans_1000_linsep_concepts", "dists"),  # 136G
        
        # Broden-Pascal
        ("Broden-Pascal", "CLIP", "patch", "avg_concepts", "cosine_similarities"),
        ("Broden-Pascal", "Llama", "patch", "avg_concepts", "cosine_similarities"),
        ("Broden-Pascal", "CLIP", "patch", "kmeans_1000_concepts", "cosine_similarities"),
        ("Broden-Pascal", "Llama", "patch", "kmeans_1000_concepts", "cosine_similarities"),
        ("Broden-Pascal", "CLIP", "patch", "linsep_concepts_BD_True_BN_False", "dists"),
        ("Broden-Pascal", "Llama", "patch", "linsep_concepts_BD_True_BN_False", "dists"),
        ("Broden-Pascal", "CLIP", "patch", "kmeans_1000_linsep_concepts", "dists"),
        ("Broden-Pascal", "Llama", "patch", "kmeans_1000_linsep_concepts", "dists"),
        
        # Broden-OpenSurfaces
        ("Broden-OpenSurfaces", "CLIP", "patch", "avg_concepts", "cosine_similarities"),
        ("Broden-OpenSurfaces", "Llama", "patch", "avg_concepts", "cosine_similarities"),
        ("Broden-OpenSurfaces", "CLIP", "patch", "kmeans_1000_concepts", "cosine_similarities"),
        ("Broden-OpenSurfaces", "Llama", "patch", "kmeans_1000_concepts", "cosine_similarities"),
        ("Broden-OpenSurfaces", "CLIP", "patch", "linsep_concepts_BD_True_BN_False", "dists"),
        ("Broden-OpenSurfaces", "Llama", "patch", "linsep_concepts_BD_True_BN_False", "dists"),
        ("Broden-OpenSurfaces", "CLIP", "patch", "kmeans_1000_linsep_concepts", "dists"),
        ("Broden-OpenSurfaces", "Llama", "patch", "kmeans_1000_linsep_concepts", "dists"),
        
        # Text datasets - only process if patch embeddings exist for them
        ("Sarcasm", "Llama", "patch", "avg_concepts", "cosine_similarities"),
        ("Sarcasm", "Llama", "patch", "kmeans_1000_concepts", "cosine_similarities"),
        ("Sarcasm", "Llama", "patch", "linsep_concepts_BD_True_BN_False", "dists"),
        ("Sarcasm", "Llama", "patch", "kmeans_1000_linsep_concepts", "dists"),
        
        ("iSarcasm", "Llama", "patch", "avg_concepts", "cosine_similarities"),
        ("iSarcasm", "Llama", "patch", "kmeans_1000_concepts", "cosine_similarities"),
        ("iSarcasm", "Llama", "patch", "linsep_concepts_BD_True_BN_False", "dists"),
        ("iSarcasm", "Llama", "patch", "kmeans_1000_linsep_concepts", "dists"),
    ]
    
    print("🚀 SIMILARITY/DISTANCE FILE SPLITTER")
    print(f"   Files to process: {len(large_files)}")
    print(f"   Using exact embedding chunk indices")
    
    success_count = 0
    skip_count = 0
    
    for i, (dataset, model, sample_type, concept_type, file_type) in enumerate(large_files):
        print(f"\n\n📋 FILE {i+1}/{len(large_files)}")
        
        # Check if the file exists first
        if file_type == 'cosine_similarities':
            folder = 'Cosine_Similarities'
            prefix = 'cosine_similarities'
        else:
            folder = 'Distances'
            prefix = 'dists'
        
        file_name = f"{prefix}_{concept_type}_{model}_{sample_type}_embeddings_percentthrumodel_100.csv"
        file_path = f"{folder}/{dataset}/{file_name}"
        
        if not os.path.exists(file_path):
            print(f"   ⏭️  Skipping - file doesn't exist: {file_path}")
            skip_count += 1
            continue
        
        success = split_similarity_distance_file(dataset, model, sample_type, concept_type, file_type)
        
        if success:
            success_count += 1
        
        # Cleanup between files
        gc.collect()
        print("\n" + "="*70)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   ✅ Successfully processed: {success_count}")
    print(f"   ⏭️  Skipped (not found): {skip_count}")
    print(f"   ❌ Failed: {len(large_files) - success_count - skip_count}")

if __name__ == "__main__":
    process_all_files()