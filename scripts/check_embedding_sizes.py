import os
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
    total_size = 0
    
    # Check image datasets
    print("IMAGE DATASETS:")
    for dataset in IMAGE_DATASETS:
        for model in IMAGE_MODELS:
            embedding_file = f"{EMBEDDINGS_DIR}/{dataset}/{model}_patch_embeddings_percentthrumodel_100.pt"
            if os.path.exists(embedding_file):
                size_gb = get_file_size_gb(embedding_file)
                total_size += size_gb
                print(f"  {dataset}/{model}: {size_gb:.2f} GB", end='')
                if size_gb > MAX_SIZE_GB:
                    print(" ⚠️  NEEDS SPLITTING!")
                    files_to_split.append((embedding_file, dataset, model, 'patch', size_gb))
                else:
                    print(" ✅")
            else:
                print(f"  {dataset}/{model}: ❌ Not found")
    
    # Check text datasets
    print("\nTEXT DATASETS:")
    for dataset in TEXT_DATASETS:
        for model in TEXT_MODELS:
            embedding_file = f"{EMBEDDINGS_DIR}/{dataset}/{model}_patch_embeddings_percentthrumodel_100.pt"
            if os.path.exists(embedding_file):
                size_gb = get_file_size_gb(embedding_file)
                total_size += size_gb
                print(f"  {dataset}/{model}: {size_gb:.2f} GB", end='')
                if size_gb > MAX_SIZE_GB:
                    print(" ⚠️  NEEDS SPLITTING!")
                    files_to_split.append((embedding_file, dataset, model, 'patch', size_gb))
                else:
                    print(" ✅")
            else:
                print(f"  {dataset}/{model}: ❌ Not found")
    
    print(f"\n📊 Summary:")
    print(f"   Total size: {total_size:.2f} GB")
    print(f"   Files needing split: {len(files_to_split)}")
    
    if files_to_split:
        print(f"\n🔴 Files larger than {MAX_SIZE_GB} GB:")
        for file_path, dataset, model, emb_type, size in files_to_split:
            print(f"   {dataset}/{model}: {size:.2f} GB")
    
    return files_to_split

if __name__ == "__main__":
    check_embedding_files()