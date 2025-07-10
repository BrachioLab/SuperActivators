#!/usr/bin/env python3
"""
Extract normalization statistics from all embedding files and save to a centralized JSON.
This allows accessing stats without loading the entire embeddings dictionaries.
Supports incremental saving and resuming from where it left off.
"""

import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_stats_from_embedding_file(filepath):
    """
    Extract statistics from a single embedding file.
    
    Returns:
        dict: Statistics including mean, norm, shape info
    """
    try:
        # Load to CPU to avoid memory issues
        data = torch.load(filepath, map_location='cpu')
        
        stats = {}
        
        # Extract mean embedding if available
        if 'mean_train_embedding' in data:
            mean_emb = data['mean_train_embedding']
            stats['mean_embedding_shape'] = list(mean_emb.shape)
            stats['mean_embedding_dim'] = mean_emb.shape[0]
            # Convert to list for JSON serialization
            stats['mean_embedding'] = mean_emb.cpu().numpy().tolist()
        
        # Extract norm if available
        if 'train_norm' in data:
            norm = data['train_norm']
            if isinstance(norm, torch.Tensor):
                stats['train_norm'] = float(norm.item())
            else:
                stats['train_norm'] = float(norm)
        
        # Extract shape information
        if 'normalized_embeddings' in data:
            emb = data['normalized_embeddings']
            stats['num_embeddings'] = emb.shape[0]
            stats['embedding_dim'] = emb.shape[1]
        elif 'embeddings' in data:
            emb = data['embeddings']
            stats['num_embeddings'] = emb.shape[0]
            stats['embedding_dim'] = emb.shape[1]
        
        # Add file size info
        stats['file_size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
        
        return stats
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_progress(progress_file='Embeddings/extraction_progress.json'):
    """Load progress from file if it exists."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_files': [], 'last_dataset': None, 'last_file': None}


def save_progress(progress, progress_file='Embeddings/extraction_progress.json'):
    """Save current progress."""
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def save_stats_incremental(all_stats, stats_file='Embeddings/embedding_stats.json'):
    """Save stats incrementally."""
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    # Save full stats
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Create lite version without embeddings
    stats_lite = {}
    for dataset, dataset_stats in all_stats.items():
        stats_lite[dataset] = {}
        for model, model_stats in dataset_stats.items():
            stats_lite[model] = {}
            for emb_type, type_stats in model_stats.items():
                stats_lite[model][emb_type] = {}
                for percent, percent_stats in type_stats.items():
                    lite_stats = {k: v for k, v in percent_stats.items() 
                                 if k != 'mean_embedding'}
                    stats_lite[model][emb_type][percent] = lite_stats
    
    # Save lite stats
    lite_file = stats_file.replace('.json', '_lite.json')
    with open(lite_file, 'w') as f:
        json.dump(stats_lite, f, indent=2)


def scan_embeddings_directory(base_dir='Embeddings'):
    """
    Scan all embeddings directories and extract statistics.
    Supports resuming from where it left off.
    
    Returns:
        dict: Nested dictionary of all statistics
    """
    # Load existing stats if they exist
    stats_file = os.path.join(base_dir, 'embedding_stats.json')
    if os.path.exists(stats_file):
        print(f"Loading existing stats from {stats_file}")
        with open(stats_file, 'r') as f:
            all_stats = json.load(f)
    else:
        all_stats = {}
    
    # Load progress
    progress = load_progress()
    completed_files = set(progress['completed_files'])
    
    # Only process specific datasets
    target_datasets = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm']
    
    # Get all dataset directories
    embeddings_path = Path(base_dir)
    if not embeddings_path.exists():
        print(f"Embeddings directory not found: {base_dir}")
        return all_stats
    
    # Filter to only target datasets
    dataset_dirs = []
    for dataset_name in target_datasets:
        dataset_path = embeddings_path / dataset_name
        if dataset_path.exists() and dataset_path.is_dir():
            dataset_dirs.append(dataset_path)
        else:
            print(f"Dataset directory not found: {dataset_path}")
    
    # Count total files for progress bar
    total_files = 0
    for dataset_dir in dataset_dirs:
        pt_files = list(dataset_dir.glob("*.pt"))
        total_files += len(pt_files)
    
    print(f"Found {total_files} total .pt files across {len(dataset_dirs)} datasets")
    print(f"Already processed {len(completed_files)} files")
    
    # Process files with single progress bar
    with tqdm(total=total_files, initial=len(completed_files), desc="Processing files") as pbar:
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            
            if dataset_name not in all_stats:
                all_stats[dataset_name] = {}
            
            dataset_stats = all_stats[dataset_name]
            
            # Find all .pt files in this dataset directory
            pt_files = list(dataset_dir.glob("*.pt"))
            
            for pt_file in pt_files:
                # Skip if already processed
                file_key = str(pt_file)
                if file_key in completed_files:
                    continue
                
                filename = pt_file.name
                
                # Skip test image files and chunk files for now
                if 'test_images' in filename or 'chunk' in filename:
                    completed_files.add(file_key)
                    pbar.update(1)
                    continue
                
                # Parse filename to extract model and percent info
                parts = filename.replace('.pt', '').split('_')
                
                # Try to identify model name and percent
                model_name = None
                percent_thru_model = None
                embedding_type = None
                
                # Common patterns
                if 'percentthrumodel' in filename:
                    # Find index of 'percentthrumodel'
                    try:
                        idx = parts.index('percentthrumodel')
                        if idx + 1 < len(parts):
                            percent_thru_model = parts[idx + 1]
                        
                        # Model name is usually the first part
                        model_name = parts[0]
                        
                        # Check if it's patch or cls
                        if 'patch' in filename:
                            embedding_type = 'patch'
                        elif 'cls' in filename:
                            embedding_type = 'cls'
                    except ValueError:
                        pass
                
                if model_name and percent_thru_model and embedding_type:
                    # Create nested structure
                    if model_name not in dataset_stats:
                        dataset_stats[model_name] = {}
                    
                    if embedding_type not in dataset_stats[model_name]:
                        dataset_stats[model_name][embedding_type] = {}
                    
                    # Extract stats
                    print(f"\nExtracting stats from: {pt_file.name}")
                    stats = extract_stats_from_embedding_file(pt_file)
                    
                    if stats:
                        dataset_stats[model_name][embedding_type][percent_thru_model] = stats
                        dataset_stats[model_name][embedding_type][percent_thru_model]['filename'] = filename
                        dataset_stats[model_name][embedding_type][percent_thru_model]['filepath'] = str(pt_file)
                        
                        # Save stats incrementally
                        save_stats_incremental(all_stats)
                        print(f"  Saved stats for {dataset_name}/{model_name}/{embedding_type}/{percent_thru_model}")
                
                # Mark as completed
                completed_files.add(file_key)
                
                # Update progress
                progress['completed_files'] = list(completed_files)
                progress['last_dataset'] = dataset_name
                progress['last_file'] = filename
                save_progress(progress)
                
                # Update progress bar
                pbar.update(1)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
    
    return all_stats


def clear_progress(progress_file='Embeddings/extraction_progress.json'):
    """Clear the progress file to start fresh."""
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"Cleared progress file: {progress_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract embedding statistics')
    parser.add_argument('--clear-progress', action='store_true', 
                       help='Clear progress and start fresh')
    args = parser.parse_args()
    
    if args.clear_progress:
        clear_progress()
    
    print("Scanning embeddings directory and extracting statistics...")
    print("This will save progress incrementally and can be resumed if interrupted.\n")
    
    # Extract all statistics
    all_stats = scan_embeddings_directory()
    
    if all_stats:
        # Final save
        save_stats_incremental(all_stats)
        
        # Print summary
        print("\n=== Summary ===")
        for dataset, dataset_stats in all_stats.items():
            print(f"\nDataset: {dataset}")
            for model, model_stats in dataset_stats.items():
                print(f"  Model: {model}")
                for emb_type, type_stats in model_stats.items():
                    print(f"    Type: {emb_type}")
                    percents = sorted(type_stats.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                    print(f"      Percentages: {', '.join(percents)}")
        
        # Clean up progress file on successful completion
        clear_progress()
        print("\nExtraction completed successfully!")