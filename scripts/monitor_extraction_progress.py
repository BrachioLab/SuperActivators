#!/usr/bin/env python3
"""Monitor the progress of embedding stats extraction."""

import json
import os
import time
from datetime import datetime

def monitor_progress():
    """Monitor extraction progress."""
    progress_file = 'Embeddings/extraction_progress.json'
    stats_file = 'Embeddings/embedding_stats.json'
    
    if not os.path.exists(progress_file):
        print("No progress file found. Extraction not running.")
        return
    
    # Load progress
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    completed = len(progress['completed_files'])
    last_dataset = progress.get('last_dataset', 'Unknown')
    last_file = progress.get('last_file', 'Unknown')
    
    # Get stats file size
    if os.path.exists(stats_file):
        stats_size = os.path.getsize(stats_file) / (1024 * 1024)  # MB
        
        # Load stats to count entries
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        # Count total entries
        total_entries = 0
        datasets_processed = []
        for dataset, dataset_stats in stats.items():
            datasets_processed.append(dataset)
            for model, model_stats in dataset_stats.items():
                for emb_type, type_stats in model_stats.items():
                    total_entries += len(type_stats)
    else:
        stats_size = 0
        total_entries = 0
        datasets_processed = []
    
    # Get file modification time
    if os.path.exists(progress_file):
        mod_time = os.path.getmtime(progress_file)
        last_update = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        time_since = time.time() - mod_time
        
        if time_since < 60:
            status = "ACTIVE"
        elif time_since < 300:
            status = "POSSIBLY STALLED"
        else:
            status = "LIKELY STOPPED"
    else:
        last_update = "Unknown"
        status = "UNKNOWN"
    
    print("=== Embedding Stats Extraction Progress ===")
    print(f"Status: {status}")
    print(f"Files processed: {completed}")
    print(f"Last dataset: {last_dataset}")
    print(f"Last file: {last_file}")
    print(f"Last update: {last_update}")
    print(f"\nStats file size: {stats_size:.2f} MB")
    print(f"Total stat entries: {total_entries}")
    print(f"Datasets with stats: {', '.join(datasets_processed)}")
    
    # Show detailed breakdown
    if datasets_processed:
        print("\n=== Detailed Breakdown ===")
        for dataset in sorted(stats.keys()):
            dataset_stats = stats[dataset]
            print(f"\n{dataset}:")
            for model in sorted(dataset_stats.keys()):
                model_stats = dataset_stats[model]
                for emb_type in sorted(model_stats.keys()):
                    type_stats = model_stats[emb_type]
                    percents = sorted(type_stats.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                    print(f"  {model}/{emb_type}: {len(percents)} percentages ({', '.join(percents[:5])}{'...' if len(percents) > 5 else ''})")


if __name__ == "__main__":
    monitor_progress()