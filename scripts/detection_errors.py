#!/usr/bin/env python3
"""
Detection Error Analysis Script
==============================

This script computes bootstrap confidence intervals for detection metrics.
It follows the same structure as all_detection_stats.py but adds error analysis.

The script:
1. Loads pre-computed detection results from all_detection_stats.py
2. Computes per-concept bootstrap confidence intervals
3. Computes dataset-level two-level bootstrap confidence intervals
4. Generates formatted reports with error bars
"""

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import os
import gc
import argparse
from collections import defaultdict
from itertools import product
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.memory_management_utils import ChunkedEmbeddingLoader, ChunkedActivationLoader
from utils.filter_datasets_utils import filter_concept_dict
from utils.default_percentthrumodels import ALL_PERCENTTHRUMODELS, get_model_default_percentthrumodels
from utils.detection_error_utils import (
    compute_per_concept_bootstrap,
    compute_dataset_level_bootstrap,
    generate_per_concept_report,
    generate_dataset_report,
    save_results_with_ci
)

# Default configuration - same as all_detection_stats.py
MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Gemma', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# Bootstrap parameters
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
SEED = 42


def get_files_for_avg(model_name, sample_type, percent_thru_model):
    """Same as in all_detection_stats.py"""
    con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}'
    return con_label


def get_files_for_linsep(model_name, sample_type, percent_thru_model):
    """Same as in all_detection_stats.py"""
    con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}'
    return con_label


def get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model):
    """Same as in all_detection_stats.py"""
    con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    return con_label


def get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model):
    """Same as in all_detection_stats.py"""
    con_label = f"{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    return con_label


def get_files_for_sae(model_name, sample_type):
    """Get SAE concept label - SAE doesn't use percent_thru_model"""
    con_label = f"{model_name}_sae_{sample_type}_dense"
    return con_label


def get_all_files(model_name, sample_type, n_clusters, percent_thru_model, concept_types=None):
    """Get all file configurations based on concept types"""
    if concept_types is None:
        concept_types = ['avg', 'linsep', 'kmeans', 'linsepkmeans']
    
    all_labels = []
    if 'avg' in concept_types:
        all_labels.append(('avg', get_files_for_avg(model_name, sample_type, percent_thru_model)))
    if 'linsep' in concept_types:
        all_labels.append(('linsep', get_files_for_linsep(model_name, sample_type, percent_thru_model)))
    if 'kmeans' in concept_types:
        all_labels.append(('kmeans', get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model)))
    if 'linsepkmeans' in concept_types:
        all_labels.append(('linsepkmeans', get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model)))
    if 'sae' in concept_types:
        all_labels.append(('sae', get_files_for_sae(model_name, sample_type)))
    
    return all_labels


def load_detection_results(dataset_name, con_label, percentile, split='test'):
    """
    Load pre-computed detection results from all_detection_stats.py
    
    Args:
        dataset_name: Dataset name
        con_label: Concept label
        percentile: Detection percentile
        split: 'test' or 'cal' for test or calibration split
        
    Returns:
        pd.DataFrame with detection results or None if not found
    """
    # For calibration split, append _cal to the con_label (matching all_detection_stats.py)
    if split == 'cal':
        con_label_with_split = f"{con_label}_cal"
    else:
        con_label_with_split = con_label
    
    # Try loading the saved detection metrics - first try .pt format (for supervised concepts)
    pt_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label_with_split}.pt'
    
    if os.path.exists(pt_path):
        return torch.load(pt_path, weights_only=False)
    
    # For calibration split with supervised concepts (avg, linsep), also try consolidated CSV files
    if split == 'cal' and ('avg' in con_label or ('linsep' in con_label and 'kmeans' not in con_label)):
        # For supervised concepts on cal split, try detectfirst consolidated format
        consolidated_path = f'Quant_Results/{dataset_name}/detectfirst_cal_{con_label}.csv'
        
        if os.path.exists(consolidated_path):
            df = pd.read_csv(consolidated_path)
            # Filter for the specific percentile
            df_filtered = df[df['percentile'] == percentile]
            
            if len(df_filtered) == 0:
                return None
                
            # Convert to dictionary format
            results = {}
            for _, row in df_filtered.iterrows():
                concept = row['concept']
                results[concept] = {
                    'tp': int(row['tp']),
                    'fp': int(row['fp']),
                    'tn': int(row['tn']),
                    'fn': int(row['fn']),
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1': row['f1']
                }
            return results
    
    
    # Try CSV format (for unsupervised concepts like kmeans)
    csv_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label_with_split}.csv'
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Regular format - convert DataFrame to dictionary
        result_dict = {}
        for _, row in df.iterrows():
            concept = row['concept']
            result_dict[concept] = row.to_dict()
        return result_dict
    
    # If CSV not found, try allpairs format for unsupervised concepts
    if 'kmeans' in con_label:
        csv_path = f'Quant_Results/{dataset_name}/allpairs_detection_per_{percentile}_{con_label_with_split}.csv'
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Load best matching clusters/units
            bestdetects_file = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label_with_split}.pt'
            if not os.path.exists(bestdetects_file):
                return None
                    
            bestdetects = torch.load(bestdetects_file, weights_only=False)
            
            # Convert to dictionary format - extract only the best cluster for each concept
            result_dict = {}
            for concept, info in bestdetects.items():
                best_cluster = str(info.get('best_cluster', ''))
                # Find the row for this (concept, cluster) pair
                concept_cluster_str = f"('{concept}', '{best_cluster}')"
                concept_rows = df[df['concept'] == concept_cluster_str]
                
                if len(concept_rows) > 0:
                    row = concept_rows.iloc[0]
                    result_dict[concept] = {
                        'tp': int(row['tp']),
                        'fp': int(row['fp']),
                        'tn': int(row['tn']),
                        'fn': int(row['fn']),
                        'precision': row['precision'],
                        'recall': row['recall'],
                        'f1': row['f1']
                    }
            return result_dict
    
    # If not found, return None
    return None


def load_baseline_detection_results(dataset_name, con_label, percentile, baseline_method):
    """
    Load pre-computed baseline detection results
    
    Args:
        dataset_name: Dataset name
        con_label: Concept label
        percentile: Detection percentile
        baseline_method: One of 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken'
        
    Returns:
        pd.DataFrame with detection results or None if not found
    """
    # For supervised concepts (avg, linsep), baseline results are in consolidated files
    # For unsupervised (kmeans), they are in per-percentile files
    
    # First try consolidated file (for supervised concepts)
    consolidated_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_method}_test_{con_label}.csv'
    
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
        # Filter for the specific percentile
        df_filtered = df[df['percentile'] == percentile]
        
        if len(df_filtered) == 0:
            return None
            
        # Convert to dictionary format
        results = {}
        for _, row in df_filtered.iterrows():
            concept = row['concept']
            # Calculate tn from the other values
            # For baseline detection, we need to calculate TN
            # Total samples = TP + FP + TN + FN
            # We can get total from the regular detection results
            # For now, use a rough estimate based on typical test set size
            tp, fp, fn = int(row['tp']), int(row['fp']), int(row['fn'])
            n_gt_positive = int(row['n_gt_positive'])
            n_detected = int(row['n_detected'])
            
            # Infer total samples (this is approximate)
            # In CLEVR test set, there are 166 images
            total_samples = 166  # This is dataset-specific
            tn = total_samples - tp - fp - fn
            
            results[concept] = {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1']
            }
        return results
    
    # Try per-percentile file (for unsupervised concepts like kmeans)
    per_percentile_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_method}_per_{percentile}_{con_label}.csv'
    
    if os.path.exists(per_percentile_path):
        df = pd.read_csv(per_percentile_path)
        # Convert to dictionary format
        results = {}
        for _, row in df.iterrows():
            concept = row['concept']
            results[concept] = {
                'tp': row['tp'],
                'fp': row['fp'], 
                'tn': row['tn'],
                'fn': row['fn'],
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1']
            }
        return results
    
    return None


def load_baseline_detection_results_v2(dataset_name, con_label, percentile, baseline_method, total_test_samples, split='test'):
    """
    Load pre-computed baseline detection results (improved version)
    
    Args:
        dataset_name: Dataset name
        con_label: Concept label
        percentile: Detection percentile
        baseline_method: One of 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken'
        total_test_samples: Total number of samples in split
        split: 'test' or 'cal' to determine which file to load
        
    Returns:
        dict with detection results or None if not found
    """
    # For supervised concepts (avg, linsep), baseline results are in consolidated files
    # For unsupervised (kmeans, sae), they are in different formats
    
    # First try consolidated file (for supervised concepts)
    if split == 'cal':
        consolidated_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_method}_cal_{con_label}.csv'
    else:
        consolidated_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_method}_test_{con_label}.csv'
    
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
        # Filter for the specific percentile
        df_filtered = df[df['percentile'] == percentile]
        
        if len(df_filtered) == 0:
            return None
            
        # Convert to dictionary format
        results = {}
        for _, row in df_filtered.iterrows():
            concept = row['concept']
            tp, fp, fn = int(row['tp']), int(row['fp']), int(row['fn'])
            
            # Calculate TN using actual test set size
            tn = total_test_samples - tp - fp - fn
            
            results[concept] = {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1']
            }
        return results
    
    # Try per-percentile file (for some unsupervised concepts)
    per_percentile_path = f'Quant_Results/{dataset_name}/detectfirst_{baseline_method}_per_{percentile}_{con_label}.csv'
    
    if os.path.exists(per_percentile_path):
        df = pd.read_csv(per_percentile_path)
        # Convert to dictionary format
        results = {}
        for _, row in df.iterrows():
            concept = row['concept']
            results[concept] = {
                'tp': row['tp'],
                'fp': row['fp'], 
                'tn': row['tn'],
                'fn': row['fn'],
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1']
            }
        return results
    
    # Try allpairs format for SAE and kmeans
    is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
    if is_unsupervised:
        if split == 'cal':
            allpairs_path = f'Quant_Results/{dataset_name}/detectionmetrics_{baseline_method}_allpairs_per_{percentile}_{con_label}_cal.csv'
            bestdetects_file = f'Unsupervised_Matches/{dataset_name}/bestdetects_{baseline_method}_{con_label}_cal.pt'
        else:
            allpairs_path = f'Quant_Results/{dataset_name}/detectionmetrics_{baseline_method}_allpairs_per_{percentile}_{con_label}.csv'
            bestdetects_file = f'Unsupervised_Matches/{dataset_name}/bestdetects_{baseline_method}_{con_label}.pt'
        
        if os.path.exists(allpairs_path):
            # Load the bestdetects file to get the best cluster for each concept
            if not os.path.exists(bestdetects_file):
                return None
                
            bestdetects = torch.load(bestdetects_file, weights_only=False)
            df = pd.read_csv(allpairs_path)
            
            # Convert to dictionary format - extract only the best cluster for each concept
            results = {}
            for concept, info in bestdetects.items():
                best_cluster = str(info.get('best_cluster', ''))
                # Find the row for this (concept, cluster) pair
                concept_cluster_str = f"('{concept}', '{best_cluster}')"
                row = df[df['concept'] == concept_cluster_str]
                
                if not row.empty:
                    row = row.iloc[0]
                    # Calculate TN
                    tp, fp, fn = int(row['tp']), int(row['fp']), int(row['fn'])
                    tn = total_test_samples - tp - fp - fn
                    
                    results[concept] = {
                        'tp': tp,
                        'fp': fp,
                        'tn': tn,
                        'fn': fn,
                        'precision': row['precision'],
                        'recall': row['recall'],
                        'f1': row['f1']
                    }
            
            return results
    
    return None


def load_sae_detection_results(dataset_name, con_label, percentile, split='test'):
    """
    Load pre-computed SAE detection results
    
    Args:
        dataset_name: Dataset name
        con_label: Concept label (e.g., 'CLIP_sae_patch_dense')
        percentile: Detection percentile
        split: 'test' or 'cal' for which data split
        
    Returns:
        dict with detection results or None if not found
    """
    # Try multiple SAE result formats
    # First try detectfirst format with invert
    pt_path = f'Quant_Results/{dataset_name}/detectfirst_{percentile}_invert_{percentile}_{con_label}.pt'
    
    if os.path.exists(pt_path):
        data = torch.load(pt_path, weights_only=False)
        # Convert to expected format if needed
        if isinstance(data, pd.DataFrame):
            results = {}
            for _, row in data.iterrows():
                concept = row['concept'] if 'concept' in row else row.get('label', row.name)
                results[concept] = row.to_dict()
            return results
        elif isinstance(data, dict):
            return data
    
    # Try allpairs format like kmeans
    csv_path = f'Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv'
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Load best matching SAE units
        best_units_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
        if not os.path.exists(best_units_path):
            return None
            
        best_units_per_concept = torch.load(best_units_path, weights_only=False)
        
        # Filter to best units
        results = {}
        for concept in best_units_per_concept:
            best_unit_info = best_units_per_concept[concept]
            # Handle both dict format and direct value format
            if isinstance(best_unit_info, dict):
                best_unit = best_unit_info.get('best_cluster', best_unit_info.get('best_unit', None))
            else:
                best_unit = best_unit_info
                
            if best_unit is None:
                continue
                
            # In SAE CSV, concepts are stored as tuples like "('concept_name', 'unit_id')"
            concept_tuple_str = f"('{concept}', '{best_unit}')"
            concept_rows = df[df['concept'] == concept_tuple_str]
            
            if len(concept_rows) > 0:
                row = concept_rows.iloc[0]
                results[concept] = {
                    'tp': int(row['tp']),
                    'fp': int(row['fp']),
                    'tn': int(row['tn']),
                    'fn': int(row['fn']),
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1': row['f1']
                }
        
        return results
    
    return None


def load_prompt_detection_results(dataset_name):
    """
    Load pre-computed prompt-based detection results
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        dict with detection results or None if not found
    """
    # Prompt results are stored in a fixed location
    csv_path = f'prompt_results/{dataset_name}/{dataset_name}_Llama-3.2-11B-Vision-Instruct_f1_scores.csv'
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convert to dictionary format
        results = {}
        for _, row in df.iterrows():
            concept = row['concept']
            # Calculate precision and recall if not in the file
            tp, fp, tn, fn = row['tp'], row['fp'], row['tn'], row['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results[concept] = {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': row['f1']
            }
        return results
    
    return None


def process_concept_configuration(
    dataset_name, model_name, model_input_size, sample_type, 
    con_label, percent_thru_model, split, n_bootstrap, confidence_level, seed
):
    """
    Process a single concept configuration for error analysis.
    Uses the OPTIMAL calibration percentile for each concept.
    
    Args:
        dataset_name: Dataset to process
        model_name: Model name
        model_input_size: Model input dimensions
        sample_type: 'patch' or 'cls'
        con_label: Concept configuration label
        percent_thru_model: Percentage through model
        split: 'test' or 'cal' for which data split to process
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
        seed: Random seed
    """
    print(f"\nProcessing {con_label} - {split} split")
    print("-" * 50)
    
    # Load the OPTIMAL calibration percentiles
    # Best percentiles files don't have _cal suffix - they're already in Best_Detection_Percentiles_Cal directory
    # Check if this is an unsupervised method (kmeans or SAE)
    is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
    
    # Use the same file pattern for both supervised and unsupervised
    best_percentiles_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    if not os.path.exists(best_percentiles_file):
        print(f"  ⚠️  No best percentiles file found at {best_percentiles_file}")
        print(f"      Run all_detection_stats.py first to generate optimal percentiles")
        return
    best_percentiles = torch.load(best_percentiles_file, weights_only=False)
    print(f"  Loaded optimal percentiles for {len(best_percentiles)} concepts")
    
    # Load ground truth - ALWAYS use sample-level (image/paragraph) GT
    if split == 'cal':
        gt_images_per_concept_split = torch.load(
            f'GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt',
            weights_only=False
        )
    else:
        gt_images_per_concept_split = torch.load(
            f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt',
            weights_only=False
        )
    
    # Filter to relevant concepts
    gt_images_per_concept_split = filter_concept_dict(gt_images_per_concept_split, dataset_name)
    
    # Get total number of samples for this split
    from utils.general_utils import get_split_df
    split_df = get_split_df(dataset_name)
    total_split_samples = len(split_df[split_df == split])
    
    # Collect detection results for all concepts at their optimal percentiles
    all_detection_results = []
    percentiles_used = set()
    
    for concept, perc_info in best_percentiles.items():
        optimal_percentile = perc_info['best_percentile']
        percentiles_used.add(optimal_percentile)
        
        # Load detection results for this percentile
        detection_results = load_detection_results(dataset_name, con_label, optimal_percentile, split)
        
        if detection_results is None:
            print(f"  ⚠️  No detection results found for percentile {optimal_percentile}, skipping concept {concept}")
            continue
        
        # Extract this concept's results
        if isinstance(detection_results, dict):
            if concept in detection_results:
                concept_data = detection_results[concept]
            else:
                print(f"  ⚠️  Concept {concept} not found in detection results")
                continue
        else:  # DataFrame
            concept_rows = detection_results[detection_results['concept'] == concept]
            if len(concept_rows) == 0:
                print(f"  ⚠️  Concept {concept} not found in detection results")
                continue
            concept_data = concept_rows.iloc[0].to_dict()
        
        # Add optimal percentile info
        concept_data['concept'] = concept
        concept_data['optimal_percentile'] = optimal_percentile
        concept_data['optimal_threshold'] = perc_info['best_threshold']
        concept_data['calibration_f1'] = perc_info['best_f1']
        
        all_detection_results.append(concept_data)
    
    if not all_detection_results:
        print("  ⚠️  No detection results found for any concepts")
        return
    
    # Create DataFrame with all concepts at their optimal percentiles
    detection_df = pd.DataFrame(all_detection_results)
    
    print(f"\n  Using optimal percentiles from calibration:")
    print(f"    Unique percentiles used: {sorted(percentiles_used)}")
    print(f"    Concepts analyzed: {len(detection_df)}")
    
    # Compute per-concept bootstrap
    print("\n  Computing per-concept bootstrap confidence intervals...")
    per_concept_results = compute_per_concept_bootstrap(
        detection_df,
        gt_images_per_concept_split,
        total_split_samples,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed
    )
    
    # Load activation loader for dataset-level bootstrap
    # Determine activation file name based on con_label type
    if 'linsep' in con_label and 'kmeans' not in con_label:
        acts_file = f"dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif 'kmeans' in con_label and 'linsep' not in con_label:
        n_clusters = int(con_label.split('_')[2])  # Extract cluster count
        acts_file = f"cosine_similarities_kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    elif 'linsep' in con_label and 'kmeans' in con_label:
        n_clusters = int(con_label.split('_')[2])
        acts_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    else:  # avg
        acts_file = f"cosine_similarities_avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR)
        
        # Compute dataset-level bootstrap
        print("  Computing dataset-level bootstrap confidence intervals...")
        dataset_results_macro = compute_dataset_level_bootstrap(
            per_concept_results,
            gt_images_per_concept_split,
            act_loader,
            None,  # No single percentile - using optimal per concept
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            aggregation='macro',
            seed=seed
        )
        
        dataset_results_weighted = compute_dataset_level_bootstrap(
            per_concept_results,
            gt_images_per_concept_split,
            act_loader,
            None,  # No single percentile - using optimal per concept
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            aggregation='weighted',
            seed=seed
        )
        
        dataset_results_micro = compute_dataset_level_bootstrap(
            per_concept_results,
            gt_images_per_concept_split,
            act_loader,
            None,  # No single percentile - using optimal per concept
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            aggregation='micro',
            seed=seed
        )
        
        # Combine all results
        dataset_results = {**dataset_results_macro, **dataset_results_weighted, **dataset_results_micro}
        
    except FileNotFoundError:
        print(f"    ⚠️  Could not load activation file for dataset-level bootstrap")
        dataset_results = {}
    
    # Save results with confidence intervals
    # Include split information in the saved filename
    if split == 'cal':
        save_label = f"{con_label}_cal"
    else:
        save_label = con_label
    save_results_with_ci(
        per_concept_results,
        dataset_results,
        dataset_name,
        save_label,
        'optimal',  # Using optimal percentiles
        output_dir="Quant_Results_with_CI"
    )
    
    # Print summary statistics
    print(f"\n  Summary Statistics:")
    print(f"    Mean F1: {per_concept_results['f1'].mean():.3f} ± {per_concept_results['f1_error'].mean():.3f}")
    print(f"    Mean Precision: {per_concept_results['precision'].mean():.3f} ± {per_concept_results['precision_error'].mean():.3f}")
    print(f"    Mean Recall: {per_concept_results['recall'].mean():.3f} ± {per_concept_results['recall_error'].mean():.3f}")
    
    if dataset_results:
        if 'macro_f1' in dataset_results:
            macro_f1 = dataset_results['macro_f1']
            print(f"    Dataset Macro F1: {macro_f1[0]:.3f} [{macro_f1[1]:.3f}, {macro_f1[2]:.3f}]")
        if 'weighted_f1' in dataset_results:
            weighted_f1 = dataset_results['weighted_f1']
            print(f"    Dataset Weighted F1: {weighted_f1[0]:.3f} [{weighted_f1[1]:.3f}, {weighted_f1[2]:.3f}]")
        if 'micro_f1' in dataset_results:
            micro_f1 = dataset_results['micro_f1']
            print(f"    Dataset Micro F1: {micro_f1[0]:.3f} [{micro_f1[1]:.3f}, {micro_f1[2]:.3f}]")


def process_baseline_configuration(
    dataset_name, model_name, model_input_size, sample_type,
    con_label, baseline_method, percent_thru_model, split, n_bootstrap, confidence_level, seed
):
    """
    Process a baseline method configuration for error analysis.
    Uses the OPTIMAL calibration percentile for each concept.
    
    Args:
        dataset_name: Dataset to process
        model_name: Model name
        model_input_size: Model input dimensions
        sample_type: 'patch' or 'cls'
        con_label: Concept configuration label
        baseline_method: 'maxtoken', 'meantoken', 'lasttoken', or 'randomtoken'
        percent_thru_model: Percentage through model
        split: 'test' or 'cal' for which data split to process
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
        seed: Random seed
    """
    print(f"\nProcessing {baseline_method} baseline for {con_label} - {split} split")
    print("-" * 50)
    
    # Load the OPTIMAL calibration percentiles for baselines
    # Check if this is an unsupervised method (kmeans or SAE)
    is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
    
    if is_unsupervised:
        # For unsupervised methods, best percentiles are in bestdetects files
        # For calibration split, append _cal to the con_label for bestdetects files
        if split == 'cal':
            con_label_with_split = f"{con_label}_cal"
        else:
            con_label_with_split = con_label
        bestdetects_file = f'Unsupervised_Matches/{dataset_name}/bestdetects_{baseline_method}_{con_label_with_split}.pt'
        if not os.path.exists(bestdetects_file):
            print(f"  ⚠️  No bestdetects file found at {bestdetects_file}")
            print(f"      Run baseline_detections.py first to generate optimal matches")
            return
        bestdetects = torch.load(bestdetects_file, weights_only=False)
        # Extract best_percentiles from bestdetects
        best_percentiles = {concept: info['best_percentile'] 
                           for concept, info in bestdetects.items() if 'best_percentile' in info}
        # For unsupervised methods, we don't have separate F1 scores stored
        best_f1_scores = {concept: info.get('best_score', 0) 
                         for concept, info in bestdetects.items()}
    else:
        # For supervised methods, use regular best_percentiles file
        # Baseline best_percentiles files are saved without _cal suffix, even though they use calibration data
        best_percentiles_file = f'Quant_Results/{dataset_name}/{baseline_method}_best_percentiles_{con_label}.pt'
        if not os.path.exists(best_percentiles_file):
            print(f"  ⚠️  No best percentiles file found at {best_percentiles_file}")
            print(f"      Run baseline_detections.py first to generate optimal percentiles")
            return
        
        best_percentiles_data = torch.load(best_percentiles_file, weights_only=False)
        best_percentiles = best_percentiles_data['best_percentiles']
        best_f1_scores = best_percentiles_data['best_f1_scores']
    
    print(f"  Loaded optimal {baseline_method} percentiles for {len(best_percentiles)} concepts")
    
    # Load ground truth - ALWAYS use sample-level (image/paragraph) GT
    if split == 'cal':
        gt_images_per_concept_split = torch.load(
            f'GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt',
            weights_only=False
        )
    else:
        gt_images_per_concept_split = torch.load(
            f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt',
            weights_only=False
        )
    
    # Filter to relevant concepts
    gt_images_per_concept_split = filter_concept_dict(gt_images_per_concept_split, dataset_name)
    
    # Get total number of samples for this split
    from utils.general_utils import get_split_df
    split_df = get_split_df(dataset_name)
    total_split_samples = len(split_df[split_df == split])
    
    # Collect detection results for all concepts at their optimal percentiles
    all_detection_results = []
    percentiles_used = set()
    
    for concept, optimal_percentile in best_percentiles.items():
        percentiles_used.add(optimal_percentile)
        
        # Load baseline detection results for this percentile
        detection_results = load_baseline_detection_results_v2(
            dataset_name, con_label, optimal_percentile, baseline_method, total_split_samples, split
        )
        
        if detection_results is None:
            print(f"  ⚠️  No detection results found for percentile {optimal_percentile}, skipping concept {concept}")
            continue
        
        # Extract this concept's results
        if concept in detection_results:
            concept_data = detection_results[concept]
        else:
            print(f"  ⚠️  Concept {concept} not found in detection results")
            continue
        
        # Add optimal percentile info
        concept_data['concept'] = concept
        concept_data['optimal_percentile'] = optimal_percentile
        concept_data['calibration_f1'] = best_f1_scores.get(concept, np.nan)
        
        all_detection_results.append(concept_data)
    
    if not all_detection_results:
        print("  ⚠️  No detection results found for any concepts")
        return
    
    # Create DataFrame with all concepts at their optimal percentiles
    detection_df = pd.DataFrame(all_detection_results)
    
    print(f"\n  Using optimal {baseline_method} percentiles from calibration:")
    print(f"    Unique percentiles used: {sorted(percentiles_used)}")
    print(f"    Concepts analyzed: {len(detection_df)}")
    
    # Compute per-concept bootstrap
    print(f"\n  Computing per-concept bootstrap confidence intervals for {baseline_method}...")
    per_concept_results = compute_per_concept_bootstrap(
        detection_df,
        gt_images_per_concept_split,
        total_split_samples,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed
    )
    
    # For baselines, we skip the dataset-level bootstrap that requires activation loader
    # Just use empty results
    dataset_results = {}
    
    # Save results with confidence intervals
    # Include split information in the saved filename
    if split == 'cal':
        save_label = f"{baseline_method}_{con_label}_cal"
    else:
        save_label = f"{baseline_method}_{con_label}"
    save_results_with_ci(
        per_concept_results,
        dataset_results,
        dataset_name,
        save_label,
        'optimal',
        output_dir="Quant_Results_with_CI"
    )
    
    # Print summary statistics
    print(f"\n  Summary Statistics for {baseline_method}:")
    print(f"    Mean F1: {per_concept_results['f1'].mean():.3f} ± {per_concept_results['f1_error'].mean():.3f}")
    print(f"    Mean Precision: {per_concept_results['precision'].mean():.3f} ± {per_concept_results['precision_error'].mean():.3f}")
    print(f"    Mean Recall: {per_concept_results['recall'].mean():.3f} ± {per_concept_results['recall_error'].mean():.3f}")


def process_sae_configuration(
    dataset_name, model_name, model_input_size, sample_type,
    con_label, percent_thru_model, split, n_bootstrap, confidence_level, seed
):
    """
    Process SAE configuration for error analysis.
    SAE doesn't have optimal percentiles per concept - it uses a single percentile.
    
    Args:
        dataset_name: Dataset to process
        model_name: Model name
        model_input_size: Model input dimensions
        sample_type: 'patch' or 'cls'
        con_label: SAE concept configuration label
        percent_thru_model: Percentage through model
        split: 'test' or 'cal' for which data split to process
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
        seed: Random seed
    """
    print(f"\nProcessing SAE: {con_label} - {split} split")
    print("-" * 50)
    
    # SAE uses a different best percentile file format
    best_percentiles_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    if not os.path.exists(best_percentiles_file):
        return
    
    # For SAE, we typically use a single percentile for all units
    # Let's check what format the file is in
    best_percentiles_data = torch.load(best_percentiles_file, weights_only=False)
    
    # Load ground truth
    gt_images_per_concept_test = torch.load(
        f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt',
        weights_only=False
    )
    gt_images_per_concept_test = filter_concept_dict(gt_images_per_concept_test, dataset_name)
    
    # Get total number of test samples
    from utils.general_utils import get_split_df
    split_df = get_split_df(dataset_name)
    total_test_samples = len(split_df[split_df == 'test'])
    
    # For SAE, we might have a single optimal percentile
    # Let's use a default of 0.1 if not found
    optimal_percentile = 0.1
    if isinstance(best_percentiles_data, dict) and 'best_percentile' in best_percentiles_data:
        optimal_percentile = best_percentiles_data['best_percentile']
    elif isinstance(best_percentiles_data, float):
        optimal_percentile = best_percentiles_data
        
    print(f"  Using percentile: {optimal_percentile}")
    
    # Load SAE detection results
    detection_results = load_sae_detection_results(dataset_name, con_label, optimal_percentile)
    
    if detection_results is None or len(detection_results) == 0:
        print(f"  No SAE detection results found")
        return
    
    print(f"  Loaded detection results for {len(detection_results)} concepts")
    
    # Convert to list format for bootstrap
    all_detection_results = []
    for concept, concept_data in detection_results.items():
        concept_data['concept'] = concept
        concept_data['optimal_percentile'] = optimal_percentile
        all_detection_results.append(concept_data)
    
    # Create DataFrame
    detection_df = pd.DataFrame(all_detection_results)
    
    print(f"    Concepts analyzed: {len(detection_df)}")
    
    # Compute per-concept bootstrap
    print("\n  Computing per-concept bootstrap confidence intervals...")
    per_concept_results = compute_per_concept_bootstrap(
        detection_df,
        gt_images_per_concept_test,
        total_test_samples,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed
    )
    
    # For SAE, the activation file is different
    acts_file = f"sae_acts_patchsae_{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt"
    
    try:
        act_loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR)
        
        # Compute dataset-level bootstrap
        print("  Computing dataset-level bootstrap confidence intervals...")
        dataset_results_macro = compute_dataset_level_bootstrap(
            per_concept_results,
            gt_images_per_concept_test,
            act_loader,
            optimal_percentile,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            aggregation='macro',
            seed=seed
        )
        
        dataset_results_weighted = compute_dataset_level_bootstrap(
            per_concept_results,
            gt_images_per_concept_test,
            act_loader,
            optimal_percentile,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            aggregation='weighted',
            seed=seed
        )
        
        dataset_results_micro = compute_dataset_level_bootstrap(
            per_concept_results,
            gt_images_per_concept_test,
            act_loader,
            optimal_percentile,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            aggregation='micro',
            seed=seed
        )
        
        # Combine all results
        dataset_results = {**dataset_results_macro, **dataset_results_weighted, **dataset_results_micro}
        
    except FileNotFoundError:
        print(f"    ⚠️  Could not load SAE activation file for dataset-level bootstrap")
        dataset_results = {}
    
    # Save results with confidence intervals
    save_results_with_ci(
        per_concept_results,
        dataset_results,
        dataset_name,
        con_label,
        'optimal',  # Using optimal percentile
        output_dir="Quant_Results_with_CI"
    )
    
    # Print summary statistics
    print(f"\n  Summary Statistics for SAE:")
    print(f"    Mean F1: {per_concept_results['f1'].mean():.3f} ± {per_concept_results['f1_error'].mean():.3f}")
    print(f"    Mean Precision: {per_concept_results['precision'].mean():.3f} ± {per_concept_results['precision_error'].mean():.3f}")
    print(f"    Mean Recall: {per_concept_results['recall'].mean():.3f} ± {per_concept_results['recall_error'].mean():.3f}")
    
    if dataset_results:
        if 'macro_f1' in dataset_results:
            macro_f1 = dataset_results['macro_f1']
            print(f"    Dataset Macro F1: {macro_f1[0]:.3f} [{macro_f1[1]:.3f}, {macro_f1[2]:.3f}]")
        if 'weighted_f1' in dataset_results:
            weighted_f1 = dataset_results['weighted_f1']
            print(f"    Dataset Weighted F1: {weighted_f1[0]:.3f} [{weighted_f1[1]:.3f}, {weighted_f1[2]:.3f}]")


def process_prompt_results(
    dataset_name, model_input_size, n_bootstrap, confidence_level, seed
):
    """
    Process prompt-based detection results for error analysis.
    
    Args:
        dataset_name: Dataset to process
        model_input_size: Model input dimensions (for loading ground truth)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
        seed: Random seed
    """
    print(f"\nProcessing prompt results for {dataset_name}")
    print("-" * 50)
    
    # Load prompt detection results
    detection_results = load_prompt_detection_results(dataset_name)
    
    if detection_results is None:
        print(f"  ⚠️  No prompt results found for {dataset_name}")
        return
    
    print(f"  Loaded prompt results for {len(detection_results)} concepts")
    
    # Load ground truth - ALWAYS use sample-level (image/paragraph) GT
    # For prompt, we use Llama vision model dimensions
    gt_images_per_concept_test = torch.load(
        f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt',
        weights_only=False
    )
    
    # Filter to relevant concepts
    gt_images_per_concept_test = filter_concept_dict(gt_images_per_concept_test, dataset_name)
    
    # Get total number of test samples
    from utils.general_utils import get_split_df
    split_df = get_split_df(dataset_name)
    total_test_samples = len(split_df[split_df == 'test'])
    
    # Create DataFrame from detection results
    all_detection_results = []
    for concept, data in detection_results.items():
        if concept in gt_images_per_concept_test:  # Only include concepts that are in GT
            data['concept'] = concept
            all_detection_results.append(data)
    
    if not all_detection_results:
        print("  ⚠️  No valid concepts found")
        return
        
    detection_df = pd.DataFrame(all_detection_results)
    print(f"  Processing {len(detection_df)} concepts")
    
    # Compute per-concept bootstrap
    print("\n  Computing per-concept bootstrap confidence intervals for prompt...")
    per_concept_results = compute_per_concept_bootstrap(
        detection_df,
        gt_images_per_concept_test,
        total_test_samples,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed
    )
    
    # For prompt, we also skip dataset-level bootstrap
    dataset_results = {}
    
    # Save results with confidence intervals
    save_results_with_ci(
        per_concept_results,
        dataset_results,
        dataset_name,
        "prompt_Llama-3.2-11B-Vision",
        'direct',  # No percentile optimization for prompt
        output_dir="Quant_Results_with_CI"
    )
    
    # Print summary statistics
    print(f"\n  Summary Statistics for prompt:")
    print(f"    Mean F1: {per_concept_results['f1'].mean():.3f} ± {per_concept_results['f1_error'].mean():.3f}")
    print(f"    Mean Precision: {per_concept_results['precision'].mean():.3f} ± {per_concept_results['precision_error'].mean():.3f}")
    print(f"    Mean Recall: {per_concept_results['recall'].mean():.3f} ± {per_concept_results['recall_error'].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute bootstrap confidence intervals for detection metrics')
    parser.add_argument('--dataset', type=str, help='Specific dataset to process')
    parser.add_argument('--datasets', nargs='+', help='Multiple datasets to process')
    parser.add_argument('--model', type=str, help='Specific model to use')
    parser.add_argument('--models', nargs='+', help='Multiple models to use')
    parser.add_argument('--sample-type', type=str, choices=['patch', 'cls'], help='Sample type to process')
    parser.add_argument('--concept-types', nargs='+', choices=['avg', 'linsep', 'kmeans', 'linsepkmeans', 'sae'], 
                        default=['avg', 'linsep'],
                        help='Concept types to compute errors for (default: avg, linsep)')
    parser.add_argument('--baselines', nargs='+', choices=['prompt', 'maxtoken', 'meantoken', 'lasttoken', 'randomtoken'],
                        help='Baseline methods to compute errors for (default: none)')
    parser.add_argument('--baselines-only', action='store_true',
                        help='Only process baselines, skip regular concept detection')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP, 
                        help=f'Number of bootstrap iterations (default: {N_BOOTSTRAP})')
    parser.add_argument('--confidence-level', type=float, default=CONFIDENCE_LEVEL,
                        help=f'Confidence level for intervals (default: {CONFIDENCE_LEVEL})')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Random seed for reproducibility (default: {SEED})')
    parser.add_argument('--percentthrumodel', type=int, default=None,
                        help='Specific percentage through model (default: use all valid percentthrumodels for each model)')
    parser.add_argument('--splits', nargs='+', choices=['test', 'cal'], default=['test', 'cal'],
                        help='Which data splits to process (default: both test and cal)')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets and exit')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        for dataset in DATASETS:
            print(f"  - {dataset}")
        sys.exit(0)
    
    # List available models if requested
    if args.list_models:
        print("Available models:")
        for model_name, input_size in MODELS:
            print(f"  - {model_name}: {input_size}")
        sys.exit(0)
    
    # Validate arguments
    if args.baselines_only and not args.baselines:
        print("Error: --baselines-only requires --baselines to be specified")
        sys.exit(1)
    
    # Determine which datasets to process
    if args.dataset:
        datasets_to_process = [args.dataset]
    elif args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = DATASETS
    
    # Determine which models to process
    all_available_models = MODELS
    
    if args.models:
        models_to_process = []
        for model_name in args.models:
            found = [(m, s) for m, s in all_available_models if m == model_name]
            if not found:
                print(f"Error: Model '{model_name}' not found")
                sys.exit(1)
            models_to_process.extend(found)
    elif args.model:
        models_to_process = [(m, s) for m, s in MODELS if m == args.model]
        if not models_to_process:
            print(f"Error: Model '{args.model}' not found")
            sys.exit(1)
    else:
        models_to_process = MODELS
    
    # Determine which sample types to process
    if args.sample_type:
        sample_types_to_process = [(s, n) for s, n in SAMPLE_TYPES if s == args.sample_type]
    else:
        sample_types_to_process = SAMPLE_TYPES
    
    print(f"\nDETECTION ERROR ANALYSIS")
    print(f"========================")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Confidence level: {args.confidence_level}")
    print(f"Random seed: {args.seed}")
    print(f"Using OPTIMAL percentiles from calibration set")
    print(f"Concept types: {args.concept_types}")
    if args.baselines:
        print(f"Baseline methods: {args.baselines}")
    if args.percentthrumodel:
        print(f"Using specific percentthrumodel: {args.percentthrumodel}")
    else:
        print(f"Using all valid percentthrumodels for each model")
    print(f"Processing splits: {args.splits}")
    print()
    
    # Process all configurations
    experiment_configs = product(models_to_process, datasets_to_process, sample_types_to_process)
    
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
        # Skip invalid dataset-input size combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        
        # Get valid percentthrumodels for this model
        if args.percentthrumodel:
            # Use specific percentthrumodel if provided
            percentthrumodels_to_process = [args.percentthrumodel]
        else:
            # Use all valid percentthrumodels for this model
            percentthrumodels_to_process = get_model_default_percentthrumodels(model_name, model_input_size)
        
        # Loop through each percentthrumodel for this model
        for percent_thru_model in percentthrumodels_to_process:
            # Loop through each split
            for split in args.splits:
                print(f"\n{'='*80}")
                print(f"Processing: {model_name} - {dataset_name} - {sample_type} - PTM {percent_thru_model} - {split.upper()} split")
                print(f"{'='*80}")
                
                # Process prompt baseline if requested (only once per dataset, not per model, not per PTM, not per split)
                if args.baselines and 'prompt' in args.baselines and model_name == models_to_process[0][0] and percent_thru_model == percentthrumodels_to_process[0] and split == args.splits[0]:
                    # Only process prompt once per dataset (with the first model, first PTM, and first split)
                    process_prompt_results(
                        dataset_name, model_input_size,
                        args.n_bootstrap, args.confidence_level, args.seed
                    )
                
                # Get all concept configurations
                all_labels = get_all_files(model_name, sample_type, n_clusters, percent_thru_model, args.concept_types)
                
                for concept_type, con_label in all_labels:
                    # Skip regular detection if baselines-only mode
                    if not args.baselines_only:
                        if concept_type == 'sae':
                            # Handle SAE differently - it doesn't use percentiles
                            process_sae_configuration(
                                dataset_name, model_name, model_input_size, sample_type,
                                con_label, percent_thru_model, split,
                                args.n_bootstrap, args.confidence_level, args.seed
                            )
                        else:
                            # Check if best percentiles file exists
                            # Best percentiles files don't have _cal suffix - they're always in Best_Detection_Percentiles_Cal directory
                            best_percentiles_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
                            if not os.path.exists(best_percentiles_file):
                                print(f"\n⚠️  No best percentiles file found for {con_label} ({split} split), skipping...")
                                print(f"   (Run all_detection_stats.py first to generate optimal percentiles)")
                                continue
                            
                            process_concept_configuration(
                                dataset_name, model_name, model_input_size, sample_type,
                                con_label, percent_thru_model, split,
                                args.n_bootstrap, args.confidence_level, args.seed
                            )
                    
                    # Process baselines if requested
                    if args.baselines and sample_type == 'patch':  # Baselines only work for patch/token
                        for baseline in args.baselines:
                            if baseline == 'prompt':
                                continue  # Handle prompt separately after all concept types
                            
                            process_baseline_configuration(
                                dataset_name, model_name, model_input_size, sample_type,
                                con_label, baseline, percent_thru_model, split,
                                args.n_bootstrap, args.confidence_level, args.seed
                            )
                    
                    # Clean up memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    print("\n✅ Detection error analysis complete!")
    print(f"Results saved to: Quant_Results_with_CI/")