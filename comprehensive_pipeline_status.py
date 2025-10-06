#!/usr/bin/env python3
"""
Comprehensive pipeline status checker that accurately tracks all pipeline stages.
"""

import os
import glob
import json
import argparse
from collections import defaultdict
from pathlib import Path

# Default configurations
IMAGE_DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
TEXT_DATASETS = ['Sarcasm', 'iSarcasm', 'GoEmotions']
ALL_DATASETS = IMAGE_DATASETS + TEXT_DATASETS

IMAGE_MODELS = ['CLIP', 'Llama']
TEXT_MODELS = ['Llama', 'Gemma', 'Qwen']

SAMPLE_TYPES = ['cls', 'patch']
CONCEPT_TYPES = ['avg', 'linsep', 'kmeans', 'linsepkmeans']

# Import default percentthrumodels
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.default_percentthrumodels import (
    CLIP_PERCENTTHRUMODELS, LLAMA_VISION_PERCENTTHRUMODELS,
    LLAMA_TEXT_PERCENTTHRUMODELS, GEMMA_TEXT_PERCENTTHRUMODELS, 
    QWEN_TEXT_PERCENTTHRUMODELS
)

# Pipeline stage locations
STAGE_LOCATIONS = {
    'embeddings': '/scratch/cgoldberg/Embeddings',
    'concepts': '/workspace/Experiments/Concepts',
    'cosine_similarities': '/scratch/cgoldberg/Cosine_Similarities',
    'distances': '/scratch/cgoldberg/Distances',
    'thresholds': '/workspace/Experiments/Thresholds',
    'detection': '/workspace/Experiments/Quant_Results',
    'inversion': '/workspace/Experiments/Quant_Results',
    'baselines': '/workspace/Experiments/Baselines',
    'error_bars': '/workspace/Experiments/Quant_Results_with_CI',
    'per_concept_ptm': '/workspace/Experiments/Per_Concept_PTM_Optimization'
}

def get_model_percentthrumodels(model, dataset):
    """Get the appropriate percentthrumodel values for a model and dataset."""
    if model == 'CLIP':
        return CLIP_PERCENTTHRUMODELS
    elif model == 'Llama':
        if dataset in IMAGE_DATASETS:
            return LLAMA_VISION_PERCENTTHRUMODELS
        else:
            return LLAMA_TEXT_PERCENTTHRUMODELS
    elif model == 'Gemma':
        return GEMMA_TEXT_PERCENTTHRUMODELS
    elif model == 'Qwen':
        return QWEN_TEXT_PERCENTTHRUMODELS
    else:
        return []

def check_embeddings(dataset, model, sample_type, percentthru):
    """Check if embeddings exist for a configuration."""
    base_path = Path(STAGE_LOCATIONS['embeddings']) / dataset
    
    # Check for main embedding file
    main_file = base_path / f"{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
    if main_file.exists():
        return True, "single_file"
    
    # Check for chunked embeddings
    chunk_pattern = str(base_path / f"{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}_chunk_*.pt")
    chunks = glob.glob(chunk_pattern)
    if chunks:
        return True, f"{len(chunks)}_chunks"
    
    # Check for info file (indicates embeddings were started)
    info_file = base_path / f"{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}_chunks_info.json"
    if info_file.exists():
        return True, "info_only"
    
    return False, None

def check_concepts(dataset, model, sample_type, percentthru, concept_types):
    """Check which concepts exist for a configuration."""
    base_path = Path(STAGE_LOCATIONS['concepts']) / dataset
    found_concepts = {}
    
    for concept_type in concept_types:
        if concept_type == 'avg':
            file_path = base_path / f"avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            found_concepts['avg'] = file_path.exists()
        
        elif concept_type == 'linsep':
            file_path = base_path / f"linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            found_concepts['linsep'] = file_path.exists()
        
        elif concept_type == 'kmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            file_path = base_path / f"kmeans_{n_clusters}_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            found_concepts['kmeans'] = file_path.exists()
        
        elif concept_type == 'linsepkmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            file_path = base_path / f"kmeans_{n_clusters}_linsep_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            found_concepts['linsepkmeans'] = file_path.exists()
        
        elif concept_type == 'sae':
            # Skip invalid SAE configurations
            if not ((model == 'CLIP' and percentthru == 92) or 
                    (model == 'Gemma' and percentthru == 81)):
                found_concepts['sae'] = False
                continue
            # SAE concepts are the SAE dictionary itself (not learned)
            found_concepts['sae'] = True  # SAE concepts are pre-defined
    
    return found_concepts

def check_activations(dataset, model, sample_type, percentthru, concept_types):
    """Check which activations exist for a configuration."""
    cos_path = Path(STAGE_LOCATIONS['cosine_similarities']) / dataset
    dist_path = Path(STAGE_LOCATIONS['distances']) / dataset
    found_activations = {}
    
    for concept_type in concept_types:
        if concept_type == 'avg':
            # Check cosine similarities
            cos_file = cos_path / f"cosine_similarities_avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            cos_chunks = glob.glob(str(cos_path / f"cosine_similarities_avg_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}_chunk_*.pt"))
            found_activations['avg_cosine'] = cos_file.exists() or len(cos_chunks) > 0
        
        elif concept_type == 'linsep':
            # Check distances
            dist_file = dist_path / f"dists_linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            dist_chunks = glob.glob(str(dist_path / f"dists_linsep_concepts_BD_True_BN_False_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}_chunk_*.pt"))
            found_activations['linsep_dist'] = dist_file.exists() or len(dist_chunks) > 0
        
        elif concept_type == 'kmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            cos_file = cos_path / f"cosine_similarities_kmeans_{n_clusters}_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            cos_chunks = glob.glob(str(cos_path / f"cosine_similarities_kmeans_{n_clusters}_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}_chunk_*.pt"))
            found_activations['kmeans_cosine'] = cos_file.exists() or len(cos_chunks) > 0
        
        elif concept_type == 'linsepkmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            dist_file = dist_path / f"dists_kmeans_{n_clusters}_linsep_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            dist_chunks = glob.glob(str(dist_path / f"dists_kmeans_{n_clusters}_linsep_concepts_{model}_{sample_type}_embeddings_percentthrumodel_{percentthru}_chunk_*.pt"))
            found_activations['linsepkmeans_dist'] = dist_file.exists() or len(dist_chunks) > 0
        
        elif concept_type == 'sae':
            # Skip invalid SAE configurations
            if not ((model == 'CLIP' and percentthru == 92) or 
                    (model == 'Gemma' and percentthru == 81)):
                found_activations['sae_dense'] = False
                continue
            # SAE dense activations have special naming
            sae_base = 'clipscope' if model == 'CLIP' else 'gemmascope'
            sae_file = cos_path / f"{sae_base}_{sample_type}_dense.pt"
            sae_chunks = glob.glob(str(cos_path / f"{sae_base}_{sample_type}_dense_chunk_*.pt"))
            found_activations['sae_dense'] = sae_file.exists() or len(sae_chunks) > 0
    
    return found_activations

def check_thresholds(dataset, model, sample_type, percentthru, concept_types):
    """Check which thresholds exist for a configuration."""
    base_path = Path(STAGE_LOCATIONS['thresholds']) / dataset
    found_thresholds = {}
    
    for concept_type in concept_types:
        if concept_type == 'avg':
            file_path = base_path / f"all_percentiles_{model}_avg_{sample_type}_embeddings_percentthrumodel_{percentthru}.pt"
            found_thresholds['avg'] = file_path.exists()
        
        elif concept_type == 'linsep':
            file_path = base_path / f"all_percentiles_{model}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthru}.pt"
            found_thresholds['linsep'] = file_path.exists()
        
        elif concept_type == 'kmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            # Check multiple patterns
            patterns = [
                f"all_percentiles_{model}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}.pt",
                f"all_percentiles_allpairs_{model}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}.pt",
                # Also check for baseline-specific threshold files
                f"maxtoken_all_percentiles_allpairs_{model}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}.pt",
                f"meantoken_all_percentiles_allpairs_{model}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}.pt"
            ]
            found_thresholds['kmeans'] = any((base_path / pattern).exists() for pattern in patterns)
        
        elif concept_type == 'linsepkmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            # Check multiple patterns
            patterns = [
                f"all_percentiles_{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}.pt",
                f"all_percentiles_allpairs_{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}.pt",
                # Also check for baseline-specific threshold files
                f"maxtoken_all_percentiles_allpairs_{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}.pt",
                f"meantoken_all_percentiles_allpairs_{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}.pt"
            ]
            found_thresholds['linsepkmeans'] = any((base_path / pattern).exists() for pattern in patterns)
        
        elif concept_type == 'sae':
            # Skip invalid SAE configurations
            if not ((model == 'CLIP' and percentthru == 92) or 
                    (model == 'Gemma' and percentthru == 81)):
                found_thresholds['sae'] = False
                continue
            file_path = base_path / f"all_percentiles_{model}_sae_{sample_type}_dense.pt"
            found_thresholds['sae'] = file_path.exists()
    
    return found_thresholds

def check_detection(dataset, model, sample_type, percentthru, concept_types):
    """Check which detection stats exist for a configuration."""
    base_path = Path(STAGE_LOCATIONS['detection']) / dataset
    found_detection = {}
    
    for concept_type in concept_types:
        if concept_type == 'avg':
            con_label = f"{model}_avg_{sample_type}_embeddings_percentthrumodel_{percentthru}"
        elif concept_type == 'linsep':
            con_label = f"{model}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthru}"
        elif concept_type == 'kmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            con_label = f"{model}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}"
        elif concept_type == 'linsepkmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            con_label = f"{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}"
        elif concept_type == 'sae':
            # Skip invalid SAE configurations
            if not ((model == 'CLIP' and percentthru == 92) or 
                    (model == 'Gemma' and percentthru == 81)):
                found_detection[concept_type] = False
                continue
            con_label = f"{model}_sae_{sample_type}_dense"
        else:
            continue
        
        # Different file types for supervised vs unsupervised
        if concept_type in ['avg', 'linsep']:
            # Supervised methods save as .pt files
            detectionmetrics_files = glob.glob(str(base_path / f"detectionmetrics_per_*_{con_label}.pt"))
            detectionmetrics_files += glob.glob(str(base_path / f"detectionmetrics_per_*_{con_label}_cal.pt"))
            detection_files = detectionmetrics_files
        else:
            # Unsupervised methods (kmeans/linsepkmeans) save as .csv files after filtering
            # First check for filtered CSV files (these are the final output)
            detectionmetrics_files = glob.glob(str(base_path / f"detectionmetrics_per_*_{con_label}.csv"))
            detectionmetrics_files += glob.glob(str(base_path / f"detectionmetrics_per_*_{con_label}_cal.csv"))
            
            # For cls level kmeans/linsepkmeans, the files may not have 'per_' in the name
            if sample_type == 'cls' and len(detectionmetrics_files) == 0:
                # Try pattern without 'per_' for cls
                detectionmetrics_files = glob.glob(str(base_path / f"detectionmetrics_0.*_{con_label}.csv"))
                detectionmetrics_files += glob.glob(str(base_path / f"detectionmetrics_0.*_{con_label}_cal.csv"))
            
            # If no filtered files, check for allpairs files (intermediate output)
            if len(detectionmetrics_files) == 0:
                allpairs_files = glob.glob(str(base_path / f"detectionmetrics_allpairs_per_*_{con_label}.csv"))
                allpairs_files += glob.glob(str(base_path / f"detectionmetrics_allpairs_per_*_{con_label}_cal.csv"))
                if allpairs_files:
                    # If allpairs exist but no filtered files, detection is partially done
                    found_detection[concept_type] = True
                    continue
            detection_files = detectionmetrics_files
        
        # Also check for detectfirst files (final summary files)
        # For supervised methods (avg, linsep), detectfirst files are CSVs with baseline prefixes
        if concept_type in ['avg', 'linsep']:
            for baseline in ['maxtoken', 'meantoken', 'lasttoken', 'randomtoken']:
                detectfirst_files = glob.glob(str(base_path / f"detectfirst_{baseline}_*_{con_label}.csv"))
                if detectfirst_files:
                    detection_files += detectfirst_files
                    break  # Found at least one baseline
        else:
            # For unsupervised methods (kmeans, linsepkmeans), detectfirst files are CSVs with baseline prefixes
            for baseline in ['maxtoken', 'meantoken', 'lasttoken', 'randomtoken']:
                detectfirst_files = glob.glob(str(base_path / f"detectfirst_{baseline}_*_{con_label}.csv"))
                detectfirst_files += glob.glob(str(base_path / f"detectfirst_{baseline}_*_{con_label}_cal.csv"))
                if detectfirst_files:
                    detection_files += detectfirst_files
                    break  # Found at least one baseline
        
        found_detection[concept_type] = len(detection_files) > 0
    
    return found_detection

def check_inversion(dataset, model, sample_type, percentthru, concept_types):
    """Check which inversion stats exist for a configuration (image datasets only)."""
    if dataset not in IMAGE_DATASETS:
        return {ct: 'N/A' for ct in concept_types}
    
    base_path = Path(STAGE_LOCATIONS['inversion']) / dataset
    found_inversion = {}
    
    for concept_type in concept_types:
        if concept_type == 'avg':
            con_label = f"{model}_avg_{sample_type}_embeddings_percentthrumodel_{percentthru}"
        elif concept_type == 'linsep':
            con_label = f"{model}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthru}"
        elif concept_type == 'kmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            con_label = f"{model}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}"
        elif concept_type == 'linsepkmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            con_label = f"{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}"
        elif concept_type == 'sae':
            # Skip invalid SAE configurations
            if not ((model == 'CLIP' and percentthru == 92) or 
                    (model == 'Gemma' and percentthru == 81)):
                found_inversion[concept_type] = False
                continue
            con_label = f"{model}_sae_{sample_type}_dense"
        else:
            continue
        
        # Check for inversion files
        inversion_files = glob.glob(str(base_path / f"*invert*_{con_label}.pt"))
        found_inversion[concept_type] = len(inversion_files) > 0
    
    return found_inversion

def check_baseline_detections(dataset, model, sample_type, percentthru, concept_types):
    """Check which baseline detection files exist for a configuration."""
    # Baseline detections are only for patch/token samples, not CLS
    if sample_type == 'cls':
        return {ct: 'N/A' for ct in concept_types}
    
    # Baseline detections are saved in Quant_Results, not a separate Baselines directory
    base_path = Path(STAGE_LOCATIONS['detection']) / dataset
    found_baselines = {}
    
    # For baseline detections, we check for different aggregation methods
    baseline_methods = ['maxtoken', 'meantoken', 'lasttoken', 'randomtoken']
    
    for concept_type in concept_types:
        if concept_type == 'avg':
            base_label = f"{model}_avg_{sample_type}_embeddings_percentthrumodel_{percentthru}"
        elif concept_type == 'linsep':
            base_label = f"{model}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthru}"
        elif concept_type == 'kmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            base_label = f"{model}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}"
        elif concept_type == 'linsepkmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            base_label = f"{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}"
        else:
            continue
        
        # Check that ALL four baseline methods exist
        methods_found = {}
        
        # Different patterns for supervised vs unsupervised
        if concept_type in ['avg', 'linsep']:
            # Supervised: detectfirst_{method}_{split}_{base_label}.csv
            for method in baseline_methods:
                # Test files
                test_files = glob.glob(str(base_path / f"detectfirst_{method}_test_{base_label}.csv"))
                # Cal files
                cal_files = glob.glob(str(base_path / f"detectfirst_{method}_cal_{base_label}.csv"))
                # Method is complete if both test and cal files exist
                methods_found[method] = len(test_files) > 0 and len(cal_files) > 0
        else:
            # Unsupervised: detectfirst_{method}_per_{percentile}_{base_label}[_cal].csv
            for method in baseline_methods:
                # Non-cal files
                files = glob.glob(str(base_path / f"detectfirst_{method}_per_*_{base_label}.csv"))
                # Cal files  
                cal_files = glob.glob(str(base_path / f"detectfirst_{method}_per_*_{base_label}_cal.csv"))
                # For unsupervised, we expect multiple percentiles (at least 10)
                methods_found[method] = len(files) >= 10 and len(cal_files) >= 10
        
        # Baseline is complete only if ALL four methods are found
        found_baselines[concept_type] = all(methods_found.values())
    
    return found_baselines

def check_error_bars(dataset, model, sample_type, percentthru, concept_types):
    """Check which error bar/bootstrap files exist for a configuration."""
    base_path = Path(STAGE_LOCATIONS['error_bars']) / dataset
    found_errors = {}
    
    for concept_type in concept_types:
        if concept_type == 'avg':
            base_label = f"{model}_avg_{sample_type}_embeddings_percentthrumodel_{percentthru}"
        elif concept_type == 'linsep':
            base_label = f"{model}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthru}"
        elif concept_type == 'kmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            base_label = f"{model}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}"
        elif concept_type == 'linsepkmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            base_label = f"{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percentthru}"
        else:
            continue
        
        # Check for per-concept confidence interval CSV files (output of detection_errors.py)
        # Pattern: per_concept_ci_optimal_{model}_{concept_type}_{sample_type}_embeddings_percentthrumodel_{ptm}[_cal].csv
        per_concept_ci_files = glob.glob(str(base_path / f"per_concept_ci_optimal_{base_label}.csv"))
        per_concept_ci_cal_files = glob.glob(str(base_path / f"per_concept_ci_optimal_{base_label}_cal.csv"))
        
        # Also check for older patterns in case they exist
        error_files = glob.glob(str(base_path / f"*error_bars*{base_label}*.csv"))
        bootstrap_files = glob.glob(str(base_path / f"*bootstrap*{base_label}*.csv"))
        
        found_errors[concept_type] = len(per_concept_ci_files) > 0 or len(per_concept_ci_cal_files) > 0 or len(error_files) > 0 or len(bootstrap_files) > 0
    
    return found_errors

def check_per_concept_ptm(dataset, model, sample_type, concept_types):
    """Check which per-concept PTM optimization files exist."""
    base_path = Path(STAGE_LOCATIONS['per_concept_ptm']) / dataset
    found_ptm_opt = {}
    
    for concept_type in concept_types:
        if concept_type == 'avg':
            base_pattern = f"optimal_ptm_*{model}_avg_{sample_type}_embeddings*.csv"
        elif concept_type == 'linsep':
            base_pattern = f"optimal_ptm_*{model}_linsep_{sample_type}_embeddings_BD_True_BN_False*.csv"
        elif concept_type == 'kmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            base_pattern = f"optimal_ptm_*{model}_kmeans_{n_clusters}_{sample_type}_embeddings*.csv"
        elif concept_type == 'linsepkmeans':
            n_clusters = 1000 if sample_type == 'patch' else 50
            base_pattern = f"optimal_ptm_*{model}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings*.csv"
        else:
            continue
        
        # Check for optimal PTM files
        ptm_files = glob.glob(str(base_path / base_pattern))
        found_ptm_opt[concept_type] = len(ptm_files) > 0
    
    return found_ptm_opt

def check_all_activations_exist(activations_dict, concept_types):
    """Check if all required activations exist for the given concept types."""
    required_activations = {
        'avg': 'avg_cosine',
        'linsep': 'linsep_dist',
        'kmeans': 'kmeans_cosine',
        'linsepkmeans': 'linsepkmeans_dist'
    }
    
    for concept_type in concept_types:
        if concept_type in required_activations:
            activation_key = required_activations[concept_type]
            if activation_key not in activations_dict or not activations_dict[activation_key]:
                return False
    
    return True

def print_status_table(results, datasets, models, sample_types, concept_types):
    """Print a formatted status table."""
    print("\nPIPELINE STATUS SUMMARY")
    print("=" * 120)
    
    # Also collect detailed missing info
    missing_details = {
        'concepts': [],
        'activations': [],
        'thresholds': [],
        'detection': []
    }
    
    for dataset in datasets:
        if dataset not in results:
            continue
            
        print(f"\n{dataset}")
        print("-" * 120)
        
        for model in models:
            if model not in results[dataset]:
                continue
                
            percentthrumodels = get_model_percentthrumodels(model, dataset)
            
            for ptm in percentthrumodels:
                if ptm not in results[dataset][model]:
                    continue
                
                print(f"\n  {model} - PTM {ptm}:")
                
                for sample_type in sample_types:
                    if sample_type not in results[dataset][model][ptm]:
                        continue
                    
                    status = results[dataset][model][ptm][sample_type]
                    
                    # Format status line
                    line_parts = [f"    {sample_type:5}"]
                    
                    # Check if all activations exist (embeddings might not be needed)
                    all_activations_exist = check_all_activations_exist(status['activations'], concept_types)
                    
                    # Embeddings
                    if not status['embeddings'][0] and all_activations_exist:
                        # Embeddings missing but not needed
                        emb_status = "✗"
                        emb_info = "(not_needed)"
                    else:
                        emb_status = "✓" if status['embeddings'][0] else "✗"
                        emb_info = f"({status['embeddings'][1]})" if status['embeddings'][1] else ""
                    line_parts.append(f"EMB:{emb_status}{emb_info}")
                    
                    # Concepts - with details
                    concept_statuses = []
                    for ct in concept_types:
                        if ct in status['concepts']:
                            if status['concepts'][ct]:
                                concept_statuses.append(f"{ct[:3]}:✓")
                            else:
                                concept_statuses.append(f"{ct[:3]}:✗")
                                if status['embeddings'][0]:  # Only if embeddings exist
                                    missing_details['concepts'].append((dataset, model, ct, sample_type, ptm))
                    if concept_statuses:
                        line_parts.append(f"CON:[{','.join(concept_statuses)}]")
                    
                    # Activations - with details
                    act_statuses = []
                    act_to_concept = {
                        'avg_cosine': 'avg',
                        'linsep_dist': 'linsep', 
                        'kmeans_cosine': 'kmeans',
                        'linsepkmeans_dist': 'linsepkmeans'
                    }
                    for key, val in status['activations'].items():
                        if val:
                            act_statuses.append(f"{key[:6]}:✓")
                        else:
                            act_statuses.append(f"{key[:6]}:✗")
                            # Check if concept exists before marking activation as missing
                            concept_type = act_to_concept.get(key)
                            if concept_type and concept_type in status['concepts'] and status['concepts'][concept_type]:
                                missing_details['activations'].append((dataset, model, concept_type, sample_type, ptm))
                    if act_statuses:
                        line_parts.append(f"ACT:[{','.join(act_statuses)}]")
                    
                    # Thresholds
                    thresh_statuses = []
                    for ct in concept_types:
                        if ct in status['thresholds']:
                            thresh_statuses.append(f"{ct[:3]}:{'✓' if status['thresholds'][ct] else '✗'}")
                    if thresh_statuses:
                        line_parts.append(f"THR:[{','.join(thresh_statuses)}]")
                    
                    # Detection
                    detect_statuses = []
                    for ct in concept_types:
                        if ct in status['detection']:
                            detect_statuses.append(f"{ct[:3]}:{'✓' if status['detection'][ct] else '✗'}")
                    if detect_statuses:
                        line_parts.append(f"DET:[{','.join(detect_statuses)}]")
                    
                    # Extended Analysis (for all PTMs when available)
                    if 'baselines' in status and status['baselines']:
                        # Baselines
                        baseline_statuses = []
                        for ct in concept_types:
                            if ct in status['baselines']:
                                if status['baselines'][ct] == 'N/A':
                                    baseline_statuses.append(f"{ct[:3]}:N/A")
                                else:
                                    baseline_statuses.append(f"{ct[:3]}:{'✓' if status['baselines'][ct] else '✗'}")
                        if baseline_statuses:
                            line_parts.append(f"BSL:[{','.join(baseline_statuses)}]")
                    
                    if 'error_bars' in status and status['error_bars']:
                        # Error Bars
                        error_statuses = []
                        for ct in concept_types:
                            if ct in status['error_bars']:
                                error_statuses.append(f"{ct[:3]}:{'✓' if status['error_bars'][ct] else '✗'}")
                        if error_statuses:
                            line_parts.append(f"ERR:[{','.join(error_statuses)}]")
                    
                    # Per-concept PTM optimization (not tied to specific PTM)
                    if 'per_concept_ptm' in status and status['per_concept_ptm']:
                        ptm_statuses = []
                        for ct in concept_types:
                            if ct in status['per_concept_ptm']:
                                ptm_statuses.append(f"{ct[:3]}:{'✓' if status['per_concept_ptm'][ct] else '✗'}")
                        if ptm_statuses:
                            line_parts.append(f"PTM:[{','.join(ptm_statuses)}]")
                    
                    print(" | ".join(line_parts))
    
    # Print detailed missing information if any
    if any(missing_details.values()):
        print("\n\nDETAILED MISSING COMPONENTS:")
        print("=" * 120)
        
        if missing_details['concepts']:
            print("\nMISSING CONCEPTS:")
            print("Format: Dataset | Model | ConceptType | SampleType | PTM")
            print("-" * 80)
            for dataset, model, concept_type, sample_type, ptm in sorted(missing_details['concepts']):
                print(f"{dataset:20} | {model:6} | {concept_type:12} | {sample_type:5} | PTM {ptm:3}")
        
        if missing_details['activations']:
            print("\nMISSING ACTIVATIONS (where concepts exist):")
            print("Format: Dataset | Model | ConceptType | SampleType | PTM")
            print("-" * 80)
            for dataset, model, concept_type, sample_type, ptm in sorted(missing_details['activations']):
                print(f"{dataset:20} | {model:6} | {concept_type:12} | {sample_type:5} | PTM {ptm:3}")

def generate_granular_commands(results, concept_types, force_embeddings=False):
    """Generate granular commands for missing components (one per PTM)."""
    commands = {
        'embeddings': [],
        'gt_samples': [],
        'concepts': [],
        'activations': [],
        'thresholds': [],
        'detection': [],
        'inversion': []
    }
    
    # Collect missing items with full details
    missing_items = {
        'embeddings': [],
        'concepts': [],
        'activations': [],
        'thresholds': [],
        'detection': []
    }
    
    for dataset in results:
        for model in results[dataset]:
            for ptm in results[dataset][model]:
                for sample_type in results[dataset][model][ptm]:
                    status = results[dataset][model][ptm][sample_type]
                    
                    # Check embeddings
                    all_activations_exist = check_all_activations_exist(status['activations'], results[dataset][model][ptm][sample_type]['concepts'].keys())
                    if not status['embeddings'][0] and (force_embeddings or not all_activations_exist):
                        missing_items['embeddings'].append((dataset, model, ptm))
                    
                    # Only check other stages if embeddings exist
                    elif status['embeddings'][0]:
                        # Check concepts per type
                        for ct in concept_types:
                            if ct in status['concepts'] and not status['concepts'][ct]:
                                missing_items['concepts'].append((dataset, model, ct, ptm))
                        
                        # Check activations per type
                        act_to_concept = {
                            'avg_cosine': 'avg',
                            'linsep_dist': 'linsep',
                            'kmeans_cosine': 'kmeans',
                            'linsepkmeans_dist': 'linsepkmeans'
                        }
                        for act_type, exists in status['activations'].items():
                            if not exists:
                                concept_type = act_to_concept.get(act_type)
                                if concept_type and concept_type in status['concepts'] and status['concepts'][concept_type]:
                                    missing_items['activations'].append((dataset, model, concept_type, ptm))
                        
                        # Check thresholds and detection per type
                        for ct in concept_types:
                            # Only check if activations exist
                            act_key = 'avg_cosine' if ct == 'avg' else 'linsep_dist' if ct == 'linsep' else 'kmeans_cosine' if ct == 'kmeans' else 'linsepkmeans_dist'
                            if act_key in status['activations'] and status['activations'][act_key]:
                                if ct in status['thresholds'] and not status['thresholds'][ct]:
                                    missing_items['thresholds'].append((dataset, model, ct, ptm))
                                if ct in status['detection'] and not status['detection'][ct]:
                                    missing_items['detection'].append((dataset, model, ct, ptm))
    
    # Generate embedding commands (one per PTM)
    for dataset, model, ptm in sorted(set(missing_items['embeddings'])):
        if dataset in IMAGE_DATASETS:
            cmd = f"python scripts/embed_image_datasets.py --model {model} --datasets {dataset} --percentthrumodels {ptm}"
        else:
            cmd = f"python scripts/embed_text_datasets.py --models {model} --datasets {dataset} --percentthrumodels {ptm}"
        commands['embeddings'].append(cmd)
    
    # Generate ground truth commands
    gt_datasets = set()
    for dataset, _, _, _ in missing_items['concepts']:
        if dataset in IMAGE_DATASETS:
            gt_datasets.add(dataset)
    if gt_datasets:
        cmd = f"python src/compute_image_gt_samples.py --datasets {' '.join(sorted(gt_datasets))}"
        commands['gt_samples'].append(cmd)
    
    # Generate concept commands (one per dataset/model/concept_type/PTM)
    for dataset, model, concept_type, ptm in sorted(set(missing_items['concepts'])):
        cmd = f"python scripts/compute_all_concepts.py --datasets {dataset} --models {model} --percentthrumodel {ptm} --concept-types {concept_type}"
        commands['concepts'].append(cmd)
    
    # Generate activation commands (one per dataset/model/concept_type/PTM)
    for dataset, model, concept_type, ptm in sorted(set(missing_items['activations'])):
        cmd = f"python scripts/compute_activations.py --datasets {dataset} --models {model} --percentthrumodels {ptm} --concept-types {concept_type}"
        commands['activations'].append(cmd)
    
    # Generate threshold commands
    for dataset, model, concept_type, ptm in sorted(set(missing_items['thresholds'])):
        cmd = f"python scripts/validation_thresholds.py --datasets {dataset} --models {model} --percentthrumodel {ptm} --concept-types {concept_type}"
        commands['thresholds'].append(cmd)
    
    # Generate detection commands
    for dataset, model, concept_type, ptm in sorted(set(missing_items['detection'])):
        cmd = f"python scripts/all_detection_stats.py --datasets {dataset} --models {model} --percentthrumodel {ptm} --concept-types {concept_type}"
        commands['detection'].append(cmd)
    
    # Generate inversion commands (image datasets only)
    for dataset, model, concept_type, ptm in sorted(set(missing_items['detection'])):
        if dataset in IMAGE_DATASETS:
            cmd = f"python scripts/all_inversion_stats.py --datasets {dataset} --models {model} --percentthrumodel {ptm} --concept-types {concept_type}"
            commands['inversion'].append(cmd)
    
    return commands

def generate_commands(results, concept_types, force_embeddings=False):
    """Generate commands for missing components."""
    commands = {
        'embeddings': [],
        'gt_samples': [],
        'concepts': [],
        'activations': [],
        'thresholds': [],
        'detection': [],
        'inversion': [],
        'baselines': [],
        'error_bars': [],
        'per_concept_ptm': []
    }
    
    # Track what needs to be done
    missing_embeddings = defaultdict(lambda: defaultdict(set))
    missing_concepts = defaultdict(lambda: defaultdict(set))
    missing_activations = defaultdict(lambda: defaultdict(set))
    missing_thresholds = defaultdict(lambda: defaultdict(set))
    missing_detection = defaultdict(lambda: defaultdict(set))
    
    for dataset in results:
        for model in results[dataset]:
            for ptm in results[dataset][model]:
                for sample_type in results[dataset][model][ptm]:
                    status = results[dataset][model][ptm][sample_type]
                    
                    # Check embeddings
                    all_activations_exist = check_all_activations_exist(status['activations'], concept_types)
                    if not status['embeddings'][0] and (force_embeddings or not all_activations_exist):
                        missing_embeddings[dataset][model].add(ptm)
                    
                    # Only check other stages if embeddings exist
                    elif status['embeddings'][0]:
                        # Check concepts
                        for ct in concept_types:
                            if ct in status['concepts'] and not status['concepts'][ct]:
                                missing_concepts[dataset][model].add(ptm)
                                break
                        
                        # Check activations
                        all_concepts_exist = all(status['concepts'].get(ct, False) for ct in concept_types if ct in status['concepts'])
                        if all_concepts_exist:
                            for act_type, exists in status['activations'].items():
                                if not exists:
                                    missing_activations[dataset][model].add(ptm)
                                    break
                        
                        # Check thresholds and detection
                        all_activations_exist = all(status['activations'].values())
                        if all_activations_exist:
                            for ct in concept_types:
                                if ct in status['thresholds'] and not status['thresholds'][ct]:
                                    missing_thresholds[dataset][model].add(ptm)
                                    break
                            for ct in concept_types:
                                if ct in status['detection'] and not status['detection'][ct]:
                                    missing_detection[dataset][model].add(ptm)
                                    break
    
    # Generate embedding commands
    for dataset, models in missing_embeddings.items():
        for model, ptms in models.items():
            ptm_list = ' '.join(map(str, sorted(ptms)))
            if dataset in IMAGE_DATASETS:
                cmd = f"python scripts/embed_image_datasets.py --model {model} --datasets {dataset} --percentthrumodels {ptm_list}"
            else:
                cmd = f"python scripts/embed_text_datasets.py --models {model} --datasets {dataset} --percentthrumodels {ptm_list}"
            commands['embeddings'].append(cmd)
    
    # Generate ground truth commands for image datasets
    gt_datasets = set()
    for dataset in missing_concepts:
        if dataset in IMAGE_DATASETS:
            gt_datasets.add(dataset)
    if gt_datasets:
        cmd = f"python src/compute_image_gt_samples.py --datasets {' '.join(sorted(gt_datasets))}"
        commands['gt_samples'].append(cmd)
    
    # Generate concept commands
    for dataset, models in missing_concepts.items():
        for model, ptms in models.items():
            ptm_list = ' '.join(map(str, sorted(ptms)))
            concept_list = ' '.join(concept_types)
            cmd = f"python scripts/compute_all_concepts.py --datasets {dataset} --models {model} --sample-types cls patch --percentthrumodel {ptm_list} --concept-types {concept_list}"
            commands['concepts'].append(cmd)
    
    # Generate activation commands
    for dataset, models in missing_activations.items():
        for model, ptms in models.items():
            ptm_list = ' '.join(map(str, sorted(ptms)))
            concept_list = ' '.join(concept_types)
            cmd = f"python scripts/compute_activations.py --datasets {dataset} --models {model} --sample-type cls patch --percentthrumodel {ptm_list} --concept-types {concept_list}"
            commands['activations'].append(cmd)
    
    # Generate threshold commands
    for dataset, models in missing_thresholds.items():
        for model, ptms in models.items():
            ptm_list = ' '.join(map(str, sorted(ptms)))
            concept_list = ' '.join(concept_types)
            cmd = f"python scripts/validation_thresholds.py --datasets {dataset} --models {model} --sample-types cls patch --percentthrumodel {ptm_list} --concept-types {concept_list}"
            commands['thresholds'].append(cmd)
    
    # Generate detection commands
    for dataset, models in missing_detection.items():
        for model, ptms in models.items():
            ptm_list = ' '.join(map(str, sorted(ptms)))
            concept_list = ' '.join(concept_types)
            cmd = f"python scripts/all_detection_stats.py --datasets {dataset} --models {model} --sample-types cls patch --percentthrumodel {ptm_list} --concept-types {concept_list}"
            commands['detection'].append(cmd)
    
    # Generate inversion commands (image datasets only)
    for dataset, models in missing_detection.items():
        if dataset in IMAGE_DATASETS:
            for model, ptms in models.items():
                ptm_list = ' '.join(map(str, sorted(ptms)))
                concept_list = ' '.join(concept_types)
                cmd = f"python scripts/all_inversion_stats.py --datasets {dataset} --models {model} --sample-types cls patch --percentthrumodel {ptm_list} --concept-types {concept_list}"
                commands['inversion'].append(cmd)
    
    # Track missing extended analysis components
    missing_baselines = defaultdict(set)
    missing_error_bars = defaultdict(set)
    missing_per_concept_ptm = defaultdict(set)
    
    # Check for missing extended analysis (for all PTMs with completed detection)
    for dataset in results:
        for model in results[dataset]:
            for ptm in results[dataset][model]:
                for sample_type in results[dataset][model][ptm]:
                    status = results[dataset][model][ptm][sample_type]
                    
                    # Check if all detections are complete
                    all_detections_complete = all(status['detection'].get(ct, False) for ct in concept_types if ct in status['detection'])
                    
                    if all_detections_complete:
                        # Check baselines
                        if 'baselines' in status:
                            for ct in concept_types:
                                if ct in status['baselines'] and not status['baselines'][ct]:
                                    missing_baselines[dataset].add(model)
                                    break
                        
                        # Check error bars
                        if 'error_bars' in status:
                            for ct in concept_types:
                                if ct in status['error_bars'] and not status['error_bars'][ct]:
                                    missing_error_bars[dataset].add(model)
                                    break
                        
                        # Check per-concept PTM
                        if 'per_concept_ptm' in status:
                            for ct in concept_types:
                                if ct in status['per_concept_ptm'] and not status['per_concept_ptm'][ct]:
                                    missing_per_concept_ptm[dataset].add(model)
                                    break
    
    # Generate baseline commands
    for dataset, models in missing_baselines.items():
        for model in models:
            concept_list = ' '.join(concept_types)
            cmd = f"python scripts/baseline_detections.py --datasets {dataset} --models {model} --concept-types {concept_list}"
            commands['baselines'].append(cmd)
    
    # Generate error bar commands
    for dataset, models in missing_error_bars.items():
        for model in models:
            concept_list = ' '.join(concept_types)
            cmd = f"python scripts/detection_errors.py --datasets {dataset} --models {model} --concept-types {concept_list}"
            commands['error_bars'].append(cmd)
    
    # Generate per-concept PTM commands
    for dataset, models in missing_per_concept_ptm.items():
        for model in models:
            concept_list = ' '.join(concept_types)
            cmd = f"python scripts/per_concept_ptm_optimization.py --datasets {dataset} --models {model} --concept-types {concept_list}"
            commands['per_concept_ptm'].append(cmd)
    
    return commands

def main():
    parser = argparse.ArgumentParser(description='Check pipeline status comprehensively')
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS, 
                        help='Datasets to check (default: all)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to check (default: appropriate for datasets)')
    parser.add_argument('--sample-types', nargs='+', default=SAMPLE_TYPES,
                        help='Sample types to check (default: cls patch)')
    parser.add_argument('--concept-types', nargs='+', default=['avg', 'linsep'],
                        help='Concept types to check (default: avg linsep)')
    parser.add_argument('--show-commands', action='store_true',
                        help='Show commands to run for missing components')
    parser.add_argument('--summary-only', action='store_true',
                        help='Show only summary statistics')
    parser.add_argument('--granular-commands', action='store_true',
                        help='Show individual commands per PTM for parallelization')
    parser.add_argument('--check-extended', action='store_true', default=True,
                        help='Check extended analysis components (baselines, error bars, per-concept PTM)')
    parser.add_argument('--force-embeddings', action='store_true',
                        help='Include embeddings in commands even when all activations exist')
    parser.add_argument('--assess-inversion', action='store_true', default=False,
                        help='Check inversion stats (default: False)')
    
    args = parser.parse_args()
    
    # Determine which models to check
    if args.models is None:
        models_to_check = set()
        for dataset in args.datasets:
            if dataset in IMAGE_DATASETS:
                models_to_check.update(IMAGE_MODELS)
            if dataset in TEXT_DATASETS:
                models_to_check.update(TEXT_MODELS)
        args.models = list(models_to_check)
    
    # Check pipeline status
    results = {}
    
    for dataset in args.datasets:
        results[dataset] = {}
        
        for model in args.models:
            # Skip invalid model-dataset combinations
            if dataset in IMAGE_DATASETS and model not in IMAGE_MODELS:
                continue
            if dataset in TEXT_DATASETS and model not in TEXT_MODELS:
                continue
            
            results[dataset][model] = {}
            percentthrumodels = get_model_percentthrumodels(model, dataset)
            
            for ptm in percentthrumodels:
                results[dataset][model][ptm] = {}
                
                for sample_type in args.sample_types:
                    # Check each stage
                    emb_exists, emb_info = check_embeddings(dataset, model, sample_type, ptm)
                    concepts = check_concepts(dataset, model, sample_type, ptm, args.concept_types)
                    activations = check_activations(dataset, model, sample_type, ptm, args.concept_types)
                    thresholds = check_thresholds(dataset, model, sample_type, ptm, args.concept_types)
                    detection = check_detection(dataset, model, sample_type, ptm, args.concept_types)
                    
                    # Only check inversion if requested
                    if args.assess_inversion:
                        inversion = check_inversion(dataset, model, sample_type, ptm, args.concept_types)
                    else:
                        inversion = {}
                    
                    # Extended analysis components (for all PTMs when check_extended is enabled)
                    if args.check_extended:
                        baselines = check_baseline_detections(dataset, model, sample_type, ptm, args.concept_types)
                        error_bars = check_error_bars(dataset, model, sample_type, ptm, args.concept_types)
                        per_concept_ptm = check_per_concept_ptm(dataset, model, sample_type, args.concept_types)
                    else:
                        baselines = {}
                        error_bars = {}
                        per_concept_ptm = {}
                    
                    results[dataset][model][ptm][sample_type] = {
                        'embeddings': (emb_exists, emb_info),
                        'concepts': concepts,
                        'activations': activations,
                        'thresholds': thresholds,
                        'detection': detection,
                        'inversion': inversion,
                        'baselines': baselines,
                        'error_bars': error_bars,
                        'per_concept_ptm': per_concept_ptm
                    }
    
    # Print results
    if not args.summary_only:
        print_status_table(results, args.datasets, args.models, args.sample_types, args.concept_types)
    
    # Generate and print commands
    if args.show_commands:
        if args.granular_commands:
            commands = generate_granular_commands(results, args.concept_types, args.force_embeddings)
        else:
            commands = generate_commands(results, args.concept_types, args.force_embeddings)
        
        print("\n\nCOMMANDS TO RUN:")
        print("=" * 120)
        
        for stage, cmd_list in commands.items():
            if cmd_list:
                print(f"\n{stage.upper()}:")
                for cmd in cmd_list:
                    print(f"  {cmd}")
    
    # Print summary statistics
    print("\n\nSUMMARY STATISTICS:")
    print("=" * 120)
    
    total_configs = 0
    missing_by_stage = defaultdict(int)
    
    for dataset in results:
        for model in results[dataset]:
            for ptm in results[dataset][model]:
                for sample_type in results[dataset][model][ptm]:
                    total_configs += 1
                    status = results[dataset][model][ptm][sample_type]
                    
                    if not status['embeddings'][0]:
                        missing_by_stage['embeddings'] += 1
                    elif any(not v for v in status['concepts'].values()):
                        missing_by_stage['concepts'] += 1
                    elif any(not v for v in status['activations'].values()):
                        missing_by_stage['activations'] += 1
                    elif any(not v for v in status['thresholds'].values()):
                        missing_by_stage['thresholds'] += 1
                    elif any(not v for v in status['detection'].values()):
                        missing_by_stage['detection'] += 1
    
    print(f"Total configurations checked: {total_configs}")
    print(f"Missing components by stage:")
    for stage, count in missing_by_stage.items():
        print(f"  {stage}: {count} ({count/total_configs*100:.1f}%)")

if __name__ == "__main__":
    main()