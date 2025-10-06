"""
Utilities for selecting best percentile thresholds based on calibration set performance
"""

import pandas as pd
import numpy as np
import os
import torch
from typing import Dict, List, Tuple, Optional
import ast


def find_best_percentile_from_calibration(dataset_name: str,
                                         con_label: str,
                                         metric: str = 'f1',
                                         percentiles: Optional[List[float]] = None) -> Tuple[float, pd.DataFrame]:
    """
    Find the best percentile threshold based on calibration set performance.
    
    Args:
        dataset_name: Name of dataset
        con_label: Concept label (will look for files with con_label + "_cal")
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        percentiles: List of percentiles to check (if None, will check standard set)
        
    Returns:
        Tuple of (best_percentile, metrics_df for that percentile)
    """
    if percentiles is None:
        percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    best_percentile = None
    best_avg_metric = -1
    best_df = None
    
    for percentile in percentiles:
        # Construct calibration result path
        cal_path = f"Quant_Results/{dataset_name}/detectfirst_{percentile*100}_per_{percentile*100}_{con_label}_cal.csv"
        
        if not os.path.exists(cal_path):
            print(f"Calibration file not found: {cal_path}")
            continue
            
        # Load calibration results
        cal_df = pd.read_csv(cal_path)
        
        # Calculate average metric across all concepts
        avg_metric = cal_df[metric].mean()
        
        if avg_metric > best_avg_metric:
            best_avg_metric = avg_metric
            best_percentile = percentile
            best_df = cal_df
    
    if best_percentile is None:
        raise ValueError(f"No calibration files found for {con_label}")
        
    print(f"Best percentile for {con_label}: {best_percentile} (avg {metric}={best_avg_metric:.4f})")
    
    return best_percentile, best_df


def find_best_percentile_from_calibration_allpairs(dataset_name: str,
                                                  con_label: str,
                                                  concepts_to_include: Optional[List[str]] = None,
                                                  metric: str = 'f1',
                                                  percentiles: Optional[List[float]] = None) -> Tuple[float, Dict[str, Tuple[str, float]]]:
    """
    Find the best percentile threshold for unsupervised methods based on calibration set performance.
    This version handles all-pairs detection metrics files.
    
    Args:
        dataset_name: Name of dataset
        con_label: Concept label (will look for files with con_label + "_cal")
        concepts_to_include: List of GT concepts to consider (if None, includes all)
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        percentiles: List of percentiles to check (if None, will check standard set)
        
    Returns:
        Tuple of (best_percentile, dict of concept -> (best_cluster, metric_value))
    """
    if percentiles is None:
        percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    best_percentile = None
    best_avg_metric = -1
    best_matches = None
    
    for percentile in percentiles:
        # Construct calibration result path for all-pairs
        cal_path = f"Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}_cal.csv"
        
        if not os.path.exists(cal_path):
            print(f"Calibration file not found: {cal_path}")
            continue
            
        # Load calibration results
        cal_df = pd.read_csv(cal_path)
        
        # Parse concept column to extract GT concept and cluster ID
        parsed_data = []
        for _, row in cal_df.iterrows():
            # Parse string representation of tuple like "('color::black-c', '0')"
            concept_tuple = ast.literal_eval(row['concept'])
            gt_concept = concept_tuple[0]
            cluster_id = concept_tuple[1]
            
            if concepts_to_include and gt_concept not in concepts_to_include:
                continue
                
            parsed_data.append({
                'gt_concept': gt_concept,
                'cluster_id': cluster_id,
                metric: row[metric]
            })
        
        if not parsed_data:
            continue
            
        parsed_df = pd.DataFrame(parsed_data)
        
        # Find best cluster for each GT concept
        concept_best_metrics = []
        for concept in parsed_df['gt_concept'].unique():
            concept_df = parsed_df[parsed_df['gt_concept'] == concept]
            best_row = concept_df.loc[concept_df[metric].idxmax()]
            concept_best_metrics.append(best_row[metric])
        
        # Calculate average of best metrics across concepts
        avg_metric = np.mean(concept_best_metrics)
        
        if avg_metric > best_avg_metric:
            best_avg_metric = avg_metric
            best_percentile = percentile
            
            # Store best matches for this percentile
            best_matches = {}
            for concept in parsed_df['gt_concept'].unique():
                concept_df = parsed_df[parsed_df['gt_concept'] == concept]
                best_row = concept_df.loc[concept_df[metric].idxmax()]
                best_matches[concept] = (best_row['cluster_id'], best_row[metric])
    
    if best_percentile is None:
        raise ValueError(f"No calibration files found for {con_label}")
        
    print(f"Best percentile for {con_label}: {best_percentile} (avg best {metric}={best_avg_metric:.4f})")
    
    return best_percentile, best_matches


def get_test_results_at_percentile(dataset_name: str,
                                  con_label: str,
                                  percentile: float,
                                  is_unsupervised: bool = False) -> pd.DataFrame:
    """
    Load test results for a specific percentile.
    
    Args:
        dataset_name: Name of dataset
        con_label: Concept label (without _cal suffix)
        percentile: Percentile to load
        is_unsupervised: Whether this is unsupervised (all-pairs) or supervised
        
    Returns:
        DataFrame with test results
    """
    if is_unsupervised:
        test_path = f"Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{percentile}_{con_label}.csv"
    else:
        test_path = f"Quant_Results/{dataset_name}/detectfirst_{percentile*100}_per_{percentile*100}_{con_label}.csv"
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test results not found: {test_path}")
        
    return pd.read_csv(test_path)


def summarize_calibration_selection(dataset_name: str,
                                   model_name: str,
                                   sample_type: str,
                                   n_clusters: Optional[int] = None,
                                   percent_thru_model: int = 100,
                                   metric: str = 'f1'):
    """
    Summarize the best percentiles selected for all concept discovery methods.
    
    Args:
        dataset_name: Name of dataset
        model_name: Model name (e.g., 'CLIP', 'Llama')
        sample_type: 'patch' or 'cls'
        n_clusters: Number of clusters for k-means methods
        percent_thru_model: Percentage through model
        metric: Metric used for selection
    """
    results = []
    
    # Supervised methods
    supervised_methods = [
        ('avg', f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}'),
        ('linsep', f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}')
    ]
    
    for method_name, con_label in supervised_methods:
        try:
            best_p, cal_df = find_best_percentile_from_calibration(dataset_name, con_label, metric)
            test_df = get_test_results_at_percentile(dataset_name, con_label, best_p, is_unsupervised=False)
            
            results.append({
                'method': method_name,
                'best_percentile': best_p,
                f'cal_avg_{metric}': cal_df[metric].mean(),
                f'test_avg_{metric}': test_df[metric].mean()
            })
        except (ValueError, FileNotFoundError) as e:
            print(f"Skipping {method_name}: {e}")
    
    # Unsupervised methods (if n_clusters provided)
    if n_clusters:
        unsupervised_methods = [
            ('kmeans', f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"),
            ('kmeans_linsep', f"{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}")
        ]
        
        for method_name, con_label in unsupervised_methods:
            try:
                best_p, best_matches = find_best_percentile_from_calibration_allpairs(dataset_name, con_label, metric=metric)
                # For test results, we'd need to parse and find best clusters again
                # Just report calibration performance for now
                
                results.append({
                    'method': method_name,
                    'best_percentile': best_p,
                    f'cal_avg_best_{metric}': np.mean([v[1] for v in best_matches.values()])
                })
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping {method_name}: {e}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    print(f"\n=== Calibration-based Percentile Selection Summary ===")
    print(f"Dataset: {dataset_name}, Model: {model_name}, Sample Type: {sample_type}")
    print(f"Optimizing for: {metric}")
    print("\n", summary_df.to_string(index=False))
    
    return summary_df