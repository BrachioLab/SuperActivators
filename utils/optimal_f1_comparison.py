import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Import the required utility functions
import sys
sys.path.append('/shared_data0/cgoldberg/Concept_Inversion/Experiments/utils')
from general_utils import get_split_df
from false_positive_extractor import get_false_positive_indices


def calculate_f1_for_percentile(dataset_name, model_name, sample_type, concept, 
                                percentile, split='test', method='linsep'):
    """Calculate F1 score for a given percentile and method."""
    results = get_false_positive_indices(
        dataset_name, model_name, sample_type, concept,
        percentile, split=split, method=method
    )
    
    if not results:
        return None, None
    
    tp = results['total_tp']
    fp = results['total_fp']
    fn = results['total_fn']
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, results


def find_optimal_percentile(dataset_name, model_name, sample_type, concept, 
                           percentiles, split='test', method='linsep'):
    """Find the percentile that yields the highest F1 score."""
    best_f1 = 0
    best_percentile = None
    best_results = None
    f1_scores = {}
    
    print(f"\nFinding optimal percentile for {method} {sample_type}...")
    
    for percentile in tqdm(percentiles):
        f1, results = calculate_f1_for_percentile(
            dataset_name, model_name, sample_type, concept,
            percentile, split=split, method=method
        )
        
        if f1 is not None:
            f1_scores[percentile] = f1
            if f1 > best_f1:
                best_f1 = f1
                best_percentile = percentile
                best_results = results
    
    # Print all F1 scores
    print(f"\nF1 scores by percentile for {method} {sample_type}:")
    for p, f1 in sorted(f1_scores.items()):
        print(f"  Percentile {p}: F1 = {f1:.4f}")
    
    print(f"\nOptimal percentile: {best_percentile} with F1 = {best_f1:.4f}")
    
    return best_percentile, best_f1, best_results


def compare_methods_at_optimal_f1(dataset_name='iSarcasm', model_name='Llama', 
                                  concept='sarcastic', method='linsep'):
    """Compare patch and CLS methods at their respective optimal F1 percentiles."""
    
    print(f"\n{'='*80}")
    print(f"FINDING OPTIMAL F1 PERCENTILES FOR '{concept.upper()}'")
    print(f"Dataset: {dataset_name}, Model: {model_name}, Method: {method}")
    print(f"{'='*80}")
    
    # Define percentiles to test
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Find optimal percentile for patch method
    patch_optimal_percentile, patch_best_f1, patch_results = find_optimal_percentile(
        dataset_name, model_name, 'patch', concept, percentiles, 'test', method
    )
    
    # Find optimal percentile for CLS method
    cls_optimal_percentile, cls_best_f1, cls_results = find_optimal_percentile(
        dataset_name, model_name, 'cls', concept, percentiles, 'test', method
    )
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL PERCENTILES SUMMARY")
    print(f"{'='*80}")
    print(f"Patch method: Percentile {patch_optimal_percentile} → F1 = {patch_best_f1:.4f}")
    print(f"CLS method: Percentile {cls_optimal_percentile} → F1 = {cls_best_f1:.4f}")
    
    # Now compare at these optimal percentiles
    print(f"\n{'='*80}")
    print(f"DETAILED COMPARISON AT OPTIMAL PERCENTILES")
    print(f"{'='*80}")
    
    # Patch results at optimal percentile
    print(f"\nPATCH METHOD (Percentile {patch_optimal_percentile}):")
    print(f"  True Positives: {patch_results['total_tp']}")
    print(f"  False Positives: {patch_results['total_fp']}")
    print(f"  False Negatives: {patch_results['total_fn']}")
    print(f"  True Negatives: {patch_results['total_tn']}")
    patch_precision = patch_results['total_tp'] / (patch_results['total_tp'] + patch_results['total_fp']) if (patch_results['total_tp'] + patch_results['total_fp']) > 0 else 0
    patch_recall = patch_results['total_tp'] / (patch_results['total_tp'] + patch_results['total_fn']) if (patch_results['total_tp'] + patch_results['total_fn']) > 0 else 0
    patch_fpr = patch_results['total_fp'] / (patch_results['total_fp'] + patch_results['total_tn']) if (patch_results['total_fp'] + patch_results['total_tn']) > 0 else 0
    print(f"  Precision: {patch_precision:.4f}")
    print(f"  Recall: {patch_recall:.4f}")
    print(f"  F1: {patch_best_f1:.4f}")
    print(f"  FPR: {patch_fpr:.4f}")
    
    # CLS results at optimal percentile
    print(f"\nCLS METHOD (Percentile {cls_optimal_percentile}):")
    print(f"  True Positives: {cls_results['total_tp']}")
    print(f"  False Positives: {cls_results['total_fp']}")
    print(f"  False Negatives: {cls_results['total_fn']}")
    print(f"  True Negatives: {cls_results['total_tn']}")
    cls_precision = cls_results['total_tp'] / (cls_results['total_tp'] + cls_results['total_fp']) if (cls_results['total_tp'] + cls_results['total_fp']) > 0 else 0
    cls_recall = cls_results['total_tp'] / (cls_results['total_tp'] + cls_results['total_fn']) if (cls_results['total_tp'] + cls_results['total_fn']) > 0 else 0
    cls_fpr = cls_results['total_fp'] / (cls_results['total_fp'] + cls_results['total_tn']) if (cls_results['total_fp'] + cls_results['total_tn']) > 0 else 0
    print(f"  Precision: {cls_precision:.4f}")
    print(f"  Recall: {cls_recall:.4f}")
    print(f"  F1: {cls_best_f1:.4f}")
    print(f"  FPR: {cls_fpr:.4f}")
    
    # Find examples where methods differ
    print(f"\n{'='*80}")
    print(f"WHERE METHODS DIFFER AT OPTIMAL PERCENTILES")
    print(f"{'='*80}")
    
    patch_fp_set = set(patch_results['false_positive_images'])
    cls_fp_set = set(cls_results['false_positive_images'])
    patch_detected = set(patch_results['detected_images'])
    cls_detected = set(cls_results['detected_images'])
    
    # False positives unique to each method
    patch_only_fp = patch_fp_set - cls_fp_set
    cls_only_fp = cls_fp_set - patch_fp_set
    
    print(f"\nFalse positives unique to PATCH: {len(patch_only_fp)}")
    if patch_only_fp:
        print(f"  Indices: {sorted(list(patch_only_fp))[:10]}")  # Show first 10
    
    print(f"\nFalse positives unique to CLS: {len(cls_only_fp)}")
    if cls_only_fp:
        print(f"  Indices: {sorted(list(cls_only_fp))[:10]}")  # Show first 10
    
    # True positives detected by one but not the other
    patch_tp = patch_detected & set(patch_results['gt_positive_images'])
    cls_tp = cls_detected & set(cls_results['gt_positive_images'])
    
    patch_only_tp = patch_tp - cls_tp
    cls_only_tp = cls_tp - patch_tp
    
    print(f"\nTrue positives detected by PATCH but not CLS: {len(patch_only_tp)}")
    print(f"True positives detected by CLS but not PATCH: {len(cls_only_tp)}")
    
    return {
        'patch_optimal_percentile': patch_optimal_percentile,
        'patch_best_f1': patch_best_f1,
        'patch_results': patch_results,
        'cls_optimal_percentile': cls_optimal_percentile,
        'cls_best_f1': cls_best_f1,
        'cls_results': cls_results
    }


if __name__ == "__main__":
    # Run the comparison
    results = compare_methods_at_optimal_f1(
        dataset_name='iSarcasm',
        model_name='Llama',
        concept='sarcastic',
        method='linsep'
    )