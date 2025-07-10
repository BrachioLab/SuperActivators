import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import json

from utils.compute_concepts_utils import compute_cosine_sims, compute_signed_distances
from utils.quant_concept_evals_utils import compute_concept_thresholds_over_percentiles, compute_detection_metrics_over_percentiles, compute_stats_from_counts
from utils.patch_alignment_utils import filter_patches_by_image_presence

def load_superdetectors(dataset_name, con_label, percentile):
    """Load superdetector patches for given dataset, concept label, and percentile."""
    # Extract model name and method from con_label
    if '_avg' in con_label:
        model_name = con_label.split('_avg')[0]
        method = 'avg'
        embeddings_type = 'patch_embeddings'
    elif '_linsep' in con_label:
        model_name = con_label.split('_linsep')[0]
        method = 'linsep'
        embeddings_type = 'patch_embeddings_BD_True_BN_False'
    else:
        # Fallback for simple model names
        model_name = con_label
        method = 'avg'
        embeddings_type = 'patch_embeddings'
    
    superdetector_file = f'Superpatches/{dataset_name}/per_{percentile}_{model_name}_{method}_{embeddings_type}_percentthrumodel_100.pt'
    if not os.path.exists(superdetector_file):
        raise FileNotFoundError(f"Required superdetector file not found: {superdetector_file}")
    
    superdetectors = torch.load(superdetector_file, weights_only=False)
    return superdetectors

def load_concept_vectors(dataset_name, con_label):
    """Load concept vectors for given concept label."""
    # Extract model name and method from con_label 
    if '_avg' in con_label:
        model_name = con_label.split('_avg')[0]
        method = 'avg'
    elif '_linsep' in con_label:
        model_name = con_label.split('_linsep')[0]
        method = 'linsep'
    else:
        # Fallback for simple model names
        model_name = con_label
        method = 'avg'  # Default to avg
    
    # Load only the concepts for the specific method
    concepts = {}
    
    if method == 'avg':
        avg_concepts_file = f'avg_concepts_{model_name}_patch_embeddings_percentthrumodel_100.pt'
        avg_concepts_path = f'Concepts/{dataset_name}/{avg_concepts_file}'
        
        if os.path.exists(avg_concepts_path):
            avg_concepts = torch.load(avg_concepts_path)
            concepts.update({k + '_avg': v for k, v in avg_concepts.items()})
        else:
            raise FileNotFoundError(f"Average concept file not found: {avg_concepts_path}")
    
    elif method == 'linsep':
        linsep_concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_patch_embeddings_percentthrumodel_100.pt'
        linsep_concepts_path = f'Concepts/{dataset_name}/{linsep_concepts_file}'
        
        if os.path.exists(linsep_concepts_path):
            linsep_concepts = torch.load(linsep_concepts_path)
            concepts.update({k + '_linsep': v for k, v in linsep_concepts.items()})
        else:
            raise FileNotFoundError(f"Linear separator concept file not found: {linsep_concepts_path}")
    
    return concepts

def get_sample_ranges(dataset_name, model_input_size, total_embeddings):
    """Get the patch ranges for each sample (image/paragraph)."""
    if model_input_size[0] == 'text':
        # For text, load token counts to determine sample boundaries
        token_counts = torch.load(f'GT_Samples/{dataset_name}/token_counts.pt', weights_only=False)
        sample_ranges = []
        start_idx = 0
        for sent_idx, word_token_counts in enumerate(token_counts):
            total_tokens = sum(word_token_counts)
            sample_ranges.append((start_idx, start_idx + total_tokens))
            start_idx += total_tokens
    else:
        # For images, calculate patches per image based on input size
        # model_input_size is (height, width), patches are 14x14
        img_height, img_width = model_input_size
        patch_size = 14
        
        patches_per_row = img_height // patch_size
        patches_per_col = img_width // patch_size
        patches_per_image = patches_per_row * patches_per_col
        
        num_images = total_embeddings // patches_per_image
        sample_ranges = []
        for img_idx in range(num_images):
            start_idx = img_idx * patches_per_image
            sample_ranges.append((start_idx, start_idx + patches_per_image))
    
    return sample_ranges


def filter_relevant_patches(indices, dataset_name, model_input_size):
    """
    Filter out padding patches/tokens using the relevant patches mask.
    
    Args:
        indices: List of patch/token indices to filter
        dataset_name: Name of dataset
        model_input_size: Model input size (for images) or 'text' identifier
    
    Returns:
        Filtered list of indices containing only real content (no padding)
    """
    if model_input_size[0] == 'text':
        # For text, load the relevant tokens mask
        mask_file = f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt'
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Required relevant tokens mask file not found: {mask_file}")
        
        relevant_tokens = torch.load(mask_file, weights_only=False)
        # Filter indices based on relevant tokens mask
        filtered_indices = [idx for idx in indices if idx < len(relevant_tokens) and relevant_tokens[idx] == 1]
        return filtered_indices
    else:
        # For images, use the existing filter function
        return filter_patches_by_image_presence(indices, dataset_name, model_input_size).tolist()

def compute_hybrid_concept_vectors(concept_vectors, superdetectors, embeddings, sample_ranges, alpha, dataset_name, model_input_size, train_patch_indices=None):
    """
    Compute hybrid concept vectors:
    concept_vector * alpha + avg_superdetectors_per_sample * (1-alpha)
    
    Each sample (image/paragraph) gets its own hybrid vectors based on which
    superdetectors are present in that specific sample.
    
    Args:
        train_patch_indices: Not used anymore - kept for backward compatibility
    
    Returns:
        Dictionary mapping concept names to lists of hybrid concept vectors (one per sample)
    """
    hybrid_concepts = {}
    
    for concept, concept_vector in concept_vectors.items():
        if concept not in superdetectors:
            # If no superdetectors for this concept, use pure concept vector for all samples
            hybrid_concepts[concept] = [concept_vector] * len(sample_ranges)
            continue
            
        concept_superdetectors = superdetectors[concept]
        sample_hybrid_vectors = []
        
        for sample_idx, (start_idx, end_idx) in enumerate(sample_ranges):
            # Find superdetectors in this specific sample
            sample_superdetector_indices = [
                idx for idx in concept_superdetectors 
                if start_idx <= idx < end_idx
            ]
            
            # Use superdetectors in this sample
            if len(sample_superdetector_indices) > 0:
                # Filter out padding patches/tokens
                filtered_superdetector_indices = filter_relevant_patches(
                    sample_superdetector_indices, dataset_name, model_input_size
                )
                
                if len(filtered_superdetector_indices) > 0:
                    # Average superdetector embeddings from this sample (excluding padding)
                    sample_superdetector_embeds = embeddings[filtered_superdetector_indices]
                    avg_superdetector = torch.mean(sample_superdetector_embeds, dim=0)
                    
                    # Compute hybrid vector for this sample
                    hybrid_vector = alpha * concept_vector + (1 - alpha) * avg_superdetector
                else:
                    # No valid superdetectors in this sample (all were padding)
                    hybrid_vector = concept_vector
            else:
                # No superdetectors in this sample, use pure concept vector
                hybrid_vector = concept_vector
            
            sample_hybrid_vectors.append(hybrid_vector)
        
        hybrid_concepts[concept] = sample_hybrid_vectors
    
    return hybrid_concepts

def compute_hybrid_activations(embeddings, hybrid_concepts, sample_ranges, dataset_name, device, method, scratch_dir='', batch_size=200):
    """
    Compute activations between embeddings and per-sample hybrid concept vectors using GPU.
    Memory optimized: processes one concept at a time and clears intermediate results.
    
    Args:
        embeddings: Patch embeddings tensor (should already be on device)
        hybrid_concepts: Dictionary mapping concept names to lists of hybrid vectors (one per sample)
        sample_ranges: List of (start_idx, end_idx) tuples for each sample
        dataset_name: Name of dataset
        device: Device to use for computation
        method: 'avg' or 'linsep'
        scratch_dir: Scratch directory path
        batch_size: Batch size for computation
    
    Returns:
        Dictionary mapping concept names to activation tensors on device
    """
    all_activations = {}
    
    for concept_idx, (concept, concept_hybrid_vectors) in enumerate(hybrid_concepts.items()):
        concept_activations = []
        
        for sample_idx, (start_idx, end_idx) in enumerate(sample_ranges):
            sample_embeddings = embeddings[start_idx:end_idx]  # Already on device
            sample_hybrid_vector = concept_hybrid_vectors[sample_idx].to(device)  # Ensure on device
            
            if method == 'linsep':
                # For linear separator, compute signed distances
                sample_activations = torch.matmul(sample_embeddings, sample_hybrid_vector)
            else:
                # For average concepts, compute cosine similarities
                sample_activations = F.cosine_similarity(
                    sample_embeddings, 
                    sample_hybrid_vector.unsqueeze(0).expand_as(sample_embeddings), 
                    dim=1
                )
            
            concept_activations.append(sample_activations)
        
        # Concatenate all sample activations for this concept
        all_activations[concept] = torch.cat(concept_activations)
        
        # Clear concept activations list to free memory
        del concept_activations
    
    return all_activations

def compute_hybrid_thresholds_for_concepts(activations, gt_samples_per_concept_cal, percentile, dataset_name, con_label, alpha):
    """
    Compute thresholds for hybrid concepts using calibration set at given percentile.
    Works at patch level for inversion F1.
    
    Args:
        activations: Dictionary mapping concept names to activation tensors (patch-level)
        gt_samples_per_concept_cal: GT samples for calibration set (patch indices)
        percentile: Percentile to use for threshold finding
        dataset_name: Name of dataset
        con_label: Concept label for saving
        alpha: Alpha value for hybrid concepts
    
    Returns:
        Dictionary mapping concept names to thresholds
    """
    thresholds = {}
    
    for concept, concept_activations in activations.items():
        if concept not in gt_samples_per_concept_cal:
            continue
            
        cal_gt_patch_indices = gt_samples_per_concept_cal[concept]
        
        if len(cal_gt_patch_indices) == 0:
            raise ValueError(f"No calibration GT patches found for concept '{concept}'. Cannot compute threshold.")
        
        # Get activations for calibration GT patches
        cal_gt_activations = concept_activations[cal_gt_patch_indices]
        
        # Find threshold that captures the desired percentile of GT activations
        # NOTE: Use (1 - percentile) to match baseline threshold computation
        threshold = torch.quantile(cal_gt_activations, 1 - percentile).item()
        
        thresholds[concept] = threshold
    
    # Save thresholds in Hybrid_Results
    os.makedirs(f'Hybrid_Results/{dataset_name}', exist_ok=True)
    threshold_file = f'Hybrid_Results/{dataset_name}/concept_thresholds_hybrid_{con_label}_alpha_{alpha:.2f}_percentile_{percentile}.pt'
    torch.save({percentile: thresholds}, threshold_file)
    
    return thresholds

def evaluate_hybrid_performance_with_thresholds(activations, thresholds, gt_patches_per_concept, gt_samples_per_concept_test, percentile, dataset_name, con_label, alpha):
    """
    Evaluate F1 performance using same patch filtering and test set as baseline.
    
    Args:
        activations: Dictionary mapping concept names to activation tensors (patch-level)
        thresholds: Dictionary mapping concept names to threshold values
        gt_samples_per_concept_test: GT samples for test set (patch indices)
        percentile: Percentile used (invert percentile)
        dataset_name: Name of dataset
        con_label: Concept label for saving
        alpha: Alpha value for hybrid concepts
    
    Returns:
        Weighted F1 score across all concepts
    """
    from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence
    from utils.general_utils import create_binary_labels
    
    # Get test patches with same filtering as baseline
    model_input_size = (224, 224) if 'CLIP' in con_label else (560, 560)
    patch_size = 14
    split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    
    # Get ground truth labels for all patches (same as baseline) 
    all_concept_labels = create_binary_labels(len(split_df), gt_patches_per_concept)
    
    # Collect counts for all concepts
    tp_counts = {}
    fp_counts = {}
    fn_counts = {}
    tn_counts = {}
    thresholds_dict = {}
    additional_info = {}
    
    for concept, concept_activations in activations.items():
        if concept not in thresholds or concept not in gt_patches_per_concept:
            continue
            
        threshold = thresholds[concept]
        
        # Apply threshold to get predictions (same threshold for detect/invert)
        activated_patches = concept_activations > threshold
        
        # Get relevant indices as boolean mask (same filtering as baseline)
        relevant_mask = torch.zeros(len(concept_activations), dtype=torch.bool, device=concept_activations.device)
        relevant_mask[relevant_indices] = True
        
        # Get GT mask (same GT processing as baseline)
        gt_values = all_concept_labels[concept] == 1
        if isinstance(gt_values, torch.Tensor):
            gt_mask = gt_values.clone().detach().to(concept_activations.device, dtype=torch.bool)
        else:
            gt_mask = torch.tensor(gt_values, dtype=torch.bool, device=concept_activations.device)
        
        # Compute confusion matrix counts only on relevant test patches (same subset as baseline)
        relevant_activated = activated_patches & relevant_mask
        relevant_gt = gt_mask & relevant_mask
        
        tp = torch.sum(relevant_activated & relevant_gt).item()
        fp = torch.sum(relevant_activated & (~relevant_gt)).item()
        fn = torch.sum((~relevant_activated) & relevant_gt).item()
        tn = torch.sum((~relevant_activated) & (~relevant_gt)).item()
        
        # Store counts in dictionaries
        tp_counts[concept] = tp
        fp_counts[concept] = fp
        fn_counts[concept] = fn
        tn_counts[concept] = tn
        thresholds_dict[concept] = threshold
        additional_info[concept] = {
            'num_gt': torch.sum(relevant_gt).item(),
            'num_pred': torch.sum(relevant_activated).item(),
            'total_patches': torch.sum(relevant_mask).item()
        }
    
    # Use existing function to compute stats from counts
    stats_df = compute_stats_from_counts(tp_counts, fp_counts, tn_counts, fn_counts)
    
    # Convert DataFrame back to dictionary format and add additional info
    all_metrics = {}
    for _, row in stats_df.iterrows():
        concept = row['concept']
        stats = row.to_dict()
        # Add threshold and additional information
        stats['threshold'] = thresholds_dict[concept]
        stats.update(additional_info[concept])
        all_metrics[concept] = stats
    
    # Compute weighted average F1 (weighted by number of test GT samples)
    f1_scores = {concept: metrics['f1'] for concept, metrics in all_metrics.items()}
    total_samples = sum(len(gt_samples_per_concept_test[concept]) 
                       for concept in f1_scores.keys() 
                       if concept in gt_samples_per_concept_test)
    
    if total_samples == 0:
        weighted_f1 = 0.0
    else:
        weighted_f1 = sum(
            f1_scores[concept] * len(gt_samples_per_concept_test[concept])
            for concept in f1_scores.keys()
            if concept in gt_samples_per_concept_test
        ) / total_samples
    
    # Add weighted F1 to metrics
    all_metrics['weighted_f1'] = weighted_f1
    all_metrics['total_test_samples'] = total_samples
    
    # Save metrics in Hybrid_Results
    os.makedirs(f'Hybrid_Results/{dataset_name}', exist_ok=True)
    metrics_file = f'Hybrid_Results/{dataset_name}/detection_metrics_hybrid_{con_label}_alpha_{alpha:.2f}_percentile_{percentile}.pt'
    torch.save({percentile: all_metrics}, metrics_file)
    
    return weighted_f1

def compute_hybrid_thresholds(acts_file, gt_samples_per_concept_cal, percentile, dataset_name, con_label, device):
    """
    Compute thresholds using calibration set and existing validation functions.
    
    Args:
        acts_file: Path to activations file
        gt_samples_per_concept_cal: GT samples for calibration set
        percentile: Percentile to use for thresholding
        dataset_name: Name of dataset
        con_label: Concept label for saving
        device: Device to use
    
    Returns:
        Path to saved thresholds file
    """
    # Load activations
    if 'linsep' in acts_file:
        act_metrics = pd.read_csv(f"Distances/{dataset_name}/{acts_file}")
    else:
        act_metrics = pd.read_csv(f"Cosine_Similarities/{dataset_name}/{acts_file}")
    
    # Use existing threshold computation function
    compute_concept_thresholds_over_percentiles(gt_samples_per_concept_cal, 
                                               act_metrics, [percentile], device, 
                                               dataset_name, f"hybrid_{con_label}", 
                                               n_vectors=1, n_concepts_to_print=0)
    
    # Return the threshold file path that was saved
    threshold_file = f"concept_thresholds_hybrid_{con_label}_percentile_{percentile}.pt"
    return threshold_file

def compute_hybrid_detection_metrics(acts_file, gt_samples_per_concept_test, percentile, dataset_name, model_input_size, device, con_label):
    """
    Compute detection metrics using test set and existing detection functions.
    
    Args:
        acts_file: Path to activations file
        gt_samples_per_concept_test: GT samples for test set
        percentile: Percentile to use
        dataset_name: Name of dataset
        model_input_size: Model input size
        device: Device to use
        con_label: Concept label
    
    Returns:
        Dictionary with F1 scores and other metrics
    """
    # Load activations
    if 'linsep' in acts_file:
        act_metrics = pd.read_csv(f"Distances/{dataset_name}/{acts_file}")
    else:
        act_metrics = pd.read_csv(f"Cosine_Similarities/{dataset_name}/{acts_file}")
    
    # Use existing detection metrics computation
    compute_detection_metrics_over_percentiles([percentile], 
                                              gt_samples_per_concept_test, 
                                              act_metrics, dataset_name, model_input_size, device, 
                                              f"hybrid_{con_label}", sample_type='patch', patch_size=14)
    
    # Load the computed metrics
    metrics_file = f"Metrics/{dataset_name}/detection_metrics_hybrid_{con_label}_percentile_{percentile}.pt"
    try:
        metrics = torch.load(metrics_file, weights_only=False)
        return metrics
    except FileNotFoundError:
        print(f"   ❌ Metrics file not found: {metrics_file}")
        return None

def get_concept_label(model_name, method):
    """Get the concept label for supervised methods."""
    if method == 'avg':
        return f'{model_name}_avg_patch_embeddings_percentthrumodel_100'
    elif method == 'linsep':
        return f'{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100'

def run_hybrid_analysis_for_config(model_name, model_input_size, dataset_name, method, detect_percentile, invert_percentile, alpha_values, device='cuda', scratch_dir=''):
    """
    Run hybrid concept analysis for a single configuration using existing pipeline functions.
    
    Args:
        detect_percentile: Percentile used to find superpatches
        invert_percentile: Percentile used for threshold computation and evaluation
    
    Returns:
        List of results for each alpha value, or None if failed
    """
    print(f"   📊 Processing {method} with detect: {detect_percentile}, invert: {invert_percentile}")
    
    try:
        # Load embeddings
        embeddings_file = f"{model_name}_patch_embeddings_percentthrumodel_100.pt"
        embeds_dic = torch.load(f"{scratch_dir}Embeddings/{dataset_name}/{embeddings_file}")
        embeddings = embeds_dic['normalized_embeddings'].to(device)
        
        # Get sample ranges
        sample_ranges = get_sample_ranges(dataset_name, model_input_size, embeddings.shape[0])
        
        # Load GT data like baseline does
        gt_patches_per_concept = torch.load(
            f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt',
            weights_only=False
        )
        gt_samples_per_concept_cal = torch.load(
            f'GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt',
            weights_only=False
        )
        gt_samples_per_concept_test = torch.load(
            f'GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt',
            weights_only=False
        )
        
        # Load concept vectors and superdetectors
        con_label = get_concept_label(model_name, method)
        concept_vectors = load_concept_vectors(dataset_name, con_label)
        superdetectors = load_superdetectors(dataset_name, con_label, detect_percentile)
        
        # Move concept vectors to device
        concept_vectors = {k: v.to(device) for k, v in concept_vectors.items()}
        
        # Sweep alpha values
        alpha_results = []
        
        for alpha in tqdm(alpha_values, desc=f"   Alpha sweep for {method}", leave=False):
            # 1. Compute hybrid concept vectors
            hybrid_concepts = compute_hybrid_concept_vectors(
                concept_vectors, superdetectors, embeddings, sample_ranges, alpha, dataset_name, model_input_size
            )
   
            # 2. Compute activations using per-sample hybrid vectors
            hybrid_activations = compute_hybrid_activations(
                embeddings, hybrid_concepts, sample_ranges, dataset_name, device, method, scratch_dir
            )
            
            # 3. Compute thresholds using calibration set
            thresholds = compute_hybrid_thresholds_for_concepts(
                hybrid_activations, gt_samples_per_concept_cal, invert_percentile, dataset_name, con_label, alpha
            )
            
            # 4. Evaluate performance on test set and save to Quant_Results
            weighted_f1 = evaluate_hybrid_performance_with_thresholds(
                hybrid_activations, thresholds, gt_patches_per_concept, gt_samples_per_concept_test, invert_percentile, dataset_name, con_label, alpha
            )
            
            alpha_results.append({
                'alpha': float(alpha),
                'weighted_f1': float(weighted_f1)
            })
        
        return alpha_results
        
    except Exception as e:
        print(f"   ❌ Error in hybrid analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_metric_over_alphas(dataset_name, model_name, method, metric='weighted_f1', save_path=None, show_plot=True):
    """
    Plot a given metric over alpha values for a specific model/dataset/method combination.
    
    Args:
        dataset_name: Name of dataset (e.g., 'CLEVR', 'Coco')
        model_name: Name of model (e.g., 'CLIP', 'Llama') 
        method: Method type ('avg' or 'linsep')
        metric: Metric to plot ('weighted_f1', 'precision', 'recall', 'accuracy', etc.)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    # Load results from individual hybrid analysis file
    results_file = f'Hybrid_Results/{dataset_name}/hybrid_analysis_{model_name}_{method}.json'
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        config_results = json.load(f)
    alpha_sweep = config_results['alpha_sweep']
    
    # Extract alpha values and metric values
    alphas = [result['alpha'] for result in alpha_sweep]
    
    if metric == 'weighted_f1':
        # Weighted F1 is stored directly in alpha_sweep
        metric_values = [result['weighted_f1'] for result in alpha_sweep]
    else:
        # For other metrics, need to load individual detection metrics files
        percentile = config_results['invert_percentile']
        con_label = get_concept_label(model_name, method)
        
        metric_values = []
        
        for alpha_result in alpha_sweep:
            alpha = alpha_result['alpha']
            
            # Load detection metrics for this alpha
            metrics_file = f'Hybrid_Results/{dataset_name}/detection_metrics_hybrid_{con_label}_alpha_{alpha:.2f}_percentile_{percentile}.pt'
            if not os.path.exists(metrics_file):
                print(f"   ⚠️  Metrics file not found: {metrics_file}")
                metric_values.append(0.0)
                continue
                
            metrics_data = torch.load(metrics_file, weights_only=False)
            alpha_metrics = metrics_data[percentile]
            
            if metric in alpha_metrics:
                # Global metric (like weighted_f1)
                metric_values.append(alpha_metrics[metric])
            else:
                # Per-concept metric - compute weighted average
                concept_values = []
                concept_weights = []
                
                for concept, concept_metrics in alpha_metrics.items():
                    if isinstance(concept_metrics, dict) and metric in concept_metrics:
                        concept_values.append(concept_metrics[metric])
                        concept_weights.append(concept_metrics.get('num_gt', 1))
                
                if concept_values:
                    # Weighted average across concepts
                    weighted_avg = sum(v * w for v, w in zip(concept_values, concept_weights)) / sum(concept_weights)
                    metric_values.append(weighted_avg)
                else:
                    metric_values.append(0.0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(alphas, metric_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Alpha\n(α=0: Pure Superdetector, α=1: Pure Concept Vector)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} vs Alpha\n{model_name} on {dataset_name} ({method})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Highlight best alpha first
    best_idx = np.argmax(metric_values)
    best_alpha = alphas[best_idx]
    best_value = metric_values[best_idx]
    
    ax.scatter(best_alpha, best_value, color='red', s=120, zorder=5, edgecolors='darkred', linewidth=2)
    
    # Get value range for positioning
    y_range = max(metric_values) - min(metric_values)
    y_min = min(metric_values)
    
    # Add vertical lines at pure endpoints
    # Pure superdetector (alpha=0)
    superdetector_value = metric_values[0] if alphas[0] == 0 else None
    if superdetector_value is not None:
        ax.axvline(x=0, color='blue', linestyle='--', alpha=0.7, linewidth=1)
    
    # Pure concept vector (alpha=1)
    concept_value = metric_values[-1] if alphas[-1] == 1 else None
    if concept_value is not None:
        ax.axvline(x=1, color='green', linestyle='--', alpha=0.7, linewidth=1)
    
    # Position annotations to avoid overlap
    # Create a list to track annotation positions
    annotations = []
    
    # Add best point annotation - position it much higher
    best_y_offset = 0.6 * y_range  # 60% of range above the point to clear the line completely
    annotations.append({
        'x': best_alpha,
        'y': best_value + best_y_offset,
        'text': f'Best: α={best_alpha:.2f}\n{metric}={best_value:.4f}',
        'color': 'yellow',
        'align': 'center'
    })
    
    # Add endpoint annotations positioned to avoid overlap
    if superdetector_value is not None:
        # Position superdetector annotation - move it left outside the plot area
        annotations.append({
            'x': -0.15,  # Move further left
            'y': superdetector_value,
            'text': f'Pure Superdetector\nα=0: {superdetector_value:.4f}',
            'color': 'lightblue',
            'align': 'center'
        })
        # Add arrow pointing to the actual point
        ax.annotate('', xy=(0, superdetector_value), xytext=(-0.15, superdetector_value),
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
    
    if concept_value is not None:
        # Position concept vector annotation - move it right outside the plot area
        annotations.append({
            'x': 1.15,  # Move further right
            'y': concept_value,
            'text': f'Pure Concept Vector\nα=1: {concept_value:.4f}',
            'color': 'lightgreen',
            'align': 'center'
        })
        # Add arrow pointing to the actual point
        ax.annotate('', xy=(1, concept_value), xytext=(1.15, concept_value),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.6))
    
    # Draw all annotations
    for ann in annotations:
        ax.text(ann['x'], ann['y'], ann['text'], 
                fontsize=9, ha=ann['align'], va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=ann['color'], alpha=0.8))
        
        # Add connecting line from annotation to point
        if 'Best' in ann['text']:
            ax.plot([ann['x'], best_alpha], [ann['y'] - 0.02 * y_range, best_value], 
                   'r--', alpha=0.5, linewidth=1)
    
    # Set alpha range with extra padding to show annotations
    ax.set_xlim(-0.25, 1.25)  # Extended to show annotations outside plot area
    ax.set_ylim(0, max(metric_values) * 1.7)  # Significantly increased upper limit for much higher annotation
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def plot_multiple_metrics_over_alphas(dataset_name, model_name, method, metrics=['weighted_f1', 'precision', 'recall'], save_path=None, show_plot=True):
    """
    Plot multiple metrics over alpha values for a specific model/dataset/method combination.
    
    Args:
        dataset_name: Name of dataset
        model_name: Name of model
        method: Method type ('avg' or 'linsep')
        metrics: List of metrics to plot
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    # Load results from individual hybrid analysis file
    results_file = f'Hybrid_Results/{dataset_name}/hybrid_analysis_{model_name}_{method}.json'
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        config_results = json.load(f)
    alpha_sweep = config_results['alpha_sweep']
    alphas = [result['alpha'] for result in alpha_sweep]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(metrics)))
    
    for metric, color in zip(metrics, colors):
        if metric == 'weighted_f1':
            metric_values = [result['weighted_f1'] for result in alpha_sweep]
        else:
            # Load individual metrics files for other metrics
            percentile = config_results['invert_percentile']
            con_label = get_concept_label(model_name, method)
            
            metric_values = []
            for alpha_result in alpha_sweep:
                alpha = alpha_result['alpha']
                metrics_file = f'Hybrid_Results/{dataset_name}/detection_metrics_hybrid_{con_label}_alpha_{alpha:.2f}_percentile_{percentile}.pt'
                
                if not os.path.exists(metrics_file):
                    metric_values.append(0.0)
                    continue
                    
                metrics_data = torch.load(metrics_file, weights_only=False)
                alpha_metrics = metrics_data[percentile]
                
                if metric in alpha_metrics:
                    metric_values.append(alpha_metrics[metric])
                else:
                    # Compute weighted average across concepts
                    concept_values = []
                    concept_weights = []
                    
                    for concept, concept_metrics in alpha_metrics.items():
                        if isinstance(concept_metrics, dict) and metric in concept_metrics:
                            concept_values.append(concept_metrics[metric])
                            concept_weights.append(concept_metrics.get('num_gt', 1))
                    
                    if concept_values:
                        weighted_avg = sum(v * w for v, w in zip(concept_values, concept_weights)) / sum(concept_weights)
                        metric_values.append(weighted_avg)
                    else:
                        metric_values.append(0.0)
        
        # Plot this metric
        ax.plot(alphas, metric_values, 'o-', linewidth=2, markersize=6, 
                label=metric.replace('_', ' ').title(), color=color)
        
        # Highlight best value for this metric
        best_idx = np.argmax(metric_values)
        best_alpha = alphas[best_idx]
        best_value = metric_values[best_idx]
        ax.scatter(best_alpha, best_value, color=color, s=80, edgecolors='black', linewidth=1, zorder=5)
    
    # Add vertical lines at pure endpoints
    ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='Pure Superdetector')
    ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Pure Concept Vector')
    
    ax.set_xlabel('Alpha\n(α=0: Pure Superdetector, α=1: Pure Concept Vector)', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(f'Multiple Metrics vs Alpha\n{model_name} on {dataset_name} ({method})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Multi-metric plot saved to: {save_path}")
    
    if show_plot:
        plt.show()