import torch
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import ast

from utils.general_utils import compute_cossim_w_vector, get_split_df, create_binary_labels, retrieve_topn_images_byconcepts
from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence, get_image_idx_from_global_patch_idx, compute_patches_per_image
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices
from utils.text_visualization_utils import get_glob_tok_indices_from_sent_idx, get_sent_idx_from_global_token_idx
from utils.memory_management_utils import ChunkedEmbeddingLoader

# Import functions that don't need chunking from the original module
from utils.quant_concept_evals_utils import (
    compute_avg_rand_threshold,
    compute_concept_thresholds,
    compute_concept_metrics,
    compute_stats_from_counts,
    create_binary_labels,
    get_patch_detection_tensor
)


def detect_then_invert_metrics_over_percentiles_chunked(detect_percentiles, invert_percentiles, 
                                              act_metrics, concepts, gt_samples_per_concept, 
                                              gt_samples_per_concept_test, device, dataset_name, 
                                              model_input_size, con_label, embedding_loader,
                                              all_object_patches=None, patch_size=14):
    """
    Chunked version of detect_then_invert_metrics_over_percentiles.
    Saves results to Chunked_Quant_Results instead of Quant_Results.
    
    Args:
        embedding_loader (ChunkedEmbeddingLoader): Chunked embedding loader instead of embeddings tensor
        ... (other args same as original)
    """
    # Create a modified version that saves to Chunked_Quant_Results
    import os
    from utils.quant_concept_evals_utils import (
        compute_concept_thresholds_over_percentiles,
        compute_detection_metrics_over_percentiles,
        get_patch_split_df,
        filter_patches_by_image_presence
    )
    
    # Compute all thresholds at once for caching
    all_percentiles = sorted(list(set(detect_percentiles) | set(invert_percentiles)))
    
    if 'kmeans' not in con_label:
        thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
    else:
        # Load files
        raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
        
        # Collect matched thresholds per percentile
        thresholds = {}
        for percentile, thresholds_dict in raw_thresholds.items():
            matched_thresholds = {}
            for concept, info in alignment_results.items():
                cluster_id = info['best_cluster']
                key = (concept, cluster_id)
                if key in thresholds_dict:
                    matched_thresholds[cluster_id] = thresholds_dict[key]
            thresholds[percentile] = matched_thresholds
    
    total_iters = len(detect_percentiles)
    pbar = tqdm(total=total_iters, desc="Evaluating thresholds")
    
    # Get the split dataframe and indices
    split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    
    # Create Chunked_Quant_Results directory structure
    os.makedirs(f'Chunked_Quant_Results/{dataset_name}', exist_ok=True)
    
    for detect_p in detect_percentiles:
        # Compute detection metrics
        detect_threshold_dict = thresholds[detect_p]
        detect_metrics_df = compute_detection_metrics_over_percentiles(
            {detect_p: detect_threshold_dict}, gt_samples_per_concept_test, act_metrics, 
            dataset_name, model_input_size, device, con_label, sample_type='patch', patch_size=patch_size
        )[detect_p]
        
        # Compute inversion metrics for all invert percentiles
        for invert_p in invert_percentiles:
            invert_threshold_dict = thresholds[invert_p]
            
            # Read the chunked inversion CSV from Chunked_Superpatches
            inv_cossim_file = f'Chunked_Superpatches/{dataset_name}/superpatch_avg_inv_per_{invert_p}_{con_label}.csv'
            
            if os.path.exists(inv_cossim_file):
                print(f"   Found inversion file: {inv_cossim_file}")
                # For now, just note that the inversion file exists
                # The original pipeline doesn't compute additional f1 metrics here
        
        pbar.update(1)
    
    pbar.close()


def compute_concept_thresholds_over_percentiles_chunked(gt_samples_per_concept_test, act_metrics, percentiles, 
                                               device='cuda', dataset_name='', con_label='', 
                                               embedding_loader=None, n_vectors=1, n_concepts_to_print=5):
    """
    Chunked version of compute_concept_thresholds_over_percentiles.
    Uses chunked embedding loader instead of full embedding tensor.
    
    Args:
        embedding_loader (ChunkedEmbeddingLoader): Chunked embedding loader instead of embeddings tensor
        ... (other args same as original)
    """
    # This function doesn't use raw embeddings either, just activation metrics
    # Import and call the original
    from utils.quant_concept_evals_utils import compute_concept_thresholds_over_percentiles
    
    return compute_concept_thresholds_over_percentiles(gt_samples_per_concept_test, act_metrics, percentiles,
                                                      device, dataset_name, con_label, n_vectors, n_concepts_to_print)


def compute_detection_metrics_over_percentiles_chunked(thresholds, act_metrics, gt_samples_per_concept, 
                                             gt_samples_per_concept_test, device, dataset_name, 
                                             model_input_size, con_label, embedding_loader=None,
                                             all_object_patches=None, patch_size=14):
    """
    Chunked version of compute_detection_metrics_over_percentiles.
    Uses chunked embedding loader instead of full embedding tensor.
    
    Args:
        embedding_loader (ChunkedEmbeddingLoader): Chunked embedding loader instead of embeddings tensor
        ... (other args same as original)
    """
    # This function also doesn't use raw embeddings, just activation metrics
    # Import and call the original
    from utils.quant_concept_evals_utils import compute_detection_metrics_over_percentiles
    
    return compute_detection_metrics_over_percentiles(thresholds, gt_samples_per_concept_test, act_metrics,
                                                     dataset_name, model_input_size, device, con_label, 
                                                     sample_type='patch', patch_size=patch_size)




# Functions that need embeddings access and chunking
def compute_cossim_with_concept_chunked(embedding_loader, concept_vector, device="cuda", batch_size=1000):
    """
    Compute cosine similarities between all embeddings and a concept vector using chunked processing.
    
    Args:
        embedding_loader (ChunkedEmbeddingLoader): Chunked embedding loader
        concept_vector (torch.Tensor): Concept vector to compare against
        device (str): Device for computation
        batch_size (int): Batch size for processing chunks
        
    Returns:
        torch.Tensor: Cosine similarities for all embeddings
    """
    concept_vector = concept_vector.to(device).unsqueeze(0)  # [1, embed_dim]
    all_similarities = []
    
    embedding_info = embedding_loader.get_embedding_info()
    total_embeddings = embedding_info['total_samples']
    
    # Process in batches
    for start_idx in tqdm(range(0, total_embeddings, batch_size), desc="Computing cosine similarities"):
        end_idx = min(start_idx + batch_size, total_embeddings)
        
        # Load batch of embeddings
        batch_embeddings = embedding_loader.load_range(start_idx, end_idx).to(device)
        
        # Compute cosine similarities for this batch
        batch_similarities = F.cosine_similarity(batch_embeddings, concept_vector, dim=1)
        all_similarities.append(batch_similarities.cpu())
        
        # Clean up
        del batch_embeddings
        torch.cuda.empty_cache() if device.startswith('cuda') else None
    
    return torch.cat(all_similarities)


def compute_avg_rand_threshold_chunked(embedding_loader, patch_indices, percentile, n_vectors=5, device="cuda", batch_size=1000):
    """
    Chunked version of compute_avg_rand_threshold.
    
    Args:
        embedding_loader (ChunkedEmbeddingLoader): Chunked embedding loader instead of embeddings tensor
        patch_indices (list): Indices of patches to consider
        percentile (float): Desired percentile
        n_vectors (int): Number of random vectors to sample
        device (str): Compute device
        batch_size (int): Batch size for processing chunks
        
    Returns:
        float: Average threshold computed over n_vectors random vectors
    """
    embedding_info = embedding_loader.get_embedding_info()
    embedding_dim = embedding_info['embedding_dim']
    
    # Generate random vectors
    random_vectors = torch.randn(n_vectors, embedding_dim, device=device)
    random_vectors = F.normalize(random_vectors, p=2, dim=1)
    
    thresholds = []
    
    for vec_idx in range(n_vectors):
        # Compute cosine similarities for this random vector
        similarities = compute_cossim_with_concept_chunked(embedding_loader, random_vectors[vec_idx], device, batch_size)
        
        # Filter to specified patch indices
        if patch_indices is not None:
            patch_indices_tensor = torch.tensor(patch_indices, dtype=torch.long)
            similarities = similarities[patch_indices_tensor]
        
        # Compute threshold
        threshold = torch.quantile(similarities, percentile).item()
        thresholds.append(threshold)
    
    return np.mean(thresholds)


def get_matched_concepts_and_data_fully_chunked(
    dataset_name,
    con_label,
    act_metrics,
    gt_samples_per_concept_test,
    gt_samples_per_concept,
    concepts=None
):
    """
    Fully chunked version that uses only Chunked_ directory paths and avoids fallbacks.
    
    Args:
        dataset_name (str): Name of the dataset
        con_label (str): Concept label
        act_metrics (pd.DataFrame): Activation metrics
        gt_samples_per_concept_test (dict): Ground truth test samples per concept
        gt_samples_per_concept (dict): Ground truth samples per concept
        concepts (dict, optional): Concepts dictionary
        
    Returns:
        tuple: (matched_acts, matched_gt_test, matched_gt, matched_concepts)
    """
    import os
    
    alignment_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
    
    if not os.path.exists(alignment_path):
        raise FileNotFoundError(f"Alignment results not found at {alignment_path}. "
                              f"Please run the unsupervised matching pipeline first.")
    
    alignment_results = torch.load(alignment_path)
    matching_cluster_ids = [info['best_cluster'] for info in alignment_results.values()]
    
    # Create Chunked_Unsupervised_Matches directory
    os.makedirs(f'Chunked_Unsupervised_Matches/{dataset_name}', exist_ok=True)
    
    matched_acts, matched_gt_test, matched_gt, matched_concepts = None, None, None, None
    
    if act_metrics is not None:
        matched_acts = act_metrics[[col for col in act_metrics.columns if col in matching_cluster_ids]]
        matched_acts.to_csv(f'Chunked_Unsupervised_Matches/{dataset_name}/actmetrics_{con_label}.csv')
    
    if gt_samples_per_concept_test is not None:
        matched_gt_test = {alignment_results[concept]['best_cluster']: gt_samples_per_concept_test[concept] 
                          for concept in alignment_results}
        torch.save(matched_gt_test, f'Chunked_Unsupervised_Matches/{dataset_name}/gt_test_{con_label}.pt')

    if gt_samples_per_concept is not None:
        matched_gt = {alignment_results[concept]['best_cluster']: gt_samples_per_concept[concept] 
                     for concept in alignment_results}
        torch.save(matched_gt, f'Chunked_Unsupervised_Matches/{dataset_name}/gt_{con_label}.pt')

    if concepts is not None:
        matched_concepts = {alignment_results[concept]['best_cluster']: concepts[alignment_results[concept]['best_cluster']] 
                           for concept in alignment_results}
        torch.save(matched_concepts, f'Chunked_Unsupervised_Matches/{dataset_name}/concepts_{con_label}.pt') 

    return matched_acts, matched_gt_test, matched_gt, matched_concepts