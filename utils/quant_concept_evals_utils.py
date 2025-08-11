import torch
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import random
import gc
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import glob

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import ast

import importlib
import utils.general_utils
import utils.text_visualization_utils
import utils.patch_alignment_utils
importlib.reload(utils.patch_alignment_utils)
importlib.reload(utils.general_utils)
importlib.reload(utils.text_visualization_utils)

from utils.general_utils import compute_cossim_w_vector, get_split_df, create_binary_labels, retrieve_topn_images_byconcepts
from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence, get_image_idx_from_global_patch_idx, compute_patches_per_image
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices, remap_text_ground_truth_indices
from utils.text_visualization_utils import get_glob_tok_indices_from_sent_idx, get_sent_idx_from_global_token_idx
from typing import Dict, List, Tuple, Union, Optional
from utils.memory_management_utils import MatchedConceptActivationLoader, ChunkedActivationLoader
from utils.loader_compatible_functions import (
    get_patch_detection_tensor as get_patch_detection_tensor_loader,
    compute_inversion_metrics
)

############# Find Thresholds for Concepts #############
def compute_avg_rand_threshold(embeddings, patch_indices, percentile, n_vectors=5, device="cuda"):
    """
    Computes the average random cosine similarity threshold over n_vectors random vectors
    in a fully vectorized manner using PyTorch.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape (N, embedding_dim).
        patch_indices (list or 1D Tensor): Indices of patches to consider.
        percentile (float): Desired percentile (e.g., 0.95).
        n_vectors (int): Number of random vectors to sample.
        device (str): Compute device (e.g., "cuda").
        
    Returns:
        float: The average threshold computed over the n_vectors random vectors.
    """
    # Ensure embeddings are on the target device.
    embeddings = embeddings.to(device)
    N, embedding_dim = embeddings.shape

    # Normalize embeddings (to compute cosine similarity via dot product)
    norm_embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    
    # Generate n_vectors random vectors and normalize them.
    random_vectors = torch.randn(n_vectors, embedding_dim, device=embeddings.device, dtype=embeddings.dtype)
    random_vectors = random_vectors / (random_vectors.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarities between each random vector and all embeddings.
    # Resulting shape: (n_vectors, N)
    cos_sims = torch.matmul(random_vectors, norm_embeddings.t())
    
    # Ensure patch_indices is a tensor on the correct device.
    if not torch.is_tensor(patch_indices):
        patch_indices = torch.tensor(patch_indices, device=device)
    
    # Select only the similarities for the specified patch indices.
    # Shape: (n_vectors, num_selected)
    relevant_cos_sims = cos_sims[:, patch_indices]
    
    # Sort each row in descending order.
    sorted_cos_sims, _ = torch.sort(relevant_cos_sims, dim=1, descending=True)
    
    # Determine the index corresponding to the desired percentile.
    n_selected = sorted_cos_sims.size(1)
    percentile_index = int(percentile * n_selected)
    percentile_index = min(percentile_index, n_selected - 1)  # safeguard against OOB
    
    # Gather the threshold from each random vector and average.
    thresholds = sorted_cos_sims[:, percentile_index]  # shape: (n_vectors,)
    avg_threshold = thresholds.mean().item()
    
    return avg_threshold

def compute_concept_thresholds(gt_samples_per_concept_cal, cos_sims, percentile, device, dataset_name, con_label, n_vectors=5,  n_concepts_to_print=0):
    """
    GPU-accelerated and vectorized computation of cosine similarity thresholds for each concept.
    For each concept, the threshold is defined as the (1 - percentile) quantile of its cosine 
    similarity scores (with NaN-padded sequences handled via torch.nanquantile). Additionally,
    an average random threshold is computed for each concept.
    
    Args:
        cos_sims (pd.DataFrame): Cosine similarity matrix (rows: patches, columns: concept names).
        gt_samples_per_concept (dict): Mapping of concept to list of patch indices.
            (The concept keys must correspond to the column names in cos_sims.)
        percentile (float): The desired percentile (e.g., 0.95).
        embeddings (torch.Tensor): Embeddings used for computing random thresholds.
        n_vectors (int): Number of random vectors for computing the random threshold.
        device (str): Compute device (e.g., "cuda").
        print_result (bool): If True, prints the computed thresholds.
        
    Returns:
        dict: Mapping from concept to a tuple (threshold, random_threshold).
              The threshold is computed from the concept's cosine similarities,
              and random_threshold is the average threshold from random vectors.
    """
    save_path = f'Thresholds/{dataset_name}/per_{percentile*100}_{con_label}.pt'
    # if os.path.exists(save_path):
    #     concept_thresholds = torch.load(f'Thresholds/{dataset_name}/per_{percentile*100}_{con_label}.pt')
    # else:
    # Convert the cosine similarity DataFrame to a torch tensor on the GPU.
    cos_sims_tensor = torch.tensor(cos_sims.values.astype(np.float32), device=device)

    concept_names = list(gt_samples_per_concept_cal.keys())
    sims_list = []

    # Gather cosine similarity scores for each concept.
    for concept in concept_names:
        # Get the column index for this concept. (Convert key to string to match DataFrame columns.)
        col_idx = cos_sims.columns.get_loc(str(concept))
        sample_indices = gt_samples_per_concept_cal[concept]
        sims = cos_sims_tensor[sample_indices, col_idx]  # shape: (num_samples_for_concept,)
        sims_list.append(sims)

    # Pad the list of tensors to form a single tensor of shape (n_concepts, max_samples),
    # using NaN for padding so that torch.nanquantile can ignore them.
    padded_sims = pad_sequence(sims_list, batch_first=True, padding_value=float('nan'))

    # Compute the (1 - percentile) quantile for each concept.
    # (For descending-sorted values, the (1 - percentile) quantile gives the threshold such that
    #  'percentile' fraction of values are above it.)
    thresholds_tensor = torch.nanquantile(padded_sims, 1 - percentile, dim=1)

    # For each concept, compute the average random threshold.
    concept_thresholds = {}
    for i, concept in enumerate(concept_names):
        sample_indices = gt_samples_per_concept_cal[concept]
        # rand_threshold = compute_avg_rand_threshold(embeddings, sample_indices, percentile, n_vectors=n_vectors, device=device)
        threshold_val = thresholds_tensor[i].item()
        concept_thresholds[concept] = (threshold_val, float('nan'))  # Use Python nan instead of numpy

    if n_concepts_to_print > 0: 
        print(f"Concept thresholds using {percentile*100:.1f}%:")
        for i, (concept, (threshold, random_threshold)) in enumerate(concept_thresholds.items()):
            if i > n_concepts_to_print:
                break
            print(f"Concept {concept}: {threshold:.4f}, (random={random_threshold:.4f})")

    torch.save(concept_thresholds, f'Thresholds/{dataset_name}/per_{percentile*100}_{con_label}.pt')

    # Clean up memory
    del cos_sims_tensor
    del padded_sims
    del sims_list

    return concept_thresholds


def compute_concept_thresholds_over_percentiles(gt_samples_per_concept_cal, loader, percentiles, device, dataset_name, con_label, n_vectors=5, n_concepts_to_print=0):
    """
    Computes thresholds for multiple percentiles using chunked activation data.
    Only loads the calibration samples needed for threshold computation.
    
    Args:
        gt_samples_per_concept_cal (dict): Mapping of concept to list of cal sample indices
        loader (ChunkedActivationLoader): Loader for chunked activation files
        percentiles (list): List of percentile values to compute thresholds for
        device (str): Compute device (e.g., "cuda")
        dataset_name (str): Name of dataset for cache file
        con_label (str): Label for cache file
        n_vectors (int): Number of random vectors (unused in current implementation)
        n_concepts_to_print (int): Number of concepts to print for debugging
        
    Returns:
        dict: Mapping from percentile -> concept -> (threshold, random_threshold)
    """
    from utils.memory_management_utils import ChunkedActivationLoader
    
    cache_file = f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt'
    all_thresholds = {}
    new_percentiles = set(percentiles)
    
    if new_percentiles:
        # Get all unique calibration indices we need
        all_cal_indices = set()
        for concept, indices in gt_samples_per_concept_cal.items():
            all_cal_indices.update(indices)
        all_cal_indices = sorted(list(all_cal_indices))
        
        print(f"   Loading activations for {len(all_cal_indices)} calibration samples...")
        
        # Determine the range of rows we need
        min_idx = min(all_cal_indices)
        max_idx = max(all_cal_indices)
        
        # Load only the range containing calibration samples
        cal_acts_tensor = loader.load_chunk_range(min_idx, max_idx + 1)
        
        # Create a mapping from original indices to tensor row positions
        idx_to_position = {idx: pos for pos, idx in enumerate(range(min_idx, max_idx + 1))}
        
        # Get concept names and info from loader
        loader_info = loader.get_activation_info() if hasattr(loader, 'get_activation_info') else loader.get_info()
        all_concept_names = loader_info['concept_names']
        concept_names = list(gt_samples_per_concept_cal.keys())
        
        # Move tensor to device if needed
        cos_sims_tensor = cal_acts_tensor.to(device)
        sims_list = []
        valid_concepts = []  # Track which concepts we actually process
        
        # Gather cosine similarity scores for each concept
        for concept in concept_names:
            # Find column index for this concept
            col_idx = all_concept_names.index(str(concept)) if str(concept) in all_concept_names else -1
            if col_idx == -1:
                continue
                
            sample_indices = gt_samples_per_concept_cal[concept]
            
            # Map original indices to positions in our loaded chunk
            chunk_positions = [idx_to_position[idx] for idx in sample_indices if idx in idx_to_position]
            
            if len(chunk_positions) > 0:
                sims = cos_sims_tensor[chunk_positions, col_idx]
                sims_list.append(sims)
                valid_concepts.append(concept)  # Add to valid concepts
            else:
                # No samples for this concept in cal set
                sims_list.append(torch.tensor([], device=device))
                valid_concepts.append(concept)  # Still add to valid concepts
        
        # Pad sequences for batch processing
        padded_sims = pad_sequence(sims_list, batch_first=True, padding_value=float('nan'))
        
        # Convert new percentiles to tensor and compute quantiles
        # For percentile p, we want the (1-p) quantile so that p fraction of values are above threshold
        percentiles_tensor = torch.tensor([(1 - p) for p in new_percentiles], device=device)
        batch_thresholds = torch.nanquantile(padded_sims, percentiles_tensor, dim=1, interpolation='linear')
        
        # Organize results
        for i, p in enumerate(new_percentiles):
            if p not in all_thresholds:
                all_thresholds[p] = {}
            
            for j, concept in enumerate(valid_concepts):
                if j < batch_thresholds.shape[1]:  # Safety check
                    threshold = batch_thresholds[i, j].item()
                    all_thresholds[p][concept] = (threshold, 0)  # (threshold, random_threshold)
        
        # Save all thresholds
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(all_thresholds, cache_file)
        
        # Clear memory
        del cos_sims_tensor, cal_acts_tensor
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return {p: all_thresholds[p] for p in percentiles}


#     try:
#         all_thresholds = torch.load(cache_file)
#         existing_percentiles = set(all_thresholds.keys())
#         new_percentiles = set(percentiles) - existing_percentiles
        
#         # If we have all percentiles, return cached results
#         if not new_percentiles:
#             return {p: all_thresholds[p] for p in percentiles}
            
#     except FileNotFoundError:
    all_thresholds = {}
    new_percentiles = set(percentiles)
    
    if new_percentiles:
        # Convert cosine similarities to tensor
        cos_sims_tensor = torch.tensor(cos_sims.values.astype(np.float32), device=device)
        concept_names = list(gt_samples_per_concept_cal.keys())
        sims_list = []

        # Gather cosine similarity scores for each concept
        for concept in concept_names:
            col_idx = cos_sims.columns.get_loc(str(concept))
            sample_indices = gt_samples_per_concept_cal[concept]
            sims = cos_sims_tensor[sample_indices, col_idx]
            sims_list.append(sims)

        # Pad sequences for batch processing
        padded_sims = pad_sequence(sims_list, batch_first=True, padding_value=float('nan'))
        
        # Convert new percentiles to tensor and compute quantiles
        percentiles_tensor = torch.tensor([(1 - p) for p in new_percentiles], device=device)
        thresholds_tensor = torch.nanquantile(padded_sims, percentiles_tensor, dim=1)
        
        # Update cache with new percentiles
        for p_idx, percentile in enumerate(new_percentiles):
            concept_thresholds = {}
            for c_idx, concept in enumerate(concept_names):
                threshold_val = thresholds_tensor[p_idx, c_idx].item()
                concept_thresholds[concept] = (threshold_val, float('nan'))  # Use Python nan
            
            all_thresholds[percentile] = concept_thresholds
            
            if n_concepts_to_print > 0:
                print(f"\nConcept thresholds using {percentile*100:.1f}%:")
                for i, (concept, (threshold, _)) in enumerate(concept_thresholds.items()):
                    if i >= n_concepts_to_print:
                        break
                    print(f"Concept {concept}: {threshold:.4f}")
        
        # Save updated cache
        if con_label is not None:
            torch.save(all_thresholds, cache_file)
        
        # Clean up memory
        del cos_sims_tensor, padded_sims, sims_list, thresholds_tensor
    
    # Return only requested percentiles
    return {p: all_thresholds[p] for p in percentiles}



def compute_nonconcept_thresholds(gt_samples_per_concept, cos_sims, percentile, device, n_vectors=5, n_concepts_to_print=0):
    """
    GPU-accelerated and vectorized computation of cosine similarity thresholds for each concept,
    considering only patches that do NOT contain the concept (based on ground truth).
    
    For each concept, the threshold is defined as the percentile quantile of its cosine similarity scores
    computed from non-concept patches. In other words, it sections off the lowest `percentile` fraction
    of non-concept scores.
    
    Args:
        cos_sims (pd.DataFrame): Cosine similarity matrix (rows: patches, columns: concept names).
        gt_samples_per_concept (dict): Mapping from concept to list of patch indices representing ground truth.\n
            (The concept keys must correspond to the column names in cos_sims.)
        percentile (float): The desired percentile (e.g., 0.1 for the lowest 10% threshold of non-concept patches).
        device (str): Compute device (e.g., "cuda").
        n_vectors (int): Number of random vectors for computing a random threshold (unused here).\n
        n_concepts_to_print (int): Number of concepts to print for debugging (optional).
        
    Returns:
        dict: Mapping from concept to a tuple (threshold, random_threshold), where threshold is computed from\n
              non-concept patches using torch.nanquantile with the given percentile, and random_threshold is set to np.nan.
    """
    # Convert the cosine similarity DataFrame to a torch tensor on the given device.
    cos_sims_tensor = torch.tensor(cos_sims.values, device=device)
    # cos_sims_tensor = torch.tensor(cos_sims.values)
    total_patches = cos_sims.shape[0]
    
    concept_names = list(gt_samples_per_concept.keys())
    sims_list = []
    
    # For each concept, compute the non-concept patch indices and extract the cosine similarity scores.
    for concept in concept_names:
        # Compute non-concept patch indices: all indices not in the ground truth set for this concept.
        concept_gt = set(gt_samples_per_concept[concept])
        all_indices = set(range(total_patches))
        nonconcept_indices = list(all_indices - concept_gt)
        
        # Get the column index for the concept (ensure key is a string to match DataFrame columns).
        col_idx = cos_sims.columns.get_loc(str(concept))
        # Extract cosine similarity scores for non-concept patches.
        sims = cos_sims_tensor[nonconcept_indices, col_idx]  # shape: (num_nonconcept_patches,)
        sims_list.append(sims)
    
    # Pad the list of tensors to form a single tensor of shape (n_concepts, max_nonconcept_patches),
    # using NaN for padding so that torch.nanquantile ignores these values.
    from torch.nn.utils.rnn import pad_sequence
    padded_sims = pad_sequence(sims_list, batch_first=True, padding_value=float('nan'))
    # sims_list_cpu = [sims.cpu() for sims in sims_list]  # Move to CPU before padding
    # padded_sims = pad_sequence(sims_list_cpu, batch_first=True, padding_value=float('nan'))
    thresholds_tensor = torch.nanquantile(padded_sims, percentile, dim=1)
    
    # Compute the quantile corresponding to the given percentile (i.e. the threshold such that 'percentile'\n
    # fraction of non-concept values are below it).
    # thresholds_tensor = torch.nanquantile(padded_sims, percentile, dim=1)
    
    # Create the output dictionary mapping each concept to its computed threshold.
    concept_thresholds = {}
    for i, concept in enumerate(concept_names):
        threshold_val = thresholds_tensor[i].item()
        concept_thresholds[concept] = (threshold_val, float('nan'))
    
    if n_concepts_to_print > 0:
        print(f"Non-concept thresholds using {percentile*100:.1f}%:")
        for i, (concept, (threshold, random_threshold)) in enumerate(concept_thresholds.items()):
            if i >= n_concepts_to_print:
                break
            print(f"Concept {concept}: {threshold:.4f}, (random={random_threshold})")
    
    # Clean up GPU memory
    del cos_sims_tensor, padded_sims, sims_list
    
    return concept_thresholds


def evaluate_thresholds_across_dataset(concept_thresholds, gt_samples_per_concept, act_metrics, 
                                       dataset_name, sample_type, model_input_size, patch_size=14, all_object_patches=None, 
                                       balance_dataset=False, n_trials=1):
    """
    Evaluate threshold-based classification performance across a dataset.
    Computes True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN)
    for each concept, averaging the results over n_trials.
    
    All computations (aside from initial DataFrame extraction) use PyTorch tensors.
    
    Args:
        concept_thresholds (dict): Mapping from concept to (threshold, random_threshold).
        gt_samples_per_concept (dict): Mapping from concept to list of ground truth patch indices.
        act_metrics (pd.DataFrame): Activation metric matrix (rows: patches, columns: concept names,
                                    value: cosine similarity or distance to boundary).
        dataset_name (str): Name of the dataset.
        sample_type (str): Type of sample ('patch' or 'image').
        all_object_patches (set, optional): If provided, only consider these patch indices.
        balance_dataset (bool): Whether to balance positive and negative test samples.
        n_trials (int): Number of trials to average over.
    
    Returns:
        tuple: (avg_fp_count, avg_fn_count, avg_tp_count, avg_tn_count), where each is a dict mapping
               concept -> average count over n_trials.
    """
    # Initialize random generator with a fixed seed for reproducibility
    rng = torch.Generator()
    rng.manual_seed(42)  # Ensures reproducibility across runs

    # Initialize dictionaries to store counts per concept over trials.
    fp_count_trials = defaultdict(list)
    fn_count_trials = defaultdict(list)
    tp_count_trials = defaultdict(list)
    tn_count_trials = defaultdict(list)
    
    # Get the split dataframe.
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    elif sample_type == 'image':
        split_df = get_split_df(dataset_name)
    
    # Get test indices as a torch tensor.
    test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    
    #filter patches that are 'padding' given the preprocessing schemes
    relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)

    # If filtering patches to ones that contain some concept, restrict to indices in all_object_patches.
    if all_object_patches is not None:
        # Use a list comprehension and then convert back to tensor.
        relevant_indices = torch.tensor([int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches])
    
    # Get ground truth labels for all concepts.
    all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept)
  
    
    # Loop over each concept.
    for concept, concept_labels in all_concept_labels.items():
        # Process each concept for n_trials.
        for trial in range(n_trials):
            # Get the labels for this concept at the relevant indices.
            relevant_labels = concept_labels[relevant_indices]
            if balance_dataset:
                # Get positions (indices within relevant_indices) of positives and negatives.
                pos_positions = torch.where(relevant_labels == 1)[0]
                neg_positions = torch.where(relevant_labels == 0)[0]
                # Determine the minimum count.
                min_count = min(len(pos_positions), len(neg_positions))
                # If both groups have at least one sample, sample equally.
                if min_count > 0:
                    perm_pos = torch.randperm(len(pos_positions), generator=rng)[:min_count]
                    sampled_pos = pos_positions[perm_pos]
                    perm_neg = torch.randperm(len(neg_positions), generator=rng)[:min_count]
                    sampled_neg = neg_positions[perm_neg]
                    # Update relevant_indices by selecting the balanced positions.
                    balanced_positions = torch.cat([sampled_pos, sampled_neg])
                    # Optional: sort indices for reproducibility.
                    balanced_positions, _ = torch.sort(balanced_positions)
                    relevant_indices = relevant_indices[balanced_positions]
            
            # Get activation values for the selected indices.
            # Convert the DataFrame values for this concept into a torch tensor.
            relevant_indices_list = relevant_indices.tolist()
            act_vals = torch.tensor(act_metrics[str(concept)].loc[relevant_indices_list].values)
            threshold = concept_thresholds[concept][0]
            
            above_threshold = act_vals >= threshold  # Boolean tensor
                
            # Compute ground truth mask for these indices using the tensor directly.
            gt_mask = (concept_labels[relevant_indices] == 1)
            
            # Compute confusion matrix counts using torch.sum.
            tp = torch.sum(above_threshold & gt_mask).item()
            fn = torch.sum((~above_threshold) & gt_mask).item()
            fp = torch.sum(above_threshold & (~gt_mask)).item()
            tn = torch.sum((~above_threshold) & (~gt_mask)).item()
            
            # Append the counts for this trial.
            tp_count_trials[concept].append(tp)
            fn_count_trials[concept].append(fn)
            fp_count_trials[concept].append(fp)
            tn_count_trials[concept].append(tn)
    
    # Average the counts over all trials.
    avg_tp_count = {k: sum(v) / len(v) for k, v in tp_count_trials.items()}
    avg_fn_count = {k: sum(v) / len(v) for k, v in fn_count_trials.items()}
    avg_fp_count = {k: sum(v) / len(v) for k, v in fp_count_trials.items()}
    avg_tn_count = {k: sum(v) / len(v) for k, v in tn_count_trials.items()}
    
    return avg_fp_count, avg_fn_count, avg_tp_count, avg_tn_count


# def detect_then_invert_metrics(detect_percentile, invert_percentile, act_metrics, concepts, 
#                                gt_samples_per_concept, gt_samples_per_concept_test, device, dataset_name, 
#                                model_input_size, con_label, all_object_patches=None, n_trials=1, 
#                                balance_dataset=False, patch_size=14):
#     """
#     Performs two-stage concept detection: (1) detect images where a concept might be present using a 
#     detection threshold, then (2) evaluate activation within those detected images using an inverted 
#     threshold for concept classification. Computes classification metrics for each concept based on 
#     patch-level predictions.

#     Args:
#         detect_percentile (float): Percentile used to compute the image-level detection thresholds.
#         invert_percentile (float): Percentile used to compute the patch-level inversion thresholds.
#         act_metrics (pd.DataFrame): Activation metric matrix (rows: patches, columns: concepts).
#         concepts (list of str): List of concept names to evaluate.
#         gt_samples_per_concept (dict): Ground truth concept labels (patch indices) across the full dataset.
#         gt_samples_per_concept_test (dict): Ground truth concept labels (patch indices) on the test set.
#         device (str): Torch device identifier (e.g., 'cuda').
#         dataset_name (str): Name of the dataset.
#         model_input_size (int): Image input size used to determine patch indexing.
#         con_label (str): String identifier used in metric saving and tracking.
#         all_object_patches (set, optional): If provided, restrict evaluation to these patch indices.
#         n_trials (int): Number of repeated trials to average metrics over.
#         balance_dataset (bool): Whether to balance the number of positive and negative examples in each trial.
#         patch_size (int): Size of each patch (default: 14).

#     Returns:
#         pd.DataFrame: A dataframe containing per-concept evaluation metrics (e.g., accuracy, precision, recall, F1).
#     """
    
#     detect_thresholds = compute_concept_thresholds(gt_samples_per_concept_test, 
#                                                 act_metrics, detect_percentile, n_vectors=1, device=device, 
#                                                 n_concepts_to_print=0, dataset_name=dataset_name, con_label=con_label)
#     invert_thresholds = compute_concept_thresholds(gt_samples_per_concept_test, 
#                                                 act_metrics, invert_percentile, n_vectors=1, device=device, 
#                                                 n_concepts_to_print=0, dataset_name=dataset_name, con_label=con_label)
    
#     concept_keys = set(detect_thresholds.keys()) & set(invert_thresholds.keys() & set(concepts.keys()))
    
#     # Initialize random generator with a fixed seed for reproducibility
#     rng = torch.Generator()
#     rng.manual_seed(42)  # Ensures reproducibility across runs
    
#     # Get the split dataframe.
#     split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    
#     # Get test indices as a torch tensor.
#     test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    
#     #filter patches that are 'padding' given the preprocessing schemes
#     relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)

#     # If filtering patches to ones that contain some concept, restrict to indices in all_object_patches.
#     if all_object_patches is not None:
#         relevant_indices = torch.tensor([int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches])
    
#     # Get ground truth labels for all concepts.
#     all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept)
  
#     # Get a boolean DataFrame indicating whether each patch is part of an image that was 'detected'
#     detected_patch_masks = get_patch_detection_tensor(act_metrics, detect_thresholds, model_input_size, dataset_name)
  
#     # Loop over each concept.
#     tp_count, fp_count, tn_count, fn_count = {}, {}, {}, {}
#     for concept, concept_labels in all_concept_labels.items():
#         if concept not in concept_keys: #fixing some weird stuff with how I filtered lowest represented concepts
#             continue
            
#         relevant_labels = concept_labels[relevant_indices]
            
#         # Get activation values for the selected indices.
#         relevant_indices_list = relevant_indices.tolist()
#         act_vals = torch.tensor(act_metrics[str(concept)].iloc[relevant_indices_list].values)
#         threshold = invert_thresholds[concept][0]

#         detected_patches = torch.tensor(detected_patch_masks[concept].iloc[relevant_indices_list].values)
#         activated_patches = (act_vals >= threshold) &  detected_patches #patch activated if activation above threshold + image is detected
#         # Compute ground truth mask for these indices using the tensor directly
#         gt_mask = (concept_labels[relevant_indices] == 1)

#         # Compute confusion matrix counts using torch.sum.
#         tp = torch.sum(activated_patches & gt_mask).item()
#         fn = torch.sum((~ activated_patches) & gt_mask).item()
#         fp = torch.sum(activated_patches & (~gt_mask)).item()
#         tn = torch.sum((~activated_patches) & (~gt_mask)).item()

#         # Append the counts 
#         tp_count[concept] = tp
#         fn_count[concept] = fn
#         fp_count[concept] = fp
#         tn_count[concept] = tn
    
#     #calculate metrics from the count
#     metrics_df = compute_concept_metrics(fp_count, fn_count, tp_count, tn_count, concept_keys,
#                                     dataset_name, con_label, just_obj = (all_object_patches is not None),
#                                          invert_percentile=invert_percentile, detect_percentile=detect_percentile)
    
#     return metrics_df
    
    
# def detect_then_invert_metrics_over_percentiles(detect_percentiles, 
#                                                 invert_percentiles, 
#                                                 act_metrics, concepts, gt_samples_per_concept, gt_samples_per_concept_test,
#                                                 device, dataset_name, model_input_size, con_label, all_object_patches=None,
#                                                 n_trials=10, balance_dataset=False, patch_size=14):
#     """ Calls detect then invert metrics performance across all percentile combinations
#     """
#     total_iters = sum(invert > detect for detect in detect_percentiles for invert in invert_percentiles)
#     pbar = tqdm(total=total_iters, desc="Evaluating thresholds")
    
#     for detect_percentile in detect_percentiles:
#         for invert_percentile in invert_percentiles:
#             if invert_percentile >= detect_percentile:
#                 try:
#                     torch.load(f'Quant_Results/{dataset_name}/detectfirst_{detect_percentile*100}_per_{invert_percentile*100}_{con_label}.csv')
#                 except:
#                     detect_then_invert_metrics(detect_percentile, invert_percentile, act_metrics, concepts, 
#                                        gt_samples_per_concept, gt_samples_per_concept_test, device, dataset_name, 
#                                        model_input_size, con_label, all_object_patches=None, n_trials=10, 
#                                        balance_dataset=False, patch_size=14)
#                 try:
#                         torch.load(f'Quant_Results/{dataset_name}/justobj_detectfirst_{detect_percentile*100}_per_{invert_percentile*100}_{con_label}.csv')
#                 except:
#                     detect_then_invert_metrics(detect_percentile, invert_percentile, act_metrics, concepts, 
#                                    gt_samples_per_concept, gt_samples_per_concept_test, device, dataset_name, 
#                                    model_input_size, con_label, all_object_patches=all_object_patches, n_trials=10, 
#                                    balance_dataset=False, patch_size=14)
#                     pbar.update(1)
#     pbar.close()
    
import time
import logging
from contextlib import contextmanager

@contextmanager
def timer(name):
    """Context manager to time code blocks"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name} took {elapsed:.2f} seconds")

# def detect_then_invert_metrics(detect_percentile, invert_percentiles, act_metrics, concepts, 
#                              gt_samples_per_concept, gt_samples_per_concept_test, relevant_indices,
#                              all_concept_labels, device, dataset_name, 
#                              model_input_size, con_label, all_object_patches=None, patch_size=14):
#     """
#     Performs two-stage concept detection using cached thresholds computation.
#     Computes metrics for multiple invert percentiles while looping through concepts only once.
#     """
#     print(f"\nProfiling detect_then_invert_metrics:")
#     print(f"Input shapes: act_metrics={act_metrics.shape}, concepts={len(concepts)}")
#     print(f"Number of invert percentiles: {len(invert_percentiles)}")

#     with timer("Computing thresholds"):
#         # Get thresholds for detection and all inversion percentiles at once
#         all_percentiles = [detect_percentile] + list(invert_percentiles)
#         thresholds = compute_concept_thresholds_over_percentiles(
#             gt_samples_per_concept_test, 
#             act_metrics,
#             all_percentiles,
#             device=device,
#             dataset_name=dataset_name,
#             con_label=con_label,
#             n_vectors=1,
#             n_concepts_to_print=0
#         )
    
#     with timer("Computing detection masks"):
#         # Get detection threshold and compute detection mask once
#         detect_thresholds = thresholds[detect_percentile]
#         detected_patch_masks = get_patch_detection_tensor(act_metrics, detect_thresholds, model_input_size, dataset_name)
    
#     # Common preprocessing steps
#     concept_keys = set(detect_thresholds.keys()) & set(concepts.keys())
#     print(f"Number of valid concepts: {len(concept_keys)}")

#     if all_object_patches is not None:
#         with timer("Filtering object patches"):
#             relevant_indices = torch.tensor([int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches])
    
#     # Initialize dictionaries for all percentiles
#     metrics_dfs = {p: {} for p in invert_percentiles}
#     for p in invert_percentiles:
#         metrics_dfs[p] = {
#             'tp_count': {}, 'fp_count': {}, 'tn_count': {}, 'fn_count': {}
#         }
    
#     concept_times = {}
#     # Loop through concepts first
#     with timer("Processing all concepts"):
#         for concept, concept_labels in all_concept_labels.items():
#             if concept not in concept_keys:
#                 continue
                
#             concept_start = time.perf_counter()
            
#             # Get concept-specific data (computed once for all percentiles)
#             relevant_labels = concept_labels[relevant_indices]
#             relevant_indices_list = relevant_indices.tolist()
            
#             with timer(f"Loading metrics for concept {concept}"):
#                 act_vals = torch.tensor(act_metrics[str(concept)].loc[relevant_indices_list].values)
#                 detected_patches = torch.tensor(detected_patch_masks[concept].iloc[relevant_indices_list].values)
#                 gt_mask = (relevant_labels == 1)
            
#             percentile_times = []
#             # Loop through invert percentiles for this concept
#             for invert_percentile in invert_percentiles:
#                 percentile_start = time.perf_counter()

#                 threshold = thresholds[invert_percentile][concept][0]
#                 activated_patches = (act_vals >= threshold) & detected_patches
                
#                 # Compute confusion matrix counts
#                 tp = torch.sum(activated_patches & gt_mask).item()
#                 fn = torch.sum((~activated_patches) & gt_mask).item()
#                 fp = torch.sum(activated_patches & (~gt_mask)).item()
#                 tn = torch.sum((~activated_patches) & (~gt_mask)).item()
                
#                 # Store counts for this percentile
#                 metrics_dfs[invert_percentile]['tp_count'][concept] = tp
#                 metrics_dfs[invert_percentile]['fn_count'][concept] = fn
#                 metrics_dfs[invert_percentile]['fp_count'][concept] = fp
#                 metrics_dfs[invert_percentile]['tn_count'][concept] = tn

#                 percentile_times.append(time.perf_counter() - percentile_start)

#             concept_times[concept] = time.perf_counter() - concept_start
#             print(f"Concept {concept} took {concept_times[concept]:.2f}s (avg {np.mean(percentile_times):.3f}s per percentile)")

#     # Compute final metrics for each percentile
#     final_metrics = {}
#     with timer("Computing final metrics"):
#         for invert_percentile in invert_percentiles:
#             counts = metrics_dfs[invert_percentile]
#             metrics_df = compute_concept_metrics(
#                 counts['fp_count'], counts['fn_count'], 
#                 counts['tp_count'], counts['tn_count'], 
#                 concept_keys, dataset_name, con_label, 
#                 just_obj=(all_object_patches is not None),
#                 invert_percentile=invert_percentile, 
#                 detect_percentile=detect_percentile
#             )
#             final_metrics[invert_percentile] = metrics_df

#     # Print summary statistics
#     print("\nPerformance Summary:")
#     print(f"Average time per concept: {np.mean(list(concept_times.values())):.2f}s")
#     print(f"Slowest concept: {max(concept_times.items(), key=lambda x: x[1])[0]} ({max(concept_times.values()):.2f}s)")
#     print(f"Memory usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
#     return final_metrics
# def detect_then_invert_metrics(detect_percentile, invert_percentiles, act_metrics, concepts, 
#                              gt_samples_per_concept, gt_samples_per_concept_test, relevant_indices,
#                              all_concept_labels, device, dataset_name, 
#                              model_input_size, con_label, all_object_patches=None, patch_size=14):
#     """
#     Performs two-stage concept detection using cached thresholds computation.
#     Computes metrics for multiple invert percentiles while processing concepts in parallel.
#     """
#     print(f"\nProfiling detect_then_invert_metrics:")
#     print(f"Input shapes: act_metrics={act_metrics.shape}, concepts={len(concepts)}")
#     print(f"Number of invert percentiles: {len(invert_percentiles)}")

#     with timer("Computing thresholds"):
#         # Get thresholds for detection and all inversion percentiles at once
#         all_percentiles = [detect_percentile] + list(invert_percentiles)
#         thresholds = compute_concept_thresholds_over_percentiles(
#             gt_samples_per_concept_test, 
#             act_metrics,
#             all_percentiles,
#             device=device,
#             dataset_name=dataset_name,
#             con_label=con_label,
#             n_vectors=1,
#             n_concepts_to_print=0
#         )
    
#     with timer("Computing detection masks"):
#         # Get detection threshold and compute detection mask once
#         detect_thresholds = thresholds[detect_percentile]
#         detected_patch_masks = get_patch_detection_tensor(act_metrics, detect_thresholds, model_input_size, dataset_name)
    
#     # Common preprocessing steps
#     concept_keys = set(detect_thresholds.keys()) & set(concepts.keys())
#     print(f"Number of valid concepts: {len(concept_keys)}")

#     if all_object_patches is not None:
#         with timer("Filtering object patches"):
#             relevant_indices = torch.tensor([int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches])
    
#     # Initialize metrics storage
#     metrics_dfs = {p: {
#         'tp_count': {}, 'fp_count': {}, 'tn_count': {}, 'fn_count': {}
#     } for p in invert_percentiles}

#     # Prepare all concept data at once
#     with timer("Processing all concepts in parallel"):
#         # Convert relevant data to tensors
#         relevant_indices_list = relevant_indices.tolist()
        
#         # Pre-allocate tensors for all concepts
#         n_concepts = len(concept_keys)
#         n_samples = len(relevant_indices)
        
#         act_vals_all = torch.zeros((n_samples, n_concepts), device=device)
#         detected_patches_all = torch.zeros((n_samples, n_concepts), device=device, dtype=torch.bool)
#         gt_masks_all = torch.zeros((n_samples, n_concepts), device=device, dtype=torch.bool)
        
#         # Load all concept data in parallel
#         for i, concept in enumerate(concept_keys):
#             act_vals_all[:, i] = torch.tensor(act_metrics[str(concept)].iloc[relevant_indices_list].values, device=device)
#             detected_patches_all[:, i] = torch.tensor(detected_patch_masks[concept].iloc[relevant_indices_list].values, device=device)
#             gt_masks_all[:, i] = torch.tensor(all_concept_labels[concept][relevant_indices] == 1, device=device)

#         # Process all percentiles for all concepts simultaneously
#         for invert_percentile in invert_percentiles:
#             # Get thresholds for all concepts at this percentile
#             thresh_tensor = torch.tensor([thresholds[invert_percentile][c][0] for c in concept_keys], device=device)
            
#             # Compute activated patches for all concepts at once
#             activated_patches = (act_vals_all >= thresh_tensor.unsqueeze(0)) & detected_patches_all
            
#             # Compute confusion matrix counts for all concepts at once
#             tp_counts = torch.sum(activated_patches & gt_masks_all, dim=0)
#             fn_counts = torch.sum((~activated_patches) & gt_masks_all, dim=0)
#             fp_counts = torch.sum(activated_patches & (~gt_masks_all), dim=0)
#             tn_counts = torch.sum((~activated_patches) & (~gt_masks_all), dim=0)
            
#             # Store results
#             for i, concept in enumerate(concept_keys):
#                 metrics_dfs[invert_percentile]['tp_count'][concept] = tp_counts[i].item()
#                 metrics_dfs[invert_percentile]['fn_count'][concept] = fn_counts[i].item()
#                 metrics_dfs[invert_percentile]['fp_count'][concept] = fp_counts[i].item()
#                 metrics_dfs[invert_percentile]['tn_count'][concept] = tn_counts[i].item()

#     # Compute final metrics for each percentile
#     final_metrics = {}
#     with timer("Computing final metrics"):
#         for invert_percentile in invert_percentiles:
#             counts = metrics_dfs[invert_percentile]
#             metrics_df = compute_concept_metrics(
#                 counts['fp_count'], counts['fn_count'], 
#                 counts['tp_count'], counts['tn_count'], 
#                 concept_keys, dataset_name, con_label, 
#                 just_obj=(all_object_patches is not None),
#                 invert_percentile=invert_percentile, 
#                 detect_percentile=detect_percentile
#             )
#             final_metrics[invert_percentile] = metrics_df

#     # Print summary statistics
#     print("\nPerformance Summary:")
#     print(f"Peak memory usage: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
#     torch.cuda.reset_peak_memory_stats()
    
#     return final_metrics
def detect_then_invert_metrics(
    detect_percentile: float,
    invert_percentiles: List[float],
    act_loader: Union[MatchedConceptActivationLoader, ChunkedActivationLoader],
    concepts: Dict,
    gt_samples_per_concept: Dict,
    gt_samples_per_concept_cal: Dict,
    relevant_indices: torch.Tensor,
    all_concept_labels: Dict,
    device: str,
    dataset_name: str,
    model_input_size: Tuple,
    con_label: str,
    all_object_patches: Optional[Dict] = None,
    patch_size: int = 14,
    embedding_loader=None,
    split: str = 'cal'
):
    """
    Memory-efficient version of detect_then_invert_metrics using activation loader.
    
    Args:
        detect_percentile: Detection threshold percentile
        invert_percentiles: List of inversion threshold percentiles
        act_loader: Activation loader (MatchedConceptActivationLoader or ChunkedActivationLoader)
        concepts: Concept vectors
        gt_samples_per_concept: Ground truth samples per concept
        gt_samples_per_concept_cal: Calibration ground truth samples
        relevant_indices: Relevant sample indices
        all_concept_labels: Binary labels for all concepts
        device: Compute device
        dataset_name: Dataset name
        model_input_size: Model input dimensions
        con_label: Concept label
        all_object_patches: Optional object patches
        patch_size: Patch size
        embedding_loader: ChunkedEmbeddingLoader instance for computing cosine similarities
        
    Returns:
        Dict of metrics for each inversion percentile
    """
    # Load thresholds
    if 'kmeans' not in con_label:
        threshold_path = f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt'
        thresholds = torch.load(threshold_path, weights_only=False)
    else:
        # Load files for kmeans
        raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
        
        # Collect matched thresholds per percentile
        # Use concept names as keys for consistency with supervised methods
        thresholds = {}
        for percentile, thresholds_dict in raw_thresholds.items():
            matched_thresholds = {}
            for concept, info in alignment_results.items():
                cluster_id = info['best_cluster']
                key = (concept, cluster_id)
                if key in thresholds_dict:
                    matched_thresholds[concept] = thresholds_dict[key]
            thresholds[percentile] = matched_thresholds

    # Get detection thresholds
    detect_thresholds = thresholds[detect_percentile]
    # Get detected patches using loader-compatible function
    # Use the specified split (cal for calibration, test for test evaluation)
    detected_patches = get_patch_detection_tensor_loader(
        act_loader, detect_thresholds, model_input_size, dataset_name, patch_size, split=split
    )
    
    # Filter to concepts that exist in thresholds, concepts, AND ground truth labels
    concept_keys = set(detect_thresholds.keys()) & set(concepts.keys()) & set(all_concept_labels.keys())
    
    # Compute metrics for each inversion percentile
    results = {}
    
    for invert_percentile in invert_percentiles:
        invert_thresholds = thresholds[invert_percentile]
        
        
        # Compute inversion metrics using loader
        inversion_metrics = compute_inversion_metrics(
            act_loader,
            {k: concepts[k] for k in concept_keys},
            invert_thresholds,
            detected_patches,
            relevant_indices,
            all_concept_labels,
            device,
            embedding_loader
        )
        
        # Save results
        os.makedirs(f'Quant_Results/{dataset_name}', exist_ok=True)
        save_path = f'Quant_Results/{dataset_name}/detectfirst_{detect_percentile}_invert_{invert_percentile}_{con_label}.pt'
        torch.save(inversion_metrics, save_path)
        
        results[invert_percentile] = inversion_metrics
        
    return results







    
# def detect_then_invert_metrics(detect_percentile, invert_percentiles, act_metrics, concepts, 
#                              gt_samples_per_concept, gt_samples_per_concept_test, relevant_indices,
#                              all_concept_labels, device, dataset_name, 
#                              model_input_size, con_label, all_object_patches=None, patch_size=14):
#     """
#     Performs two-stage concept detection using cached thresholds computation.
#     Computes metrics for multiple invert percentiles while looping through concepts only once.
#     """
#     # Get thresholds for detection and all inversion percentiles at once
#     all_percentiles = [detect_percentile] + list(invert_percentiles)
#     thresholds = compute_concept_thresholds_over_percentiles(
#         gt_samples_per_concept_test, 
#         act_metrics,
#         all_percentiles,
#         device=device,
#         dataset_name=dataset_name,
#         con_label=con_label,
#         n_vectors=1,
#         n_concepts_to_print=0
#     )
    
#     # Get detection threshold and compute detection mask once
#     detect_thresholds = thresholds[detect_percentile]
#     detected_patch_masks = get_patch_detection_tensor(act_metrics, detect_thresholds, model_input_size, dataset_name)
    
#     # Common preprocessing steps
#     concept_keys = set(detect_thresholds.keys()) & set(concepts.keys())

#     if all_object_patches is not None:
#         relevant_indices = torch.tensor([int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches])
    
#     # Initialize dictionaries for all percentiles
#     metrics_dfs = {p: {} for p in invert_percentiles}
#     for p in invert_percentiles:
#         metrics_dfs[p] = {
#             'tp_count': {}, 'fp_count': {}, 'tn_count': {}, 'fn_count': {}
#         }
    
#     # Loop through concepts first
#     for concept, concept_labels in all_concept_labels.items():
#         if concept not in concept_keys:
#             continue
            
#         # Get concept-specific data (computed once for all percentiles)
#         relevant_labels = concept_labels[relevant_indices]
#         relevant_indices_list = relevant_indices.tolist()
#         act_vals = torch.tensor(act_metrics[str(concept)].iloc[relevant_indices_list].values)
#         detected_patches = torch.tensor(detected_patch_masks[concept].iloc[relevant_indices_list].values)
#         gt_mask = (relevant_labels == 1)
        
#         # Loop through invert percentiles for this concept
#         for invert_percentile in invert_percentiles:

#             threshold = thresholds[invert_percentile][concept][0]
#             activated_patches = (act_vals >= threshold) & detected_patches
            
#             # Compute confusion matrix counts
#             tp = torch.sum(activated_patches & gt_mask).item()
#             fn = torch.sum((~activated_patches) & gt_mask).item()
#             fp = torch.sum(activated_patches & (~gt_mask)).item()
#             tn = torch.sum((~activated_patches) & (~gt_mask)).item()
            
#             # Store counts for this percentile
#             metrics_dfs[invert_percentile]['tp_count'][concept] = tp
#             metrics_dfs[invert_percentile]['fn_count'][concept] = fn
#             metrics_dfs[invert_percentile]['fp_count'][concept] = fp
#             metrics_dfs[invert_percentile]['tn_count'][concept] = tn
    
#     # Compute final metrics for each percentile
#     final_metrics = {}
#     for invert_percentile in invert_percentiles:
#         counts = metrics_dfs[invert_percentile]
#         metrics_df = compute_concept_metrics(
#             counts['fp_count'], counts['fn_count'], 
#             counts['tp_count'], counts['tn_count'], 
#             concept_keys, dataset_name, con_label, 
#             just_obj=(all_object_patches is not None),
#             invert_percentile=invert_percentile, 
#             detect_percentile=detect_percentile
#         )
#         final_metrics[invert_percentile] = metrics_df
    
#     return final_metrics


def detect_then_invert_metrics_over_percentiles(
    detect_percentiles: List[float],
    invert_percentiles: List[float],
    act_loader: Union[MatchedConceptActivationLoader, ChunkedActivationLoader],
    concepts: Dict,
    gt_samples_per_concept: Dict,
    gt_samples_per_concept_cal: Dict,
    device: str,
    dataset_name: str,
    model_input_size: Tuple,
    con_label: str,
    all_object_patches: Optional[Dict] = None,
    patch_size: int = 14,
    use_best_detection_percentiles: bool = True,
    embedding_loader=None
):
    """
    Memory-efficient version using activation loader instead of full DataFrame.
    Evaluates metrics across all detect/invert percentile combinations on calibration set.
    
    Args:
        detect_percentiles: List of detection percentiles to evaluate
        invert_percentiles: List of inversion percentiles to evaluate
        act_loader: Activation loader instead of act_metrics DataFrame
        concepts: Concept vectors
        gt_samples_per_concept: Ground truth samples per concept
        gt_samples_per_concept_cal: Calibration ground truth samples
        device: Compute device
        dataset_name: Dataset name
        model_input_size: Model input dimensions
        con_label: Concept label
        all_object_patches: Optional object patches
        patch_size: Patch size
        use_best_detection_percentiles: If True, use best detection percentiles from calibration
        embedding_loader: ChunkedEmbeddingLoader instance for computing cosine similarities
    """
    # Load best detection percentiles if requested (REQUIRED when use_best_detection_percentiles=True)
    best_detection_percentiles = None
    if use_best_detection_percentiles:
        best_detection_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
        if not os.path.exists(best_detection_file):
            raise FileNotFoundError(
                f"Best detection percentiles file not found at {best_detection_file}\n"
                f"Please run all_detection_stats.py first to generate best detection percentiles."
            )
        best_detection_percentiles = torch.load(best_detection_file, weights_only=False)
        print(f"Using best detection percentiles from {best_detection_file}")
    
    # Get the split dataframe and indices - USE CALIBRATION SET  
    from utils.general_utils import get_split_df
    split_df = get_split_df(dataset_name)
    
    # For text datasets, get calibration sentence indices
    if model_input_size[0] == 'text':
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        token_counts_per_sentence = torch.load(token_counts_file, weights_only=False)
        cal_sentence_indices = [i for i in range(len(token_counts_per_sentence)) if split_df.get(i) == 'cal']
        cal_indices = torch.tensor(cal_sentence_indices)
    else:
        # For image datasets, use patch-based approach  
        split_df_patch = get_patch_split_df(dataset_name, model_input_size=model_input_size, patch_size=patch_size)
        cal_indices = torch.tensor(split_df_patch.index[split_df_patch == 'cal'].tolist())
    
    if model_input_size[0] == 'text':
        # GLOBAL INDICES APPROACH: Use ALL dataset token indices, filter by split in ground truth
        loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
        total_tokens = loader_info['total_samples']
        
        # Use all token indices - no split-specific filtering here
        relevant_indices = torch.arange(total_tokens)  # 0, 1, 2, ..., total_tokens-1
        
    else:
        relevant_indices = filter_patches_by_image_presence(cal_indices, dataset_name, model_input_size)
    
    # Get ground truth labels from calibration set
    loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
    total_samples = loader_info['total_samples']
    
    # GLOBAL INDICES: Keep ground truth in global indices, no remapping!
    if model_input_size[0] == 'text':
        # Filter ground truth to only include tokens from calibration sentences
        filtered_gt = {}
        for concept, indices in gt_samples_per_concept_cal.items():
            # Keep original global indices - no remapping needed!
            filtered_gt[concept] = set(indices)
        
        all_concept_labels = create_binary_labels(total_samples, filtered_gt)
    else:
        all_concept_labels = create_binary_labels(total_samples, gt_samples_per_concept_cal)
    
    if use_best_detection_percentiles:
        # Process with batch optimization but NO persistent caching (to overwrite old files)
        print(f"Processing {len(invert_percentiles)} inversion percentiles...")
        
        # Load all thresholds we need
        if 'kmeans' not in con_label:
            all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
        else:
            raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
            alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
        
        # Process each inversion percentile
        pbar = tqdm(invert_percentiles, desc="Evaluating inversion thresholds")
        for invert_percentile in pbar:
            # Group concepts by detection percentile for batch processing
            concepts_by_detect = {}
            
            for concept, info in best_detection_percentiles.items():
                if concept not in concepts:
                    continue
                detect_perc = info['best_percentile']
                
                # Only process if invert >= detect
                if invert_percentile >= detect_perc:
                    if detect_perc not in concepts_by_detect:
                        concepts_by_detect[detect_perc] = []
                    concepts_by_detect[detect_perc].append(concept)
            
            # Process each detection group
            for detect_perc, concept_list in concepts_by_detect.items():
                # OPTIMIZATION: Batch process all concepts with same detection percentile
                filtered_concepts = {c: concepts[c] for c in concept_list if c in concepts}
                
                if filtered_concepts:
                    # Call detect_then_invert_metrics once for this batch
                    metrics = detect_then_invert_metrics(
                        detect_perc, [invert_percentile],
                        act_loader, filtered_concepts,
                        gt_samples_per_concept, gt_samples_per_concept_cal,
                        relevant_indices, all_concept_labels,
                        device, dataset_name, model_input_size, con_label,
                        all_object_patches=all_object_patches,
                        patch_size=patch_size,
                        embedding_loader=embedding_loader
                    )
    else:
        # Original behavior: evaluate all combinations
        total_iters = len(detect_percentiles)
        pbar = tqdm(total=total_iters, desc="Evaluating thresholds")
        
        for detect_percentile in detect_percentiles:
            # Get valid invert percentiles for this detect percentile
            valid_invert_percentiles = [p for p in invert_percentiles if p >= detect_percentile]
            
            if not valid_invert_percentiles:
                continue
            
            # Compute metrics for all valid invert percentiles at once
            metrics = detect_then_invert_metrics(
                detect_percentile, valid_invert_percentiles,
                act_loader, concepts,
                gt_samples_per_concept, gt_samples_per_concept_cal,
                relevant_indices, all_concept_labels,
                device, dataset_name, model_input_size, con_label,
                all_object_patches=all_object_patches,
                patch_size=patch_size,
                embedding_loader=embedding_loader
            )
            
            pbar.update(1)
    
    pbar.close()


def find_optimal_detect_invert_thresholds(invert_percentiles, dataset_name, 
                                        con_label, optimization_metric='f1'):
    """
    Find optimal detect/invert percentile pairs for each concept from calibration results.
    Uses detection thresholds from Best_Detection_Percentiles_Cal and finds best inversion thresholds.
    Must be run AFTER detect_then_invert_metrics_over_percentiles.
    """
    import os
    import numpy as np
    
    optimal_thresholds = {}
    results_dir = f"Quant_Results/{dataset_name}"
    
    # Load best detection percentiles from calibration (REQUIRED)
    best_detection_file = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    if not os.path.exists(best_detection_file):
        raise FileNotFoundError(
            f"Best detection percentiles file not found at {best_detection_file}\n"
            f"Please run all_detection_stats.py first to generate best detection percentiles."
        )
    
    best_detection_percentiles = torch.load(best_detection_file, weights_only=False)
    print(f"Loaded best detection percentiles from {best_detection_file}")
    
    # Get all concepts from any results file (.pt files now)
    concept_names = set()
    
    # Use concepts from best detection percentiles
    concept_names = set(best_detection_percentiles.keys())
    
    print(f"Finding optimal thresholds for {len(concept_names)} concepts using {optimization_metric}...")
    
    # For each concept, find best invert threshold using fixed detection threshold
    for concept in tqdm(concept_names, desc="Optimizing concepts"):
        best_score = -1
        best_detect = None
        best_invert = None
        
        if concept not in best_detection_percentiles:
            print(f"  Warning: Concept {concept} not found in best detection percentiles, skipping...")
            continue
            
        # Use the best detection percentile from calibration
        fixed_detect_p = best_detection_percentiles[concept]['best_percentile']
        
        # Search only over inversion thresholds >= detection threshold
        for invert_p in invert_percentiles:
            if invert_p >= fixed_detect_p:  # Valid combination
                # Only use .pt files
                pt_filename = f"{results_dir}/detectfirst_{fixed_detect_p}_invert_{invert_p}_{con_label}.pt"
                
                try:
                    if os.path.exists(pt_filename):
                        results = torch.load(pt_filename, weights_only=False)
                        if str(concept) in results and optimization_metric in results[str(concept)]:
                            score = results[str(concept)][optimization_metric]
                            # Handle NaN scores
                            if not math.isnan(score) and score > best_score:
                                best_score = score
                                best_detect = fixed_detect_p
                                best_invert = invert_p
                except Exception as e:
                    continue
        
        # Store optimal thresholds for this concept
        if best_detect is not None:
            # Load the actual threshold values used for this concept
            if 'kmeans' not in con_label:
                all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
                detect_threshold = all_thresholds[best_detect][str(concept)]
                invert_threshold = all_thresholds[best_invert][str(concept)]
            else:
                # For unsupervised concepts, load matched thresholds
                raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
                alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
                
                # Get the cluster ID for this concept
                cluster_id = alignment_results[str(concept)]['best_cluster']
                key = (str(concept), cluster_id)
                
                detect_threshold = raw_thresholds[best_detect][key] if key in raw_thresholds[best_detect] else None
                invert_threshold = raw_thresholds[best_invert][key] if key in raw_thresholds[best_invert] else None
            
            optimal_thresholds[str(concept)] = {
                'detect_percentile': best_detect,
                'invert_percentile': best_invert, 
                'detect_threshold': detect_threshold,
                'invert_threshold': invert_threshold,
                f'best_{optimization_metric}': best_score
            }
            
            if concept == 'sarcasm':
                print(f"\n🔍 DEBUG - Optimal thresholds for '{concept}':")
                print(f"   Best detection percentile: {best_detect}")
                print(f"   Best inversion percentile: {best_invert}")
                print(f"   Best {optimization_metric} score: {best_score:.4f}")
        else:
            print(f"  Warning: No valid thresholds found for concept {concept}")
    
    # Save optimal thresholds to both locations
    # Save inversion percentiles to Best_Inversion_Percentiles_Cal
    os.makedirs(f'Best_Inversion_Percentiles_Cal/{dataset_name}', exist_ok=True)
    best_inversion_percentiles = {}
    for concept, info in optimal_thresholds.items():
        best_inversion_percentiles[concept] = {
            'best_percentile': info['invert_percentile'],
            'best_threshold': info['invert_threshold'],
            f'best_{optimization_metric}': info[f'best_{optimization_metric}']
        }
    inversion_file = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_{con_label}.pt'
    torch.save(best_inversion_percentiles, inversion_file)
    print(f"  Saved best inversion percentiles to: {inversion_file}")
    
    # Print summary
    if optimal_thresholds:
        detect_percs = [v['detect_percentile'] for v in optimal_thresholds.values()]
        invert_percs = [v['invert_percentile'] for v in optimal_thresholds.values()]
        scores = [v[f'best_{optimization_metric}'] for v in optimal_thresholds.values()]
        
        print(f"\nOptimization Summary ({len(optimal_thresholds)} concepts):")
        detect_tensor = torch.tensor(detect_percs)
        invert_tensor = torch.tensor(invert_percs)
        scores_tensor = torch.tensor(scores)
        
        if len(optimal_thresholds) > 1:
            print(f"  Avg detect percentile: {detect_tensor.mean():.3f} ± {detect_tensor.std():.3f}")
            print(f"  Avg invert percentile: {invert_tensor.mean():.3f} ± {invert_tensor.std():.3f}")
            print(f"  Avg {optimization_metric}: {scores_tensor.mean():.3f} ± {scores_tensor.std():.3f}")
        else:
            # For single concept, don't show std
            print(f"  Detect percentile: {detect_tensor.mean():.3f}")
            print(f"  Invert percentile: {invert_tensor.mean():.3f}")
            print(f"  {optimization_metric}: {scores_tensor.mean():.3f}")
        print(f"  Saved to: {inversion_file}")
    
    return optimal_thresholds


def detect_then_invert_with_optimal_thresholds(
    act_loader: Union[MatchedConceptActivationLoader, ChunkedActivationLoader],
    concepts: Dict,
    gt_samples_per_concept: Dict,
    gt_samples_per_concept_test: Dict,
    device: str,
    dataset_name: str,
    model_input_size: Tuple,
    con_label: str,
    optimization_metric: str = 'f1',
    embedding_loader=None
):
    """
    Memory-efficient version using activation loader.
    Evaluate detection and inversion with optimal thresholds on TEST set.
    Must be run AFTER find_optimal_detect_invert_thresholds.
    
    Args:
        act_loader: Activation loader
        concepts: Concept vectors
        gt_samples_per_concept: Ground truth samples per concept
        gt_samples_per_concept_test: Test set ground truth samples
        device: Compute device
        dataset_name: Dataset name
        model_input_size: Model input dimensions
        con_label: Concept label
        optimization_metric: Metric used for optimization
        embedding_loader: ChunkedEmbeddingLoader instance for computing cosine similarities
    """
    # Load optimal inversion thresholds
    optimal_path = f'Best_Inversion_Percentiles_Cal/{dataset_name}/best_inversion_percentiles_{con_label}.pt'
    optimal_inversion_thresholds = torch.load(optimal_path, weights_only=False)
    
    # Load detection thresholds  
    detection_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    optimal_detection_thresholds = torch.load(detection_path, weights_only=False)
    
    # Get unique percentile combinations
    unique_combinations = set()
    for concept in optimal_inversion_thresholds.keys():
        if concept in optimal_detection_thresholds:
            detect_perc = optimal_detection_thresholds[concept]['best_percentile']
            invert_perc = optimal_inversion_thresholds[concept]['best_percentile']
            unique_combinations.add((detect_perc, invert_perc))
    
    print(f"Found {len(unique_combinations)} unique threshold combinations to evaluate")
    
    # MATCH CALIBRATION APPROACH: Use global indices for test set too
    from utils.general_utils import get_split_df
    split_df = get_split_df(dataset_name)
    
    # For text datasets, get test sentence indices
    if model_input_size[0] == 'text':
        token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
        token_counts_per_sentence = torch.load(token_counts_file, weights_only=False)
        test_sentence_indices = [i for i in range(len(token_counts_per_sentence)) if split_df.get(i) == 'test']
        test_indices = torch.tensor(test_sentence_indices)
        
        # GLOBAL INDICES APPROACH: Use ALL dataset token indices, filter by split in ground truth
        loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
        total_tokens = loader_info['total_samples']
        relevant_indices = torch.arange(total_tokens)  # Use all token indices
        
    else:
        # For image datasets, use patch-based approach  
        split_df_patch = get_patch_split_df(dataset_name, model_input_size=model_input_size, patch_size=14)
        test_indices = torch.tensor(split_df_patch.index[split_df_patch == 'test'].tolist())
        relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    
    # Get ground truth labels from test set
    loader_info = act_loader.get_activation_info() if hasattr(act_loader, 'get_activation_info') else act_loader.get_info()
    total_samples = loader_info['total_samples']
    
    # GLOBAL INDICES: Keep ground truth in global indices, no remapping!
    if model_input_size[0] == 'text':
        filtered_gt = {}
        for concept, indices in gt_samples_per_concept_test.items():
            # Keep original global indices - no remapping needed!
            filtered_gt[concept] = set(indices)
        all_concept_labels = create_binary_labels(total_samples, filtered_gt)
    else:
        all_concept_labels = create_binary_labels(total_samples, gt_samples_per_concept_test)
    
    # Process each unique combination
    all_results = {}
    
    for detect_per, invert_per in tqdm(unique_combinations, desc="Evaluating optimal thresholds"):
        # Load thresholds for this combination
        if 'kmeans' not in con_label:
            all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
        else:
            # Handle kmeans thresholds
            raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
            alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
            
            all_thresholds = {}
            for percentile, thresholds_dict in raw_thresholds.items():
                matched_thresholds = {}
                for concept, info in alignment_results.items():
                    cluster_id = info['best_cluster']
                    key = (concept, cluster_id)
                    if key in thresholds_dict:
                        matched_thresholds[concept] = thresholds_dict[key]
                all_thresholds[percentile] = matched_thresholds
        
        detect_thresholds = all_thresholds[detect_per]
        invert_thresholds = all_thresholds[invert_per]
        
        # Get concepts that use this combination
        concepts_for_combo = [c for c in optimal_inversion_thresholds.keys()
                             if c in optimal_detection_thresholds and 
                             optimal_detection_thresholds[c]['best_percentile'] == detect_per and
                             optimal_inversion_thresholds[c]['best_percentile'] == invert_per]
        
        # Filter thresholds and concepts
        filtered_detect = {c: detect_thresholds[c] for c in concepts_for_combo if c in detect_thresholds}
        filtered_invert = {c: invert_thresholds[c] for c in concepts_for_combo if c in invert_thresholds}
        filtered_concepts = {c: concepts[c] for c in concepts_for_combo if c in concepts}
        
        if not filtered_detect:
            continue
        
        # Compute metrics using loader - SPECIFY TEST SPLIT
        results = detect_then_invert_metrics(
            detect_per, [invert_per],
            act_loader, filtered_concepts,
            gt_samples_per_concept, gt_samples_per_concept_test,
            relevant_indices, all_concept_labels,
            device, dataset_name, model_input_size, con_label,
            patch_size=14,
            embedding_loader=embedding_loader,
            split='test'
        )
        
        # Store results
        for concept in concepts_for_combo:
            if concept in results.get(invert_per, {}):
                all_results[concept] = results[invert_per][concept]
                all_results[concept]['detect_percentile'] = detect_per
                all_results[concept]['invert_percentile'] = invert_per
    
    # Save final results
    save_path = f'Quant_Results/{dataset_name}/optimal_test_results_{con_label}_{optimization_metric}.pt'
    torch.save(all_results, save_path)
    
    # Print summary statistics
    if all_results:
        avg_precision = np.mean([r['precision'] for r in all_results.values()])
        avg_recall = np.mean([r['recall'] for r in all_results.values()])
        avg_f1 = np.mean([r['f1'] for r in all_results.values()])
        
        print(f"\nTest Set Results with Optimal Thresholds:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Results saved to: {save_path}")
    else:
        print("No results computed")

    
def detect_then_invert_performance_heatmap(metric_name, gt_samples_per_concept_test, dataset_name, con_label, 
                                           detect_percentiles, invert_percentiles, just_obj=False):
    """
    Plots a triangular heatmap of a selected metric over detect/invert percentile combinations.
    Only (invert > detect) regions are shown. Invert percentiles are ordered top-down.

    Args:
        metric_name (str): Metric to visualize (e.g., 'f1', 'accuracy', 'fpr').
        dataset_name (str): Dataset name used in filenames.
        con_label (str): Concept label identifier.
        detect_percentiles (list of float): List of detect percentiles.
        invert_percentiles (list of float): List of invert percentiles.
    """
    prefix = "" if not just_obj else "justobj_"
    heatmap_data = []
    mask_data = []

    # Reverse the invert percentiles for top-down visualization
    invert_percentiles = list(sorted(invert_percentiles, reverse=True))

    for invert_p in invert_percentiles:
        row = []
        mask_row = []
        for detect_p in detect_percentiles:
            if invert_p >= detect_p:
                filename = f"Quant_Results/{dataset_name}/{prefix}detectfirst_{detect_p*100}_per_{invert_p*100}_{con_label}.csv"
                try:
                    df = pd.read_csv(filename)
                    df['concept'] = df['concept'].astype(str)
                    df = df[df['concept'].isin(list(gt_samples_per_concept_test.keys()))]
                    if metric_name in df.columns:
                        value = np.average(df[metric_name], weights=[len(gt_samples_per_concept_test[c]) for c in df['concept']]) #weight by freq
                    else:
                        value = np.nan
                except FileNotFoundError:
                    print(f"Missing file: {filename}")
                    value = np.nan
                mask_row.append(False)
            else:
                value = np.nan
                mask_row.append(True)

            row.append(value)
        heatmap_data.append(row)
        mask_data.append(mask_row)

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=[f"{p:.2f}" for p in invert_percentiles],
        columns=[f"{p:.2f}" for p in detect_percentiles]
    )

    mask = np.array(mask_data)

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".2f", 
        cmap="plasma", 
        cbar_kws={"label": metric_name},
        mask=mask,
        vmin=0, vmax=1
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Find max value and location
    max_val = np.nanmax(heatmap_df.values)
    if not np.isnan(max_val):
        max_idx = np.unravel_index(np.nanargmax(heatmap_df.values), heatmap_df.shape)
        max_detect = heatmap_df.columns[max_idx[1]]
        max_invert = heatmap_df.index[max_idx[0]]
        max_label = f" (Max: {max_val:.2f} @ detect={max_detect}, invert={max_invert})"
    else:
        max_label = ""

    title = f"{metric_name} over Detect/Inversion Percentiles{max_label}"
    if just_obj:
        title += " (Just Obj Patches)"
    plt.title(title, pad=10)

    plt.ylabel("Invert Percentile")
    plt.xlabel("Detect Percentile")
    plt.tight_layout()
    plt.show()
    
    
def detect_then_invert_performance_heatmap_per_concept(metric_name, gt_samples_per_concept_test, dataset_name, con_label, 
                                                       detect_percentiles, invert_percentiles, just_obj=False):
    """
    Plots a triangular heatmap of a selected metric over detect/invert percentile combinations,
    for each individual concept. Only (invert > detect) regions are shown.

    Args:
        metric_name (str): Metric to visualize (e.g., 'f1', 'accuracy', 'fpr').
        gt_samples_per_concept_test (dict): Mapping from concept name to number of test samples.
        dataset_name (str): Dataset name used in filenames.
        con_label (str): Concept label identifier.
        detect_percentiles (list of float): List of detect percentiles.
        invert_percentiles (list of float): List of invert percentiles.
        just_obj (bool): Whether to use the "justobj_" prefix in file paths.
    """
    prefix = "" if not just_obj else "justobj_"
    invert_percentiles = list(sorted(invert_percentiles, reverse=True))  # top-down heatmap

    all_concepts = list(gt_samples_per_concept_test.keys())

    for concept in all_concepts:
        heatmap_data = []
        mask_data = []

        for invert_p in invert_percentiles:
            row = []
            mask_row = []
            for detect_p in detect_percentiles:
                if invert_p >= detect_p:
                    filename = f"Quant_Results/{dataset_name}/{prefix}detectfirst_{detect_p*100}_per_{invert_p*100}_{con_label}.csv"
                    try:
                        df = pd.read_csv(filename)
                        concept_row = df[df['concept'] == concept]
                        if metric_name in concept_row.columns and not concept_row.empty:
                            value = concept_row[metric_name].values[0]
                        else:
                            value = np.nan
                    except FileNotFoundError:
                        print(f"Missing file: {filename}")
                        value = np.nan
                    mask_row.append(False)
                else:
                    value = np.nan
                    mask_row.append(True)
                row.append(value)
            heatmap_data.append(row)
            mask_data.append(mask_row)

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[f"{p:.2f}" for p in invert_percentiles],
            columns=[f"{p:.2f}" for p in detect_percentiles]
        )

        mask = np.array(mask_data)

        # Plotting
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            heatmap_df, 
            annot=True, 
            fmt=".2f", 
            cmap="plasma", 
            cbar_kws={"label": metric_name},
            mask=mask,
            vmin=0, vmax=1
        )
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        # Find max value and location
        max_val = np.nanmax(heatmap_df.values)
        if not np.isnan(max_val):
            max_idx = np.unravel_index(np.nanargmax(heatmap_df.values), heatmap_df.shape)
            max_detect = heatmap_df.columns[max_idx[1]]
            max_invert = heatmap_df.index[max_idx[0]]
            max_label = f" (Max: {max_val:.2f} @ detect={max_detect}, invert={max_invert})"
        else:
            max_label = ""

        title_prefix = "Just Obj Patches" if just_obj else ""
        plt.title(f"{metric_name} Heatmap - Concept: {concept} {title_prefix}{max_label}", pad=10)
        plt.ylabel("Invert Percentile")
        plt.xlabel("Detect Percentile")
        plt.tight_layout()
        plt.show()
        


def evaluate_baseline_models_across_dataset(gt_samples_per_concept, dataset_name, sample_type, model_input_size, 
                                             patch_size=14, n_trials=1):
    """
    Evaluate baseline predictions (Always Yes, Always No, and Random) across a dataset.
    For each concept, computes True Positives (TP), False Negatives (FN), False Positives (FP), 
    and True Negatives (TN) for three baseline prediction strategies, averaging over n_trials.
    
    Baselines:
        - Always Yes: Predict 1 for every patch.
        - Always No: Predict 0 for every patch.
        - Random: Randomly predict 0 or 1 for each patch (with p=0.5 each).
    
    Args:
        gt_samples_per_concept (dict): Mapping from concept to list of ground truth patch indices.
        dataset_name (str): Name of the dataset.
        sample_type (str): Type of sample ('patch' or 'image').
        model_input_size (tuple): The final padded size in pixels (e.g., (560,560)).
        patch_size (int): Size of each patch (assumed square).
        all_object_patches (set, optional): If provided, only consider these patch indices.
        balance_dataset (bool): Whether to balance positive and negative test samples.
        n_trials (int): Number of trials to average over.
        
    Returns:
        dict: A dictionary with keys 'always_yes', 'always_no', and 'random'. Each value is a dictionary
              mapping concept -> (avg_tp, avg_fn, avg_fp, avg_tn).
    """
    # For reproducibility
    rng = torch.Generator()
    rng.manual_seed(42)
    
    # Dictionaries to accumulate counts per concept for each baseline.
    always_yes_tp = defaultdict(list)
    always_yes_fn = defaultdict(list)
    always_yes_fp = defaultdict(list)
    always_yes_tn = defaultdict(list)
    
    always_no_tp = defaultdict(list)
    always_no_fn = defaultdict(list)
    always_no_fp = defaultdict(list)
    always_no_tn = defaultdict(list)
    
    random_tp = defaultdict(list)
    random_fn = defaultdict(list)
    random_fp = defaultdict(list)
    random_tn = defaultdict(list)
    
    # Get the split dataframe.
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    elif sample_type == 'cls':
        split_df = get_split_df(dataset_name)
    
    # Get test indices as a torch tensor.
    test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    
    # Filter patches to only those that are relevant given the preprocessing scheme.
    relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    
    # Get ground truth labels for all concepts.
    all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept)
    
    # Loop over each concept.
    for concept, concept_labels in tqdm(all_concept_labels.items()):
        # Process each concept for n_trials.
        for trial in range(n_trials):
            # Get the ground truth labels for relevant patches.
            relevant_labels = concept_labels[relevant_indices]
            
            # Ground truth mask (1 for yes, 0 for no).
            gt_mask = (relevant_labels == 1)
            
            # Baseline Predictions:
            # Always Yes: Predict 1 for every patch.
            pred_always_yes = torch.ones_like(relevant_labels, dtype=torch.bool)
            # Always No: Predict 0 for every patch.
            pred_always_no = torch.zeros_like(relevant_labels, dtype=torch.bool)
            # Random: Predict 1 or 0 randomly with probability 0.5.
            pred_random = torch.bernoulli(torch.full(relevant_labels.shape, 0.5, dtype=torch.float)).bool()
            
            # Compute confusion matrix counts.
            # For Always Yes:
            tp = torch.sum(pred_always_yes & gt_mask).item()
            fn = torch.sum((~pred_always_yes) & gt_mask).item()
            fp = torch.sum(pred_always_yes & (~gt_mask)).item()
            tn = torch.sum((~pred_always_yes) & (~gt_mask)).item()
            always_yes_tp[concept].append(tp)
            always_yes_fn[concept].append(fn)
            always_yes_fp[concept].append(fp)
            always_yes_tn[concept].append(tn)
            
            # For Always No:
            tp = torch.sum(pred_always_no & gt_mask).item()
            fn = torch.sum((~pred_always_no) & gt_mask).item()
            fp = torch.sum(pred_always_no & (~gt_mask)).item()
            tn = torch.sum((~pred_always_no) & (~gt_mask)).item()
            always_no_tp[concept].append(tp)
            always_no_fn[concept].append(fn)
            always_no_fp[concept].append(fp)
            always_no_tn[concept].append(tn)
            
            # For Random:
            tp = torch.sum(pred_random & gt_mask).item()
            fn = torch.sum((~pred_random) & gt_mask).item()
            fp = torch.sum(pred_random & (~gt_mask)).item()
            tn = torch.sum((~pred_random) & (~gt_mask)).item()
            random_tp[concept].append(tp)
            random_fn[concept].append(fn)
            random_fp[concept].append(fp)
            random_tn[concept].append(tn)
    
    # For each baseline, compute the average for each concept.
    avg_always_yes_fp = {k: sum(always_yes_fp[k]) / len(always_yes_fp[k]) for k in always_yes_fp}
    avg_always_yes_fn = {k: sum(always_yes_fn[k]) / len(always_yes_fn[k]) for k in always_yes_fn}
    avg_always_yes_tp = {k: sum(always_yes_tp[k]) / len(always_yes_tp[k]) for k in always_yes_tp}
    avg_always_yes_tn = {k: sum(always_yes_tn[k]) / len(always_yes_tn[k]) for k in always_yes_tn}

    avg_always_no_fp = {k: sum(always_no_fp[k]) / len(always_no_fp[k]) for k in always_no_fp}
    avg_always_no_fn = {k: sum(always_no_fn[k]) / len(always_no_fn[k]) for k in always_no_fn}
    avg_always_no_tp = {k: sum(always_no_tp[k]) / len(always_no_tp[k]) for k in always_no_tp}
    avg_always_no_tn = {k: sum(always_no_tn[k]) / len(always_no_tn[k]) for k in always_no_tn}

    avg_random_fp = {k: sum(random_fp[k]) / len(random_fp[k]) for k in random_fp}
    avg_random_fn = {k: sum(random_fn[k]) / len(random_fn[k]) for k in random_fn}
    avg_random_tp = {k: sum(random_tp[k]) / len(random_tp[k]) for k in random_tp}
    avg_random_tn = {k: sum(random_tn[k]) / len(random_tn[k]) for k in random_tn}
    
    results = {
    'always_yes': (avg_always_yes_fp, avg_always_yes_fn, avg_always_yes_tp, avg_always_yes_tn),
    'always_no': (avg_always_no_fp, avg_always_no_fn, avg_always_no_tp, avg_always_no_tn),
    'random':    (avg_random_fp, avg_random_fn, avg_random_tp, avg_random_tn)
    }
    return results


def compute_stats_from_counts(tp_count, fp_count, tn_count, fn_count):
    metrics = []
    
    for concept in tp_count.keys():
        # Retrieve counts for each concept
        tp = tp_count[concept]
        fp = fp_count[concept]
        tn = tn_count[concept]
        fn = fn_count[concept]
        
        # Compute precision, recall, accuracy, f1-score, fpr, tpr, tnr, fnr
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Add metrics to the list
        metrics.append({
            "concept": concept,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "fpr": fpr,
            "tpr": tpr,
            "tnr": tnr,
            "fnr": fnr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,  
        })
    
    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


def compute_concept_metrics(fp_count, fn_count, tp_count, tn_count, concepts, dataset_name, con_label, invert_percentile=None, just_obj=False, baseline_type=None, detect_percentile=None, save_label=None):
    metrics_df = compute_stats_from_counts(tp_count, fp_count, tn_count, fn_count)
    
    # Use save_label for file naming if provided, otherwise use con_label
    if save_label is None:
        save_label = con_label
    
    # Add concept column with the keys from the count dictionaries
    metrics_df['concept'] = list(tp_count.keys())
    
    # Map cluster IDs to concept names for unsupervised methods
    if 'kmeans' in con_label:
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
        cluster_to_concept = {str(info['best_cluster']): concept_name for concept_name, info in alignment_results.items()}
        metrics_df['concept'] = metrics_df['concept'].map(lambda x: cluster_to_concept.get(str(x), str(x)))
    
    # Save the DataFrame as a CSV file
    if just_obj:
        if baseline_type:
            if 'inversion_' in baseline_type and invert_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/{baseline_type}_justobj_per_{invert_percentile*100}.csv'
            else:
                save_path = f'Quant_Results/{dataset_name}/{baseline_type}_justobj_{save_label}.csv'
        else:
            if detect_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/justobj_detectfirst_{detect_percentile*100}_per_{invert_percentile*100}_{save_label}.csv'
            else:
                save_path = f'Quant_Results/{dataset_name}/justobj_per_{invert_percentile*100}_{save_label}.csv'
    else:
        if baseline_type:
            if 'inversion_' in baseline_type and invert_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/{baseline_type}_per_{invert_percentile*100}.csv'
            else:
                save_path = f'Quant_Results/{dataset_name}/{baseline_type}_{save_label}.csv'
        else:
            if detect_percentile is not None and invert_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/detectfirst_{detect_percentile*100}_per_{invert_percentile*100}_{save_label}.csv'
            elif invert_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/per_{invert_percentile*100}_{save_label}.csv'
            else:
                save_path = f'Quant_Results/{dataset_name}/{save_label}.csv'


    metrics_df.to_csv(save_path, index=False)
#     print(f"Metrics saved to {save_path} :)")
    
    return metrics_df


def inversion_baselines(
    dataset_name,
    model_input_size,
    con_label,
    device,
    patch_size=14
):
    """
    Compute inversion baseline metrics for random, always positive, and always negative predictions
    at the patch level. Handles data loading and filtering internally.
    
    Args:
        dataset_name: Name of dataset
        model_input_size: Model input dimensions
        con_label: Concept label (e.g., 'CLIP_patch')
        device: Torch device
        patch_size: Patch size for vision models
    """
    print(f"Computing inversion baselines for {dataset_name}")
    
    # Load ALL gt_patches (not just test) to create complete labels
    all_gt_patches_file = f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt'
    concepts = torch.load(all_gt_patches_file, weights_only=False)
    print(f"Loaded ALL gt_patches with {len(concepts)} concepts")
    
    # Determine total dataset size D for create_binary_labels
    if model_input_size[0] == 'text':
        # For text models, get total number of tokens
        # Load model-specific token counts
        token_counts = torch.load(f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt')
        D = sum(sum(x) for x in token_counts)
    else:
        # For vision models, calculate total patches from max patch index
        all_patch_indices = set()
        for concept_patches in concepts.values():
            all_patch_indices.update(concept_patches)
        D = max(all_patch_indices) + 1 if all_patch_indices else 0
    
    print(f"Total dataset size D = {D}")
    
    all_concept_labels = create_binary_labels(D, concepts)
    print(f"Created labels for all patches: {[f'{concept}: {labels.sum().item()}/{len(labels)} positive' for concept, labels in list(all_concept_labels.items())[:3]]}")
    
    # Get patch split information and filter to test split only
    split_df = get_patch_split_df(dataset_name, model_input_size, patch_size=patch_size)
    print(f"Loaded split info for {len(split_df)} patches/tokens")
    
    # Filter to test split only
    test_mask = split_df == 'test'
    test_indices = test_mask[test_mask].index
    
    # Further filter to exclude padding patches (for vision models)
    if model_input_size[0] != 'text':
        relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    else:
        relevant_indices = torch.tensor(test_indices)
    
    print(f"Using {len(relevant_indices)} test patches/tokens (after filtering)")
    
    relevant_indices_list = relevant_indices.tolist()
    concept_keys = list(concepts.keys())
    n_concepts = len(concept_keys)
    n_samples = len(relevant_indices)
    
    # Prepare ground truth masks
    gt_masks_all = torch.zeros((n_samples, n_concepts), dtype=torch.bool, device=device)
    for i, concept in enumerate(concept_keys):
        vals = all_concept_labels[concept][relevant_indices_list]
        if isinstance(vals, torch.Tensor):
            mask = (vals == 1).clone().detach().to(device)
        else:
            mask = torch.tensor(vals == 1, device=device)
        gt_masks_all[:, i] = mask
    
    # Baseline types to compute
    baseline_types = ['random', 'always_positive', 'always_negative']
    
    for baseline_type in baseline_types:
        print(f"Computing {baseline_type} baseline...")
        
        # Generate baseline predictions based on type
        if baseline_type == 'random':
            # Random 50/50 predictions
            activated_patches = torch.rand((n_samples, n_concepts), device=device) < 0.5
        elif baseline_type == 'always_positive':
            # All predictions above threshold (all positive)
            activated_patches = torch.ones((n_samples, n_concepts), dtype=torch.bool, device=device)
        elif baseline_type == 'always_negative':
            # All predictions below threshold (all negative)
            activated_patches = torch.zeros((n_samples, n_concepts), dtype=torch.bool, device=device)
        
        # Compute confusion matrix elements
        tp_counts = torch.sum(activated_patches & gt_masks_all, dim=0)
        fn_counts = torch.sum((~activated_patches) & gt_masks_all, dim=0)
        fp_counts = torch.sum(activated_patches & (~gt_masks_all), dim=0)
        tn_counts = torch.sum((~activated_patches) & (~gt_masks_all), dim=0)
        
        # Convert to dictionary format
        tp_count = {concept_keys[i]: tp_counts[i].item() for i in range(n_concepts)}
        fn_count = {concept_keys[i]: fn_counts[i].item() for i in range(n_concepts)}
        fp_count = {concept_keys[i]: fp_counts[i].item() for i in range(n_concepts)}
        tn_count = {concept_keys[i]: tn_counts[i].item() for i in range(n_concepts)}
        
        # Compute metrics using existing function
        baseline_name = f"inversion_{baseline_type}_{con_label}_baseline"
        compute_concept_metrics(
            fp_count, fn_count, tp_count, tn_count,
            concept_keys, dataset_name, con_label,
            just_obj=False,
            baseline_type=baseline_name
        )
        
        print(f"✓ Completed {baseline_type} baseline for {dataset_name}")
    
    print(f"✓ All inversion baselines completed for {dataset_name}")


def compute_metrics_across_percentiles(gt_patches_per_concept_test, concepts, sim_metrics, model_input_size, dataset_name, 
                                       device, con_label, sample_type='patch', 
                                       percentiles=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95, 1.0],
                                      all_object_patches=None):
    """Computes metrics across dataset using different thresholds"""
    if 'kmeans' not in con_label:
        all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
    else:
        # Load files
        raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)

        # Collect matched thresholds per percentile
        all_thresholds = {}

        for percentile, thresholds_dict in raw_thresholds.items():
            matched_thresholds = {}

            for concept, info in alignment_results.items():
                cluster_id = info['best_cluster']
                key = (concept, cluster_id)

                if key in thresholds_dict:
                    matched_thresholds[cluster_id] = thresholds_dict[key]  # keep full (val, nan) tuple

            all_thresholds[percentile] = matched_thresholds
            

    for percentile in tqdm(percentiles):
#         try:
#             metrics_df = torch.load(f'Quant_Results/{dataset_name}/per_{percentile*100}_{con_label}.csv')
# #             if set(metrics_df.columns) != set(concepts.keys()):
# #                 filtered_df = metrics_df[[col for col in metrics_df.columns if col in concepts.keys()]]
# #                 torch.save(filtered_df, f'Quant_Results/{dataset_name}/per_{percentile*100}_{con_label}.csv')
            
#         except:
        # concept_thresholds = compute_concept_thresholds(gt_patches_per_concept_test, 
        #                                             sim_metrics, percentile, n_vectors=1, device=device, 
        #                                             n_concepts_to_print=0, dataset_name=f'{dataset_name}-Cal',
        #                                                 con_label=con_label)
        concept_thresholds = all_thresholds[percentile]
        fp_count, fn_count, tp_count, tn_count = evaluate_thresholds_across_dataset(concept_thresholds, 
                                                                                gt_patches_per_concept_test, 
                                                                                sim_metrics, model_input_size=model_input_size,
                                                                                dataset_name=dataset_name,
                                                                               sample_type=sample_type,
                                                                                all_object_patches=None)
        metrics_df = compute_concept_metrics(fp_count, fn_count, tp_count, tn_count, concepts,
                                        dataset_name, con_label, percentile, just_obj=False)

        # try:
        #     metrics_df = torch.load(f'Quant_Results/{dataset_name}/justobj_per_{percentile*100}_{con_label}.csv')
        #     # if set(metrics_df.columns) != set(concepts.keys()):
        #     #     filtered_df = metrics_df[[col for col in metrics_df.columns if col in concepts.keys()]]
        #     #     torch.save(filtered_df, f'Quant_Results/{dataset_name}/justobj_per_{percentile*100}_{con_label}.csv')
        # except:
        #     concept_thresholds = compute_concept_thresholds(gt_patches_per_concept_test, 
        #                                                 sim_metrics, percentile, n_vectors=1, device=device, 
        #                                                 n_concepts_to_print=0, dataset_name=f'{dataset_name}-Cal',
        #                                                     con_label=con_label)
        #     fp_count, fn_count, tp_count, tn_count = evaluate_thresholds_across_dataset(concept_thresholds, 
        #                                                                             gt_patches_per_concept_test, 
        #                                                                             sim_metrics, model_input_size=model_input_size,
        #                                                                             dataset_name=dataset_name,
        #                                                                            sample_type=sample_type,
        #                                                                             all_object_patches=all_object_patches)
        #     metrics_df = compute_concept_metrics(fp_count, fn_count, tp_count, tn_count, concepts,
        #                                     dataset_name, con_label, percentile, just_obj=True)

            
def print_threshold_eval_results(metrics_df, print_types):
    """
    Print metrics such as counts, rates, and summaries from the DataFrame.
    """
    # Print per-concept metrics
    if 'rate' in print_types:
        for _, row in metrics_df.iterrows():
            print(f"Concept: {row['concept']}")
            print(f"TPR: {row['tpr']:.4f}, FPR: {row['fpr']:.4f}, TNR: {row['tnr']:.4f}, FNR: {row['fnr']:.4f}\n")
    
    if 'count' in print_types:
        for _, row in metrics_df.iterrows():
            print(f"Concept: {row['concept']}")
            print(f"TP: {row['precision']:.4f}, FP: {row['fpr']:.4f}, TN: {row['tnr']:.4f}, FN: {row['fnr']:.4f}\n")
    
    # Print summary statistics if enabled
    if 'summary' in print_types:
        # Top and Bottom Concepts for Precision
        top_precision = metrics_df.sort_values(by="precision", ascending=False).head(5)[["concept", "precision"]]
        bottom_precision = metrics_df.sort_values(by="precision", ascending=True).head(5)[["concept", "precision"]]

        # Top and Bottom Concepts for Recall
        top_recall = metrics_df.sort_values(by="recall", ascending=False).head(5)[["concept", "recall"]]
        bottom_recall = metrics_df.sort_values(by="recall", ascending=True).head(5)[["concept", "recall"]]
        
        # Top and Bottom Concepts for F1
        top_f1 = metrics_df.sort_values(by="f1", ascending=False).head(5)[["concept", "f1"]]
        bottom_f1 = metrics_df.sort_values(by="f1", ascending=True).head(5)[["concept", "f1"]]

        # Top and Bottom Concepts for FPR
        top_fpr = metrics_df.sort_values(by="fpr", ascending=True).head(5)[["concept", "fpr"]]
        bottom_fpr = metrics_df.sort_values(by="fpr", ascending=False).head(5)[["concept", "fpr"]]

        # Displaying them side by side for each metric
        print("\nBest and Worst 5 Concepts by Precision (how many of the predicted positives are actually correct):")
        print(pd.concat([top_precision.reset_index(drop=True), bottom_precision.reset_index(drop=True)], axis=1))

        print("\nBest and Worst 5 Concepts by Recall (how many of the actual positives were correctly identified):")
        print(pd.concat([top_recall.reset_index(drop=True), bottom_recall.reset_index(drop=True)], axis=1))
        
        print("\nBest and Worst 5 Concepts by F1 (harmonic mean of precision and recall):")
        print(pd.concat([top_f1.reset_index(drop=True), bottom_f1.reset_index(drop=True)], axis=1))

        print("\nBest and Worst 5 Concepts by FPR (how many of the actual negatives were incorrectly predicted as positives):")
        print(pd.concat([top_fpr.reset_index(drop=True), bottom_fpr.reset_index(drop=True)], axis=1))


def plot_metric(df, metric, y_min=None, y_max=None):
    """
    Plots either a distribution of a given metric for all concepts
    or individual bars for each concept's metric value.

    Args:
        df (pd.DataFrame): DataFrame containing concept metrics.
        metric (str): The metric to plot (e.g., 'precision', 'recall', 'f1', etc.).
        plot_individual (bool): Whether to plot individual concept bars (True) or a distribution (False).
    """
    # Bar plot for individual concepts, sorted by metric
    sorted_df = df.sort_values(by=metric, ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='concept', y=metric, data=sorted_df, palette='viridis')
    plt.xticks(rotation=90)  # Rotate the concept names for better visibility
    plt.title(f'{metric.capitalize()} for Each Concept')
    plt.xlabel('Concept')
    plt.ylabel(f'{metric.capitalize()}')
    
    # Apply y-axis limit if specifieddf
    if y_max is not None:
        if y_min is not None:
            plt.ylim(y_min, y_max)
        else:
            plt.ylim(0, y_max)
        
    plt.show()
        
def plot_metric_distribution(df, metric):
    """
    Plots either a distribution of a given metric.

    Args:
        df (pd.DataFrame): DataFrame containing concept metrics.
        metric (str): The metric to plot (e.g., 'precision', 'recall', 'f1', etc.).
        plot_individual (bool): Whether to plot individual concept bars (True) or a distribution (False).
    """
    # Distribution plot for the selected metric across all concepts
    plt.figure(figsize=(12, 8))
    sns.histplot(df[metric], bins=20, color='purple')
    plt.title(f'Distribution of {metric.capitalize()} Across Concepts')
    plt.xlabel(f'{metric.capitalize()}')
    plt.ylabel('Number of Concepts')
    plt.show()


def compute_avg_rand_mean_and_std(embeddings, patch_indices, n_vectors=5, device='cuda'):
    embeddings = embeddings.to(device)

    # Generate n_vectors random vectors and normalize them
    random_vectors = torch.randn(n_vectors, embeddings.shape[1], device=device, dtype=embeddings.dtype)
    random_vectors = F.normalize(random_vectors, p=2, dim=1)  # Normalize each random vector

    # Normalize embeddings before computing cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarities directly between embeddings and random vectors
    cos_sim_matrix = torch.matmul(embeddings, random_vectors.t())  # [n_samples, n_vectors]

    # Select relevant cosine similarities for the given patch indices
    relevant_rand_cos_sims = cos_sim_matrix[patch_indices]

    # Compute mean and std for each patch across all random vectors
    rand_means = relevant_rand_cos_sims.mean(dim=1)  # Mean across random vectors
    rand_stds = relevant_rand_cos_sims.std(dim=1)    # Std across random vectors

    # Calculate the average mean and std over all patches
    avg_mean = rand_means.mean().item()  # Averaging over all patches
    avg_std = rand_stds.mean().item()    # Averaging over all patches

    return avg_mean, avg_std


def compute_concept_cosine_stats(gt_patches_per_concept, cos_sims, embeddings, results_to_print=0, device='cuda', print_random=True):
    """
    Computes the mean and standard deviation of cosine similarities for each concept
    based on the patches that are known to contain the concept (using object masks).
    """
    if results_to_print > 0:
        print(f"Mean and Std of Cossims:")

    # Step 2: Initialize dictionary to store mean and std for each concept
    concept_cosine_stats = {}

    # Step 3: Convert cos_sims DataFrame to tensor
    cos_sims_tensor = torch.tensor(cos_sims.values, device=device)  # Convert the DataFrame to a tensor
    cos_sims_tensor = cos_sims_tensor.float()  # Ensure it's of float type (important for cos similarity)

    # Step 4: Calculate cosine similarities between embeddings and concepts
    for i, (concept, patch_indices) in enumerate(gt_patches_per_concept.items()):
        # Use precomputed cos_sims_tensor to extract relevant cosine similarities
        relevant_cos_sims = cos_sims_tensor[patch_indices, cos_sims.columns.get_loc(str(concept))]

        # Compute mean and standard deviation for the relevant cosine similarities
        mean_sim = relevant_cos_sims.mean().item()
        std_sim = relevant_cos_sims.std().item()

        # Do the same thing for a random vector (average over n_vectors)
        rand_mean_sim, rand_std_sim = compute_avg_rand_mean_and_std(embeddings, patch_indices, n_vectors=5, device=device)

        # Store the results in the dictionary
        concept_cosine_stats[concept] = (mean_sim, std_sim, rand_mean_sim, rand_std_sim)

        if i < results_to_print:
            print(f"Concept {concept}: mean cossim={mean_sim:.4f}, std={std_sim:.4f}")
            if print_random:
                print(f"          (random: mean cossim={rand_mean_sim:.4f}, std={rand_std_sim:.4f})")

    return concept_cosine_stats


### Visualizations of Quantitative Results ###
def plot_heatmap(concept_names, cosine_similarity_matrix, heatmap_title, 
                   save_label=None, dataset_name='CLEVR'):
    """
    Creates and displays a heatmap of cosine similarities between concept embeddings.

    Args:
        concept_names (list of str): A list of concept names to be displayed on the heatmap axes.
        cosine_similarity_matrix (ndarray): A 2D NumPy array representing the cosine similarity values 
        between concept embeddings.
        heatmap_title (str): The title of the heatmap to be displayed.
        save_label (str): label to put in path of saved image.
        dataset_name (str) : Name of the dataset

    Returns:
        None: The function directly displays the heatmap using `matplotlib` and `seaborn`.
    """
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_similarity_matrix, 
                xticklabels=concept_names, 
                yticklabels=concept_names, 
                cmap='coolwarm', 
                cbar=True, 
                annot=True, 
                fmt=".2f")

    plt.title(heatmap_title)
    
    if save_label:
        save_path = f'../Figs/{dataset_name}/concepts_heatmap/{save_label}.jpg'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        
    plt.show()
    
def concept_heatmap(concept_embeddings, con_label, dataset_name='CLEVR', normalize=False):
    """
    Plots a heatmap of cosine similarities between concept embeddings from a dataset.

    This function loads the concept embeddings from the specified dataset, selects a subset of 
    concepts (up to 10), calculates the cosine similarities between them, and displays the result 
    as a heatmap.

    Args:
        concepts_file (str): File where concept dictionary is stored.
        con_label (str): label to put in path of saved image.
        dataset_name (str, optional): The name of the dataset to load concept embeddings from.
                
    Returns:
        None: The function generates and displays the heatmap.
    """
    concept_names = list(concept_embeddings.keys())
    concept_names.sort()
    
    # Get concept names and embeddings
    embeddings = torch.stack([concept_embeddings[name] for name in concept_names])
    
    # Normalize embeddings to unit norm
    if normalize:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings = embeddings / norms
    
    #compute cosine similarities
    cosine_similarity_matrix = torch.matmul(embeddings.float(), embeddings.float().T).cpu().numpy()
    
    heatmap_title = 'Cosine Similarity Between Concept Embeddings'
    # save_label = f'{heatmap_title.replace(" ", "_")}__{con_label}'
    plot_heatmap(concept_names, cosine_similarity_matrix, heatmap_title,
                   dataset_name=dataset_name)


def concept_heatmap_groupedby_concept(concepts_file, con_label, dataset_name='CLEVR', normalize=False):
    """
    Plots a heatmap of cosine similarities between concept embeddings grouped by a specific concept category.

    This function allows the user to choose a concept category (e.g., color, shape) and generates a 
    heatmap of cosine similarities between the embeddings of concepts in that category.

    Args:
        concepts_file (str): File where concept dictionary is stored.
        con_label (str): label to put in path of saved image.
        dataset_name (str, optional): The name of the dataset to load concept embeddings from. 

    Returns:
        None: The function generates and displays the heatmap based on the chosen concept category.
    """
    concept_embeddings = torch.load(f'Concepts/{dataset_name}/{concepts_file}', weights_only=False)
    
    # Have user choose concept category
    potential_concept_categories = [key for key in concept_embeddings.keys() if key not in ['class', 'image_filename', 'split']]
    concept_category = get_user_category(potential_concept_categories)[0]
    
    # Make heatmap just based on those categories
    concept_names = [key for key in list(concept_embeddings.keys()) if key.startswith(concept_category)]
    embeddings = torch.stack([concept_embeddings[name] for name in concept_names])
    
    # Normalize embeddings to unit norm
    if normalize:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings = embeddings / norms
    
    cosine_similarity_matrix = torch.matmul(embeddings, embeddings.T).cpu().numpy()
    
    heatmap_title = f'Cosine Similarities Between {concept_category} Concepts'
    save_label = f'{heatmap_title.replace(" ", "_")}__{con_label}'
    plot_heatmap(concept_names, cosine_similarity_matrix, heatmap_title, 
                   dataset_name=dataset_name, save_label=save_label)


def concept_heatmap_random_samples(concept_embeddings, con_label, num_samples=15, dataset_name='CLEVR', normalize=False):
    """
    Plots a heatmap of cosine similarities between a random subset of concept embeddings from a dataset.

    This function loads the concept embeddings from the specified dataset, selects a random subset 
    of concepts (up to `num_samples`), calculates the cosine similarities between them, and displays the result 
    as a heatmap.
    
    THE SELECTED EMBEDDINGS ARE NORMALIZED WRT TO EACH OTHER BEFORE COMPUTING THEIR SIMILARITIES

    Args:
        concepts_file (str): File where the concept dictionary is stored.
        con_label (str): Label to include in the path of the saved image.
        num_samples (int, optional): The number of random concepts to sample for visualization. Default is 15.
        dataset_name (str, optional): The name of the dataset to load concept embeddings from. 
                
    Returns:
        None
    """
    # Randomly sample `num_samples` concepts for visualization
    print(f"Sampling {num_samples} random concepts for visualization purposes.")
    concept_names = random.sample(list(concept_embeddings.keys()), min(num_samples, len(concept_embeddings)))
    concept_names.sort()
    
    # Extract embeddings for the sampled concepts
    embeddings = torch.stack([concept_embeddings[name] for name in concept_names])
    
    # Normalize embeddings to unit norm
    # Normalize embeddings to unit norm
    if normalize:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        embeddings = embeddings / norms

    # Now compute the cosine similarity matrix
    cosine_similarity_matrix = torch.matmul(embeddings, embeddings.T).cpu().numpy()
    
    # Heatmap title
    heatmap_title = f'Cosine Similarity Between {num_samples} Random Concept Embeddings'
    
    # Create and optionally save the heatmap
    save_label = f'{heatmap_title.replace(" ", "_")}__{con_label}'
    plot_heatmap(concept_names, cosine_similarity_matrix, heatmap_title,
                   dataset_name=dataset_name, save_label=save_label)
    

def compute_cossim_hist_stats(gt_samples_per_concept, acts_loader, dataset_name, percentile, sample_type, model_input_size, con_label, patch_size=14, all_object_patches=None, concepts_to_process=None):
    """
    Computes in-sample and out-of-sample cosine similarity statistics for each concept, separated by train and test splits.

    Args:
        concept_thresholds (dict): Dictionary mapping concepts to (threshold, random_threshold).
        gt_samples_per_concept (dict): Dictionary mapping concepts to sets of true concept patch indices.
        acts_loader: ChunkedActivationLoader instance or DataFrame with activations.
        dataset_name (str): The name of the dataset, used to load the correct metadata file.
        percentile (float): Percentile of in-sample patches to compute the threshold.
        all_object_patches (set, optional): Set of patch indices to consider. If provided, only these patches are considered.
        concepts_to_process (list, optional): List of specific concepts to process. If None, processes all concepts.

    Returns:
        dict: A dictionary with per-concept cosine similarity stats, separated by train and test splits.
    """
    # Handle both ChunkedActivationLoader and DataFrame inputs
    if hasattr(acts_loader, 'load_full_dataframe'):
        # For ChunkedActivationLoader, we'll process in chunks to save memory
        is_chunked_loader = True
    else:
        # Assume it's already a DataFrame
        is_chunked_loader = False
        cos_sims = acts_loader
    
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    else:
        split_df = get_split_df(dataset_name)

    train_mask = split_df == 'train'
    test_mask = split_df != 'train'

    stats = {'train': {}, 'test': {}}
    
    # Filter concepts if specified
    if concepts_to_process is not None:
        # Convert to set for faster lookup
        concepts_to_process = set(str(c) for c in concepts_to_process)
        # Filter gt_samples_per_concept to only include requested concepts
        filtered_gt_samples = {k: v for k, v in gt_samples_per_concept.items() 
                             if str(k) in concepts_to_process}
    else:
        filtered_gt_samples = gt_samples_per_concept
    
    # Pre-process all concept indices
    processed_concept_indices = {}
    for concept, concept_indices in filtered_gt_samples.items():
        concept = str(concept)
        concept_indices = set(concept_indices)
        
        # Filter patches that are irrelevant given the preprocessing scheme
        concept_indices = set(filter_patches_by_image_presence(concept_indices, dataset_name, model_input_size).tolist())
        
        # Apply object patches filter if provided
        if all_object_patches is not None:
            concept_indices &= all_object_patches
        
        processed_concept_indices[concept] = concept_indices
    
    # Initialize storage
    for concept in processed_concept_indices:
        stats['train'][concept] = {'in_concept_sims': [], 'out_concept_sims': []}
        stats['test'][concept] = {'in_concept_sims': [], 'out_concept_sims': []}
    
    if is_chunked_loader:
        # Process all concepts in one pass through chunks
        chunk_size = 50000
        total_samples = acts_loader.total_samples
        
        
        for start_idx in tqdm(range(0, total_samples, chunk_size), desc="Processing chunks"):
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # Load chunk once
            chunk_df = acts_loader.load_chunk_range(start_idx, end_idx)
            
            # Get indices and masks for this chunk
            chunk_indices = list(range(start_idx, end_idx))
            chunk_train_mask = train_mask.iloc[start_idx:end_idx].reset_index(drop=True)
            chunk_test_mask = test_mask.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # Process each concept for this chunk
            for concept, concept_indices in processed_concept_indices.items():
                if concept not in chunk_df.columns:
                    continue
                
                # Create masks for this concept
                chunk_in_gt_mask = pd.Series(chunk_indices).isin(concept_indices)
                if all_object_patches is not None:
                    chunk_out_gt_mask = pd.Series(chunk_indices).isin(all_object_patches - concept_indices)
                else:
                    chunk_out_gt_mask = ~chunk_in_gt_mask
                
                # Get cosine similarities
                cos_vals = chunk_df[concept]
                
                # Extract and store values
                train_in = cos_vals[chunk_train_mask & chunk_in_gt_mask].tolist()
                test_in = cos_vals[chunk_test_mask & chunk_in_gt_mask].tolist()
                train_out = cos_vals[chunk_train_mask & chunk_out_gt_mask].tolist()
                test_out = cos_vals[chunk_test_mask & chunk_out_gt_mask].tolist()
                
                stats['train'][concept]['in_concept_sims'].extend(train_in)
                stats['test'][concept]['in_concept_sims'].extend(test_in)
                stats['train'][concept]['out_concept_sims'].extend(train_out)
                stats['test'][concept]['out_concept_sims'].extend(test_out)
                
            
            # Clear chunk from memory
            del chunk_df
            gc.collect()
    
    else:
        # Original non-chunked logic
        for concept, concept_indices in tqdm(processed_concept_indices.items(), desc="Processing concepts"):
            # Create masks
            in_gt_mask = cos_sims.index.to_series().isin(concept_indices)
            if all_object_patches is not None:
                out_gt_mask = cos_sims.index.to_series().isin(all_object_patches - concept_indices)
            else:
                out_gt_mask = ~in_gt_mask
            
            # Get cosine similarities
            cos_vals = cos_sims[concept]
            
            # Extract values
            stats['train'][concept]['in_concept_sims'] = cos_vals[train_mask & in_gt_mask].tolist()
            stats['test'][concept]['in_concept_sims'] = cos_vals[test_mask & in_gt_mask].tolist()
            stats['train'][concept]['out_concept_sims'] = cos_vals[train_mask & out_gt_mask].tolist()
            stats['test'][concept]['out_concept_sims'] = cos_vals[test_mask & out_gt_mask].tolist()
    
    if all_object_patches is not None:
        torch.save(stats, f'Hist_Stats/{dataset_name}/histstats_justobj_{con_label}.pt')
    else:
        torch.save(stats, f'Hist_Stats/{dataset_name}/histstats_{con_label}.pt')

    return stats


def plot_cosine_similarity_histograms(stats, concept_thresholds, sample_type, plot_type="both", metric_type='Cosine Similarity', percentile=None, bins=50, concepts=None, save_path=None, vmin=None, vmax=None):
    """
    Plots histograms of cosine similarity values for each concept using precomputed statistics.

    Args:
        stats (dict): Dictionary containing in-sample and out-of-sample cosine similarity stats for both train and test splits.
                      Expected structure:
                      {
                        'train': { concept: {'in_concept_sims': [...], 'out_concept_sims': [...]}, ... },
                        'test': { concept: {'in_concept_sims': [...], 'out_concept_sims': [...]}, ... }
                      }
        concept_thresholds (dict): Dictionary mapping concepts to (threshold, random_threshold).
        sample_type (str): Label for the sample type (e.g., "patch" or "image").
        plot_type (str): Option to plot "train", "test", or "both" datasets.
        percentile (float, optional): Percentile value for threshold line.
        bins (int): Number of bins for the histogram.

    Returns:
        None: Displays the histograms.
    """
    plt.rcParams.update({'font.size': 8})
    # Extract train and test stats
    train_stats = stats['train']
    test_stats = stats['test']
    
    # Use the keys from the train split (assume same keys in test)
    if not concepts:
        concepts = list(train_stats.keys())
    num_concepts = len(concepts)
    
    fig, axes = plt.subplots(nrows=num_concepts, figsize=(5.5, num_concepts*2), sharex=True)
    if num_concepts == 1:
        axes = [axes]
        
    # Define KDE plotting helper
    from scipy.stats import gaussian_kde
    def plot_kde(data, color, label):
        if len(data) > 1 and np.std(data) > 0:
            kde = gaussian_kde(data)
            xs = np.linspace(vmin if vmin is not None else min(data) - 0.01,
                             vmax if vmax is not None else max(data) + 0.01, 300)
            # ax.plot(xs, kde(xs), color=color, label=label)
            ax.fill_between(xs, 0, kde(xs), color=color, alpha=0.6, label=label)

    for i, concept in enumerate(concepts):
        ax = axes[i]

        # Retrieve similarity values
        in_concept_sims_train = train_stats[concept]['in_concept_sims']
        out_concept_sims_train = train_stats[concept]['out_concept_sims']
        in_concept_sims_test = test_stats[concept]['in_concept_sims']
        out_concept_sims_test = test_stats[concept]['out_concept_sims']
        
#         # Plot histograms based on plot_type
#         if plot_type in {"both", "train"}:
#             ax.hist(out_concept_sims_train, bins=bins, alpha=0.5, color='lightblue', label='Train - Out-of-Concept', density=True, edgecolor='none', histtype='stepfilled',)
#             ax.hist(in_concept_sims_train, bins=bins, alpha=0.5, color='lightcoral', label='Train - In-Concept', density=True, edgecolor='none', histtype='stepfilled',)

#         if plot_type in {"both", "test"}:
#             ax.hist(out_concept_sims_test, bins=bins, alpha=0.5, color='blue', label='Out-of-Concept', density=True)
#             ax.hist(in_concept_sims_test, bins=bins, alpha=0.5, color='red', label='In-Concept', density=True)
        # Plot KDE curves
        if plot_type in {"both", "train"}:
            plot_kde(out_concept_sims_train, 'lightblue', 'Train - Out-of-Concept')
            plot_kde(in_concept_sims_train, 'lightcoral', 'Train - In-Concept')

        if plot_type in {"both", "test"}:
            plot_kde(out_concept_sims_test, 'blue', 'Out-of-Concept')
            plot_kde(in_concept_sims_test, 'red', 'In-Concept')

        # Set labels and title
        ax.set_xlabel(metric_type)
        # ax.set_ylabel(f'{sample_type.capitalize()} {"Density"}')
        ax.set_title(f'{concept.capitalize()}', fontstyle='italic')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Plot percentile threshold if available
        if percentile is not None and concept in concept_thresholds:
            ax.axvline(concept_thresholds[concept][0], color='green', linestyle='--', linewidth=2, label=f'{percentile:.2f}% Threshold')
        
        # Force x-axis tick labels
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_yticks([])
        
        ax.set_xlim([vmin, vmax])

        ax.legend(loc='upper left')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, format='pdf', bbox_inches='tight')
        
    plt.show()


    
### tools for comparing image activations using different patch thresholds   

def find_activated_images_bypatch(loader, curr_thresholds, model_input_size, dataset_name, patch_size=14):
    """
    Computes per-image activations by max-pooling over patches using chunked data.
    Memory-efficient version that processes one split at a time.
    """
    from utils.memory_management_utils import ChunkedActivationLoader
    
    split_df = get_split_df(dataset_name)
    patches_per_image = (model_input_size[0] // patch_size) ** 2
    
    # Initialize result dictionaries
    activated_images_train = defaultdict(set)
    activated_images_test = defaultdict(set)
    activated_images_cal = defaultdict(set)
    
    # Get concepts from loader
    info = loader.get_activation_info()
    concepts = info['concept_names']
    
    # Process thresholds
    thresholds = {c: curr_thresholds[c][0] for c in curr_thresholds}
    threshold_tensor = torch.tensor([thresholds.get(c, float('inf')) for c in concepts])
    
    # Process each split separately to save memory
    for split_name, activated_dict in [('train', activated_images_train), 
                                       ('test', activated_images_test), 
                                       ('cal', activated_images_cal)]:
        
        # Load only the data for this split
        split_tensor = loader.load_split_tensor(split_name, dataset_name, model_input_size, patch_size)
        
        # Get the indices for this split
        split_indices = split_df[split_df == split_name].index.tolist()
        num_split_images = len(split_indices)
        
        if num_split_images == 0:
            continue
        
        # Reshape to image-level max activations
        reshaped_sims = split_tensor.reshape(num_split_images, patches_per_image, -1)
        max_activations = torch.max(reshaped_sims, dim=1)[0]  # [num_split_images, n_concepts]
        
        # Ensure threshold tensor is on same device as activations
        if threshold_tensor.device != max_activations.device:
            threshold_tensor = threshold_tensor.to(max_activations.device)
        
        # Thresholding
        activated = max_activations >= threshold_tensor  # [num_split_images, num_concepts]
        
        # Process each image in this split
        for local_img_idx in range(num_split_images):
            # Find activated concepts for this image
            activated_concepts = torch.where(activated[local_img_idx])[0]
            
            if len(activated_concepts) > 0:
                global_img_idx = split_indices[local_img_idx]
                
                for concept_idx in activated_concepts:
                    concept = concepts[concept_idx]
                    activated_dict[concept].add(global_img_idx)
        
        # Clear memory for this split
        del split_tensor, reshaped_sims, max_activations, activated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return activated_images_train, activated_images_test, activated_images_cal


def find_activated_images_byimage(loader, curr_thresholds, model_input_size, dataset_name):
    """
    Finds activated images for cls-level embeddings using chunked data.
    Memory-efficient version that processes one split at a time.
    """
    from utils.memory_management_utils import ChunkedActivationLoader
    
    split_df = get_split_df(dataset_name)
    info = loader.get_activation_info()
    concepts = info['concept_names']
    
    # Process thresholds
    thresholds = {c: curr_thresholds[c][0] for c in curr_thresholds}
    threshold_tensor = torch.tensor([thresholds.get(c, float('inf')) for c in concepts])
    
    # Initialize result dictionaries
    activated_images_train = defaultdict(set)
    activated_images_test = defaultdict(set)
    activated_images_cal = defaultdict(set)
    
    # Process each split separately to save memory
    for split_name, activated_dict in [('train', activated_images_train), 
                                       ('test', activated_images_test), 
                                       ('cal', activated_images_cal)]:
        
        # Load only the data for this split
        split_tensor = loader.load_split_tensor(split_name, dataset_name, model_input_size)
        
        # Get the indices for this split
        split_indices = split_df[split_df == split_name].index.tolist()
        
        if len(split_indices) == 0:
            continue
        
        # Ensure threshold tensor is on same device as split tensor
        if threshold_tensor.device != split_tensor.device:
            threshold_tensor = threshold_tensor.to(split_tensor.device)
        
        # Apply thresholds
        activated = split_tensor >= threshold_tensor
        
        # Process each image in this split
        for local_idx in range(len(split_tensor)):
            # Find activated concepts
            activated_concepts = torch.where(activated[local_idx])[0]
            
            if len(activated_concepts) > 0:
                global_idx = split_indices[local_idx]
                
                for concept_idx in activated_concepts:
                    concept = concepts[concept_idx]
                    activated_dict[concept].add(global_idx)
        
        # Clear memory for this split
        del split_tensor, activated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return activated_images_train, activated_images_test, activated_images_cal


def find_activated_sentences_bytoken(loader, curr_thresholds, model_input_size, dataset_name):
    """
    Finds activated sentences by token using chunked data.
    TODO: Implement chunked version for text data
    """
    # For now, fall back to loading all data
    # This would need a proper implementation based on how text tokens are organized
    acts_df = loader.load_full_dataframe()
    return find_activated_sentences_bytoken(acts_df, curr_thresholds, model_input_size, dataset_name)


# def find_activated_images_bypatch(cos_sims, curr_thresholds, model_input_size, dataset_name, patch_size=14):
#     """Vectorized version using torch operations (maybe could take out padding from consideration)"""
#     split_df = get_split_df(dataset_name)
#     patches_per_image = (model_input_size[0] // patch_size) ** 2
#     num_images = len(cos_sims) // patches_per_image
    
#     # Convert to tensors for faster operations
#     thresholds = {c: curr_thresholds[c][0] for c in curr_thresholds}
#     cos_sims_tensor = torch.tensor(cos_sims.values)
#     threshold_tensor = torch.tensor([thresholds[c] for c in cos_sims.columns])
    
#     # Reshape to [num_images, patches_per_image, num_concepts]
#     reshaped_sims = cos_sims_tensor.reshape(num_images, patches_per_image, -1)
    
#     # Max over patches dimension
#     max_activations = torch.max(reshaped_sims, dim=1)[0]  # [num_images, num_concepts]
    
#     # Compare with thresholds
#     activated = max_activations >= threshold_tensor
    
#     # Split by train/test
#     split_array = np.array(split_df)
#     train_mask = torch.tensor(split_array == 'train')
#     test_mask = torch.tensor(split_array == 'test')
    
#     activated_images_train = defaultdict(set)
#     activated_images_test = defaultdict(set)
    
#     for i, concept in enumerate(cos_sims.columns):
#         train_indices = torch.where(activated[:, i] & train_mask)[0].tolist()
#         test_indices = torch.where(activated[:, i] & test_mask)[0].tolist()
#         activated_images_train[concept].update(train_indices)
#         activated_images_test[concept].update(test_indices)
        
#     return activated_images_train, activated_images_test








def get_patch_detection_tensor(act_metrics, detect_thresholds, model_input_size, dataset_name, patch_size=14):
    """Optimized version using torch operations"""
    # Pre-compute sample indices
    if model_input_size[0] == 'text':
        token_counts_per_sentence = torch.load(f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt')
        num_tokens_per_sentence = [sum(x) for x in token_counts_per_sentence]
        sample_indices = torch.repeat_interleave(torch.arange(len(num_tokens_per_sentence)), torch.tensor(num_tokens_per_sentence))
    else:
        num_patches_per_image = (model_input_size[0] // patch_size) ** 2
        sample_indices = torch.tensor(act_metrics.index) // num_patches_per_image
    
    # Get activated samples using optimized functions
    if model_input_size[0] == 'text':
        detected_samples_train, detected_samples_test, detected_samples_cal = find_activated_sentences_bytoken(
            act_metrics, detect_thresholds, model_input_size, dataset_name)
    else:
        detected_samples_train, detected_samples_test, detected_samples_cal = find_activated_images_bypatch(
            act_metrics, detect_thresholds, model_input_size, dataset_name)
    
    # Initialize detection mask as tensor
    detection_mask = torch.zeros((len(act_metrics), len(act_metrics.columns)), dtype=torch.bool)
    
    # Update mask for all concepts at once - combine all detected samples (train, test, cal)
    for concept in detect_thresholds.keys():
        # Find the correct column index for this concept
        if concept in act_metrics.columns:
            col_idx = act_metrics.columns.get_loc(concept)
            
            all_detected_samples = set()
            all_detected_samples.update(detected_samples_train.get(concept, set()))
            all_detected_samples.update(detected_samples_test.get(concept, set()))
            all_detected_samples.update(detected_samples_cal.get(concept, set()))
            
            detected_sample_ids = torch.tensor(list(all_detected_samples))
            mask = torch.isin(sample_indices, detected_sample_ids)
            detection_mask[:, col_idx] = mask
    
    return pd.DataFrame(detection_mask.numpy(), index=act_metrics.index, columns=act_metrics.columns)



def compute_detection_metrics_for_per(per, gt_images_per_concept_split, 
                                      activated_images_split, 
                                      dataset_name, con_label, split='test'):
    """
    Compute detection metrics (TP, FP, TN, FN) for a specific percentile.
    Saves to disk if not already computed.

    Args:
        per: Percentile
        gt_images_per_concept_split: {concept: set of GT image indices} for the split
        activated_images_split: {concept: set of activated image indices} for the split
        dataset_name: Dataset name
        con_label: Concept label for saving
        split: Which split to evaluate ('test', 'cal', 'train')
    Returns:
        metrics_df: pd.DataFrame with TP, FP, TN, FN, F1, TPR, FPR per concept
    """
    save_path = f'Quant_Results/{dataset_name}/detectionmetrics_per_{per}_{con_label}.pt'
    
    # try:
    #     metrics_df = torch.load(save_path)
    # except:
    fp_count, tp_count, tn_count, fn_count = {}, {}, {}, {}
    
    # Collect all image indices present in the current split
    split_df = get_split_df(dataset_name)
    all_indices = set(split_df[split_df == split].index)

    for concept in gt_images_per_concept_split.keys():
        # gt_images = set(gt_images_per_concept_split[concept]) & set(relevant_indices)
        # activated_images = activated_images_test.get(concept, set()) & set(relevant_indices)
        gt_images = set(gt_images_per_concept_split[concept])
        activated_images = activated_images_split.get(concept, set())

        tp = len(gt_images & activated_images)
        fp = len(activated_images - gt_images)
        fn = len(gt_images - activated_images)
        tn = len(all_indices) - (tp + fp + fn)

        tp_count[concept] = tp
        fp_count[concept] = fp
        fn_count[concept] = fn
        tn_count[concept] = tn

    metrics_df = compute_stats_from_counts(tp_count, fp_count, tn_count, fn_count)
    torch.save(metrics_df, save_path)

    return metrics_df

    

def compute_detection_metrics_over_percentiles(percentiles, gt_images_per_concept_split, 
                                               loader, dataset_name, model_input_size, device, 
                                               con_label, sample_type='patch', patch_size=14):
    """
    Loads activations only once and processes all percentiles.
    Works for both supervised and unsupervised methods.

    Args:
        percentiles: List of percentiles
        gt_images_per_concept_split: {concept: image indices} - can be test or cal set
        loader: ChunkedActivationLoader
        dataset_name: Dataset name
        model_input_size: (width, height) tuple
        device: CUDA/CPU device
        con_label: Label for saving
        sample_type: 'patch' or 'cls'
        patch_size: Patch size
    Returns:
        all_metrics: dict mapping per -> metrics_df
    """
    from collections import defaultdict
    import gc
    from utils.memory_management_utils import ChunkedActivationLoader
    
    # Determine split being evaluated
    if con_label.endswith('_cal'):
        eval_split = 'cal'
    else:
        eval_split = 'test'
    
    # Load thresholds for all percentiles
    threshold_label = con_label.replace('_cal', '') if con_label.endswith('_cal') else con_label
    
    if 'kmeans' not in con_label:
        # Supervised methods
        all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{threshold_label}.pt', weights_only=False)
        concept_names = list(all_thresholds[percentiles[0]].keys())
    else:
        # Unsupervised methods - need to handle concept-cluster mapping
        raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{threshold_label}.pt', weights_only=False)
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{threshold_label}.pt', weights_only=False)
        
        # Create matched thresholds
        all_thresholds = {}
        for percentile, thresholds_dict in raw_thresholds.items():
            matched_thresholds = {}
            for concept, info in alignment_results.items():
                cluster_id = info['best_cluster']
                key = (concept, cluster_id)
                if key in thresholds_dict:
                    matched_thresholds[cluster_id] = thresholds_dict[key]
            all_thresholds[percentile] = matched_thresholds
        
        concept_names = list(alignment_results.keys())
    
    # Get split dataframe
    split_df = get_split_df(dataset_name)
    all_indices = set(split_df[split_df == eval_split].index)
    
    # Precompute max activations for all images with GPU optimization
    # print(f"Precomputing max activations for {eval_split} split...")
    
    if sample_type == 'patch':
        # Check if this is a text dataset
        if isinstance(model_input_size[0], str) and model_input_size[0] == 'text':
            # For text datasets, load token counts to determine tokens per paragraph
            token_counts_file = f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt'
            
            if not os.path.exists(token_counts_file):
                raise FileNotFoundError(f"Token counts file not found: {token_counts_file}")
            
            token_counts = torch.load(token_counts_file, weights_only=False)
            
            # Convert to tokens per sentence/paragraph
            tokens_per_paragraph = [sum(sent_tokens) if isinstance(sent_tokens, list) else sent_tokens 
                                  for sent_tokens in token_counts]
            
            info = loader.get_activation_info()
            total_tokens = info['total_samples']
            num_paragraphs = len(tokens_per_paragraph)
            num_concepts = info['num_concepts']
            
            # For text, we'll use variable-size handling below
            patches_per_image = None  # Variable for text
            num_images = num_paragraphs
        else:
            # For image datasets, use fixed patch grid
            patches_per_image = (model_input_size[0] // patch_size) ** 2
            info = loader.get_activation_info()
            total_patches = info['total_samples']
            num_images = total_patches // patches_per_image
            num_concepts = info['num_concepts']
        
        # Check GPU memory availability
        keep_on_gpu = False
        if device == 'cuda':
            torch.cuda.empty_cache()
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            required_memory = num_images * num_concepts * 4  # float32
            
            if required_memory < available_memory * 0.8:  # Use 80% of available memory
                # print(f"Allocating {required_memory / 1e9:.2f}GB on GPU")
                max_activations_all = torch.zeros(num_images, num_concepts, dtype=torch.float32, device=device)
                keep_on_gpu = True
            else:
                # print(f"Not enough GPU memory ({required_memory / 1e9:.2f}GB needed, {available_memory / 1e9:.2f}GB available). Using CPU.")
                max_activations_all = torch.zeros(num_images, num_concepts, dtype=torch.float32)
        else:
            max_activations_all = torch.zeros(num_images, num_concepts, dtype=torch.float32)
        
        # Process in chunks with GPU optimization
        if patches_per_image is not None:
            # Fixed patch size (image datasets)
            images_per_chunk = 500
            for img_start_idx in tqdm(range(0, num_images, images_per_chunk), desc="Loading activations"):
                img_end_idx = min(img_start_idx + images_per_chunk, num_images)
                
                # Calculate patch range
                patch_start_idx = img_start_idx * patches_per_image
                patch_end_idx = img_end_idx * patches_per_image
                
                # Load chunk and move to GPU if beneficial
                chunk_tensor = loader.load_chunk_range(patch_start_idx, patch_end_idx)
                if device == 'cuda' and chunk_tensor.device != device:
                    chunk_tensor = chunk_tensor.to(device)
                    
                num_images_in_chunk = img_end_idx - img_start_idx
                
                # Reshape and compute max on GPU if possible
                reshaped = chunk_tensor.reshape(num_images_in_chunk, patches_per_image, -1)
                max_acts = torch.max(reshaped, dim=1)[0]
                
                # Store results
                if keep_on_gpu:
                    max_activations_all[img_start_idx:img_end_idx] = max_acts
                else:
                    max_activations_all[img_start_idx:img_end_idx] = max_acts.cpu()
                
                # Clean up
                del chunk_tensor, reshaped, max_acts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Variable token size (text datasets)
            paragraphs_per_chunk = 100
            
            # Calculate cumulative token indices
            cumulative_tokens = [0]
            for tokens in tokens_per_paragraph:
                cumulative_tokens.append(cumulative_tokens[-1] + tokens)
            
            for para_start_idx in tqdm(range(0, num_paragraphs, paragraphs_per_chunk), desc="Loading activations"):
                para_end_idx = min(para_start_idx + paragraphs_per_chunk, num_paragraphs)
                
                # Calculate token range for this chunk of paragraphs
                token_start_idx = cumulative_tokens[para_start_idx]
                token_end_idx = cumulative_tokens[para_end_idx]
                
                # Load chunk and move to GPU if beneficial
                chunk_tensor = loader.load_chunk_range(token_start_idx, token_end_idx)
                if device == 'cuda' and chunk_tensor.device != device:
                    chunk_tensor = chunk_tensor.to(device)
                
                # Process each paragraph separately due to variable lengths
                chunk_start = 0
                for para_idx in range(para_start_idx, para_end_idx):
                    para_tokens = tokens_per_paragraph[para_idx]
                    para_acts = chunk_tensor[chunk_start:chunk_start + para_tokens]
                    
                    # Compute max activation for this paragraph
                    max_act = torch.max(para_acts, dim=0)[0]
                    
                    # Store results
                    if keep_on_gpu:
                        max_activations_all[para_idx] = max_act
                    else:
                        max_activations_all[para_idx] = max_act.cpu()
                    
                    chunk_start += para_tokens
                
                # Clean up
                del chunk_tensor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        # For CLS, activations are already at image level
        # print("Loading image-level activations...")
        full_tensor = loader.load_full_tensor()
        
        # Keep on GPU if memory allows
        if device == 'cuda':
            torch.cuda.empty_cache()
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            required_memory = full_tensor.numel() * 4  # float32
            
            if required_memory < available_memory * 0.8:
                # print(f"Keeping CLS activations on GPU ({required_memory / 1e9:.2f}GB)")
                max_activations_all = full_tensor.to(device)
                keep_on_gpu = True
            else:
                # print(f"Using CPU for CLS activations ({required_memory / 1e9:.2f}GB needed)")
                max_activations_all = full_tensor.cpu()
                keep_on_gpu = False
        else:
            max_activations_all = full_tensor.cpu()
            keep_on_gpu = False
            
        num_images = max_activations_all.shape[0]
    
    # print(f"Max activations shape: {max_activations_all.shape}")
    
    # Get concept/cluster labels from loader
    loader_info = loader.get_activation_info() if hasattr(loader, 'get_activation_info') else loader.get_info()
    
    if 'kmeans' not in con_label:
        # For supervised, concept names are column names
        concept_columns = loader_info['concept_names']
    else:
        # For unsupervised, columns are cluster IDs
        cluster_columns = loader_info['concept_names']
    
    # Process each percentile using precomputed activations with GPU optimization
    all_metrics = {}
    
    # Precompute split mask for efficient filtering
    split_mask = torch.tensor([split_df.iloc[i] == eval_split if i < len(split_df) else False 
                              for i in range(len(split_df))], dtype=torch.bool)
    if keep_on_gpu and device == 'cuda':
        split_mask = split_mask.to(device)
    
    for per in tqdm(percentiles, desc="Computing metrics for each percentile"):
        curr_thresholds = all_thresholds[per]
        
        # Find activated images with vectorized operations
        activated_images_split = defaultdict(set)
        
        if 'kmeans' not in con_label:
            # Supervised: direct concept mapping with vectorized operations
            for concept, (threshold, _) in curr_thresholds.items():
                if str(concept) in concept_columns:
                    concept_idx = concept_columns.index(str(concept))
                    
                    # Vectorized activation detection
                    concept_activations = max_activations_all[:, concept_idx]
                    activated_mask = concept_activations >= threshold
                    
                    # Combine with split mask for efficient filtering
                    if len(split_mask) > len(activated_mask):
                        split_mask_truncated = split_mask[:len(activated_mask)]
                    else:
                        split_mask_truncated = split_mask
                        
                    final_mask = activated_mask & split_mask_truncated
                    activated_indices = torch.where(final_mask)[0]
                    
                    # Convert to set on CPU
                    if activated_indices.device.type == 'cuda':
                        activated_indices = activated_indices.cpu()
                    activated_images_split[concept] = set(activated_indices.tolist())
                    
        else:
            # Unsupervised: vectorized cluster to concept mapping
            for concept, info in alignment_results.items():
                cluster_id = str(info['best_cluster'])
                if cluster_id in curr_thresholds:
                    threshold, _ = curr_thresholds[cluster_id]
                    
                    if cluster_id in cluster_columns:
                        cluster_idx = cluster_columns.index(cluster_id)
                        
                        # Vectorized activation detection
                        cluster_activations = max_activations_all[:, cluster_idx]
                        activated_mask = cluster_activations >= threshold
                        
                        # Combine with split mask for efficient filtering
                        if len(split_mask) > len(activated_mask):
                            split_mask_truncated = split_mask[:len(activated_mask)]
                        else:
                            split_mask_truncated = split_mask
                            
                        final_mask = activated_mask & split_mask_truncated
                        activated_indices = torch.where(final_mask)[0]
                        
                        # Convert to set on CPU
                        if activated_indices.device.type == 'cuda':
                            activated_indices = activated_indices.cpu()
                        activated_images_split[concept] = set(activated_indices.tolist())
        
        # Compute metrics
        metrics_df = compute_detection_metrics_for_per(
            per, gt_images_per_concept_split, activated_images_split,
            dataset_name, con_label, split=eval_split
        )
        
        all_metrics[per] = metrics_df
    
    # Clean up
    del max_activations_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_metrics


def compute_percentages_per_concept(concepts, activated_images_train, activated_images_test,
                                    gt_images_per_concept_train, gt_images_per_concept_test,
                                    total_train_images, total_test_images, percentile, dataset_name, con_label):
    """
    For a given threshold, compute activation percentages per concept.

    Args:
        concepts (list): List of concept names.
        activated_images_train (dict): Dictionary (key: concept) of sets of activated train image indices.
        activated_images_test (dict): Dictionary (key: concept) of sets of activated test image indices.
        gt_images_per_concept_train (dict): Dictionary mapping concept to list of ground truth train image indices.
        gt_images_per_concept_test (dict): Dictionary mapping concept to list of ground truth test image indices.
        total_train_images (int): Total number of train images.
        total_test_images (int): Total number of test images.

    Returns:
        Tuple of four dicts:
          - percent_train_inconcept: activation % for train images containing the concept.
          - percent_train_outconcept: activation % for train images not containing the concept.
          - percent_test_inconcept: activation % for test images containing the concept.
          - percent_test_outconcept: activation % for test images not containing the concept.
    """
    percent_train_inconcept = {}
    percent_train_outconcept = {}
    percent_test_inconcept = {}
    percent_test_outconcept = {}

    for concept in concepts:
        # Number of images known to contain the concept.
        n_train_images_w_concept = len(gt_images_per_concept_train.get(concept, []))
        n_test_images_w_concept = len(gt_images_per_concept_test.get(concept, []))
        
        # Number of images known NOT to contain the concept.
        n_train_images_wo_concept = total_train_images - n_train_images_w_concept
        n_test_images_wo_concept = total_test_images - n_test_images_w_concept
        
        # Count activated images (unique image indices are assumed in activated_images_* dictionaries).
        n_inconcept_activated_train = len(activated_images_train.get(concept, set()) & set(gt_images_per_concept_train.get(concept, [])))
        n_outconcept_activated_train = len(activated_images_train.get(concept, set())) - n_inconcept_activated_train

        n_inconcept_activated_test = len(activated_images_test.get(concept, set()) & set(gt_images_per_concept_test.get(concept, [])))
        n_outconcept_activated_test = len(activated_images_test.get(concept, set())) - n_inconcept_activated_test
        
        # Compute percentages with division-by-zero checks.
        percent_train_inconcept[concept] = (n_inconcept_activated_train / n_train_images_w_concept * 100) if n_train_images_w_concept > 0 else 0
        percent_train_outconcept[concept] = (n_outconcept_activated_train / n_train_images_wo_concept * 100) if n_train_images_wo_concept > 0 else 0
        percent_test_inconcept[concept]  = (n_inconcept_activated_test / n_test_images_w_concept * 100) if n_test_images_w_concept > 0 else 0
        percent_test_outconcept[concept] = (n_outconcept_activated_test / n_test_images_wo_concept * 100) if n_test_images_wo_concept > 0 else 0
    
    torch.save(percent_train_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_train_inconcept_{con_label}.pt')
    torch.save(percent_test_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_test_inconcept_{con_label}.pt')
    torch.save(percent_train_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_train_outconcept_{con_label}.pt')
    torch.save(percent_test_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_test_outconcept_{con_label}.pt')
        
    return percent_train_inconcept, percent_train_outconcept, percent_test_inconcept, percent_test_outconcept

     
def plot_activation_percentages_over_thresholds(cos_sims, gt_patches_per_concept_train, 
                                                gt_patches_per_concept_test, gt_samples_per_concept_train,
                                                gt_samples_per_concept_test, dataset_name, 
                                                model_input_size, device, con_label, sample_type):
    """
    Plots the average activation percentages (over concepts) a range of threshold percentiles.
    
    Args:
        cos_sims (pd.DataFrame): DataFrame of cosine similarity values (columns: concepts, rows: patches).
        gt_patches_per_concept_train (dict): Mapping from concept to ground truth patch indices for the train set.
        gt_patches_per_concept_test (dict): Mapping from concept to ground truth patch indices for the test set.
        gt_images_per_concept_train (dict): Mapping from concept to ground truth image indices for the train set.
        gt_images_per_concept_test (dict): Mapping from concept to ground truth image indices for the test set.
        dataset_name (str): Name of the dataset used to obtain the train/test split.
        model_input_size (int): Size of the model input (used for image patch indexing).
        device (str): Device identifier for any GPU-based operations (e.g., 'cuda').
    
    Returns:
        None. The function plots the activation percentages.
    """
    # Sorted list of concepts.
    concepts = sorted(gt_patches_per_concept_train.keys())
    
    # Lists to store threshold values and per-threshold percentage dictionaries.
    train_inconcept_dicts = []
    train_outconcept_dicts = []
    test_inconcept_dicts = []
    test_outconcept_dicts = []
    
    split_df = get_split_df(dataset_name)
    total_train_samples = int((split_df == 'train').sum())
    total_test_samples = int((split_df == 'test').sum())
        
    # Define threshold percentiles to test (e.g., from 5% to 100% in steps of 5%).
    # in_concept_patch_thresholds = [round(x, 2) for x in np.arange(0.05, 1.05, 0.05)]
    in_concept_patch_thresholds = [0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    
    for percentile in tqdm(in_concept_patch_thresholds):
        try: #just load if you already did this computation
            pct_train_in = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_train_inconcept_{con_label}.pt', weights_only=False)
            pct_test_in = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_test_inconcept_{con_label}.pt', weights_only=False)
            pct_train_out = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_train_outconcept_{con_label}.pt', weights_only=False)
            pct_test_out = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_test_outconcept_{con_label}.pt', weights_only=False)
        except:
            curr_thresholds = compute_concept_thresholds(gt_patches_per_concept_test,  
                                                         cos_sims, percentile, device=device, 
                                                         dataset_name=f'{dataset_name}', 
                                                         con_label=con_label,
                                                         n_vectors=1,
                                                         n_concepts_to_print=0)

            if sample_type == 'patch':
                activated_samples_train, activated_samples_test = find_activated_images_bypatch(cos_sims, 
                                                                                  curr_thresholds, 
                                                                                  model_input_size, 
                                                                                    dataset_name)
            elif sample_type == 'cls':
                activated_samples_train, activated_samples_test = find_activated_images_byimage(cos_sims, 
                                                                                  curr_thresholds, 
                                                                                  model_input_size, 
                                                                                  dataset_name)
            elif sample_type == 'token':
                activated_samples_train, activated_samples_test = find_activated_sentences_bytoken(cos_sims, 
                                                                                  curr_thresholds, 
                                                                                  model_input_size, 
                                                                                  dataset_name)


            # Compute per-concept activation percentages using the helper function.
            pct_train_in, pct_train_out, pct_test_in, pct_test_out = compute_percentages_per_concept(
                concepts, activated_samples_train, activated_samples_test,
                gt_samples_per_concept_train, gt_samples_per_concept_test,
                total_train_samples, total_test_samples, percentile, dataset_name, con_label)
        
        train_inconcept_dicts.append(pct_train_in)
        train_outconcept_dicts.append(pct_train_out)
        test_inconcept_dicts.append(pct_test_in)
        test_outconcept_dicts.append(pct_test_out)
    
    # Average the percentages over all concepts for each threshold.
    avg_train_inconcept = [sum(d.values()) / len(d) for d in train_inconcept_dicts]
    avg_train_outconcept = [sum(d.values()) / len(d) for d in train_outconcept_dicts]
    avg_test_inconcept = [sum(d.values()) / len(d) for d in test_inconcept_dicts]
    avg_test_outconcept = [sum(d.values()) / len(d) for d in test_outconcept_dicts]
    
    # Plot the averaged percentages.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data with shared colors but different markers for train/test.
    in_concept_color = 'b'
    out_concept_color = 'orangered'

    train_marker = 'x'
    test_marker = 'o'

    train_inconcept_plot, = ax.plot(in_concept_patch_thresholds, avg_train_inconcept, color=in_concept_color, marker=train_marker, linestyle='-', alpha=0.7)
    test_inconcept_plot, = ax.plot(in_concept_patch_thresholds, avg_test_inconcept, color=in_concept_color, marker=test_marker, linestyle='-', alpha=0.7)
    train_outconcept_plot, = ax.plot(in_concept_patch_thresholds, avg_train_outconcept, color=out_concept_color, marker=train_marker, linestyle='-', alpha=0.7)
    test_outconcept_plot, = ax.plot(in_concept_patch_thresholds, avg_test_outconcept, color=out_concept_color, marker=test_marker, linestyle='-', alpha=0.7)

    ax.set_xlabel('In-Concept Patch Percentile')
    if sample_type == 'token':
        ax.set_ylabel('Average Sentence Activation Percentage')
        ax.set_title(f'Sentence Activation Percentages over In-Concept Patch Thresholds ({dataset_name})')
    else:  
        ax.set_ylabel('Average Image Activation Percentage')
        ax.set_title(f'Image Activation Percentages over In-Concept Patch Thresholds ({dataset_name})')
    ax.grid(True)

    # Custom legend with separate entries for concept type and data split.
    legend_elements = [
        Line2D([0], [0], color=in_concept_color, lw=2, label='In-Concept'),
        Line2D([0], [0], color=out_concept_color, lw=2, label='Out-of-Concept'),
        Line2D([0], [0], marker=train_marker, color='black', linestyle='None', markersize=8, label='Train'),
        Line2D([0], [0], marker=test_marker, color='black', linestyle='None', markersize=8, label='Test')
    ]

    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

    
def plot_activation_percentages_per_concept(percentile, cos_sims, gt_images_per_concept_train, 
                                gt_images_per_concept_test, gt_patches_per_concept_test, dataset_name, 
                                            model_input_size, device, con_label, sample_type, curr_concepts=None,
                                           force_compute=False):
    """
    Plots a horizontal bar chart showing the percentage of in-concept and out-of-concept
    train/test images activated for each concept at a given threshold percentile.
    
    Args:
        percentile (float): Percentile to determine concept activation threshold.
        cos_sims (pd.DataFrame): Cosine similarity values (columns: concepts, rows: patches).
        gt_images_per_concept_train (dict): Mapping from concept to ground truth train image indices.
        gt_images_per_concept_test (dict): Mapping from concept to ground truth test image indices.
        gt_patches_per_concept_test (dict): Mapping from concept to ground truth patch indices for the test set.
        dataset_name (str): Dataset name for obtaining train/test splits.
        model_input_size (int): Input size used for computing image index from a patch index.
        device (str): Device identifier for any GPU-based operations (e.g., 'cuda').
    
    Returns:
        None. Displays the bar plot.
    """
    concepts = sorted(gt_images_per_concept_train.keys())
    
    try: #just load if you already did this computation
        if force_compute:
            raise Exception("Forced Error")
        pct_train_in = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_train_inconcept_{con_label}.pt', weights_only=False)
        pct_test_in = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_test_inconcept_{con_label}.pt', weights_only=False)
        pct_train_out = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_train_outconcept_{con_label}.pt', weights_only=False)
        pct_test_out = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_test_outconcept_{con_label}.pt', weights_only=False)
    except:
        # Compute thresholds for the given percentile
        curr_thresholds = compute_concept_thresholds(gt_patches_per_concept_test,  
                                                         cos_sims, percentile, device=device, 
                                                         dataset_name=f'{dataset_name}', con_label=con_label,
                                                         n_vectors=1,
                                                         n_concepts_to_print=0)

        if sample_type == 'patch':
            activated_images_train, activated_images_test = find_activated_images_bypatch(cos_sims, 
                                                                                  curr_thresholds, 
                                                                                  model_input_size, 
                                                                                    dataset_name)
        elif sample_type == 'cls':
            activated_images_train, activated_images_test = find_activated_images_byimage(cos_sims, 
                                                                                  curr_thresholds, 
                                                                                  model_input_size, 
                                                                                  dataset_name)

        # Get train/test split counts
        split_df = get_split_df(dataset_name)
        total_train_images = int((split_df == 'train').sum())
        total_test_images = int((split_df == 'test').sum())

        # Compute percentages
        pct_train_in, pct_train_out, pct_test_in, pct_test_out = compute_percentages_per_concept(
            concepts, activated_images_train, activated_images_test,
            gt_images_per_concept_train, gt_images_per_concept_test,
            total_train_images, total_test_images, percentile, dataset_name, con_label)

    if curr_concepts is None:
        curr_concepts = concepts

    # Prepare data for plotting
    spacing = 1.2  # Increase spacing between concept groups
    y_pos = np.arange(len(curr_concepts)) * spacing  # Spread out the concepts
    width = 0.2  # Keep bar width the same
    fig, ax = plt.subplots(figsize=(12, len(curr_concepts) * 0.6)) 
    ax.barh(y_pos + 1.5 * width, [pct_train_in[c] for c in curr_concepts], width, label='Train In-Concept', color='lightblue')
    ax.barh(y_pos + 0.5 * width, [pct_test_in[c] for c in curr_concepts], width, label='Test In-Concept', color='blue')
    ax.barh(y_pos - 0.5 * width, [pct_train_out[c] for c in curr_concepts], width, label='Train Out-Concept', color='lightsalmon')
    ax.barh(y_pos - 1.5 * width, [pct_test_out[c] for c in curr_concepts], width, label='Test Out-Concept', color='orangered')

    ax.set_xlabel('Percentage of Activated Images')
    ax.set_ylabel('Concepts')
    ax.set_title(f'Image Activation Percentages for Each Concept at {percentile * 100:.0f}th Percentile for In-Concept Patches')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(curr_concepts)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Move legend to the right outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()
    
    
def count_activated_patches(cos_sims, curr_thresholds, model_input_size, dataset_name):
    """
    Counts the number of activated patches per image for each concept.

    Args:
        cos_sims (pd.DataFrame): DataFrame with cosine similarity values (rows: patches, columns: concepts).
        curr_thresholds (dict): Mapping from concept to a tuple where the first element is the threshold.
        model_input_size (int): Input size used to compute the image index from a patch index.
        dataset_name (str): Name of the dataset to obtain the train/test split.

    Returns:
        tuple: Two dictionaries mapping each concept to a dictionary of image indices and their activated patch counts 
               (one for train and one for test).
    """
    split_df = get_split_df(dataset_name)

    activated_patch_counts_train = defaultdict(lambda: defaultdict(int))
    activated_patch_counts_test = defaultdict(lambda: defaultdict(int))

    # Filter patches that are 'padding' given the preprocessing schemes
    relevant_indices = filter_patches_by_image_presence(cos_sims.index, dataset_name, model_input_size).tolist()
    cos_sims = cos_sims.loc[relevant_indices]

    concepts = curr_thresholds.keys()
    for patch_idx, cossim_vals in tqdm(cos_sims.iterrows(), total=len(cos_sims)):
        if model_input_size[0] == 'text':
            sample_idx = get_sent_idx_from_global_token_idx(patch_idx, dataset_name)
        else:
            sample_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size=14)
        for concept in concepts:
            threshold = curr_thresholds[concept][0]  
            if cossim_vals[concept] >= threshold:
                if split_df[sample_idx] == 'train':
                    activated_patch_counts_train[concept][sample_idx] += 1
                elif split_df[sample_idx] == 'test':
                    activated_patch_counts_test[concept][sample_idx] += 1
                    
    activated_patch_counts_train = {k: dict(v) for k, v in activated_patch_counts_train.items()}
    activated_patch_counts_test = {k: dict(v) for k, v in activated_patch_counts_test.items()}

    return activated_patch_counts_train, activated_patch_counts_test



def plot_activation_count_distributions(cos_sims, gt_patches_per_concept_train, 
                                             gt_patches_per_concept_test, gt_images_per_concept_train,
                                             gt_images_per_concept_test, dataset_name, 
                                             model_input_size, device, con_label, show_zero_count=False):
    """
    Plots line plots of the distributions (histograms) over activated patch counts (pooled across all concepts)
    for both train and test sets, for various threshold percentiles. For each percentile, the test distribution is
    plotted as a solid line and the train distribution as a dotted line, in the same color. The x-axis represents 
    the patch count, and the legend shows one set of entries for the threshold percentiles (colors) and one entry for 
    the line style (solid = Test, dotted = Train).
    
    Args:
        cos_sims (pd.DataFrame): DataFrame of cosine similarity values (rows: patches, columns: concepts).
        gt_patches_per_concept_train (dict): Mapping from concept to ground truth patch indices for the train set.
        gt_patches_per_concept_test (dict): Mapping from concept to ground truth patch indices for the test set.
        gt_images_per_concept_train (dict): Mapping from concept to ground truth image indices for the train set.
        gt_images_per_concept_test (dict): Mapping from concept to ground truth image indices for the test set.
        dataset_name (str): Name of the dataset used to obtain the train/test split.
        model_input_size (int): Model input size (used for computing image indices from patch indices).
        device (str): Device identifier (e.g., 'cuda').
        con_label (str): Label used for saving/loading computed counts.
        show_zero_count (bool): Whether to plot the images that have no activated patches
    
    Returns:
        None. Displays a line plot of the distributions.
    """
    # Get sorted list of concepts.
    concepts = sorted(gt_patches_per_concept_train.keys())
    
    split_df = get_split_df(dataset_name)
    total_train_images = int((split_df == 'train').sum())
    total_test_images = int((split_df == 'test').sum())
    
    # Define the threshold percentiles to test.
    threshold_percentiles = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
    
    # To store pooled distributions for each percentile for train and test.
    distributions_train = {}
    distributions_test = {}
    
    for percentile in tqdm(threshold_percentiles):
        try:
            # Attempt to load precomputed activated patch counts (each is a dict: concept -> {img_idx: count})
            activated_counts_train = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_{con_label}.pt', weights_only=False)
            activated_counts_test  = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_{con_label}.pt', weights_only=False)
        except:
            curr_thresholds = compute_concept_thresholds(gt_patches_per_concept_test,  
                                                         cos_sims, percentile, device=device, 
                                                         dataset_name=f'{dataset_name}', con_label=con_label,
                                                         n_vectors=1,
                                                         n_concepts_to_print=0)
            activated_counts_train, activated_counts_test = count_activated_patches(cos_sims, curr_thresholds, model_input_size, dataset_name)
            # Convert defaultdicts to regular dictionaries for saving.
            activated_counts_train = {k: dict(v) for k, v in activated_counts_train.items()}
            activated_counts_test  = {k: dict(v) for k, v in activated_counts_test.items()}

            torch.save(activated_counts_train, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_{con_label}.pt')
            torch.save(activated_counts_test, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_{con_label}.pt')
        
        # Pool counts across all concepts:
        all_counts_train = []
        all_counts_test = []
        for concept in concepts:
            concept_counts_train = np.array(list(activated_counts_train.get(concept, {}).values()))
            concept_counts_test  = np.array(list(activated_counts_test.get(concept, {}).values()))
            
            if show_zero_count:
                all_counts_train.extend(concept_counts_train[concept_counts_train > 0])
                all_counts_test.extend(concept_counts_test[concept_counts_test > 0])
            else:
                # Filter to only include images with at least one activated patch
                all_counts_train.extend(concept_counts_train[concept_counts_train > 0])
                all_counts_test.extend(concept_counts_test[concept_counts_test > 0])
        
        distributions_train[percentile] = np.array(all_counts_train)
        distributions_test[percentile]  = np.array(all_counts_test)
    
    # Determine common bins based on the overall max count from both train and test across all percentiles.
    max_count = 0
    for percentile in threshold_percentiles:
        if distributions_train[percentile].size > 0:
            max_count = max(max_count, distributions_train[percentile].max())
        if distributions_test[percentile].size > 0:
            max_count = max(max_count, distributions_test[percentile].max())
    bins = np.arange(0, max_count + 2) - 0.5  # bins for histogram
    
    # Prepare plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a colormap to choose a color for each percentile.
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(threshold_percentiles), max(threshold_percentiles))
    
    for percentile in threshold_percentiles:
        # Plot train distribution with dotted line.
        counts_train = distributions_train[percentile]
        if counts_train.size > 0:
            hist_train, bin_edges = np.histogram(counts_train, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            ax.plot(bin_centers, hist_train, linestyle=':', marker=None, 
                    color=cmap(norm(percentile)), alpha=0.8)
        
        # Plot test distribution with solid line.
        counts_test = distributions_test[percentile]
        if counts_test.size > 0:
            hist_test, bin_edges = np.histogram(counts_test, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            ax.plot(bin_centers, hist_test, linestyle='-', marker=None, 
                    color=cmap(norm(percentile)), alpha=0.8, label=f'{percentile*100:.0f}%')
    
    ax.set_xlabel('Activated Patch Count Per Image (Averaged Across Concepts)')
    ax.set_ylabel('Normalized Frequency')
    ax.set_title(f'Distribution of Average Activated Patch Counts Per Image ({dataset_name})')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Create custom legend entries.
    handles_percentiles = [Line2D([0], [0], color=cmap(norm(p)), lw=2, label=f'{p*100:.0f}%') 
                           for p in threshold_percentiles]
    handles_styles = [Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test (Solid)'),
                      Line2D([0], [0], color='black', linestyle=':', lw=2, label='Train (Dotted)')]
    
    # Adjust figure to make space for the legends
    plt.subplots_adjust(right=0.75)  # Leave space on the right

    # Place both legends outside the plot, stacked
    legend1 = ax.legend(handles=handles_percentiles, title='In-Concept Patch Percentile', loc='upper left', 
                        bbox_to_anchor=(1.02, 1))
    legend2 = ax.legend(handles=handles_styles, title='Data Split', loc='upper left', 
                        bbox_to_anchor=(1.02, 0.55))
    
    ax.add_artist(legend1)  # Ensure both legends appear
    
    plt.tight_layout()
    plt.show()
    
    
def plot_activation_count_by_concept(percentile, cos_sims, gt_patches_per_concept_train, 
                                     gt_patches_per_concept_test, dataset_name, 
                                     model_input_size, device, con_label, sample_type,
                                     show_zero_count=False, curr_concepts=None):
    """
    Plots histograms of activated patch counts per image for multiple concepts at a fixed percentile threshold.

    Args:
        cos_sims (pd.DataFrame): DataFrame of cosine similarity values (rows: patches, columns: concepts).
        gt_patches_per_concept_train (dict): Mapping from concept to ground truth patch indices for the train set.
        gt_patches_per_concept_test (dict): Mapping from concept to ground truth patch indices for the test set.
        dataset_name (str): Name of the dataset used to obtain the train/test split.
        model_input_size (int): Model input size (used for computing image indices from patch indices).
        device (str): Device identifier (e.g., 'cuda').
        con_label (str): Label used for saving/loading computed counts.
        percentile (float): Chosen percentile for threshold selection (default: 0.1).
        show_zero_count (bool): Whether to include images with zero activated patches in the distribution.

    Returns:
        None. Displays a histogram plot of activated patch counts per concept.
    """
    concepts = sorted(gt_patches_per_concept_train.keys())
    split_df = get_split_df(dataset_name)
    
    try:
        # Load precomputed activated patch counts
        activated_counts_train = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_{con_label}.pt', weights_only=False)
        activated_counts_test  = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_{con_label}.pt', weights_only=False)
    except:
        # Compute thresholds if not already stored
        curr_thresholds = compute_concept_thresholds(gt_patches_per_concept_test,  
                                                         cos_sims, percentile, device=device, 
                                                         dataset_name=f'{dataset_name}', con_label=con_label,
                                                         n_vectors=1,
                                                         n_concepts_to_print=0)
        activated_counts_train, activated_counts_test = count_activated_patches(cos_sims, curr_thresholds, model_input_size, dataset_name)
        torch.save(activated_counts_train, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_{con_label}.pt')
        torch.save(activated_counts_test, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_{con_label}.pt')
    
    if curr_concepts is None:
        curr_concepts = concepts
        
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(curr_concepts)))

    # Histogram bins based on max activation count
    max_count = max([max(activated_counts_train.get(c, {}).values(), default=0) for c in curr_concepts] +
                    [max(activated_counts_test.get(c, {}).values(), default=0) for c in curr_concepts])
    bins = np.arange(0, max_count + 2) - 0.5  # Align bins to integer counts

    for concept, color in zip(curr_concepts, colors):
        counts_train = np.array(list(activated_counts_train.get(concept, {}).values()))
        counts_test  = np.array(list(activated_counts_test.get(concept, {}).values()))

        if not show_zero_count:
            counts_train = counts_train[counts_train > 0]
            counts_test = counts_test[counts_test > 0]

        # Plot train set (dotted)
        if counts_train.size > 0:
            hist_train, _ = np.histogram(counts_train, bins=bins, density=True)
            ax.plot(bins[:-1], hist_train, linestyle=':', color=color, alpha=0.8)

        # Plot test set (solid)
        if counts_test.size > 0:
            hist_test, _ = np.histogram(counts_test, bins=bins, density=True)
            ax.plot(bins[:-1], hist_test, linestyle='-', color=color, label=concept)

    ax.set_xlabel('Activated Patch Count Per Image')
    ax.set_ylabel('Normalized Frequency')
    ax.set_title(f'Distribution of Activated Patch Counts Per Concept ({dataset_name}, {percentile*100:.0f}%)')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legend with concept names and line styles
    handles = [Line2D([0], [0], color=color, linestyle='-', lw=2, label=concept) for concept, color in zip(curr_concepts, colors)]
    handles.append(Line2D([0], [0], color='black', linestyle='-', lw=2, label='Test (Solid)'))
    handles.append(Line2D([0], [0], color='black', linestyle=':', lw=2, label='Train (Dotted)'))
    
    ax.legend(handles=handles, title="Concepts", loc='upper right')

    plt.tight_layout()
    plt.show()

    
def count_gt_activated_patches_per_image(gt_patches_per_concept_train, gt_patches_per_concept_test, model_input_size, dataset_name):
    """
    Computes the number of patches per image that are associated with the ground truth (GT) concept.

    Args:
        gt_patches_per_concept_train (dict): Mapping from concept to a list of patch indices containing the concept in the training set.
        gt_patches_per_concept_test (dict): Mapping from concept to a list of patch indices containing the concept in the test set.
        dataset_name (str): Dataset name for retrieving patch and image metadata.
        model_input_size (int): Input resolution of the model (e.g., 224).

    Returns:
        tuple: Two dictionaries mapping each concept to a dictionary of image indices and their GT patch counts 
               (one for train and one for test).
    """
    patch_split_df = get_patch_split_df(dataset_name,
                                        patch_size=14,
                                        model_input_size=model_input_size)
    relevant_indices = filter_patches_by_image_presence(patch_split_df.index,
                                                dataset_name,
                                                model_input_size).tolist()

    # Precompute sets for fast membership tests
    gt_train_sets = {c: set(v) for c, v in gt_patches_per_concept_train.items()}
    gt_test_sets  = {c: set(v) for c, v in gt_patches_per_concept_test.items()}

    gt_counts_train = defaultdict(lambda: defaultdict(int))
    gt_counts_test  = defaultdict(lambda: defaultdict(int))

    for patch_idx in tqdm(relevant_indices):
        if model_input_size[0] == 'text':
            sample_idx = get_sent_idx_from_global_token_idx(patch_idx, dataset_name)
        else:
            sample_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size=14)
            
        for concept in gt_train_sets:
            if patch_idx in gt_train_sets[concept]:
                gt_counts_train[concept][sample_idx] += 1
            elif patch_idx in gt_test_sets[concept]:
                gt_counts_test[concept][sample_idx] += 1
            else:
                if patch_split_df[patch_idx] == 'train':
                    gt_counts_train[concept][sample_idx] += 0
                elif patch_split_df[patch_idx] == 'test':
                    gt_counts_test[concept][sample_idx] += 0

    # Convert inner defaultdicts to normal dicts (optional)
    gt_counts_train = {c: dict(d) for c, d in gt_counts_train.items()}
    gt_counts_test  = {c: dict(d) for c, d in gt_counts_test.items()}

    return gt_counts_train, gt_counts_test
    
    

def count_activated_patches_splitby_inconcept(gt_images_per_concept_train, gt_images_per_concept_test, cos_sims, curr_thresholds, model_input_size, dataset_name):
    """
    Counts the number of activated patches per image for each concept.

    Args:
        cos_sims (pd.DataFrame): DataFrame with cosine similarity values (rows: patches, columns: concepts).
        curr_thresholds (dict): Mapping from concept to a tuple where the first element is the threshold.
        model_input_size (int): Input size used to compute the image index from a patch index.
        dataset_name (str): Name of the dataset to obtain the train/test split.

    Returns:
        tuple: Two dictionaries mapping each concept to a dictionary of image indices and their activated patch counts 
               (one for train and one for test).
    """
    split_df = get_split_df(dataset_name)

    activated_patch_counts_train_inconcept = defaultdict(lambda: defaultdict(int))
    activated_patch_counts_test_inconcept = defaultdict(lambda: defaultdict(int))
    activated_patch_counts_train_outconcept = defaultdict(lambda: defaultdict(int))
    activated_patch_counts_test_outconcept = defaultdict(lambda: defaultdict(int))

    # Filter patches that are 'padding' given the preprocessing schemes
    relevant_indices = filter_patches_by_image_presence(cos_sims.index, dataset_name, model_input_size).tolist()
    cos_sims = cos_sims.loc[relevant_indices]

    concepts = curr_thresholds.keys()
    for patch_idx, cossim_vals in tqdm(cos_sims.iterrows(), total=len(cos_sims)):
        
        if model_input_size[0] == 'text':
            sample_idx = get_sent_idx_from_global_token_idx(patch_idx, dataset_name)
        else:
            sample_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size=14)
        
        for concept in concepts:
            threshold = curr_thresholds[concept][0] 
            if cossim_vals[concept] >= threshold: #case where patch is activated
                if split_df[sample_idx] == 'train':
                    if sample_idx in gt_images_per_concept_train[concept]:
                        activated_patch_counts_train_inconcept[concept][sample_idx] += 1
                    else:
                        activated_patch_counts_train_outconcept[concept][sample_idx] += 1
                elif split_df[sample_idx] == 'test':
                    if sample_idx in gt_images_per_concept_test[concept]:
                        activated_patch_counts_test_inconcept[concept][sample_idx] += 1
                    else:
                        activated_patch_counts_test_outconcept[concept][sample_idx] += 1
            else: #case where patch isn't activation
                if split_df[sample_idx] == 'train':
                    if sample_idx in gt_images_per_concept_train[concept]:
                        activated_patch_counts_train_inconcept[concept][sample_idx] += 0
                    else:
                        activated_patch_counts_train_outconcept[concept][sample_idx] += 0
                elif split_df[sample_idx] == 'test':
                    if sample_idx in gt_images_per_concept_test[concept]:
                        activated_patch_counts_test_inconcept[concept][sample_idx] += 0
                    else:
                        activated_patch_counts_test_outconcept[concept][sample_idx] += 0
                            
    activated_patch_counts_train_inconcept = {k: dict(v) for k, v in activated_patch_counts_train_inconcept.items()}
    activated_patch_counts_test_inconcept = {k: dict(v) for k, v in activated_patch_counts_test_inconcept.items()}
    activated_patch_counts_train_outconcept = {k: dict(v) for k, v in activated_patch_counts_train_outconcept.items()}
    activated_patch_counts_test_outconcept = {k: dict(v) for k, v in activated_patch_counts_test_outconcept.items()}
    return activated_patch_counts_train_inconcept, activated_patch_counts_train_outconcept, activated_patch_counts_test_inconcept, activated_patch_counts_test_outconcept
  
    
def compute_multiple_activation_analyses(cos_sims, gt_patches_per_concept_train, gt_patches_per_concept_test, gt_images_per_concept_train, gt_images_per_concept_test, model_input_size, dataset_name, con_label, device):
    try:
        gt_activated_counts_train = torch.load(f'Quant_Results/{dataset_name}/gt_activated_counts_train_{con_label}.pt', weights_only=False)
        gt_activated_counts_test = torch.load(f'Quant_Results/{dataset_name}/gt_activated_counts_test_{con_label}.pt', weights_only=False)
    except:
        gt_activated_counts_train, gt_activated_counts_test = count_gt_activated_patches_per_image(
            gt_patches_per_concept_train, gt_patches_per_concept_test, model_input_size, dataset_name
        )
        torch.save(gt_activated_counts_train, f'Quant_Results/{dataset_name}/gt_activated_counts_train_{con_label}.pt', weights_only=False)
        torch.save(gt_activated_counts_test, f'Quant_Results/{dataset_name}/gt_activated_counts_test_{con_label}.pt', weights_only=False)
    
    for percentile in tqdm([0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]):
        try:
            activated_counts_train_inconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_inconcept_{con_label}.pt', weights_only=False)   
            activated_counts_test_inconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_inconcept_{con_label}.pt', weights_only=False)
            activated_counts_train_outconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_outconcept_{con_label}.pt', weights_only=False)
            activated_counts_test_outconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_outconcept_{con_label}.pt', weights_only=False)
        except:
            curr_thresholds = compute_concept_thresholds(gt_patches_per_concept_test,  
                                                         cos_sims, percentile, device=device, 
                                                         dataset_name=f'{dataset_name}', con_label=con_label,
                                                         n_vectors=1,
                                                         n_concepts_to_print=0)
            activated_counts_train_inconcept, activated_counts_train_outconcept, activated_counts_test_inconcept, \
                activated_counts_test_outconcept = count_activated_patches_splitby_inconcept(
                    gt_images_per_concept_train, gt_images_per_concept_test, 
                    cos_sims, curr_thresholds, model_input_size, dataset_name
                )
            torch.save(activated_counts_train_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_inconcept_{con_label}.pt')
            torch.save(activated_counts_test_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_inconcept_{con_label}.pt')
            torch.save(activated_counts_train_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_outconcept_{con_label}.pt')
            torch.save(activated_counts_test_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_outconcept_{con_label}.pt')
            
            
def plot_patch_activation_counts_per_concept(
    cos_sims, dataset_name, gt_patches_per_concept_train, 
    gt_patches_per_concept_test, 
    gt_images_per_concept_train,
    gt_images_per_concept_test,
    model_input_size, device, con_label, 
    percentile, concepts=None, show_zero_count=True, split="test"
):
    if concepts is None:
        concepts = sorted(gt_patches_per_concept_train.keys())

    # Load or compute GT patch counts
    try:
        gt_activated_counts_train = torch.load(f'Quant_Results/{dataset_name}/gt_activated_counts_train_{con_label}.pt', weights_only=False)
        gt_activated_counts_test = torch.load(f'Quant_Results/{dataset_name}/gt_activated_counts_test_{con_label}.pt', weights_only=False)
    except:
        gt_activated_counts_train, gt_activated_counts_test = count_gt_activated_patches_per_image(
            gt_patches_per_concept_train, gt_patches_per_concept_test, model_input_size, dataset_name
        )
        torch.save(gt_activated_counts_train, f'Quant_Results/{dataset_name}/gt_activated_counts_train_{con_label}.pt', weights_only=False)
        torch.save(gt_activated_counts_test, f'Quant_Results/{dataset_name}/gt_activated_counts_test_{con_label}.pt', weights_only=False)

    # Load or compute activated patch counts
    try:
        activated_counts_train_inconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_inconcept_{con_label}.pt', weights_only=False)   
        activated_counts_test_inconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_inconcept_{con_label}.pt', weights_only=False)
        activated_counts_train_outconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_outconcept_{con_label}.pt', weights_only=False)
        activated_counts_test_outconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_outconcept_{con_label}.pt', weights_only=False)
    except:
        curr_thresholds = compute_concept_thresholds(gt_patches_per_concept_test,  
                                                         cos_sims, percentile, device=device, 
                                                         dataset_name=f'{dataset_name}', con_label=con_label,
                                                         n_vectors=1,
                                                         n_concepts_to_print=0)
        activated_counts_train_inconcept, activated_counts_train_outconcept, activated_counts_test_inconcept, \
        activated_counts_test_outconcept = count_activated_patches_splitby_inconcept(
            gt_images_per_concept_train, gt_images_per_concept_test,
            cos_sims, curr_thresholds, model_input_size, dataset_name
        )
        torch.save(activated_counts_train_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_inconcept_{con_label}.pt')
        torch.save(activated_counts_test_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_inconcept_{con_label}.pt')
        torch.save(activated_counts_train_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_outconcept_{con_label}.pt')
        torch.save(activated_counts_test_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_outconcept_{con_label}.pt')

    n_concepts = len(concepts)
    fig, axes = plt.subplots(nrows=n_concepts, figsize=(8, (3 * n_concepts) + 1))
    if n_concepts == 1:
        axes = [axes]

    for ax, concept in zip(axes, concepts):
        if split == "train":
            data_groups = {
                "In-Concept": np.array(list(activated_counts_train_inconcept.get(concept, {}).values())),
                "Out-Concept": np.array(list(activated_counts_train_outconcept.get(concept, {}).values())),
                "GT": np.array(list(gt_activated_counts_train.get(concept, {}).values())),
            }
        else:
            data_groups = {
                "In-Concept": np.array(list(activated_counts_test_inconcept.get(concept, {}).values())),
                "Out-Concept": np.array(list(activated_counts_test_outconcept.get(concept, {}).values())),
                "GT": np.array(list(gt_activated_counts_test.get(concept, {}).values())),
            }

        colors = {
            "In-Concept": "blue",
            "Out-Concept": "red",
            "GT": "gray"
        }

        for k in data_groups:
            if not show_zero_count:
                data_groups[k] = data_groups[k][data_groups[k] > 0]

        all_counts = np.concatenate(list(data_groups.values()))
        xmax = int(np.percentile(all_counts, 100))

        if show_zero_count:
            bins_array = np.arange(0, xmax + 2) - 0.5
        else:
            bins_array = np.arange(1, xmax + 2) - 0.5
        bin_centers = (bins_array[:-1] + bins_array[1:]) / 2

        for label, values in data_groups.items():
            if len(values) > 0:
                hist, _ = np.histogram(values, bins=bins_array)
                if hist.sum() > 0:
                    hist = hist / hist.sum()
                ax.plot(bin_centers, hist, label=label, color=colors[label])

        if show_zero_count:
            ax.set_xlim(-0.5, xmax + 1)
            ax.set_xticks(np.arange(0, xmax + 1, max(1, xmax // 10)))
        else:
            ax.set_xlim(0.5, xmax + 1)
            ax.set_xticks(np.arange(1, xmax + 1, max(1, xmax // 10)))

        ax.set_yticks([])
        ax.set_ylabel("# of Images (Normalized)")
        ax.set_title(f"Concept: {concept}")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Activated Patch Count Per Image")
    plt.suptitle(f"Patch Count Distributions ({split.capitalize()}) - {dataset_name}, {percentile*100:.0f}th Percentile")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()
    
    
def plot_patch_activation_counts_averaged(cos_sims, dataset_name, gt_patches_per_concept_train, 
                                    gt_patches_per_concept_test, 
                                    gt_images_per_concept_train,
                                    gt_images_per_concept_test,
                                    model_input_size, device, con_label, 
                                    percentile, concepts=None, show_zero_count=True, split="test"):

    if concepts is None:
        concepts = sorted(gt_patches_per_concept_train.keys())

    try:
        activated_counts_train_inconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_inconcept_{con_label}.pt', weights_only=False)   
        activated_counts_test_inconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_inconcept_{con_label}.pt', weights_only=False)
        activated_counts_train_outconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_outconcept_{con_label}.pt', weights_only=False)
        activated_counts_test_outconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_outconcept_{con_label}.pt', weights_only=False)
        gt_activated_counts_train = torch.load(f'Quant_Results/{dataset_name}/gt_activated_counts_train_{con_label}.pt', weights_only=False)
        gt_activated_counts_test = torch.load(f'Quant_Results/{dataset_name}/gt_activated_counts_test_{con_label}.pt', weights_only=False)
    except:
        curr_thresholds = compute_concept_thresholds(gt_patches_per_concept_test,  
                                                         cos_sims, percentile, device=device, 
                                                         dataset_name=f'{dataset_name}', con_label=con_label,
                                                         n_vectors=1,
                                                         n_concepts_to_print=0)
        activated_counts_train_inconcept, activated_counts_train_outconcept, activated_counts_test_inconcept, \
            activated_counts_test_outconcept = count_activated_patches_splitby_inconcept(
                gt_images_per_concept_train, gt_images_per_concept_test, 
                cos_sims, curr_thresholds, model_input_size, dataset_name
            )
        torch.save(activated_counts_train_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_inconcept_{con_label}.pt')
        torch.save(activated_counts_test_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_inconcept_{con_label}.pt')
        torch.save(activated_counts_train_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_outconcept_{con_label}.pt')
        torch.save(activated_counts_test_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_outconcept_{con_label}.pt')
        gt_activated_counts_train, gt_activated_counts_test = count_gt_activated_patches_per_image(
            gt_patches_per_concept_train, gt_patches_per_concept_test, model_input_size, dataset_name
        )
        torch.save(gt_activated_counts_train, f'Quant_Results/{dataset_name}/gt_activated_counts_train_{con_label}.pt')
        torch.save(gt_activated_counts_test, f'Quant_Results/{dataset_name}/gt_activated_counts_test_{con_label}.pt')

    if split == "train":
        counts_in = activated_counts_train_inconcept
        counts_out = activated_counts_train_outconcept
        gt_counts = gt_activated_counts_train
    else:
        counts_in = activated_counts_test_inconcept
        counts_out = activated_counts_test_outconcept
        gt_counts = gt_activated_counts_test

    all_in, all_out, all_gt = [], [], []

    for concept in concepts:
        c_in = np.array(list(counts_in.get(concept, {}).values()))
        c_out = np.array(list(counts_out.get(concept, {}).values()))
        c_gt = np.array(list(gt_counts.get(concept, {}).values()))

        if not show_zero_count:
            c_in = c_in[c_in > 0]
            c_out = c_out[c_out > 0]
            c_gt = c_gt[c_gt > 0]

        all_in.extend(c_in)
        all_out.extend(c_out)
        all_gt.extend(c_gt)

    all_counts = np.array(all_in + all_out + all_gt)
    xmax = int(np.percentile(all_counts, 100))
    if show_zero_count:
        bins_array = np.arange(0, xmax + 2) - 0.5
    else:
        bins_array = np.arange(1, xmax + 2) - 0.5
    bin_centers = (bins_array[:-1] + bins_array[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 4))

    def plot_hist_line(data, label, color):
        if len(data) == 0:
            return
        hist_vals, _ = np.histogram(data, bins=bins_array)
        hist_vals = hist_vals / hist_vals.sum() if hist_vals.sum() > 0 else hist_vals
        ax.plot(bin_centers, hist_vals, label=label, color=color)

    plot_hist_line(all_in, f"{split.title()} In-Concept", "blue")
    plot_hist_line(all_out, f"{split.title()} Out-Concept", "red")
    plot_hist_line(all_gt, f"{split.title()} GT", "gray")
    
    if show_zero_count:
        ax.set_xlim(-0.5, xmax + 1)
        ax.set_xticks(np.arange(0, xmax + 1, max(1, xmax // 10)))
    else:
        ax.set_xlim(0.5, xmax + 1)
        ax.set_xticks(np.arange(1, xmax + 1, max(1, xmax // 10)))

    ax.set_xlabel("Activated Patch Count Per Image")
    ax.set_ylabel("# of Images (Normalized)")
    ax.set_title(f"Activated Patch Count Distribution (Averaged over Concepts) ({split.title()} Split)")
    ax.legend()
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_avg_activation_distribution_across_percentiles(cos_sims, dataset_name, gt_patches_per_concept_train,
                                                        gt_patches_per_concept_test, gt_images_per_concept_train,
                                                        gt_images_per_concept_test,
                                                        model_input_size, device, con_label, percentiles,
                                                        concepts=None, show_zero_count=True, split="test"):
    
    if concepts is None:
        concepts = sorted(gt_patches_per_concept_train.keys())

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))

    for percentile, color in zip(percentiles, colors):
        try:
            activated_counts_train_inconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_inconcept_{con_label}.pt', weights_only=False)
            activated_counts_test_inconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_inconcept_{con_label}.pt', weights_only=False)
            activated_counts_train_outconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_outconcept_{con_label}.pt', weights_only=False)
            activated_counts_test_outconcept = torch.load(f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_outconcept_{con_label}.pt', weights_only=False)
            gt_activated_counts_train = torch.load(f'Quant_Results/{dataset_name}/gt_activated_counts_train_{con_label}.pt', weights_only=False)
            gt_activated_counts_test = torch.load(f'Quant_Results/{dataset_name}/gt_activated_counts_test_{con_label}.pt', weights_only=False)
        except:
            curr_thresholds = compute_concept_thresholds(gt_patches_per_concept_test,  
                                                         cos_sims, percentile, device=device, 
                                                         dataset_name=f'{dataset_name}', con_label=con_label,
                                                         n_vectors=1,
                                                         n_concepts_to_print=0)
            activated_counts_train_inconcept, activated_counts_train_outconcept, activated_counts_test_inconcept, \
                activated_counts_test_outconcept = count_activated_patches_splitby_inconcept(
                    gt_images_per_concept_train, gt_images_per_concept_test,
                    cos_sims, curr_thresholds, model_input_size, dataset_name
                )
            torch.save(activated_counts_train_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_inconcept_{con_label}.pt')
            torch.save(activated_counts_test_inconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_inconcept_{con_label}.pt')
            torch.save(activated_counts_train_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_train_outconcept_{con_label}.pt')
            torch.save(activated_counts_test_outconcept, f'Quant_Results/{dataset_name}/percentile_{percentile}_activated_counts_test_outconcept_{con_label}.pt')
            gt_activated_counts_train, gt_activated_counts_test = count_gt_activated_patches_per_image(
                gt_patches_per_concept_train, gt_patches_per_concept_test, model_input_size, dataset_name
            )
            torch.save(gt_activated_counts_train, f'Quant_Results/{dataset_name}/gt_activated_counts_train_{con_label}.pt')
            torch.save(gt_activated_counts_test, f'Quant_Results/{dataset_name}/gt_activated_counts_test_{con_label}.pt')

        if split == "train":
            counts_in = activated_counts_train_inconcept
            counts_out = activated_counts_train_outconcept
            gt_counts = gt_activated_counts_train
        else:
            counts_in = activated_counts_test_inconcept
            counts_out = activated_counts_test_outconcept
            gt_counts = gt_activated_counts_test

        all_in, all_out, all_gt = [], [], []

        for concept in concepts:
            c_in = np.array(list(counts_in.get(concept, {}).values()))
            c_out = np.array(list(counts_out.get(concept, {}).values()))
            c_gt = np.array(list(gt_counts.get(concept, {}).values()))

            if not show_zero_count:
                c_in = c_in[c_in > 0]
                c_out = c_out[c_out > 0]
                c_gt = c_gt[c_gt > 0]

            all_in.extend(c_in)
            all_out.extend(c_out)
            all_gt.extend(c_gt)

        all_counts = np.array(all_in + all_out + all_gt)
        if len(all_counts) == 0:
            continue

        xmax = int(np.percentile(all_counts, 100))
        if show_zero_count:
            bins_array = np.arange(0, xmax + 2) - 0.5
        else:
            bins_array = np.arange(1, xmax + 2) - 0.5
        bin_centers = (bins_array[:-1] + bins_array[1:]) / 2

        def plot_line(data, style, color):
            if len(data) == 0:
                return
            hist_vals, _ = np.histogram(data, bins=bins_array)
            hist_vals = hist_vals / hist_vals.sum() if hist_vals.sum() > 0 else hist_vals
            ax.plot(bin_centers, hist_vals, linestyle=style, color=color)

        plot_line(all_in, "-", color)
        plot_line(all_out, ":", color)

    # Plot ground truth (same for all percentiles)
    if len(all_gt) > 0:
        hist_vals, _ = np.histogram(all_gt, bins=bins_array)
        hist_vals = hist_vals / hist_vals.sum() if hist_vals.sum() > 0 else hist_vals
        ax.plot(bin_centers, hist_vals, linestyle="-", color="gray", label="GT", linewidth=3)

    ax.set_xlabel("Activated Patch Count Per Image")
    ax.set_ylabel("# of Images (Normalized)")
    ax.set_title(f"Average Activation Distributions ({split.title()} Split)")

    # Custom legend
    percentile_lines = [mlines.Line2D([], [], color=color, linestyle='-', label=f"{p:.2f}") for p, color in zip(percentiles, colors)]
    line_example = [
        mlines.Line2D([], [], color="gray", linestyle="-", label="GT"),
        mlines.Line2D([], [], color="black", linestyle="-", label="In-Concept"),
        mlines.Line2D([], [], color="black", linestyle=":", label="Out-Concept")
    ]
    ax.legend(handles=percentile_lines + line_example, title="Percentiles")

    if show_zero_count:
        ax.set_xlim(-0.5, xmax + 1)
        ax.set_xticks(np.arange(0, xmax + 1, max(1, xmax // 10)))
    else:
        ax.set_xlim(0.5, xmax + 1)
        ax.set_xticks(np.arange(1, xmax + 1, max(1, xmax // 10)))
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    
    
def compute_cooccurrence_detection_iou(concept1, concept2, percentile, act_metrics, gt_patches_per_concept_test, gt_images_per_concept_test, dataset_name, model_input_size, device, con_label):
    """
    Computes the IoU of detected patches for two concepts, restricted to locations where either occurs.

    Args:
        concept1 (str): The first concept.
        concept2 (str): The second concept.
        percentile (float): Percentile threshold for detection.
        act_metrics (pd.DataFrame): Activation metrics for all patches.
        gt_patches_per_concept_test (dict): Ground truth patch indices for each concept (test set).
        dataset_name (str): Dataset name for patch filtering.
        model_input_size (int): Input image size for patch indexing.
        device (str): Device used (e.g., 'cuda').

    Returns:
        float: Intersection over Union score.
    """
    detect_thresholds = compute_concept_thresholds(
        gt_patches_per_concept_test, act_metrics, percentile, n_vectors=1, device=device, n_concepts_to_print=0,
        dataset_name=f'{dataset_name}', con_label=con_label
    )

    #get the raw binary patch activations for each concept
    patch_activations1 = act_metrics[concept1] >= detect_thresholds[concept1][0]
    patch_activations2 = act_metrics[concept2] >= detect_thresholds[concept2][0]
    
    #cooccur_set = tells you all of the images where {concept1} and {concept2} cooccur
    gt1_set = set(gt_images_per_concept_test[concept1])
    gt2_set = set(gt_images_per_concept_test[concept2])
    cooccur_set = gt1_set & gt2_set
    
    #Filter patches to those in co-occuring images
    patch_indices = act_metrics.index
    if model_input_size[0] == 'text':
        cooccur_mask = [
            get_sent_idx_from_global_token_idx(patch_idx, dataset_name) in cooccur_set
            for patch_idx in patch_indices
        ]
    else:
        cooccur_mask = [
            get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size=14) in cooccur_set
            for patch_idx in patch_indices
        ]
    cooccur_mask = pd.Series(cooccur_mask, index=patch_indices)

    #Mask patch activations
    patch_activations1 = patch_activations1[cooccur_mask]
    patch_activations2 = patch_activations2[cooccur_mask]
    
    #filter out padding
    relevant_indices = filter_patches_by_image_presence(patch_activations1.index, dataset_name, model_input_size)
    patch_activations1 = patch_activations1.loc[relevant_indices]
    patch_activations2 = patch_activations2.loc[relevant_indices]

    #Compute IoU
    t1_vals = patch_activations1.values.astype(bool)
    t2_vals = patch_activations2.values.astype(bool)
    intersection = (t1_vals & t2_vals).sum()
    union = (t1_vals | t2_vals).sum()
    iou = intersection / (union + 1e-6)

    return iou, intersection, union
     

def plot_cooccurrence_detection_iou_over_percentiles(percentiles, concept1, concept2, act_metrics, 
                                                     gt_patches_per_concept_test, gt_images_per_concept_test,
                                                     dataset_name, model_input_size, device, con_label):
    ious, intersections, unions = [], [], []

    for percentile in tqdm(percentiles, desc="Computing IOUs"):
        iou, intersection, union = compute_cooccurrence_detection_iou(
            concept1, concept2, percentile, act_metrics,
            gt_patches_per_concept_test, gt_images_per_concept_test, 
            dataset_name, model_input_size, device, con_label
        )
        ious.append(iou)
        intersections.append(intersection)
        unions.append(union)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Left y-axis: IOU
    ax1.plot(percentiles, ious, marker='o', color='purple', label="IOU")
    ax1.set_xlabel("Detection Percentile Threshold")
    ax1.set_ylabel("IOU", color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.grid(True)
    ax1.set_ylim(-0.1, 1.1)

    # Right y-axis: Intersection and Union (linear scale)
    ax2 = ax1.twinx()
    ax2.plot(percentiles, intersections, linestyle='--', color='gray', label="Intersection")
    ax2.plot(percentiles, unions, linestyle='--', color='lightgray', label="Union")
    ax2.set_ylabel("Count (Intersection / Union)", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Optionally set custom y-ticks if needed
    # all_counts = intersections + unions
    # max_count = max(all_counts)
    # ax2.set_yticks([1, 10, 100, 1000] if max_count >= 1000 else sorted(set(all_counts)))
    ax2.set_ylim(-10, max(unions) + 10)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title(f"Patch Co-occurrence IOU for '{concept1}' and '{concept2}'")
    plt.tight_layout()
    plt.show()


def find_best_detection_percentiles_cal(dataset_name, con_label, percentiles, sample_type='patch'):
    """
    Find the best percentile threshold for each concept based on calibration set F1 scores.
    
    Args:
        dataset_name: Name of the dataset
        con_label: Concept label identifier
        percentiles: List of percentiles to search over
        sample_type: 'patch' or 'cls' 
        
    Returns:
        dict: Mapping of concept -> (best_percentile, best_f1_score, best_threshold)
    """
    import os
    import ast
    
    # Results directory
    base_dir = f'Quant_Results/{dataset_name}'
    save_dir = f'Best_Detection_Percentiles_Cal/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize results dictionary
    best_per_concept = {}
    
    # For unsupervised methods, we need to handle (concept, cluster) pairs
    is_unsupervised = 'kmeans' in con_label or 'sae' in con_label
    
    # Load calibration detection metrics for each percentile
    all_metrics = {}
    missing_count = 0
    for per in percentiles:
        # Look for calibration results
        # For unsupervised: CSV files, for supervised: PT files
        if is_unsupervised:
            cal_metrics_path = f"{base_dir}/detectionmetrics_allpairs_per_{per}_{con_label}_cal.csv"
            if os.path.exists(cal_metrics_path):
                metrics_df = pd.read_csv(cal_metrics_path)
                all_metrics[per] = metrics_df
            else:
                missing_count += 1
        else:
            # Supervised saves as .pt files
            cal_metrics_path = f"{base_dir}/detectionmetrics_per_{per}_{con_label}_cal.pt"
            if os.path.exists(cal_metrics_path):
                metrics_df = torch.load(cal_metrics_path, weights_only=False)
                all_metrics[per] = metrics_df
            else:
                missing_count += 1
    
    if not all_metrics:
        # Check what files actually exist in the directory
        import glob
        if is_unsupervised:
            pattern = f"{base_dir}/detectionmetrics_allpairs_per_*_{con_label}_cal.csv"
        else:
            pattern = f"{base_dir}/detectionmetrics_per_*_{con_label}_cal.pt"
        
        existing_files = glob.glob(pattern)
        
        raise FileNotFoundError(
            f"No calibration metrics found for {con_label}.\n"
            f"Expected pattern: {pattern}\n"
            f"Found {len(existing_files)} matching files: {existing_files[:3]}..."
        )
    
    # Load thresholds
    threshold_file = f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt' if is_unsupervised else f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt'
    if os.path.exists(threshold_file):
        all_thresholds = torch.load(threshold_file, weights_only=False)
    else:
        print(f"Warning: Threshold file not found: {threshold_file}")
        all_thresholds = {}
    
    # Process each concept
    if is_unsupervised:
        # For unsupervised, group by concept from (concept, cluster) pairs
        concepts_seen = set()
        for per, metrics_df in all_metrics.items():
            for idx, row in metrics_df.iterrows():
                concept, cluster = ast.literal_eval(row['concept'])
                if concept not in concepts_seen:
                    concepts_seen.add(concept)
        
        # Find best percentile for each concept
        for concept in concepts_seen:
            best_f1 = -1
            best_per = None
            best_threshold = None
            best_cluster = None
            
            for per, metrics_df in all_metrics.items():
                # Find all clusters for this concept
                concept_rows = metrics_df[metrics_df['concept'].apply(lambda x: ast.literal_eval(x)[0] == concept)]
                
                for idx, row in concept_rows.iterrows():
                    _, cluster = ast.literal_eval(row['concept'])
                    f1 = row['f1']
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_per = per
                        best_cluster = cluster
                        # Get threshold
                        if per in all_thresholds and (concept, str(cluster)) in all_thresholds[per]:
                            best_threshold = all_thresholds[per][(concept, str(cluster))][0]
                        else:
                            best_threshold = None
            
            if best_per is not None:
                best_per_concept[concept] = {
                    'best_percentile': best_per,
                    'best_f1': best_f1,
                    'best_threshold': best_threshold,
                    'best_cluster': best_cluster
                }
    else:
        # For supervised, process each concept directly
        for per, metrics_df in all_metrics.items():
            for idx, row in metrics_df.iterrows():
                concept = row['concept']
                f1 = row['f1']
                
                if concept not in best_per_concept or f1 > best_per_concept[concept]['best_f1']:
                    # Get threshold
                    if per in all_thresholds and concept in all_thresholds[per]:
                        threshold = all_thresholds[per][concept][0]
                    else:
                        threshold = None
                        
                    best_per_concept[concept] = {
                        'best_percentile': per,
                        'best_f1': f1,
                        'best_threshold': threshold
                    }
    
    # Save results
    save_path = os.path.join(save_dir, f'best_percentiles_{con_label}.pt')
    torch.save(best_per_concept, save_path)
    
    # Also save as readable CSV
    csv_data = []
    for concept, info in best_per_concept.items():
        csv_data.append({
            'concept': concept,
            'best_percentile': info['best_percentile'],
            'best_f1': info['best_f1'],
            'best_threshold': info['best_threshold'],
            'best_cluster': info.get('best_cluster', 'N/A')
        })
    
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_df = csv_df.sort_values('concept')
        csv_path = os.path.join(save_dir, f'best_percentiles_{con_label}.csv')
        csv_df.to_csv(csv_path, index=False)
        
        # print(f"Saved best calibration percentiles to {save_path}")
        # print(f"Summary: {len(best_per_concept)} concepts analyzed")
        # print(f"Average best percentile: {np.mean([info['best_percentile'] for info in best_per_concept.values()]):.3f}")
        # print(f"Average best F1: {np.mean([info['best_f1'] for info in best_per_concept.values()]):.3f}")
    
    return best_per_concept
    
    

    
    