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

import importlib
import utils.general_utils
import utils.text_visualization_utils
import utils.patch_alignment_utils
importlib.reload(utils.patch_alignment_utils)
importlib.reload(utils.general_utils)
importlib.reload(utils.text_visualization_utils)

from utils.general_utils import compute_cossim_w_vector, get_split_df, create_binary_labels, retrieve_topn_images_byconcepts
from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence, get_image_idx_from_global_patch_idx, compute_patches_per_image
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices
from utils.text_visualization_utils import get_glob_tok_indices_from_sent_idx, get_sent_idx_from_global_token_idx

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
        concept_thresholds[concept] = (threshold_val, np.nan)#rand_threshold)

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


def compute_concept_thresholds_over_percentiles(gt_samples_per_concept_cal, cos_sims, percentiles, device, dataset_name, con_label, n_vectors=5, n_concepts_to_print=0):
    """
    Computes thresholds for multiple percentiles with efficient caching.
    
    Args:
        gt_samples_per_concept (dict): Mapping of concept to list of patch indices
        cos_sims (pd.DataFrame): Cosine similarity matrix (rows: patches, columns: concepts)
        percentiles (list): List of percentile values to compute thresholds for
        device (str): Compute device (e.g., "cuda")
        dataset_name (str): Name of dataset for cache file
        con_label (str): Label for cache file
        n_vectors (int): Number of random vectors (unused in current implementation)
        n_concepts_to_print (int): Number of concepts to print for debugging
        
    Returns:
        dict: Mapping from percentile -> concept -> (threshold, random_threshold)
    """
    cache_file = f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt'
    
    # Try to load existing thresholds
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
                concept_thresholds[concept] = (threshold_val, np.nan)
            
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
    detect_percentile,
    invert_percentiles,
    act_metrics,
    concepts,
    gt_samples_per_concept,
    gt_samples_per_concept_cal,
    relevant_indices,
    all_concept_labels,
    device,
    dataset_name,
    model_input_size,
    con_label,
    all_object_patches=None,
    patch_size=14
):
    # Compute all thresholds
    all_percentiles = [detect_percentile] + list(invert_percentiles)
    # thresholds = compute_concept_thresholds_over_percentiles(
    #     gt_samples_per_concept_test,
    #     act_metrics,
    #     all_percentiles,
    #     device=device,
    #     dataset_name=f'{dataset_name}-Cal',
    #     con_label=con_label,
    #     n_vectors=1,
    #     n_concepts_to_print=0
    # )
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
                    matched_thresholds[cluster_id] = thresholds_dict[key]  # keep full (val, nan) tuple

            thresholds[percentile] = matched_thresholds

    # Detection mask from detection threshold
    detect_thresholds = thresholds[detect_percentile]
    detected_patch_masks = get_patch_detection_tensor(
        act_metrics, detect_thresholds, model_input_size, dataset_name
    )

    concept_keys = set(detect_thresholds.keys()) & set(concepts.keys())

    if all_object_patches is not None:
        relevant_indices = torch.tensor(
            [int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches]
        )

    relevant_indices_list = relevant_indices.tolist()
    n_concepts = len(concept_keys)
    n_samples = len(relevant_indices)

    act_vals_all = torch.zeros((n_samples, n_concepts), device=device)
    detected_patches_all = torch.zeros((n_samples, n_concepts), dtype=torch.bool, device=device)
    gt_masks_all = torch.zeros((n_samples, n_concepts), dtype=torch.bool, device=device)

    for i, concept in enumerate(concept_keys):
        act_vals_all[:, i] = torch.tensor(
            act_metrics[str(concept)].loc[relevant_indices_list].values, device=device
        )
        detected_patches_all[:, i] = torch.tensor(
            detected_patch_masks[concept].loc[relevant_indices_list].values, device=device
        )
        vals = all_concept_labels[concept][relevant_indices_list]
        if isinstance(vals, torch.Tensor):
            mask = (vals == 1).clone().detach().to(device)
        else:
            mask = torch.tensor(vals == 1, device=device)

        gt_masks_all[:, i] = mask

    metrics_dfs = {p: {'tp_count': {}, 'fp_count': {}, 'tn_count': {}, 'fn_count': {}} for p in invert_percentiles}

    for invert_percentile in invert_percentiles:
        thresh_tensor = torch.tensor(
            [thresholds[invert_percentile][c][0] for c in concept_keys], device=device
        )

        activated_patches = (act_vals_all >= thresh_tensor.unsqueeze(0)) & detected_patches_all

        tp_counts = torch.sum(activated_patches & gt_masks_all, dim=0)
        fn_counts = torch.sum((~activated_patches) & gt_masks_all, dim=0)
        fp_counts = torch.sum(activated_patches & (~gt_masks_all), dim=0)
        tn_counts = torch.sum((~activated_patches) & (~gt_masks_all), dim=0)

        for i, concept in enumerate(concept_keys):
            metrics_dfs[invert_percentile]['tp_count'][concept] = tp_counts[i].item()
            metrics_dfs[invert_percentile]['fn_count'][concept] = fn_counts[i].item()
            metrics_dfs[invert_percentile]['fp_count'][concept] = fp_counts[i].item()
            metrics_dfs[invert_percentile]['tn_count'][concept] = tn_counts[i].item()

    final_metrics = {}
    for invert_percentile in invert_percentiles:
        counts = metrics_dfs[invert_percentile]
        metrics_df = compute_concept_metrics(
            counts['fp_count'], counts['fn_count'],
            counts['tp_count'], counts['tn_count'],
            concept_keys, dataset_name, con_label,
            just_obj=(all_object_patches is not None),
            invert_percentile=invert_percentile,
            detect_percentile=detect_percentile
        )
        final_metrics[invert_percentile] = metrics_df

    return final_metrics







    
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


def detect_then_invert_metrics_over_percentiles(detect_percentiles, invert_percentiles, 
                                              act_metrics, concepts, gt_samples_per_concept, 
                                              gt_samples_per_concept_cal, device, dataset_name, 
                                              model_input_size, con_label, all_object_patches=None,
                                              patch_size=14):
    """
    Evaluates metrics across all detect percentile combinations on CALIBRATION set.
    This is used to find optimal thresholds - always uses calibration split.
    More efficient version that handles multiple invert percentiles at once.
    """
    # Compute all thresholds at once for caching
    all_percentiles = sorted(list(set(detect_percentiles) | set(invert_percentiles)))
    # thresholds = compute_concept_thresholds_over_percentiles(
    #     gt_samples_per_concept_test,
    #     act_metrics,
    #     all_percentiles,
    #     device=device,
    #     dataset_name=f'{dataset_name}-Cal',
    #     con_label=con_label,
    #     n_vectors=1,
    #     n_concepts_to_print=0
    # )
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
                    matched_thresholds[cluster_id] = thresholds_dict[key]  # keep full (val, nan) tuple

            thresholds[percentile] = matched_thresholds
    
    total_iters = len(detect_percentiles)  # Now we only iterate over detect percentiles
    pbar = tqdm(total=total_iters, desc="Evaluating thresholds")

    # Get the split dataframe and indices - USE CALIBRATION SET
    split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    cal_indices = torch.tensor(split_df.index[split_df == 'cal'].tolist())
    if model_input_size[0] == 'text':
        relevant_indices = cal_indices
    else:
        relevant_indices = filter_patches_by_image_presence(cal_indices, dataset_name, model_input_size)

    # Get ground truth labels from calibration set
    all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept_cal)
    
    for detect_percentile in detect_percentiles:
        # Get valid invert percentiles for this detect percentile
        valid_invert_percentiles = [p for p in invert_percentiles if p >= detect_percentile]
        
        if not valid_invert_percentiles:
            continue
            

        # Compute metrics for all valid invert percentiles at once
        metrics = detect_then_invert_metrics(
            detect_percentile, valid_invert_percentiles,
            act_metrics, concepts,
            gt_samples_per_concept, gt_samples_per_concept_cal,
            relevant_indices, all_concept_labels,
            device, dataset_name, model_input_size, con_label,
            all_object_patches=None,
            patch_size=patch_size
        )
        
        
#     # Compute metrics with object patches for all valid invert percentiles
#     if all_object_patches is not None:
#         metrics = detect_then_invert_metrics(
#             detect_percentile, valid_invert_percentiles,
#             act_metrics, concepts,
#             gt_samples_per_concept, gt_samples_per_concept_cal,
#             device, dataset_name, model_input_size, con_label,
#             all_object_patches=all_object_patches, n_trials=n_trials,
#             balance_dataset=balance_dataset, patch_size=patch_size
#         )
        
        pbar.update(1)
    
    pbar.close()


def find_optimal_detect_invert_thresholds(detect_percentiles, invert_percentiles, dataset_name, 
                                        con_label, optimization_metric='f1'):
    """
    Find optimal detect/invert percentile pairs for each concept from calibration results.
    Must be run AFTER detect_then_invert_metrics_over_percentiles.
    """
    import os
    import numpy as np
    
    optimal_thresholds = {}
    results_dir = f"Quant_Results/{dataset_name}"
    
    # Get all concepts from any results file
    concept_names = set()
    
    for detect_p in detect_percentiles:
        for invert_p in invert_percentiles:
            if invert_p >= detect_p:
                filename = f"{results_dir}/detectfirst_{detect_p*100}_per_{invert_p*100}_{con_label}.csv"
                try:
                    df = pd.read_csv(filename)
                    concept_names.update(df['concept'].astype(str))
                    break
                except FileNotFoundError:
                    continue
        if concept_names:
            break
    
    print(f"Finding optimal thresholds for {len(concept_names)} concepts using {optimization_metric}...")
    
    # For each concept, find best detect/invert combination
    for concept in tqdm(concept_names, desc="Optimizing concepts"):
        best_score = -1
        best_detect = None
        best_invert = None
        
        for detect_p in detect_percentiles:
            for invert_p in invert_percentiles:
                if invert_p >= detect_p:  # Valid combination
                    filename = f"{results_dir}/detectfirst_{detect_p*100}_per_{invert_p*100}_{con_label}.csv"
                    try:
                        df = pd.read_csv(filename)
                        # Find the row for this concept
                        concept_row = df[df['concept'] == str(concept)]
                        if not concept_row.empty and optimization_metric in concept_row.columns:
                            score = concept_row[optimization_metric].iloc[0]
                            # Handle NaN scores
                            if pd.notna(score) and score > best_score:
                                best_score = score
                                best_detect = detect_p
                                best_invert = invert_p
                    except FileNotFoundError:
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
        else:
            print(f"  Warning: No valid thresholds found for concept {concept}")
    
    # Save optimal thresholds
    os.makedirs(f'Detect_Invert_Thresholds/{dataset_name}', exist_ok=True)
    threshold_file = f'Detect_Invert_Thresholds/{dataset_name}/optimal_{optimization_metric}_{con_label}.pt'
    torch.save(optimal_thresholds, threshold_file)
    
    # Print summary
    if optimal_thresholds:
        detect_percs = [v['detect_percentile'] for v in optimal_thresholds.values()]
        invert_percs = [v['invert_percentile'] for v in optimal_thresholds.values()]
        scores = [v[f'best_{optimization_metric}'] for v in optimal_thresholds.values()]
        
        print(f"\nOptimization Summary ({len(optimal_thresholds)} concepts):")
        print(f"  Avg detect percentile: {np.mean(detect_percs):.3f} ± {np.std(detect_percs):.3f}")
        print(f"  Avg invert percentile: {np.mean(invert_percs):.3f} ± {np.std(invert_percs):.3f}")
        print(f"  Avg {optimization_metric}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        print(f"  Saved to: {threshold_file}")
    
    return optimal_thresholds


def detect_then_invert_with_optimal_thresholds(act_metrics, concepts, gt_samples_per_concept, 
                                              gt_samples_per_concept_test, device, dataset_name, 
                                              model_input_size, con_label, optimization_metric='f1',
                                              all_object_patches=None, patch_size=14):
    """
    Evaluate on TEST set using per-concept optimal detect/invert thresholds found on calibration set.
    """
    # Load optimal thresholds
    threshold_file = f'Detect_Invert_Thresholds/{dataset_name}/optimal_{optimization_metric}_{con_label}.pt'
    try:
        optimal_thresholds = torch.load(threshold_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Optimal thresholds not found: {threshold_file}. Run calibration optimization first.")
    
    # We now use actual threshold values saved in optimal_thresholds, not percentiles
    # Get test split indices
    split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    if model_input_size[0] == 'text':
        relevant_indices = test_indices
    else:
        relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)

    # Get ground truth labels from test set - USE TEST GT
    all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept_test)
    
    # Prepare results storage
    fp_counts = {}
    fn_counts = {}
    tp_counts = {}
    tn_counts = {}
    
    # Process each concept with its optimal thresholds
    print(f"Evaluating {len(optimal_thresholds)} concepts on test set with optimal thresholds...")
    
    # Check for concepts without optimal thresholds
    all_concepts = set(str(c) for c in concepts.keys())
    missing_concepts = all_concepts - set(optimal_thresholds.keys())
    if missing_concepts:
        print(f"Warning: These concepts have no optimal thresholds (likely all F1=0): {missing_concepts}")
    
    for concept_str, threshold_info in tqdm(optimal_thresholds.items(), desc="Evaluating concepts"):
        # Skip if concept not in current concepts
        if concept_str not in [str(c) for c in concepts.keys()]:
            continue
            
        # Use the actual threshold values that were saved
        detect_threshold = threshold_info['detect_threshold']
        invert_threshold = threshold_info['invert_threshold']
        
        if detect_threshold is None or invert_threshold is None:
            print(f"Warning: Skipping {concept_str} due to missing thresholds")
            continue
        
        # Get detection mask
        detected_patches = get_patch_detection_tensor(
            act_metrics, {concept_str: detect_threshold}, model_input_size, dataset_name
        )[concept_str]
        
        # Get ground truth and activations for relevant indices
        concept_labels = all_concept_labels[concept_str]
        relevant_labels = concept_labels[relevant_indices.tolist()]
        
        concept_acts = torch.tensor(
            act_metrics[concept_str].loc[relevant_indices.tolist()].values, device=device
        )
        
        detected_mask = torch.tensor(
            detected_patches.loc[relevant_indices.tolist()].values, device=device
        )
        
        # Compute predictions: detected AND above invert threshold
        predictions = detected_mask & (concept_acts >= invert_threshold[0])
        
        # Ground truth mask
        gt_mask = (relevant_labels == 1)
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.to(device)
        else:
            gt_mask = torch.tensor(gt_mask == 1, device=device)
        
        # Compute confusion matrix
        tp = torch.sum(predictions & gt_mask).item()
        fn = torch.sum((~predictions) & gt_mask).item()
        fp = torch.sum(predictions & (~gt_mask)).item()
        tn = torch.sum((~predictions) & (~gt_mask)).item()
        
        # Store results
        tp_counts[concept_str] = tp
        fn_counts[concept_str] = fn
        fp_counts[concept_str] = fp
        tn_counts[concept_str] = tn
    
    # Compute metrics
    metrics_df = compute_concept_metrics(
        fp_counts, fn_counts, tp_counts, tn_counts,
        list(optimal_thresholds.keys()), dataset_name, f'{con_label}_optimal_test'
    )
    
    # Add threshold information to results (only if concept column exists)
    if 'concept' in metrics_df.columns:
        for concept_str, threshold_info in optimal_thresholds.items():
            if concept_str in metrics_df['concept'].values:
                idx = metrics_df[metrics_df['concept'] == concept_str].index[0]
                metrics_df.loc[idx, 'detect_percentile'] = threshold_info['detect_percentile']
                metrics_df.loc[idx, 'invert_percentile'] = threshold_info['invert_percentile']
                metrics_df.loc[idx, f'cal_{optimization_metric}'] = threshold_info[f'best_{optimization_metric}']
    
    return metrics_df

    
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


def compute_concept_metrics(fp_count, fn_count, tp_count, tn_count, concepts, dataset_name, con_label, invert_percentile=None, just_obj=False, baseline_type=None, detect_percentile=None):
    metrics_df = compute_stats_from_counts(tp_count, fp_count, tn_count, fn_count)
    
    # Map cluster IDs to concept names for unsupervised methods
    if 'kmeans' in con_label:
        alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
        cluster_to_concept = {info['best_cluster']: concept_name for concept_name, info in alignment_results.items()}
        metrics_df['concept'] = metrics_df['concept'].map(lambda x: cluster_to_concept.get(str(x), str(x)))
    
    # Save the DataFrame as a CSV file
    if just_obj:
        if baseline_type:
            if 'inversion_' in baseline_type and invert_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/{baseline_type}_justobj_per_{invert_percentile*100}.csv'
            else:
                save_path = f'Quant_Results/{dataset_name}/{baseline_type}_justobj_{con_label}.csv'
        else:
            if detect_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/justobj_detectfirst_{detect_percentile*100}_per_{invert_percentile*100}_{con_label}.csv'
            else:
                save_path = f'Quant_Results/{dataset_name}/justobj_per_{invert_percentile*100}_{con_label}.csv'
    else:
        if baseline_type:
            if 'inversion_' in baseline_type and invert_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/{baseline_type}_per_{invert_percentile*100}.csv'
            else:
                save_path = f'Quant_Results/{dataset_name}/{baseline_type}_{con_label}.csv'
        else:
            if detect_percentile is not None and invert_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/detectfirst_{detect_percentile*100}_per_{invert_percentile*100}_{con_label}.csv'
            elif invert_percentile is not None:
                save_path = f'Quant_Results/{dataset_name}/per_{invert_percentile*100}_{con_label}.csv'
            else:
                save_path = f'Quant_Results/{dataset_name}/{con_label}.csv'


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
    

def compute_cossim_hist_stats(gt_samples_per_concept, cos_sims, dataset_name, percentile, sample_type, model_input_size, con_label, patch_size=14, all_object_patches=None):
    """
    Computes in-sample and out-of-sample cosine similarity statistics for each concept, separated by train and test splits.

    Args:
        concept_thresholds (dict): Dictionary mapping concepts to (threshold, random_threshold).
        gt_samples_per_concept (dict): Dictionary mapping concepts to sets of true concept patch indices.
        cos_sims (pd.DataFrame): DataFrame where each column is a concept and rows are patch cosine similarities.
        dataset_name (str): The name of the dataset, used to load the correct metadata file.
        percentile (float): Percentile of in-sample patches to compute the threshold.
        all_object_patches (set, optional): Set of patch indices to consider. If provided, only these patches are considered.

    Returns:
        dict: A dictionary with per-concept cosine similarity stats, separated by train and test splits.
    """
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    else:
        split_df = get_split_df(dataset_name)

    train_mask = split_df == 'train'
    test_mask = split_df != 'train'

    stats = {'train': {}, 'test': {}}

    # Loop over each concept; vectorized operations occur per concept
    for concept, concept_indices in tqdm(gt_samples_per_concept.items()):
        concept = str(concept)
        concept_indices = set(concept_indices)
        
        #filter patches that are irrelevant given the preprocessing scheme
        concept_indices = set(filter_patches_by_image_presence(concept_indices, dataset_name, model_input_size).tolist())
        
        # Apply object patches filter if provided
        if all_object_patches is not None:
            concept_indices &= all_object_patches
        
        # Create a boolean mask for samples belonging to this concept.
        in_gt_mask = cos_sims.index.to_series().isin(concept_indices)
        out_gt_mask = cos_sims.index.to_series().isin(all_object_patches - concept_indices) if all_object_patches is not None else ~in_gt_mask

        # Get the cosine similarity column for this concept.
        cos_vals = cos_sims[concept]

        # Vectorized extraction of cosine similarity values for each combination.
        in_concept_sims_train = cos_vals[train_mask & in_gt_mask].tolist()
        in_concept_sims_test = cos_vals[test_mask & in_gt_mask].tolist()
        out_concept_sims_train = cos_vals[train_mask & out_gt_mask].tolist()
        out_concept_sims_test = cos_vals[test_mask & out_gt_mask].tolist()

        # Store results for train and test splits.
        stats['train'][concept] = {
            'in_concept_sims': in_concept_sims_train,
            'out_concept_sims': out_concept_sims_train
        }
        stats['test'][concept] = {
            'in_concept_sims': in_concept_sims_test,
            'out_concept_sims': out_concept_sims_test
        }
        
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

def find_activated_images_bypatch(cos_sims, curr_thresholds, model_input_size, dataset_name, patch_size=14):
    """
    Computes per-image activations by max-pooling over patches.
    Filters out irrelevant padded image indices using relevant_indices.
    """
    split_df = get_split_df(dataset_name)

    patches_per_image = (model_input_size[0] // patch_size) ** 2
    total_patches = len(cos_sims)
    num_images = total_patches // patches_per_image

    # === Filter: valid indices only
    relevant_indices = set(
        filter_patches_by_image_presence(cos_sims.index, dataset_name, model_input_size).tolist()
    )

    # Convert similarities and thresholds - handle missing concepts
    thresholds = {c: curr_thresholds[c][0] for c in curr_thresholds}
    cos_sims_tensor = torch.tensor(cos_sims.values)
    threshold_tensor = torch.tensor([thresholds[c] if c in thresholds else float('inf') for c in cos_sims.columns])  # [num_concepts]

    # Reshape: [num_images, patches_per_image, num_concepts]
    reshaped_sims = cos_sims_tensor.reshape(num_images, patches_per_image, -1)

    # Max over patches → [num_images, num_concepts]
    max_activations = torch.max(reshaped_sims, dim=1)[0]

    # Thresholding
    activated = max_activations >= threshold_tensor  # [num_images, num_concepts]

    # Get train/test/cal split labels for each image index
    train_mask = torch.tensor([
        (i in relevant_indices) and (split_df.get(i) == 'train') for i in range(num_images)
    ])
    test_mask = torch.tensor([
        (i in relevant_indices) and (split_df.get(i) == 'test') for i in range(num_images)
    ])
    cal_mask = torch.tensor([
        (i in relevant_indices) and (split_df.get(i) == 'cal') for i in range(num_images)
    ])

    activated_images_train = defaultdict(set)
    activated_images_test = defaultdict(set)
    activated_images_cal = defaultdict(set)

    for i, concept in enumerate(cos_sims.columns):
        train_indices = torch.where(activated[:, i] & train_mask)[0].tolist()
        test_indices = torch.where(activated[:, i] & test_mask)[0].tolist()
        cal_indices = torch.where(activated[:, i] & cal_mask)[0].tolist()
        activated_images_train[concept].update(train_indices)
        activated_images_test[concept].update(test_indices)
        activated_images_cal[concept].update(cal_indices)

    return activated_images_train, activated_images_test, activated_images_cal



def find_activated_images_byimage(cos_sims, curr_thresholds, model_input_size=None, dataset_name=None):
    """Vectorized version using torch operations"""
    split_df = get_split_df(dataset_name)
    
    # Convert to tensors
    thresholds = {c: curr_thresholds[c][0] for c in curr_thresholds}
    cos_sims_tensor = torch.tensor(cos_sims.values)
    threshold_tensor = torch.tensor([thresholds[c] for c in cos_sims.columns])
    
    # Compare with thresholds
    activated = cos_sims_tensor >= threshold_tensor.unsqueeze(0)
    
    # Split by train/test
    train_mask = torch.tensor([split_df[i] == 'train' for i in cos_sims.index])
    test_mask = torch.tensor([split_df[i] == 'test' for i in cos_sims.index])
    
    activated_images_train = defaultdict(set)
    activated_images_test = defaultdict(set)
    
    for i, concept in enumerate(cos_sims.columns):
        train_indices = torch.where(activated[:, i] & train_mask)[0].tolist()
        test_indices = torch.where(activated[:, i] & test_mask)[0].tolist()
        activated_images_train[concept].update(train_indices)
        activated_images_test[concept].update(test_indices)
        
    return activated_images_train, activated_images_test


def find_activated_sentences_bytoken(act_metrics, curr_thresholds, model_input_size, dataset_name):
    """Optimized version using torch operations"""
    split_df = get_split_df(dataset_name)
    token_counts_per_sentence = torch.load(f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt', weights_only=False)  # List[List[int]]
    
    # Each inner list gives token counts per word → sum across to get total tokens per sentence
    token_counts_flat = torch.tensor([sum(x) for x in token_counts_per_sentence])
    sentence_starts = torch.cat([torch.tensor([0]), token_counts_flat.cumsum(0)[:-1]])
    sentence_ends = token_counts_flat.cumsum(0)
    
    # Convert activation metrics and thresholds
    metrics_tensor = torch.tensor(act_metrics.values)
    threshold_tensor = torch.tensor([curr_thresholds[c][0] for c in act_metrics.columns], dtype=metrics_tensor.dtype)
    
    # Compute max activations per sentence (vectorized with list comprehension)
    max_activations = torch.stack([
        metrics_tensor[start:end].amax(dim=0) for start, end in zip(sentence_starts, sentence_ends)
    ])
    
    # Compare with thresholds
    activated = max_activations >= threshold_tensor
    
    # Convert split to boolean masks
    split_array = split_df.values if isinstance(split_df, pd.Series) else split_df
    train_mask = (split_array == "train")
    test_mask = (split_array == "test")
    
    activated_sentences_train = defaultdict(set)
    activated_sentences_test = defaultdict(set)
    
    for i, concept in enumerate(act_metrics.columns):
        train_indices = torch.where(activated[:, i] & torch.tensor(train_mask))[0].tolist()
        test_indices = torch.where(activated[:, i] & torch.tensor(test_mask))[0].tolist()
        activated_sentences_train[concept].update(train_indices)
        activated_sentences_test[concept].update(test_indices)

    return activated_sentences_train, activated_sentences_test


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
        detected_samples_train, detected_samples_test = find_activated_sentences_bytoken(
            act_metrics, detect_thresholds, model_input_size, dataset_name)
        detected_samples_cal = {}  # No cal support for text yet
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



def compute_detection_metrics_for_per(per, gt_images_per_concept_test, 
                                      activated_images_test, 
                                      dataset_name, con_label):
    """
    Compute detection metrics (TP, FP, TN, FN) for a specific percentile.
    Saves to disk if not already computed.

    Args:
        per: Percentile
        gt_images_per_concept_test: {concept: set of GT image indices}
        activated_images_test: {concept: set of activated image indices}
        dataset_name: Dataset name
        con_label: Concept label for saving
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
    all_indices = set(split_df[split_df == 'test'].index)

    for concept in gt_images_per_concept_test.keys():
        # gt_images = set(gt_images_per_concept_test[concept]) & set(relevant_indices)
        # activated_images = activated_images_test.get(concept, set()) & set(relevant_indices)
        gt_images = set(gt_images_per_concept_test[concept])
        activated_images = activated_images_test.get(concept, set())

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

    
def compute_detection_metrics_over_percentiles(percentiles, gt_images_per_concept_test, 
                                               sim_metrics, dataset_name, model_input_size, device, 
                                               con_label, sample_type='patch', patch_size=14):
    """
    Computes detection metrics over multiple percentiles.

    Args:
        percentiles: List of percentiles
        gt_samples_per_concept_test: {concept: patch indices}
        gt_images_per_concept_test: {concept: image indices}
        sim_metrics: Cosine similarities
        dataset_name: Dataset name
        model_input_size: (width, height) tuple
        device: CUDA/CPU device
        con_label: Label for saving
        sample_type: 'patch' or 'cls'
        patch_size: Patch size
    Returns:
        all_metrics: dict mapping per -> metrics_df
    """
    # if sample_type == 'patch':
    #     relevant_indices = set(filter_patches_by_image_presence(sim_metrics.index, dataset_name, model_input_size).tolist())
    # elif sample_type == 'cls':
    #     relevant_indices = sim_metrics.index
        
    all_metrics = {}
    
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

    for per in tqdm(percentiles):
        # === Thresholds for current percentile
        # curr_thresholds = compute_concept_thresholds(
        #     gt_samples_per_concept_test, sim_metrics, per, 
        #     device=device, dataset_name=f'{dataset_name}-Cal', con_label=con_label,
        #     n_vectors=1, n_concepts_to_print=0
        # )
        curr_thresholds = all_thresholds[per]

        # === Activation
        if sample_type == 'patch':
            if 'text' in model_input_size:
                _, activated_images_test = find_activated_sentences_bytoken(
                    sim_metrics, curr_thresholds, model_input_size, dataset_name
                )
            else:
                _, activated_images_test = find_activated_images_bypatch(
                    sim_metrics, curr_thresholds, model_input_size, dataset_name, patch_size=patch_size
                )
        elif sample_type == 'cls':
            _, activated_images_test = find_activated_images_byimage(
                sim_metrics, curr_thresholds, model_input_size, dataset_name
            )
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

        # === Compute detection metrics
        metrics_df = compute_detection_metrics_for_per(
            per, gt_images_per_concept_test, activated_images_test, 
            dataset_name, con_label
        )

        all_metrics[per] = metrics_df

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
    
    

    
    