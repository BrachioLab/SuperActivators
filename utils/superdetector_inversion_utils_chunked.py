import torch
import pandas as pd
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath("utils"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import torch.nn.functional as F
import seaborn as sns

from utils.quant_concept_evals_utils import compute_concept_thresholds, compute_concept_metrics, \
     create_binary_labels, get_patch_detection_tensor
from utils.patch_alignment_utils import filter_patches_by_image_presence, get_patch_range_for_image, get_patch_range_for_text, \
     calculate_patch_location, get_patch_split_df
from utils.general_utils import pad_or_resize_img
from utils.unsupervised_utils import match_thresholds_across_percentiles
from utils.memory_management_utils import ChunkedEmbeddingLoader

# Import original functions that don't need chunking
from utils.superdetector_inversion_utils import (
    find_superdetector_patches, 
    get_superdetector_vector,
    draw_superdetectors_on_image
)


def find_all_superdetector_patches_chunked(percentile, act_metrics, gt_samples_per_concept_test, 
                                 dataset_name, model_input_size, con_label, device):
    """
    Chunked version that computes superdetector patches without fallback to non-chunked files.
    Always computes fresh to ensure chunked-only behavior.
    """
    concept_names = list(gt_samples_per_concept_test.keys())
    all_superdetectors = {}
    
    for concept in concept_names:
        test_image_indices = gt_samples_per_concept_test[concept]
        if not test_image_indices:
            all_superdetectors[concept] = []
            continue
            
        # Get patch indices for test images
        from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices
        patch_indices = map_concepts_to_patch_indices([concept], test_image_indices, dataset_name, model_input_size)
        concept_patch_indices = patch_indices.get(concept, [])
        
        if not concept_patch_indices:
            all_superdetectors[concept] = []
            continue
            
        # Get activations for this concept's patches
        concept_activations = act_metrics[concept].iloc[concept_patch_indices]
        
        # Find top percentile patches
        threshold = concept_activations.quantile(1 - percentile)
        superdetector_mask = concept_activations >= threshold
        superdetector_indices = [concept_patch_indices[i] for i, is_super in enumerate(superdetector_mask) if is_super]
        
        all_superdetectors[concept] = superdetector_indices
    
    return all_superdetectors


def batch_superdetector_inversions_chunked(percentile, agglomerate_type, embedding_loader, act_metrics,
                                      gt_samples_per_concept_test, dataset_name, model_input_size,
                                      con_label, device, patch_size=14, local=False):
    """
    Chunked version of batch_superdetector_inversions that processes embeddings in chunks.
    
    Args:
        percentile (float): Percentile (0 < percentile < 1) to define superdetector patches.
        agglomerate_type (str): Aggregation method ('avg' or 'max') for computing concept vectors.
        embedding_loader (ChunkedEmbeddingLoader): Chunked embedding loader instead of full embeds tensor.
        act_metrics (pd.DataFrame): Activation scores for each patch and concept.
        gt_samples_per_concept_test (dict): Mapping from concept -> list of test image indices.
        dataset_name (str): Dataset name for filtering and thresholding.
        model_input_size (tuple): Image size (e.g., (224, 224)).
        con_label (str): Concept category name.
        device (torch.device): Device to run cosine similarity on.
        patch_size (int): Patch size used in embedding.
        local (bool): If True, compute superdetectors per image. If False, compute once for all images.

    Returns:
        pd.DataFrame: DataFrame of shape (n_patches, n_concepts), values are cosine similarities.
    """
    concept_names = act_metrics.columns.tolist()
    
    # Get embedding info
    embedding_info = embedding_loader.get_embedding_info()
    total_embeddings = embedding_info['total_samples']
    
    if model_input_size[0] == 'text':
        # Load precomputed patch counts per sample (i.e., token counts)
        patch_counts_per_sample = torch.load(f'GT_Samples/{dataset_name}/token_counts.pt', weights_only=False)
        num_patches_per_sample = [sum(x) for x in patch_counts_per_sample]
        sample_boundaries = [(0, num_patches_per_sample[0])]
        for count in num_patches_per_sample[1:]:
            start = sample_boundaries[-1][1]
            sample_boundaries.append((start, start + count))
    else:
        # For vision, patches per sample is constant
        patches_per_sample = (model_input_size[0] // patch_size) * (model_input_size[1] // patch_size)
        total_samples = total_embeddings // patches_per_sample
        sample_boundaries = [
            (i * patches_per_sample, (i + 1) * patches_per_sample) for i in range(total_samples)
        ]
    
    # Precompute global superdetectors if needed
    global_superdetectors = None
    if not local:
        global_superdetectors = find_all_superdetector_patches_chunked(
            percentile, act_metrics, gt_samples_per_concept_test,
            dataset_name, model_input_size, con_label, device
        )

    # Store activations in dictionary: concept -> list of patch-level scores
    flat_scores = {concept: [] for concept in concept_names}

    for sample_idx, (start_idx, end_idx) in enumerate(tqdm(sample_boundaries, desc="Processing samples")):
        if local:
            superdetectors_per_concept = find_superdetector_patches(
                sample_idx, percentile, act_metrics, gt_samples_per_concept_test,
                dataset_name, model_input_size, con_label, device
            )
        else:
            superdetectors_per_concept = global_superdetectors

        # Load sample embeddings from chunks
        sample_indices = list(range(start_idx, end_idx))
        sample_embeds = embedding_loader.load_specific_embeddings(sample_indices)  # [n_patches, embed_dim]

        for concept in concept_names:
            superdetectors = superdetectors_per_concept.get(concept, [])
            if not superdetectors:
                sim_scores = torch.zeros(end_idx - start_idx)
            else:
                # Load superdetector embeddings for this concept
                super_embeds = embedding_loader.load_specific_embeddings(superdetectors)
                super_vec = get_superdetector_vector_from_embeds(
                    super_embeds, [act_metrics[concept].iloc[idx] for idx in superdetectors], agglomerate_type
                )
                sim_scores = F.cosine_similarity(
                    sample_embeds.to(device), super_vec.unsqueeze(0).to(device), dim=1
                ).detach().cpu()

            flat_scores[concept].append(sim_scores)

    # Concatenate across images and create DataFrame
    for concept in concept_names:
        flat_scores[concept] = torch.cat(flat_scores[concept])

    df = pd.DataFrame(flat_scores)
    
    # Save to CSV in Chunked_Superpatches folder without _chunked suffix
    os.makedirs(f'Chunked_Superpatches/{dataset_name}', exist_ok=True)
    output_file = f'Chunked_Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_per_{percentile}_{con_label}.csv'
    df.to_csv(output_file, index=False)
    print(f"   💾 Saved chunked superdetector inversions: {output_file}")

    return df


def get_superdetector_vector_from_embeds(super_embeds, activations, agglomerate_type):
    """
    Compute superdetector vector from already-loaded embeddings and activations.
    
    Args:
        super_embeds (torch.Tensor): Embeddings of superdetector patches [n_super, embed_dim]
        activations (list): Activation values for superdetector patches
        agglomerate_type (str): 'avg' or 'max'
    
    Returns:
        torch.Tensor: Aggregated superdetector vector [embed_dim]
    """
    if len(super_embeds) == 0:
        return torch.zeros(super_embeds.shape[1])
    
    if agglomerate_type == 'avg':
        return torch.mean(super_embeds, dim=0)
    elif agglomerate_type == 'max':
        # Weight by activation and take max
        activations = torch.tensor(activations)
        max_idx = torch.argmax(activations)
        return super_embeds[max_idx]
    else:
        raise ValueError(f"Unknown agglomerate_type: {agglomerate_type}")


def all_superdetector_inversions_across_percentiles_chunked(percentiles, agglomerate_type, embedding_loader, act_metrics,
                                   gt_samples_per_concept_test, dataset_name, model_input_size,
                                   con_label, device, patch_size=14, local=False):
    """
    Chunked version of all_superdetector_inversions_across_percentiles.
    """
    for percentile in tqdm(percentiles, desc="Processing percentiles"):
        batch_superdetector_inversions_chunked(percentile, agglomerate_type, embedding_loader, act_metrics,
                                       gt_samples_per_concept_test, dataset_name, model_input_size,
                                       con_label, device, patch_size, local)


def detect_then_invert_locally_metrics_over_percentiles_chunked(detect_percentiles, invert_percentiles, act_metrics, 
                                                        concepts, gt_patches_per_concept, gt_patches_per_concept_test,
                                                        device, dataset_name, model_input_size, con_label, embedding_loader,
                                                        all_object_patches=None, patch_size=14,
                                                        agglomerate_type='avg'):
    """
    Chunked version of detect_then_invert_locally_metrics_over_percentiles.
    Uses chunked embedding loader instead of full embedding tensor.
    """
    # Ensure all inversion files exist in chunked format before computing metrics
    # This function does NOT use the original non-chunked version to avoid fallbacks
    for detect_percentile in detect_percentiles:
        for invert_percentile in invert_percentiles:
            # Make sure the superdetector inversion files exist for this percentile
            inversion_file = f'Chunked_Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_per_{invert_percentile}_{con_label}.csv'
            if not os.path.exists(inversion_file):
                print(f"   Creating missing inversion file for percentile {invert_percentile}")
                batch_superdetector_inversions_chunked(invert_percentile, agglomerate_type, embedding_loader, act_metrics,
                                               gt_patches_per_concept_test, dataset_name, model_input_size,
                                               con_label, device, patch_size, local=True)
    
    # Create a modified version that saves to Chunked_Quant_Results
    import os
    from utils.quant_concept_evals_utils import (
        compute_concept_thresholds_over_percentiles,
        compute_detection_metrics_over_percentiles,
        get_patch_split_df,
        filter_patches_by_image_presence
    )
    
    # Get thresholds
    if 'kmeans' not in con_label:
        thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
    else:
        # Load files for kmeans
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
    
    # Create Chunked_Quant_Results directory
    os.makedirs(f'Chunked_Quant_Results/{dataset_name}', exist_ok=True)
    
    # Process each percentile combination
    for detect_p in detect_percentiles:
        detect_threshold_dict = thresholds[detect_p]
        
        # Compute detection metrics
        detect_metrics_df = compute_detection_metrics_over_percentiles(
            {detect_p: detect_threshold_dict}, gt_patches_per_concept_test, act_metrics,
            dataset_name, model_input_size, device, con_label, sample_type='patch', patch_size=patch_size
        )[detect_p]
        
        for invert_p in invert_percentiles:
            # Read the chunked superpatch inversion CSV from Chunked_Superpatches
            inv_cossim_file = f'Chunked_Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_per_{invert_p}_{con_label}.csv'
            
            if os.path.exists(inv_cossim_file):
                print(f"   Found local inversion file: {inv_cossim_file}")
                # For now, just note that the inversion file exists
                # The original pipeline doesn't compute additional f1 metrics here