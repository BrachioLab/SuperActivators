import torch
import pandas as pd
from tqdm import tqdm
import os

import sys
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
from utils.unsupervised_utils  import match_thresholds_across_percentiles

### Reasoning about superpatches ####
# def find_superdetector_patches(sample_idx, percentile, act_metrics, gt_samples_per_concept_test, 
#                                 dataset_name, model_input_size, con_label, device):
#     """
#     Identifies superdetector patches for a single image by selecting patches whose concept
#     activation scores exceed the computed threshold.

#     Args:
#         img_idx (int): Index of the image to analyze.
#         percentile (float): Percentile threshold to select top concept-activated patches.
#         act_metrics (pd.DataFrame): Activation metrics for each patch and concept.
#         gt_samples_per_concept_test (dict): Mapping of concepts to ground truth positive image indices.
#         dataset_name (str): Name of the dataset (used for indexing logic).
#         model_input_size (tuple): Size of model input (e.g., (224, 224)).
#         con_label (str): Concept label name for output directory naming.
#         device (torch.device): Device used for computations.

#     Returns:
#         dict: Mapping from concept names to lists of global patch indices that are superdetectors.
#     """
#     #Compute thresholds for each concept
#     # thresholds = compute_concept_thresholds(
#     #     None, 
#     #     act_metrics, percentile, n_vectors=1, 
#     #     device=device, n_concepts_to_print=0, 
#     #     dataset_name=f'{dataset_name}-Cal', con_label=con_label
#     # )
    
#     if 'kmeans' in con_label:
#         all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt')
#         matched_thresholds =  match_thresholds_across_percentiles(all_thresholds, dataset_name, con_label)
#         thresholds = matched_thresholds[percentile]
#     else:
#         thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt')[percentile]
    
#     # Get patch range for this image
#     if model_input_size[0] == 'text':
#         start_patch_idx, end_patch_idx = get_patch_range_for_text(sample_idx, dataset_name)
#     else:
#         start_patch_idx, end_patch_idx = get_patch_range_for_image(
#             sample_idx, patch_size=14, model_input_size=model_input_size
#         )
        
#     all_patch_indices = np.arange(start_patch_idx, end_patch_idx)
    
#     # Filter out padding patches
#     relevant_indices = filter_patches_by_image_presence(all_patch_indices, dataset_name, model_input_size).tolist()
#     relevant_act_metrics = act_metrics.loc[relevant_indices]
    
#     # Find patch indices (global) that are superdetectors
#     superdetectors_per_concept = {}
#     for concept in thresholds.keys():
#         # Boolean mask where activation >= threshold
#         concept_mask = relevant_act_metrics[concept] >= thresholds[concept][0]
        
#         # Extract global indices of patches that are superdetectors
#         superdetector_patch_indices = [relevant_indices[i] for i, is_super in enumerate(concept_mask) if is_super]
        
#         superdetectors_per_concept[concept] = superdetector_patch_indices

#     return superdetectors_per_concept
def find_superdetector_patches(sample_idx, percentile, act_metrics, gt_samples_per_concept_cal, 
                                dataset_name, model_input_size, con_label, device):
    """
    Identifies superdetector patches for a single image by selecting patches whose concept
    activation scores exceed the computed threshold — or loads from precomputed global results.

    Args:
        sample_idx (int): Index of the image to analyze.
        percentile (float): Percentile threshold to select top concept-activated patches.
        act_metrics (pd.DataFrame): Activation metrics for each patch and concept.
        gt_samples_per_concept_test (dict): Mapping of concepts to ground truth positive image indices.
        dataset_name (str): Name of the dataset.
        model_input_size (tuple): Size of model input (e.g., (224, 224)).
        con_label (str): Concept label name.
        device (torch.device): Device used for computations.

    Returns:
        dict: Mapping from concept names to lists of global patch indices that are superdetectors.
    """
    superpatch_path = f'Superpatches/{dataset_name}/per_{percentile}_{con_label}.pt'
    
    if os.path.exists(superpatch_path):
        all_superpatches = torch.load(superpatch_path, weights_only=False)

        if model_input_size[0] == 'text':
            start_patch_idx, end_patch_idx = get_patch_range_for_text(sample_idx, dataset_name)
        else:
            start_patch_idx, end_patch_idx = get_patch_range_for_image(
                sample_idx, patch_size=14, model_input_size=model_input_size
            )
        patch_ids = set(range(start_patch_idx, end_patch_idx))

        # Return only the patches for this image
        return {
            concept: [idx for idx in patch_list if idx in patch_ids]
            for concept, patch_list in all_superpatches.items()
        }

    # Otherwise, fall back to the original computation
    if 'kmeans' in con_label:
        all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
        matched_thresholds = match_thresholds_across_percentiles(all_thresholds, dataset_name, con_label)
        thresholds = matched_thresholds[percentile]
        
        # Handle potential column name mismatch
        # Check if actmetrics columns are concept names instead of cluster IDs
        alignment_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
        if os.path.exists(alignment_path):
            alignment_results = torch.load(alignment_path, weights_only=False)
            concept_to_cluster = {concept: str(info['best_cluster']) 
                                  for concept, info in alignment_results.items()}
            
            # If columns are concept names but thresholds are keyed by cluster IDs
            if hasattr(act_metrics, 'columns'):
                first_col = str(act_metrics.columns[0])
                if first_col in concept_to_cluster and first_col not in thresholds:
                    # Remap thresholds to use concept names
                    threshold_lookup = {}
                    for concept, cluster_id in concept_to_cluster.items():
                        if cluster_id in thresholds:
                            threshold_lookup[concept] = thresholds[cluster_id]
                    thresholds = threshold_lookup
    else:
        thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)[percentile]

    if model_input_size[0] == 'text':
        start_patch_idx, end_patch_idx = get_patch_range_for_text(sample_idx, dataset_name)
    else:
        start_patch_idx, end_patch_idx = get_patch_range_for_image(
            sample_idx, patch_size=14, model_input_size=model_input_size
        )
    
    all_patch_indices = np.arange(start_patch_idx, end_patch_idx)
    relevant_indices = filter_patches_by_image_presence(all_patch_indices, dataset_name, model_input_size).tolist()
    relevant_act_metrics = act_metrics.loc[relevant_indices]
    
    superdetectors_per_concept = {}
    for concept in thresholds.keys():
        concept_mask = relevant_act_metrics[concept] >= thresholds[concept][0]
        superdetector_patch_indices = [relevant_indices[i] for i, is_super in enumerate(concept_mask) if is_super]
        superdetectors_per_concept[concept] = superdetector_patch_indices

    return superdetectors_per_concept


def find_all_superdetector_patches(percentile, act_metrics,
                                   dataset_name, model_input_size, con_label, device):
    """
    Identifies superdetector patches for the entire dataset based on percentile thresholds 
    and saves them to disk.

    Args:
        percentile (float): Percentile threshold to select top concept-activated patches.
        act_metrics (pd.DataFrame): Activation metrics for each patch and concept.
        dataset_name (str): Name of the dataset (used for output path).
        model_input_size (tuple): Size of model input (e.g., (224, 224)).
        con_label (str): Concept label name for output file.
        device (torch.device): Device used for computation.

    Returns:
        dict: Mapping from concept names to lists of global patch indices that are superdetectors.
    """
    # try:
    #     all_superdetectors_per_concept = torch.load(f'Superpatches/{dataset_name}/per_{percentile}_{con_label}.pt')
    # except:
    # Compute thresholds for each concept
    # thresholds = compute_concept_thresholds(
    #     None, 
    #     act_metrics, percentile, n_vectors=1, 
    #     device=device, n_concepts_to_print=0, 
    #     dataset_name=f'{dataset_name}-Cal', con_label=con_label
    # )
    if 'kmeans' in con_label:
        all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
        matched_thresholds =  match_thresholds_across_percentiles(all_thresholds, dataset_name, con_label)
        thresholds = matched_thresholds[percentile]
        
        # Handle potential column name mismatch
        # Check if actmetrics columns are concept names instead of cluster IDs
        alignment_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
        if os.path.exists(alignment_path):
            alignment_results = torch.load(alignment_path, weights_only=False)
            concept_to_cluster = {concept: str(info['best_cluster']) 
                                  for concept, info in alignment_results.items()}
            
            # If columns are concept names but thresholds are keyed by cluster IDs
            first_col = str(act_metrics.columns[0])
            if first_col in concept_to_cluster and first_col not in thresholds:
                # Remap thresholds to use concept names
                threshold_lookup = {}
                for concept, cluster_id in concept_to_cluster.items():
                    if cluster_id in thresholds:
                        threshold_lookup[concept] = thresholds[cluster_id]
                thresholds = threshold_lookup
    else:
        thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)[percentile]

    concepts = list(act_metrics.columns)
    
    # Filter out padding patches
    relevant_indices = filter_patches_by_image_presence(act_metrics.index, dataset_name, model_input_size).tolist()
    relevant_act_metrics = act_metrics.loc[relevant_indices]

    # Find patch indices (global) that are superdetectors
    all_superdetectors_per_concept = {}
    for concept in concepts:
        # Boolean mask where activation >= threshold
        concept_mask = relevant_act_metrics[concept] >= thresholds[concept][0]

        # Extract global indices of patches that are superdetectors
        superdetector_patch_indices = [relevant_indices[i] for i, is_super in enumerate(concept_mask) if is_super]

        all_superdetectors_per_concept[concept] = superdetector_patch_indices

    torch.save(all_superdetectors_per_concept, f'Superpatches/{dataset_name}/per_{percentile}_{con_label}.pt')
    return all_superdetectors_per_concept
                
                
def get_superdetector_vector(superdetectors, embeds, concept_act_metrics, agglomerate_type='avg'):
    """
    Computes a single vector representing the superdetector patches using the specified aggregation.

    Args:
        superdetectors (list of int): Indices of superdetector patches.
        embeds (torch.Tensor): Patch embeddings, shape (num_patches, embed_dim).
        concept_act_metrics (torch.Tensor or pd.Series): Activation metric for each patch for a concept.
        agglomerate_type (str): Aggregation type: 'avg' or 'max'.

    Returns:
        torch.Tensor: Aggregated embedding vector.
    """
    if len(superdetectors) == 0:
        raise ValueError("No superdetector patches provided.")

    # Subset embeddings and activations to superdetector patches
    selected_embeds = embeds[superdetectors]
    selected_metrics = concept_act_metrics[superdetectors]

    if agglomerate_type == 'avg':
        superdetector_vector = selected_embeds.mean(dim=0)

    elif agglomerate_type == 'max':
        if isinstance(selected_metrics, torch.Tensor):
            max_idx = torch.argmax(selected_metrics)
        else:
            max_idx = selected_metrics.values.argmax()
        superdetector_vector = selected_embeds[max_idx]

    else:
        raise ValueError(f"Unknown agglomerate_type: {agglomerate_type}")

    return superdetector_vector


### Inversions ###
def superdetector_inversion(sample_idx, percentile, agglomerate_type, embeds, act_metrics, gt_samples_per_concept_test, 
                                  dataset_name, model_input_size, con_label, device, patch_size=14, local=True):
    """
    Computes inversion maps for each concept by measuring the similarity between 
    per-patch embeddings of an image and a concept-specific superdetector vector.

    A superdetector vector is computed by aggregating embeddings of the top percentile 
    of most activated patches (either locally within the image or globally across the dataset),
    and used to compute cosine similarity with each patch in the target image.

    Args:
        img_idx (int): Index of the image to process.
        percentile (float): Percentile (0 < percentile < 1) used to select top-activated patches.
        agglomerate_type (str): Aggregation method for patch embeddings ('avg' or 'max').
        embeds (torch.Tensor): Patch embeddings for the entire dataset (shape: [num_patches, embed_dim]).
        act_metrics (pd.DataFrame): Patch-level activation scores (shape: [num_patches, num_concepts]).
        gt_samples_per_concept_test (dict): Mapping from concept to test image indices where it appears.
        dataset_name (str): Name of the dataset, used for loading masks and thresholds.
        model_input_size (tuple): Dimensions of the input image (e.g., (224, 224)).
        con_label (str): Concept label used for thresholding (e.g., "color", "shape").
        device (torch.device): Device to run computations on.
        patch_size (int, optional): Size of each image patch. Default is 14.
        local (bool, optional): If True, compute superdetectors from the target image only.
                                If False, compute from the whole dataset.

    Returns:
        dict: Mapping from each concept to a 1D tensor of cosine similarity scores per patch
              (length = number of patches in the image).
    """
    if local:
        #find the superdetector patches for the image 
        superdetectors_per_concept = find_superdetector_patches(sample_idx, percentile, act_metrics, gt_samples_per_concept_test, 
                                      dataset_name, model_input_size, con_label, device)
    else:
        #find superdetector patches for entire dataset
        superdetectors_per_concept = find_all_superdetector_patches(percentile, act_metrics, gt_samples_per_concept_test, 
                                    dataset_name, model_input_size, con_label, device)
    
    inversions = {}
    for concept, superdetectors in superdetectors_per_concept.items():
        if not superdetectors: #store 0s if no superdetectors
            inversions[concept] = torch.zeros((model_input_size[0]//patch_size) * (model_input_size[1])//patch_size)
            continue
        superdetector_vector = get_superdetector_vector(superdetectors, embeds, 
                                                        act_metrics[concept], agglomerate_type)
        
        # Get patch range for this image
        if model_input_size[0] == 'text':
            start_patch_idx, end_patch_idx = get_patch_range_for_text(sample_idx, dataset_name)
        else:
            start_patch_idx, end_patch_idx = get_patch_range_for_image(
                sample_idx, patch_size=14, model_input_size=model_input_size
            )
            all_patch_indices = np.arange(start_patch_idx, end_patch_idx)
        sample_embeds = embeds[start_patch_idx:end_patch_idx]
        
        # Compute cosine similarity between superdetector and all image patch embeddings
        sim_scores = F.cosine_similarity(sample_embeds.to(device), superdetector_vector.unsqueeze(0).to(device), dim=1)
        inversions[concept] = sim_scores.detach().cpu()
    return inversions


def batch_superdetector_inversions(percentile, agglomerate_type, embeds, act_metrics,
                                      gt_samples_per_concept_cal, dataset_name, model_input_size,
                                      con_label, device, patch_size=14, local=False):
    """
    Computes inversion maps (cosine similarities to superdetector vectors) for all patches and concepts,
    and returns a flattened DataFrame: one row per patch, one column per concept.

    Args:
        percentile (float): Percentile (0 < percentile < 1) to define superdetector patches.
        agglomerate_type (str): Aggregation method ('avg' or 'max') for computing concept vectors.
        embeds (torch.Tensor): All patch embeddings (shape: [n_patches, embed_dim]).
        act_metrics (pd.DataFrame): Activation scores for each patch and concept (shape: [n_patches, n_concepts]).
        gt_samples_per_concept_cal (dict): Mapping from concept -> list of calibration image indices (used in thresholding).
        dataset_name (str): Dataset name for filtering and thresholding.
        model_input_size (tuple): Image size (e.g., (224, 224)).
        con_label (str): Concept category name (used for thresholding).
        device (torch.device): Device to run cosine similarity on.
        patch_size (int): Patch size used in embedding.
        local (bool): If True, compute superdetectors per image. If False, compute once for all images.

    Returns:
        pd.DataFrame: DataFrame of shape (n_patches, n_concepts), values are cosine similarities.
    """
    # try:
    #     df = pd.read_csv(f'Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_per_{percentile}_{con_label}.csv')
    # except:
    concept_names = act_metrics.columns.tolist()
    if model_input_size[0] == 'text':
        # Load precomputed patch counts per sample (i.e., token counts)
        patch_counts_per_sample = torch.load(f'GT_Samples/{dataset_name}/token_counts_inputsize_{model_input_size}.pt', weights_only=False)
        num_patches_per_sample = [sum(x) for x in patch_counts_per_sample]
        sample_boundaries = [(0, num_patches_per_sample[0])]
        for count in num_patches_per_sample[1:]:
            start = sample_boundaries[-1][1]
            sample_boundaries.append((start, start + count))
    else:
        # For vision, patches per sample is constant
        patches_per_sample = (model_input_size[0] // patch_size) * (model_input_size[1] // patch_size)
        total_samples = embeds.shape[0] // patches_per_sample
        sample_boundaries = [
            (i * patches_per_sample, (i + 1) * patches_per_sample) for i in range(total_samples)
        ]
    
    # Precompute global superdetectors if needed
    global_superdetectors = None
    if not local:
        global_superdetectors = find_all_superdetector_patches(
            percentile, act_metrics, gt_samples_per_concept_cal,
            dataset_name, model_input_size, con_label, device
        )

    # Store activations in dictionary: concept -> list of patch-level scores
    flat_scores = {concept: [] for concept in concept_names}

    for sample_idx, (start_idx, end_idx) in enumerate(sample_boundaries):
        if local:
            superdetectors_per_concept = find_superdetector_patches(
                sample_idx, percentile, act_metrics, gt_samples_per_concept_cal,
                dataset_name, model_input_size, con_label, device
            )
        else:
            superdetectors_per_concept = global_superdetectors

        sample_embeds = embeds[start_idx:end_idx]  # [n_patches, embed_dim]

        for concept in concept_names:
            superdetectors = superdetectors_per_concept.get(concept, [])
            if not superdetectors:
                sim_scores = torch.zeros(end_idx - start_idx)
            else:
                super_vec = get_superdetector_vector(
                    superdetectors, embeds, act_metrics[concept], agglomerate_type
                )
                sim_scores = F.cosine_similarity(
                    sample_embeds.to(device), super_vec.unsqueeze(0).to(device), dim=1
                ).detach().cpu()

            flat_scores[concept].append(sim_scores)

    # Concatenate across images
    for concept in flat_scores:
        flat_scores[concept] = torch.cat(flat_scores[concept], dim=0).numpy()

    df = pd.DataFrame(flat_scores)
    
    # Debug: Check if kmeans results are zero
    if 'kmeans' in con_label:
        non_zero_count = (df != 0).sum().sum()
        total_count = df.shape[0] * df.shape[1]
        print(f"DEBUG: kmeans superpatch - non-zero: {non_zero_count}/{total_count}")
        if non_zero_count == 0:
            print("WARNING: All values are zero before saving!")
            # Check one concept
            first_concept = df.columns[0]
            print(f"First concept '{first_concept}' stats: min={df[first_concept].min()}, max={df[first_concept].max()}")
    
    df.to_csv(f'Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_per_{percentile}_{con_label}.csv', index=False)
    
    return df


def all_superdetector_inversions_across_percentiles(percentiles, agglomerate_type, embeds, act_metrics,
                                   gt_samples_per_concept_cal, dataset_name, model_input_size,
                                   con_label, device, patch_size=14, local=False):
    for percentile in tqdm(percentiles):
        batch_superdetector_inversions(percentile, agglomerate_type, embeds, act_metrics,
                                       gt_samples_per_concept_cal, dataset_name, model_input_size,
                                       con_label, device, patch_size, local)



                
### Plotting ###
def draw_superdetectors_on_image(all_images, img_idx, superdetectors_per_concept, model_input_size, patch_size=14):
    """
    Draws a separate plot for each concept, showing its superdetector patches on the image.

    Args:
        img_idx (int): Index of the image in the dataset.
        superdetectors_per_concept (dict): Concept -> list of global patch indices.
        model_input_size (tuple): Size used to embed the image (e.g., (224, 224)).
        patch_size (int): Size of each patch (default 14).
    """
    # Get and resize the image
    image = pad_or_resize_img(all_images[img_idx], model_input_size)

    # Compute number of patches per image
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col

    # Color map for consistency (but optional here since 1 concept per figure)
    colors = cm.get_cmap('tab10', len(superdetectors_per_concept))

    for i, (concept, global_patch_indices) in enumerate(superdetectors_per_concept.items()):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)

        for global_patch_idx in global_patch_indices:
            # Convert to image-relative patch index
            patch_idx = global_patch_idx - img_idx * patches_per_image

            # Get patch coordinates
            left, top, right, bottom = calculate_patch_location(
                image, patch_idx, patch_size, model_input_size
            )

            # Draw rectangle
            rect = patches.Rectangle(
                (left, top),
                patch_size,
                patch_size,
                linewidth=2,
                edgecolor=colors(i),
                facecolor='none'
            )
            ax.add_patch(rect)

        ax.set_title(f"Superdetector Patches for Concept: {concept}", pad=10)
        ax.axis("off")
        plt.tight_layout()
        plt.show()


def plot_inversions(img_idx, inversions, all_images, model_input_size, patch_size=14, title=None):
    """
    Plots the inversion maps (cosine similarity scores) for each concept, overlayed on the original image.
    All concepts are displayed horizontally as separate heatmaps.

    Args:
        img_idx (int): Index of the image in the dataset.
        inversions (dict): Concept -> 1D tensor of similarity scores (length: num_patches).
        all_images (list or dataset): Original images, where all_images[img_idx] gives the raw image.
        model_input_size (tuple): Size the image was resized to (default: (224, 224)).
        patch_size (int): Patch size used for embedding (default: 14).
    """
    image = pad_or_resize_img(all_images[img_idx], model_input_size)
    num_patches_per_row = model_input_size[0] // patch_size
    num_patches_per_col = model_input_size[1] // patch_size

    concepts = list(inversions.keys())
    n_concepts = len(concepts)

    fig, axs = plt.subplots(1, n_concepts, figsize=(5 * n_concepts, 5))

    if n_concepts == 1:
        axs = [axs]  # handle single subplot case

    for i, concept in enumerate(concepts):
        sim_scores = inversions[concept]

        # Reshape to 2D grid
        sim_grid = sim_scores.reshape(num_patches_per_col, num_patches_per_row)

        axs[i].imshow(image)
        im = axs[i].imshow(
            sim_grid, cmap="hot", alpha=0.5,
            extent=(0, model_input_size[0], model_input_size[1], 0),
            vmin=0, vmax=1
        )
        axs[i].set_title(f"{concept}", pad=10)
        axs[i].axis("off")
        
    # Add colorbar at the right
    cbar = fig.colorbar(im, ax=axs, location='right', shrink=0.8, label="Cosine Similarity")

    if title is not None:
        fig.suptitle(title)
    plt.show()
    
    
### Higher Level Fxns ###
def superdetector_inversion_across_percentiles(percentiles, img_idx, all_images, agglomerate_type, embeds, 
                                                     act_metrics, gt_samples_per_concept_test, 
                                  dataset_name, model_input_size, con_label, device, do_plot=True, local=True):
    """
    Computes and optionally visualizes inversion maps for multiple superdetector percentile thresholds
    on a given image.

    For each percentile, a superdetector vector is computed (either locally or globally), and the cosine
    similarity between that vector and each patch embedding in the image is computed to produce an 
    inversion heatmap per concept.

    Args:
        percentiles (list of float): List of top-k percentile values (0 < p < 1) used to define superdetectors.
        img_idx (int): Index of the image to analyze.
        all_images (list or dataset): List of all original images, used for visualization.
        agglomerate_type (str): Aggregation method used to combine patch embeddings ('avg' or 'max').
        embeds (torch.Tensor): Patch embeddings for all images (shape: [num_patches, embed_dim]).
        act_metrics (pd.DataFrame): Patch-level activation scores (shape: [num_patches, num_concepts]).
        gt_samples_per_concept_test (dict): Mapping from concept to test image indices where it appears.
        dataset_name (str): Name of the dataset, used for filtering and paths.
        model_input_size (tuple): Target size of input images, e.g., (224, 224).
        con_label (str): Concept label used to compute thresholds.
        device (torch.device): Device to run computations on (e.g., 'cuda').
        do_plot (bool, optional): If True, display heatmaps of cosine similarities for each concept and percentile.
        local (bool, optional): If True, compute superdetectors using only the current image. 
                                If False, use the whole dataset.

    Returns:
        dict: Mapping from each percentile to a concept-to-similarity-score dictionary (same format as `superdetector_inversion`).
    """
    
    
    inversions_per_percentile = {}                                     
    for percentile in percentiles:
        inversions = superdetector_inversion(img_idx, percentile, agglomerate_type, embeds, act_metrics, gt_samples_per_concept_test, 
                                  dataset_name, model_input_size, con_label, device, patch_size=14, local=local)

            
        if do_plot:
            title = f'Using Top {percentile*100}% Superdetectors'
            if local:
                title += ' (local)'
            else:
                title += ' (global)'
            plot_inversions(img_idx, inversions, all_images, model_input_size, title=title)

        inversions_per_percentile[percentile] = inversions
    return inversions_per_percentile


### Quantiative Evals ###
def detect_then_invert_locally_metrics(detect_percentile, invert_percentile, act_metrics, concepts, 
                               gt_samples_per_concept, gt_samples_per_concept_cal, device, dataset_name, 
                               model_input_size, con_label, all_object_patches=None, 
                               patch_size=14, agglomerate_type='avg'):
    """
    Performs two-stage concept detection: (1) detect images where a concept might be present using a 
    detection threshold, then (2) evaluate activation within those detected images using an inverted 
    threshold for concept classification. Computes classification metrics for each concept based on 
    patch-level predictions.

    Args:
        detect_percentile (float): Percentile used to compute the image-level detection thresholds.
        invert_percentile (float): Percentile used to compute the patch-level inversion thresholds.
        act_metrics (pd.DataFrame): Activation metric matrix (rows: patches, columns: concepts).
        concepts (list of str): List of concept names to evaluate.
        gt_samples_per_concept (dict): Ground truth concept labels (patch indices) across the full dataset.
        gt_samples_per_concept_test (dict): Ground truth concept labels (patch indices) on the test set.
        device (str): Torch device identifier (e.g., 'cuda').
        dataset_name (str): Name of the dataset.
        model_input_size (int): Image input size used to determine patch indexing.
        con_label (str): String identifier used in metric saving and tracking.
        all_object_patches (set, optional): If provided, restrict evaluation to these patch indices.
        n_trials (int): Number of repeated trials to average metrics over.
        balance_dataset (bool): Whether to balance the number of positive and negative examples in each trial.
        patch_size (int): Size of each patch (default: 14).

    Returns:
        pd.DataFrame: A dataframe containing per-concept evaluation metrics (e.g., accuracy, precision, recall, F1).
    """
    #detect using superpatches of given percentile
    # detect_thresholds = compute_concept_thresholds(gt_samples_per_concept_test, 
    #                                             act_metrics, detect_percentile, n_vectors=1, device=device, 
    #                                             n_concepts_to_print=0, dataset_name=dataset_name, con_label=con_label)
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
    
    detect_thresholds = all_thresholds[detect_percentile]
    
    #invert based on superpatches for given percentile
    inversion_activations = pd.read_csv(f'Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_per_{invert_percentile}_{con_label}.csv')
    
    # Handle column name mismatch for unsupervised methods
    if 'kmeans' in con_label:
        # Check if there's a mismatch between activation columns and ground truth keys
        gt_keys = set(str(k) for k in gt_samples_per_concept_cal.keys())
        act_cols = set(inversion_activations.columns)
        
        # If no overlap, we need to map columns
        if not gt_keys.intersection(act_cols):
            # Load alignment to get mapping
            alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
            
            # Create mapping from current column names to what ground truth expects
            col_mapping = {}
            for concept_name, info in alignment_results.items():
                cluster_id = str(info['best_cluster'])
                if cluster_id in act_cols:
                    col_mapping[cluster_id] = concept_name  # Map cluster ID to concept name
            
            # Rename columns if needed
            if col_mapping:
                inversion_activations = inversion_activations.rename(columns=col_mapping)
    
    inversion_thresholds = compute_concept_thresholds(gt_samples_per_concept_cal, 
                                                inversion_activations, invert_percentile, n_vectors=1, device=device, 
                                                n_concepts_to_print=0, dataset_name=dataset_name, con_label=con_label)
    
    
    # Initialize dictionaries to store counts per concept
    fp_counts = {}
    fn_counts = {}
    tp_counts = {}
    tn_counts = {}
    
    # Get the split dataframe.
    split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    
    # Get calibration indices as a torch tensor.
    cal_indices = torch.tensor(split_df.index[split_df == 'cal'].tolist())
    
    #filter patches that are 'padding' given the preprocessing schemes
    if model_input_size[0] == 'text':
        relevant_indices = cal_indices
    else:
        relevant_indices = filter_patches_by_image_presence(cal_indices, dataset_name, model_input_size)

    # If filtering patches to ones that contain some concept, restrict to indices in all_object_patches.
    if all_object_patches is not None:
        relevant_indices = torch.tensor([int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches])
    
    # Get ground truth labels for all concepts - USE CALIBRATION GT
    all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept_cal)
  
    # Get a boolean DataFrame indicating whether each patch is part of an image that was 'detected'
    detected_patch_masks = get_patch_detection_tensor(act_metrics, detect_thresholds, model_input_size, dataset_name)
  
    # Loop over each concept.
    for concept, concept_labels in all_concept_labels.items():
        # Get activation values for the selected indices.
        relevant_indices_list = relevant_indices.tolist()
        act_vals = torch.tensor(inversion_activations[concept].loc[relevant_indices_list].values)
        
        #only count patches from images that were first detected
        detected_patches = torch.tensor(detected_patch_masks[concept].loc[relevant_indices_list].values)
        activated_patches = (act_vals >= inversion_thresholds[concept][0]) &  detected_patches

        # Compute ground truth mask for these indices using the tensor directly.
        gt_mask = (concept_labels[relevant_indices] == 1)

        # Compute confusion matrix counts using torch.sum.
        tp = torch.sum(activated_patches & gt_mask).item()
        fn = torch.sum((~ activated_patches) & gt_mask).item()
        fp = torch.sum(activated_patches & (~gt_mask)).item()
        tn = torch.sum((~activated_patches) & (~gt_mask)).item()

        # Append the counts for this trial.
        tp_counts[concept] = tp
        fn_counts[concept] = fn
        fp_counts[concept] = fp
        tn_counts[concept] = tn
    
    #calculate metrics from the count
    metrics_df = compute_concept_metrics(fp_counts, fn_counts, tp_counts, tn_counts, act_metrics.columns,
                                    dataset_name, f'superpatch_{agglomerate_type}_inv_{con_label}', 
                                         just_obj = (all_object_patches is not None),
                                         invert_percentile=invert_percentile, detect_percentile=detect_percentile)
    
    return metrics_df


def detect_then_invert_locally_metrics_over_percentiles(detect_percentiles, invert_percentiles, act_metrics, 
                                                        concepts, gt_samples_per_concept, gt_samples_per_concept_cal,
                                                        device, dataset_name, model_input_size, con_label,
                                                        all_object_patches=None, patch_size=14,
                                                        agglomerate_type='avg'):
    """ Calls detect then invert metrics performance across all percentile combinations
    """
    total_iters = sum(invert > detect for detect in detect_percentiles for invert in invert_percentiles)
    pbar = tqdm(total=total_iters, desc="Evaluating thresholds")
    
    for detect_percentile in detect_percentiles:
        for invert_percentile in invert_percentiles:
            if detect_percentile <= invert_percentile:
                # try:
                #     torch.load(f'Quant_Results/{dataset_name}/detectfirst_{detect_percentile*100}_per_{invert_percentile*100}_{con_label}.csv')
                # except:
                detect_then_invert_locally_metrics(detect_percentile, invert_percentile, act_metrics, concepts, 
                                   gt_samples_per_concept, gt_samples_per_concept_cal, device, dataset_name, 
                                   model_input_size, con_label, all_object_patches=None, 
                                   patch_size=14, agglomerate_type='avg')
                # try:
                #     torch.load(f'Quant_Results/{dataset_name}/justobj_detectfirst_{detect_percentile*100}_per_{invert_percentile*100}_{con_label}.csv')
                # except:
                # detect_then_invert_locally_metrics(detect_percentile, invert_percentile, act_metrics, concepts, 
                #                    gt_samples_per_concept, gt_samples_per_concept_cal, device, dataset_name, 
                #                    model_input_size, con_label, all_object_patches=all_object_patches, 
                #                                    patch_size=14, agglomerate_type='avg')
                pbar.update(1)
    pbar.close()


def find_optimal_superdetector_thresholds(detect_percentiles, invert_percentiles, dataset_name, con_label, 
                                          model_input_size, optimization_metric='f1', agglomerate_type='avg'):
    """
    Finds optimal detect/invert percentile pairs for superdetector method based on calibration results.
    
    Args:
        detect_percentiles: List of detect percentiles to search over
        invert_percentiles: List of invert percentiles to search over  
        dataset_name: Name of dataset
        con_label: Concept label
        model_input_size: Model input size for loading GT data
        optimization_metric: Metric to optimize (default: 'f1')
        agglomerate_type: Aggregation type for superdetector ('avg' or 'max')
    """
    print(f"Finding optimal superdetector thresholds based on {optimization_metric}...")
    
    # Collect all concepts from calibration results
    concepts = set()
    
    for detect_p in detect_percentiles:
        for invert_p in invert_percentiles:
            if detect_p <= invert_p:
                filename = f"Quant_Results/{dataset_name}/detectfirst_{detect_p*100}_per_{invert_p*100}_superpatch_{agglomerate_type}_inv_{con_label}.csv"
                try:
                    df = pd.read_csv(filename)
                    concepts.update(df['concept'].tolist())
                    break
                except FileNotFoundError:
                    continue
        if concepts:
            break
    
    if not concepts:
        print("No calibration results found for superdetector method. Cannot find optimal thresholds.")
        return
    
    # Create directory for saving thresholds
    os.makedirs(f'Detect_Invert_Thresholds/{dataset_name}', exist_ok=True)
    
    # Find optimal thresholds for each concept
    for concept in concepts:
        best_score = -float('inf')
        best_detect_p = None
        best_invert_p = None
        
        for detect_p in detect_percentiles:
            for invert_p in invert_percentiles:
                if detect_p <= invert_p:
                    filename = f"Quant_Results/{dataset_name}/detectfirst_{detect_p*100}_per_{invert_p*100}_superpatch_{agglomerate_type}_inv_{con_label}.csv"
                    try:
                        df = pd.read_csv(filename)
                        
                        concept_row = df[df['concept'] == concept]
                        if not concept_row.empty and optimization_metric in concept_row.columns:
                            score = concept_row[optimization_metric].iloc[0]
                            if score > best_score:
                                best_score = score
                                best_detect_p = detect_p
                                best_invert_p = invert_p
                    except FileNotFoundError:
                        continue
        
        if best_detect_p is not None:
            # Load the actual threshold values used for this concept
            if 'kmeans' not in con_label:
                all_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt', weights_only=False)
                detect_threshold = all_thresholds[best_detect_p][concept]
            else:
                # For unsupervised concepts, load matched thresholds
                raw_thresholds = torch.load(f'Thresholds/{dataset_name}/all_percentiles_allpairs_{con_label}.pt', weights_only=False)
                alignment_results = torch.load(f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt', weights_only=False)
                
                # Get the cluster ID for this concept
                cluster_id = alignment_results[concept]['best_cluster']
                key = (concept, cluster_id)
                
                detect_threshold = raw_thresholds[best_detect_p][key] if key in raw_thresholds[best_detect_p] else None
            
            # For invert threshold, load from superdetector inversion results
            inversion_filename = f'Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_per_{best_invert_p}_{con_label}.csv'
            try:
                inversion_activations = pd.read_csv(inversion_filename)
                # Compute invert threshold from calibration data
                from utils.patch_alignment_utils import get_patch_split_df
                from utils.general_utils import get_split_df
                
                # Load calibration GT
                if model_input_size[0] == 'text':
                    model_input_size_key = model_input_size
                else:
                    model_input_size_key = model_input_size
                gt_patches_per_concept_cal = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size_key}.pt')
                
                invert_threshold = compute_concept_thresholds(gt_patches_per_concept_cal, 
                                                            inversion_activations, best_invert_p, n_vectors=1, 
                                                            device='cpu', n_concepts_to_print=0, 
                                                            dataset_name=dataset_name, con_label=con_label)[concept]
            except FileNotFoundError:
                print(f"Warning: Could not load inversion activations for {concept}")
                invert_threshold = None
            
            # Save optimal thresholds
            threshold_file = f'Detect_Invert_Thresholds/{dataset_name}/optimal_{optimization_metric}_superdetector_{con_label}_{concept}.pt'
            optimal_thresholds_data = {
                'detect_percentile': best_detect_p,
                'invert_percentile': best_invert_p,
                'detect_threshold': detect_threshold,
                'invert_threshold': invert_threshold,
                f'best_{optimization_metric}': best_score
            }
            torch.save(optimal_thresholds_data, threshold_file)
            print(f"Saved optimal thresholds for {concept}: detect_p={best_detect_p}, invert_p={best_invert_p}, {optimization_metric}={best_score:.4f}")
        else:
            print(f"Warning: Could not find optimal thresholds for concept {concept}")


def detect_then_invert_locally_with_optimal_thresholds(act_metrics, concepts, gt_samples_per_concept, 
                                                      gt_samples_per_concept_test, device, dataset_name, 
                                                      model_input_size, con_label, agglomerate_type='avg',
                                                      all_object_patches=None, patch_size=14):
    """
    Evaluates superdetector inversion on test set using per-concept optimal thresholds.
    
    Args:
        act_metrics: Activation metrics DataFrame
        concepts: Concept vectors dictionary
        gt_samples_per_concept: Full GT samples per concept
        gt_samples_per_concept_test: Test split GT samples per concept
        device: Device for computation
        dataset_name: Name of dataset
        model_input_size: Model input size
        con_label: Concept label
        agglomerate_type: Aggregation type for superdetector ('avg' or 'max')
        all_object_patches: Object patches filter (optional)
        patch_size: Patch size
    """
    print(f"Evaluating superdetector method on test set with optimal thresholds...")
    
    # Load optimal thresholds
    optimal_thresholds = {}
    concepts_list = list(concepts.keys()) if hasattr(concepts, 'keys') else list(gt_samples_per_concept_test.keys())
    
    for concept in concepts_list:
        try:
            threshold_file = f'Detect_Invert_Thresholds/{dataset_name}/optimal_f1_superdetector_{con_label}_{concept}.pt'
            optimal_thresholds[concept] = torch.load(threshold_file, weights_only=False)
        except FileNotFoundError:
            print(f"Warning: No optimal threshold found for concept {concept}, skipping...")
            continue
    
    if not optimal_thresholds:
        print("No optimal thresholds found for any concept. Skipping superdetector test evaluation.")
        return
    
    # Create a custom version of detect_then_invert_locally_metrics that uses actual thresholds instead of percentiles
    # This avoids modifying the existing function signature
    from utils.patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence
    from utils.general_utils import create_binary_labels
    from utils.quant_concept_evals_utils import get_patch_detection_tensor, compute_concept_metrics
    
    # Get test split indices
    split_df = get_patch_split_df(dataset_name, patch_size=patch_size, model_input_size=model_input_size)
    test_indices = torch.tensor(split_df.index[split_df == 'test'].tolist())
    
    if model_input_size[0] == 'text':
        relevant_indices = test_indices
    else:
        relevant_indices = filter_patches_by_image_presence(test_indices, dataset_name, model_input_size)
    
    if all_object_patches is not None:
        relevant_indices = torch.tensor([int(idx.item()) for idx in relevant_indices if int(idx.item()) in all_object_patches])
    
    # Get ground truth labels - USE TEST GT
    all_concept_labels = create_binary_labels(len(split_df), gt_samples_per_concept_test)
    
    # Initialize results storage
    fp_counts = {}
    fn_counts = {}
    tp_counts = {}
    tn_counts = {}
    
    # Process each concept with its optimal thresholds
    for concept, threshold_info in optimal_thresholds.items():
        detect_threshold = threshold_info['detect_threshold']
        invert_threshold = threshold_info['invert_threshold']
        
        if detect_threshold is None or invert_threshold is None:
            print(f"Skipping {concept} due to missing thresholds")
            continue
            
        print(f"Evaluating concept {concept} with optimal thresholds on test set")
        
        # Get detection mask using detect threshold
        detected_patches = get_patch_detection_tensor(
            act_metrics, {concept: detect_threshold}, model_input_size, dataset_name
        )[concept]
        
        # Load inversion activations for this concept at the optimal invert percentile
        invert_p = threshold_info['invert_percentile']
        inversion_filename = f'Superpatches/{dataset_name}/superpatch_{agglomerate_type}_inv_per_{invert_p}_{con_label}.csv'
        try:
            inversion_activations = pd.read_csv(inversion_filename)
            
            # Get inversion activations for relevant indices
            concept_inv_acts = torch.tensor(
                inversion_activations[concept].loc[relevant_indices.tolist()].values, device=device
            )
            
            # Get detection mask for relevant indices
            detected_mask = torch.tensor(
                detected_patches.loc[relevant_indices.tolist()].values, device=device
            )
            
            # Compute predictions: detected AND above invert threshold
            predictions = detected_mask & (concept_inv_acts >= invert_threshold[0])
            
            # Ground truth mask
            concept_labels = all_concept_labels[concept]
            gt_mask = torch.tensor(concept_labels[relevant_indices] == 1, device=device)
            
            # Compute confusion matrix
            tp = torch.sum(predictions & gt_mask).item()
            fn = torch.sum((~predictions) & gt_mask).item()
            fp = torch.sum(predictions & (~gt_mask)).item()
            tn = torch.sum((~predictions) & (~gt_mask)).item()
            
            # Store results
            tp_counts[concept] = tp
            fn_counts[concept] = fn
            fp_counts[concept] = fp
            tn_counts[concept] = tn
            
        except FileNotFoundError:
            print(f"Warning: Could not load inversion activations for {concept}")
            continue
    
    # Compute and save metrics
    if tp_counts:
        suffix = '_optimal_test'
        if all_object_patches is not None:
            suffix = 'justobj' + suffix
            
        metrics_df = compute_concept_metrics(
            fp_counts, fn_counts, tp_counts, tn_counts,
            list(tp_counts.keys()), dataset_name, 
            f'superpatch_{agglomerate_type}_inv_{con_label}{suffix}',
            just_obj=(all_object_patches is not None)
        )
    
    print("Completed superdetector test evaluation with optimal thresholds.")
    

def detect_then_invert_locally_performance_heatmap(metric_name, gt_samples_per_concept_test, dataset_name, con_label, 
                                           detect_percentiles, invert_percentiles, agglomerate_type='avg', just_obj=False):
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
                filename = f"Quant_Results/{dataset_name}/{prefix}detectfirst_{detect_p*100}_per_{invert_p*100}_superpatch_{agglomerate_type}_inv_{con_label}.csv"
                try:
                    df = pd.read_csv(filename)
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

    title = f"{metric_name} over Detect/Inversion Percentiles{max_label}\n(Local Superpatch Inversions)"
    if just_obj:
        title += " (Just Obj Patches)"
    plt.title(title, pad=10)

    plt.ylabel("Invert Percentile")
    plt.xlabel("Detect Percentile")
    plt.tight_layout()
    plt.show()
