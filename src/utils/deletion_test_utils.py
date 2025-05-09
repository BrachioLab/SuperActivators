"""Utils for Deletion Test"""
import torch
import pandas as pd
from PIL import Image
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

import importlib
import general_utils
importlib.reload(general_utils)
from general_utils import retrieve_image, plot_image_with_attributes

### Helper Functions for Deletion Test ###
def blackout_patches(image, patch_size, stride_ratio):
    """
    Extracts patches from the given PIL image and blackouts each patch according to the specified size and stride ratio.

    Args:
        image (PIL.Image): Image from which to extract and blackout patches.
        patch_size (tuple): Size of each patch as (height, width). Default is (16, 16).
        stride_ratio (float): Overlap ratio between patches. Default is 0.5.

    Returns:
        List[PIL.Image]: List of modified PIL images with blacked-out patches.
    """
    width, height = image.size
    patch_height, patch_width = patch_size

    step_y = int(patch_height * stride_ratio)
    step_x = int(patch_width * stride_ratio)

    # Iterate through the coordinates to blackout patches
    imgs_w_blackedout_patches = []
    patch_coords = []
    for y1 in range(0, height - patch_height + 1, step_y):
        for x1 in range(0, width - patch_width + 1, step_x):
            # Make a copy of the image to modify
            image_copy = image.copy()

            # Define the coordinates for the patch to blackout
            y2 = min(y1 + patch_height, height)
            x2 = min(x1 + patch_width, width)

            # Blackout the patch in the copied image (fill with black color)
            for y in range(y1, y2):
                for x in range(x1, x2):
                    image_copy.putpixel((x, y), (0, 0, 0))  # Set to black

            # Append the modified image and coordinates
            imgs_w_blackedout_patches.append(image_copy)
            patch_coords.append((x1, x2, y1, y2))
            
    return imgs_w_blackedout_patches, patch_coords


def compute_change_in_sim_heatmap(original_sim, blacked_patch_sims, patch_coords, image_size):
    """
    Computes the impact scores and generates the heatmap based on the change in concept activation
    for blacked-out patches.

    Args:
        original_sim (Tensor): The original cosine similarity between the image and the concept vector.
        blacked_patch_sims (Tensor): The cosine similarities after blacking out patches.
        patch_coords (List[Tuple[int, int, int, int]]): Coordinates of the patches that were blacked out.
        image_size (Tuple[int, int]): The size of the original image (height, width).

    Returns:
        Tensor: Heatmap of importance scores based on the change in similarity.
    """
    # Compute importance scores based on the difference between original similarity and modified similarity
    importance_scores = (original_sim - blacked_patch_sims).to('cpu')
    
    # Initialize the impact scores and counter scores to accumulate the impact of blacking out regions
    impact_scores = torch.zeros((image_size[1], image_size[0]))  # Same shape as image
    counter_scores = torch.zeros_like(impact_scores)

    # Loop over the patches to assign impact scores to each region
    for i, (x1, x2, y1, y2) in enumerate(patch_coords):
        # Calculate the change in similarity for this patch
        change = importance_scores[i]

        # Update the impact scores and counter scores for the corresponding patch region
        impact_scores[y1:y2, x1:x2] += change
        counter_scores[y1:y2, x1:x2] += 1

    # Normalize the impact scores by dividing by the counter scores to avoid multiple updates
    heatmap = torch.divide(impact_scores, counter_scores)
    
    return heatmap


def compute_avg_similarity_heatmap(original_sim, blacked_patch_sims, patch_coords, image_size):
    """
    Computes a heatmap based on the average concept similarity for each blacked-out patch.

    Args:
        original_sim (Tensor): The original cosine similarity between the image and the concept vector.
        blacked_patch_sims (Tensor): The cosine similarities after blacking out patches.
        patch_coords (List[Tuple[int, int, int, int]]): Coordinates of the patches that were blacked out.
        image_size (Tuple[int, int]): The size of the original image (height, width).

    Returns:
        Tensor: Heatmap based on the average concept similarity for each blacked-out patch.
    """
    # Initialize the heatmap with zeros
    heatmap = torch.zeros((image_size[1], image_size[0]))
    counter_scores = torch.zeros_like(heatmap)

    # Loop over the patches to compute average similarity for each region
    for i, (x1, x2, y1, y2) in enumerate(patch_coords):
        # Calculate the average similarity for this patch
        avg_similarity = blacked_patch_sims[i].item()  # Convert tensor to scalar value
        
        # Update the heatmap for the corresponding patch region with the average similarity
        heatmap[y1:y2, x1:x2] += avg_similarity
        counter_scores[y1:y2, x1:x2] += 1

    # Normalize the heatmap by dividing by the counter scores to avoid multiple updates
    heatmap = torch.divide(heatmap, counter_scores)
    
    return heatmap


def center_and_normalize(embeddings, mean_train_embedding, train_norm, device):
    """
    Center and normalize embeddings using statistics from the training set.

    Args:
        train_embeddings (torch.Tensor): Tensor of training set embeddings.
        test_embeddings (torch.Tensor): Tensor of test set embeddings.

    Returns:
        tuple: (normalized_train_embeddings, normalized_test_embeddings)
    """
    
    centered_embeddings = embeddings - mean_train_embedding.to(device)
    norm_embeddings = centered_embeddings / train_norm.to(device)
    return norm_embeddings
#     all_embeddings = torch.load(f'Embeddings/{dataset_name}/{embeds_file}')
    
#     # Load metadata
#     metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')

#     # Compute number of embeddings per image
#     n_embeddings_per_sample = all_embeddings.shape[0] // len(metadata) 

#     # Create boolean masks for train/test
#     train_mask = metadata["split"] == "train"

#     # Repeat each split mask for all patches per image
#     train_mask = train_mask.repeat(n_embeddings_per_sample).to_numpy()

#     # Apply masks to embeddings
#     train_embeddings = all_embeddings[train_mask]
    
#     #center embeddings
#     mean_train_embedding = train_embeddings.mean(dim=0)
#     centered_embeddings = embeddings - mean_train_embedding.to(device)

#     #normalize embeddings
#     train_norm = train_embeddings.norm(dim=1, keepdim=True).mean()  # Compute mean L2 norm from training set
#     norm_embeddings = centered_embeddings / train_norm
    
#     mean_train_embedding, train_norm = compute_train_avg_and_norm_and(all_embeddings, dataset_name)
    
#     print("Mean Train Embedding (Individual Method):",  mean_train_embedding)
#     print("Mean Train Norm (Individual Method):", train_norm)

#     return norm_embeddings


###For Visualizing Results of Deletion Tests###
def plot_heatmap(img_idx, heatmap, concept, heatmap_type='change_sim', dataset_name='CLEVR', save_file=None):
    """
    Plots heatmap visualization of 'important' patches for concept activation for a given concept and image.
    Args:
        img_idx (int): Index of the image in the metadata.
        heatmap (Tensor): Values of the heatmap
        heatmap_type (str): Type of heatmap ('change_sim' or 'avg_sim').
        dataset_name (str): Dataset name (default is 'CLEVR').
        save_file (str): Where to save heatmap png.
    """
    
    print(f"Visualizing importance for concept: {concept}")
    image = retrieve_image(img_idx, dataset_name)
    image = image.resize((224, 224))

    plt.figure(figsize=(16, 8))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot the grayscale image with the heatmap overlay
    ax = plt.subplot(1, 2, 2)  # Store the axis of the heatmap plot
    grayscale_image = image.convert('L')
    plt.imshow(grayscale_image, cmap='gray', alpha=0.5)
    heatmap_overlay = plt.imshow(heatmap, cmap='hot', alpha=0.5, extent=[0, grayscale_image.size[0], grayscale_image.size[1], 0])
    plt.title(f'Importance Heatmap for Concept {concept}')
    plt.axis('off')

    # Add a color bar specific to the heatmap
    cbar = plt.colorbar(heatmap_overlay, ax=ax, fraction=0.046, pad=0.04)
    if heatmap_type == 'change_sim':
        value_type = 'Avg Decrease in CosSim to Concept when Patch Removed'
    elif heatmap_type == 'avg_sim':
        value_type = 'Avg CosSim to Concept when Patch Removed'
    cbar.set_label(value_type)
        
    if save_file:
        save_path = f'../Figs/{dataset_name}/deltest_heatmaps/{save_file}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        
    plt.show()
    

def plot_heatmaps_all_concepts(img_idx, heatmaps, heatmap_type='change_sim', dataset_name='CLEVR', save_file=None):
    """
    Plots all heatmaps for all concepts with a consistent color scale.

    Args:
        img_idx (int): Index of the image in the metadata.
        heatmaps (dict): Dictionary where keys are concepts and values are heatmaps.
        heatmap_type (str): Type of heatmap ('change_sim' or 'avg_sim').
        dataset_name (str): Dataset name (default is 'CLEVR').
        save_file (str): Where to save heatmap png.
    """
    # Retrieve the image and convert to grayscale
    image = retrieve_image(img_idx, dataset_name)
    grayscale_image = image.convert('L')

    # Determine the global color scale range
    all_values = [value for heatmap in heatmaps.values() for row in heatmap for value in row]
    vmin, vmax = min(all_values), max(all_values)

    # Plot each heatmap
    concepts = list(heatmaps.keys())
    n_concepts = len(concepts)
    plt.figure(figsize=(16, 4 * ((n_concepts + 2) // 3)))  # Adjust figure size dynamically

    for i, concept in enumerate(concepts):
        heatmap = heatmaps[concept]
        ax = plt.subplot((n_concepts + 2) // 3, 3, i + 1)

        # Plot the grayscale image with heatmap overlay
        plt.imshow(grayscale_image, cmap='gray', alpha=0.5)
        heatmap_overlay = plt.imshow(heatmap, cmap='hot', alpha=0.5, vmin=vmin, vmax=vmax, extent=[0, grayscale_image.size[0], grayscale_image.size[1], 0])
        plt.title(f'{concept}')
        plt.axis('off')

    # Add a single color bar for the entire plot
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    if heatmap_type == 'change_sim':
        cbar.set_label('Avg Decrease in CosSim to Concept when Patch Removed')
    elif heatmap_type == 'avg_sim':
        cbar.set_label('Avg CosSim to Concept when Patch Removed')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    
    if save_file:
        save_path = f'../Figs/{dataset_name}/deltest_heatmaps/allconcepts_{save_file}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        
    plt.show()
