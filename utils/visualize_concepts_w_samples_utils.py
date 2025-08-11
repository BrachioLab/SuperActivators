import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import math
import ast

import torch.nn.functional as F
import torch

import importlib
import general_utils
importlib.reload(general_utils)
import patch_alignment_utils
importlib.reload(patch_alignment_utils)

from general_utils import retrieve_image, get_split_df, pad_or_resize_img, load_images, create_image_loader_function
from patch_alignment_utils import compute_patches_per_image, calculate_patch_location, compute_patch_similarities_to_vector, get_image_idx_from_global_patch_idx, get_patch_split_df, calculate_patch_indices

######### only when there's gt labels #########
def get_user_category(concept_columns):
    """
    Helper function to get the user's choice for concept category.

    Args:
        concept_columns (list): A list of concept column names.

    Returns:
        tuple: The selected category and the concept columns.
    """
    print("Available concept categories:")
    categories = sorted(set(col.split('::')[0] for col in concept_columns))  # Get unique concept categories
    for idx, category in enumerate(categories):
        print(f"{idx + 1}. {category}")
    
    while True:
        try:
            choice = int(input(f"Enter the number of the concept category you want to choose (1-{len(categories)}): "))
            if 1 <= choice <= len(categories):
                return categories[choice - 1], concept_columns
            else:
                print("Invalid choice, please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_user_concept(category, concept_columns):
    """
    Helper function to get the user's choice for specific concept within a category.

    Args:
        category (str): The selected concept category.
        concept_columns (list): A list of concept column names.

    Returns:
        str: The selected concept.
    """
    # Filter concepts based on selected category
    concepts = [col for col in concept_columns if col.startswith(category)]
    print(f"\nAvailable concepts in category '{category}':")
    for idx, concept in enumerate(concepts):
        print(f"{idx + 1}. {concept.split('::')[1]}")  # Display the specific concept (e.g., 'red', 'cube', etc.)
    
    while True:
        try:
            choice = int(input(f"Enter the number of the specific concept you want to choose (1-{len(concepts)}): "))
            if 1 <= choice <= len(concepts):
                return concepts[choice - 1]
            else:
                print("Invalid choice, please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def plot_aligned_images(acts_loader, con_label, concept_key=None, k=5, dataset_name='CLEVR', metric_type='Cosine Similarity', save_image=False, test_only=True):
    """
    Plot images that align well with a selected concept.

    Args:
        acts_loader: ChunkedActivationLoader instance or DataFrame with activations.
        con_label (str): label to put in path of saved image.
        concept_key (str): The concept to visualize. If None, will prompt user.
        k (int): Number of top images to display. Defaults to 5.
        dataset_name (str): The name of the dataset. Defaults to 'CLEVR'.
        metric_type (str): Type of metric being visualized.
        save_image (bool): Whether to save png of plots.
        test_only (bool): Whether to only consider test samples.

    Returns:
        None
    """
    # Load activations - handle both loader and DataFrame inputs
    if hasattr(acts_loader, 'load_full_dataframe'):
        comp_df = acts_loader.load_full_dataframe()
    else:
        comp_df = acts_loader  # Assume it's already a DataFrame
    
    # Filter for test samples if requested
    if test_only:
        metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
        test_indices = metadata[metadata['split'] == 'test'].index
        comp_df = comp_df.loc[comp_df.index.intersection(test_indices)]
    
    # Get the user's choice of concept if not provided
    concept_columns = list(comp_df.columns)
    if not concept_key:
        category, concept_columns = get_user_category(concept_columns)
        concept_key = get_user_concept(category, concept_columns)
    
    # Check if concept exists
    if concept_key not in comp_df.columns:
        print(f"Concept '{concept_key}' not found. Available concepts:")
        print(sorted(comp_df.columns)[:10], "...")
        return
    
    # Sort by cosine similarity and get the top k highest values for the specified concept
    top_k_indices = comp_df.nlargest(k, concept_key).index.tolist()
    
    # Calculate the number of rows and columns for the plot
    n_cols = k  # All images in one row
    n_rows = 1  # Single row

    # Plot the top k images based on cosine similarity
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * k, 5))
    if k == 1:
        axes = [axes]  # Ensure axes is always a list
    elif n_rows == 1:
        axes = axes.flatten()  # Flatten for consistency

    plt.suptitle(f"Top {k} Images with Highest {metric_type} to: Concept {concept_key}", fontsize=16)
    
    # Load only the images we need
    loaded_images = {}
    for idx in top_k_indices:
        loaded_images[idx] = retrieve_image(idx, dataset_name, test_only=False)
    
    for rank, idx in enumerate(top_k_indices):
        if rank >= len(axes):  # In case there are fewer images than axes
            break
        
        # Get the image from our loaded images
        img = loaded_images[idx]

        value = comp_df.loc[idx, concept_key]
        axes[rank].imshow(img)
        axes[rank].set_title(f"Rank {rank+1}: Image {idx}\n{metric_type} = {value:.4f}")
        axes[rank].axis('off')
    
    # Hide unused axes
    for rank in range(len(top_k_indices), len(axes)):
        axes[rank].axis('off')

    plt.tight_layout()
    
    if save_image:
        save_path = f'../Figs/{dataset_name}/most_aligned_w_concepts/concept_{concept_key}_{k}__{con_label}.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        
    plt.show()
    
    
###### Patch Similarities #####
def neighboring_patch_comparisons(image_index, patch_index_in_image, loader, 
                                  dataset_name, model_input_size, patch_size=14, 
                                  save_path=None):
    """
    Plots a heatmap of cosine similarity between a chosen patch's embedding and all other patches in an image.
    The heatmap is overlayed on the original image, and the original image is shown separately to the left.

    Args:
        image_index (int): The index of the image in the dataset.
        patch_index_in_image (int): The index of the patch within the image.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        save_path (str) : Where to save path.

    Returns:
        None: Displays the heatmap overlayed on the image.
    """
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)

    # Calculate patch indices
    patches_per_row, patches_per_col, global_patch_idx = calculate_patch_indices(
        image_index, patch_index_in_image, patch_size, model_input_size
    )

    # Calculate indices for all patches in the image
    patches_per_image = patches_per_row * patches_per_col
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all necessary embeddings in one go
    all_indices_needed = image_patch_indices + [global_patch_idx]
    all_indices_needed = list(set(all_indices_needed))  # Remove duplicates
    embeddings = loader.load_specific_embeddings(all_indices_needed)
    
    # Create a mapping from global indices to loaded position
    idx_to_position = {idx: pos for pos, idx in enumerate(all_indices_needed)}
    
    # Get the selected patch embedding
    selected_patch_position = idx_to_position[global_patch_idx]
    selected_patch_embedding = embeddings[selected_patch_position]
    
    # Get embeddings for all patches in the image
    image_patch_embeddings = torch.stack([
        embeddings[idx_to_position[idx]] for idx in image_patch_indices
    ])

    # Compute cosine similarities
    cos_sims = F.cosine_similarity(
        selected_patch_embedding.unsqueeze(0),
        image_patch_embeddings
    ).cpu()

    # Reshape similarities to match the patch grid
    cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)

    # Plot the heatmap
    plot_patches_sim_to_vector(
        cos_sim_grid, resized_image, patch_size, image_index, patch_index_in_image, save_path=save_path,
        plot_title = f'Patch Similarity (Image {image_index}, Patch {patch_index_in_image})',
        bar_title='Cosine Similarity with Chosen Patch'
    )
    

def make_image_with_highlighted_patch(image, left, top, right, bottom, model_input_size=(224, 224), plot_image_title=None, grayscale=False):
    """
    Helper function to return the original image with a red rectangle highlighting the patch.

    Args:
        image (PIL.Image): The original image to return.
        left, top, right, bottom (int): The coordinates of the patch.
        model_input_size (tuple): The size to which the image is resized during embedding.
        plot_image_title (str, optional): Title for the plot.

    Returns:
        PIL.Image: The image with the highlighted patch.
    """
    # Resize the image to match the embedding process
    resized_image = pad_or_resize_img(image, model_input_size)

    # Draw the rectangle on the resized image
    image_with_patch = resized_image.copy()
    if grayscale:
        image_with_patch = image_with_patch.convert('L').convert('RGB') 
    draw = ImageDraw.Draw(image_with_patch)
    draw.rectangle([left, top, right, bottom], outline="blue", width=5)

    if plot_image_title is not None:
        plt.imshow(image_with_patch)
        plt.title(plot_image_title)
        plt.axis('off')
        plt.show()

    return image_with_patch
    
    
def plot_patches_w_corr_images(patch_indices, concept_cos_sims, images, overall_title, model_input_size,
                               save_path=None, patch_size=14, metric_type='CosSim'):
    """
    Helper function to plot the original images with highlighted patches and the patches themselves.

    Args:
        patch_indices (list): List of patch indices to plot.
        concept_cos_sims (pd.Series): Cosine similarity values for the concept.
        images (list of PIL.Image): List of original images.
        overall_title (str): Title of the figure.
        save_path (str): Where to save plots.
        patch_size (int): Size of each patch.
        model_input_size (tuple): The size to which the image is resized during embedding.

    Returns:
        None: Returns the images with highlighted patches and corresponding patches.
    """
    #just display figure if it already was computed
    # if os.path.exists(save_path):
    #     plt.figure(figsize=(15, 10))
    #     plt.imshow(Image.open(save_path))
    #     plt.axis('off')
    #     plt.show()
    #     return
    
    top_n = len(patch_indices)
    
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))

    for i, patch_idx in enumerate(patch_indices):
        # Determine the image index
        #image_idx = patch_idx // ((model_input_size[0] // patch_size) * (model_input_size[1] // patch_size))get_
        image_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size)
        image = images[image_idx]

        # Calculate the patch location
        left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size, model_input_size)

        # Highlight the patch
        image_with_patch = make_image_with_highlighted_patch(image, left, top, right, bottom, model_input_size)

        # Plot the image with highlighted patch
        axes[0, i].imshow(image_with_patch)
        axes[0, i].set_title(f'Image {image_idx}')
        axes[0, i].axis('off')

        # Crop the patch from the resized image
        resized_image = pad_or_resize_img(image, model_input_size)
        patch = resized_image.crop((left, top, right, bottom))

        # Plot the cropped patch
        axes[1, i].imshow(patch)
        try:
            axes[1, i].set_title(f'Patch {patch_idx} ({metric_type}: {concept_cos_sims[patch_idx]:.2f})')
        except:
            axes[1, i].set_title(f'Patch {patch_idx} ({metric_type}: {concept_cos_sims.iloc[patch_idx]:.2f})')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.suptitle(overall_title, fontsize=16, y=1.05)
                   
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
      
    plt.show()

    

def plot_top_patches_for_concept(concept_label, cos_sims, images, dataset_name, save_path='', 
                                 top_n=5, patch_size=14, model_input_size=(224, 224), metric_type='CosSim',
                                 test_only=True):
    """
    Given a concept label, plot the given patches that align most with concept 
    with their original images and highlight the patch locations.

    Args:
        concept_label (str): The concept label for which to plot patches.
        cos_sims (pd.DataFrame): DataFrame with cosine similarities, where rows are patch indices
                                 and columns are concept labels.
        images (list of PIL.Image): List of PIL Image objects, the original images.
        save_path (str): Where to save the resulting figure.
        top_n (int): The number of patches to plot for each concept.
        patch_size (int): Size of each patch (default is 14).
        model_input_size (tuple): The size to which the image is resized during embedding.

    Returns:
        None: Displays the patches and their respective images with highlighted locations.
    """   
    split_df = get_patch_split_df(dataset_name, patch_size=14, model_input_size=model_input_size)
    
    if test_only:
        test_image_indices = split_df[split_df == 'test'].index
        cos_sims = cos_sims.loc[test_image_indices]
    
    #Get the cosine similarity values for the specified concept
    concept_cos_sims = cos_sims[concept_label]
    
    #Sort the patches by cosine similarity in descending order
    top_patch_indices = concept_cos_sims.nlargest(top_n).index 

    #Call the helper function to plot the images and patches
    overall_title = f'{top_n} Test Patches Most Activated by Concept {concept_label}'
    plot_patches_w_corr_images(top_patch_indices, concept_cos_sims, images, overall_title, save_path=save_path, patch_size=patch_size, model_input_size=model_input_size, metric_type=metric_type)
    

def plot_most_similar_patches_w_heatmaps_and_corr_images(concept_label, acts_loader, con_label, dataset_name, model_input_size, vmin=None, vmax=None, save_path="", patch_size=14, top_n=5, metric_type='Cosine Similarity', test_only=True):
    """
    Plots the most similar patches with a chosen concept, as well as the heatmaps for that concept and the corresponding image.
    
    Args:
        concept_label (str): The concept to visualize.
        acts_loader: ChunkedActivationLoader instance or DataFrame with activations.
        con_label (str): Label for saving.
        dataset_name (str): Name of the dataset.
        model_input_size (tuple): Model input size.
        vmin (float): Minimum value for heatmap color scale.
        vmax (float): Maximum value for heatmap color scale.
        save_path (str): Where to save the figure.
        patch_size (int): Size of patches.
        top_n (int): Number of top patches to show.
        metric_type (str): Type of metric.
        test_only (bool): Whether to only use test samples.
    """
    
    # Load activations - handle both loader and DataFrame inputs
    if hasattr(acts_loader, 'load_full_dataframe'):
        cos_sims = acts_loader.load_full_dataframe()
    else:
        cos_sims = acts_loader  # Assume it's already a DataFrame
    
    # Filter for test samples if needed
    if test_only:
        split_df = get_patch_split_df(dataset_name, model_input_size, patch_size)
        test_indices = split_df[split_df == 'test'].index
        cos_sims_filtered = cos_sims.loc[cos_sims.index.intersection(test_indices)]
    else:
        cos_sims_filtered = cos_sims
    
    # Check if concept exists
    if concept_label not in cos_sims.columns:
        print(f"Concept '{concept_label}' not found. Available concepts:")
        print(sorted(cos_sims.columns)[:10], "...")
        return
    
    # Get top patches
    most_similar_patches = cos_sims_filtered[concept_label].sort_values(ascending=False).head(top_n).index
    
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))
    if top_n == 1:
        axes = axes.reshape(-1, 1)
    
    heatmaps = {}
    images_w_patches = []
    
    # Calculate patches per image
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Determine which images we need to load
    image_indices_needed = set()
    for patch_idx in most_similar_patches:
        image_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size)
        image_indices_needed.add(image_idx)
    
    # Load only the needed images
    loaded_images = {}
    for image_idx in image_indices_needed:
        loaded_images[image_idx] = retrieve_image(image_idx, dataset_name, test_only=False)
    
    for patch_idx in most_similar_patches:
        # Determine the image index
        image_idx = get_image_idx_from_global_patch_idx(patch_idx, model_input_size, patch_size)
        image = loaded_images[image_idx]
        
        # Get patch activations for this image to create heatmap
        start_idx = image_idx * patches_per_image
        end_idx = start_idx + patches_per_image
        image_patch_acts = cos_sims[concept_label].iloc[start_idx:end_idx]
        
        # Reshape to 2D heatmap
        heatmap = torch.tensor(image_patch_acts.values).reshape(patches_per_col, patches_per_row)
        
        # Calculate the patch location
        left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size, model_input_size)
        
        # Highlight the patch
        image_with_patch = make_image_with_highlighted_patch(image, left, top, right, bottom, model_input_size, grayscale=True)
        
        heatmaps[image_idx] = heatmap
        images_w_patches.append(image_with_patch)
    
    # Determine the global color scale range across all heatmaps
    if vmin is None or vmax is None:
        all_values = [value.item() for heatmap in heatmaps.values() for row in heatmap for value in row]
        vmin, vmax = min(all_values), max(all_values)
    
    for i, patch_idx in enumerate(most_similar_patches):
        image_idx = patch_idx // patches_per_image
        image = loaded_images[image_idx]
        resized_image = pad_or_resize_img(image, model_input_size)
        
        # Plot the original image
        axes[0, i].imshow(resized_image)
        axes[0, i].set_title(f'Image {image_idx}')
        axes[0, i].axis('off')
        
        # Plot the image with highlighted patch and heatmap
        heatmap = heatmaps[image_idx]
        axes[1, i].imshow(images_w_patches[i], alpha=0.6)
        heatmap_overlay = axes[1, i].imshow(heatmap, cmap='hot', alpha=0.4, 
                                           extent=(0, model_input_size[0], model_input_size[1], 0),
                                           vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Max = {round(heatmap.max().item(), 4)}\nMin = {round(heatmap.min().item(), 4)}')
        axes[1, i].axis('off')
    
    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label(metric_type)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    plt.suptitle(f"Most Activated {'Test' if test_only else ''} Patches by Concept {concept_label}", fontsize=16, y=1.05)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    
    plt.show()
    

def plot_patchsims_for_concept(concept_label, heatmaps, image_indices, images, model_input_size, 
                               dataset_name, top_n=7, save_file=None, 
                               metric_type='Cosine Similarity', vmin=None, vmax=None):
    """
    Plots patch similarities for a single concept across multiple images with a consistent color scale,
    using precomputed heatmaps provided in the heatmaps parameter. The original image is displayed above its corresponding heatmap.

    Args:
        concept_label (str): The name of the concept to be visualized.
        image_indices (list of int): A list of image indices to visualize for the given concept.
        heatmaps (dict): A dictionary where keys are image indices and values are precomputed heatmaps.
        images (list of PIL.Image): A list of images.
        model_input_size (tuple): The dimensions (width, height) to which the image is resized for model input.
        dataset_name (str): Dataset name (default is 'CLEVR').
        save_file (str): Where to save the heatmap png. If None, the plot is not saved.
    """
    
    # Determine the global color scale range across all heatmaps
    if not vmin or not vmax:
        all_values = [value for heatmap in heatmaps.values() for row in heatmap for value in row]
        vmin, vmax = min(all_values), max(all_values)

    # Create a figure with a size based on the number of images
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))  # 2 rows: 1 for images, 1 for heatmaps

    for i, image_index in enumerate(image_indices):
        # Retrieve the image and corresponding heatmap
        image = retrieve_image(image_index, dataset_name)
        heatmap = heatmaps[image_index]

        # Resize the image
        resized_image = pad_or_resize_img(image, model_input_size)

        # Get the axes for the image and heatmap
        ax_image = axes[0, i]  # Top row for the image
        ax_heatmap = axes[1, i]  # Bottom row for the heatmap

        # Plot the image in the top row (ax_image)
        ax_image.imshow(resized_image)
        ax_image.set_title(f'Image {image_index}')
        ax_image.axis('off')

        # Plot the heatmap in the bottom row (ax_heatmap)
        heatmap_overlay = ax_heatmap.imshow(heatmap, cmap='hot', alpha=0.6, vmin=vmin, vmax=vmax, extent=[0, model_input_size[0], model_input_size[1], 0])

        ax_heatmap.set_title(f'Heatmap Max = {round(heatmap.max().item(), 4)}\nHeatmap Min = {round(heatmap.min().item(), 4)}')
        #ax_heatmap.set_title(f'Heatmap Max = {round(heatmap.max().item(), 2)}')
        ax_heatmap.imshow(resized_image.convert('L'), cmap='gray', alpha=0.4)
        ax_heatmap.axis('off')

    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label(metric_type)

    # Adjust layout to prevent overlap
    plt.suptitle(f"Concept {concept_label} Activations", fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    

    # Optionally save the figure
    if save_file:
        save_path = f'../Figs/{dataset_name}/patch_alignment_to_concept/{concept_label}_allimages.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)

    plt.show()
    

def plot_patchsims_heatmaps_all_concepts(concept_labels, heatmaps, image_indices,
                                           model_input_size, dataset_name, top_n=7, 
                                           save_file=None, metric_type='Cosine Similarity', vmin=None, vmax=None):
    """
    Plots patch similarities for multiple concepts across multiple images with a consistent color scale.
    The original images are displayed only once at the top, with heatmaps for each concept in subsequent rows.

    Args:
        concept_labels (list of str): List of concept names to be visualized.
        image_indices (list of int): A list of image indices to visualize for the given concepts.
        heatmaps (dict): A dictionary where keys are image indices and values are precomputed heatmaps.
        model_input_size (tuple): The dimensions (width, height) to which the image is resized for model input.
        dataset_name (str): Dataset name (default is 'CLEVR').
        save_file (str): Where to save the heatmap png. If None, the plot is not saved.
    """

    # Determine the global color scale range across all heatmaps
    if vmin is None or vmax is None:
        vmin, vmax = np.inf, -np.inf
        for k, v in heatmaps.items():
            for k2, v2 in v.items():
                vmin = min(vmin, v2.min())
                vmax = max(vmax, v2.max())
#     if vmin is None or vmax is None:
#         all_vals = []
#         for concept_dict in heatmaps.values():
#             for heatmap in concept_dict.values():
#                 all_vals.append(heatmap.flatten())
#         all_vals = np.concatenate(all_vals)

#         # Set vmin/vmax using percentiles to ignore extreme outliers
#         vmin = np.percentile(all_vals, 1)
#         vmax = np.percentile(all_vals, 99)

    # Create a figure with a size based on the number of concepts and images
    num_concepts = len(concept_labels)
    fig, axes = plt.subplots(num_concepts + 1, top_n, figsize=(top_n * 3, (num_concepts + 1) * 3))  # 1 row for images, other rows for heatmaps

    # First, plot the original images on the top row
    for i, image_index in enumerate(image_indices):
        image = retrieve_image(image_index, dataset_name)
        resized_image = pad_or_resize_img(image, model_input_size)

        ax_image = axes[0, i]  # Top row for images
        ax_image.imshow(resized_image)
        ax_image.set_title(f'Image {image_index}')
        ax_image.axis('off')

    # Set the row labels for each concept once (to the left of the row)
    axes[0, 0].text(-0.1, 0.5, 'Original\nImage', fontsize=20,
                              verticalalignment='center', horizontalalignment='right',
                              transform=axes[0, 0].transAxes)
    for j, concept_label in enumerate(concept_labels):
        axes[j + 1, 0].text(-0.1, 0.5, f'{concept_label}', fontsize=20,
                              verticalalignment='center', horizontalalignment='right',
                              transform=axes[j + 1, 0].transAxes)

    # Then, loop through images and concepts to plot the heatmaps
    for i, image_index in enumerate(image_indices):
        image = retrieve_image(image_index, dataset_name)
        resized_image = pad_or_resize_img(image, model_input_size)

        for j, concept_label in enumerate(concept_labels):
            heatmap = heatmaps[concept_label][image_index]  # Access heatmap for this concept and image

            ax_heatmap = axes[j + 1, i]  # Heatmap rows are below the image row
            ax_heatmap.imshow(heatmap, cmap='hot', alpha=0.6, vmin=vmin, vmax=vmax,
                              extent=[0, model_input_size[0], model_input_size[1], 0])
            ax_heatmap.set_title(f'Heatmap Max = {round(heatmap.max().item(), 4)}\n'
                                 f'Heatmap Min = {round(heatmap.min().item(), 4)}')
            ax_heatmap.imshow(resized_image.convert('L'), cmap='gray', alpha=0.4)  # Overlay the image in gray
            ax_heatmap.axis('off')

    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='hot')
    sm.set_array([])  # You don't need to set any specific array data.
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(metric_type)

    # Adjust layout to prevent overlap
    plt.suptitle(f"Concept Activations on Random Test {dataset_name} Images", fontsize=16, y=1)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    
    # Optionally save the figure
    if save_file:
        save_path = f'../Figs/{dataset_name}/patch_alignment_to_concept/{concept_label}_allimages.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)

    plt.show()
    

def plot_patchsims_all_concepts(img_idx, heatmaps, model_input_size, dataset_name, 
                                metric_type = 'Cosine Similarity', vmin=None, vmax=None, 
                                sort_by_act=False, save_file=None):
    """
    Plots all patch similarities for all concepts with a consistent color scale.

    Args:
        img_idx (int): Index of the image in the metadata.
        heatmaps (dict): Dictionary where keys are concepts and values are heatmaps.
        model_input_size (tuple): Resized image size.
        dataset_name (str): Dataset name (default is 'CLEVR').
        save_file (str): Where to save heatmap png.
    """
    print(f"Plotting image {img_idx}:")
    # Retrieve the image and convert to grayscale
    image = retrieve_image(img_idx, dataset_name)
    resized_image = pad_or_resize_img(image, model_input_size)
    grayscale_image = resized_image.convert('L')

    # Determine the global color scale range across all heatmaps
    if not vmin or not vmax:
        all_values = [value for heatmap in heatmaps.values() for row in heatmap for value in row]
        vmin, vmax = min(all_values), max(all_values)

    # Plot each heatmap
    if sort_by_act:
        concepts = sorted(heatmaps.keys(), key=lambda x: heatmaps[x].max().item(), reverse=True)
    else:
        concepts = list(heatmaps.keys())
    n_concepts = len(concepts)
    plt.figure(figsize=(16, 4 * ((n_concepts + 2) // 3)))  # Adjust figure size dynamically

    for i, concept in enumerate(concepts):
        heatmap = heatmaps[concept]
        ax = plt.subplot((n_concepts + 2) // 3, 3, i + 1)

        # Plot the grayscale image with heatmap overlay
        image_width, image_height = grayscale_image.size
        heatmap_overlay = ax.imshow(heatmap, cmap='hot', alpha=0.4, extent=(0, image_width, image_height, 0),
                                   vmin=vmin, vmax=vmax)
        
        # ax.set_title(f'Heatmap Avg = {round(heatmap.mean().item(), 4)}\nHeatmap Max = {round(heatmap.max().item(), 4)}\nHeatmap Min = {round(heatmap.min().item(), 4)}')
        
        # Plot the grayscale image with heatmap overlay
        ax.imshow(grayscale_image, cmap='gray', alpha=0.6)
        plt.title(f'{concept}\nHeatmap Max = {round(heatmap.max().item(), 4)}\nHeatmap Min = {round(heatmap.min().item(), 4)}')
        plt.axis('off')

    # Add a single color bar for the entire plot
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label(metric_type)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    
    if save_file:
        save_path = f'../Figs/{dataset_name}/patch_alignment_to_concept/allconcepts_{save_file}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        
    plt.show()


def animate_patch_similarities(image_index, loader, dataset_name, model_input_size, 
                              patch_size=14, save_path=None, fps=2, show_animation=True,
                              skip_patches=1, max_frames=None, figsize=(12, 6)):
    """
    Creates an animation cycling through all patches in an image, showing similarity heatmaps.
    
    Args:
        image_index (int): The index of the image in the dataset.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        save_path (str): Where to save the animation (as .gif or .mp4).
        fps (int): Frames per second for the animation.
        show_animation (bool): Whether to display the animation in the notebook.
        skip_patches (int): Skip every N patches to reduce animation size (1 = no skip).
        max_frames (int): Maximum number of frames to include (None = all patches).
        figsize (tuple): Figure size (width, height) in inches.
        
    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    import matplotlib
    
    # Set matplotlib parameters to reduce animation size
    if show_animation:
        matplotlib.rcParams['animation.embed_limit'] = 40  # Increase limit to 40MB
    
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Calculate patch dimensions
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Calculate indices for all patches in the image
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all embeddings for this image
    embeddings = loader.load_specific_embeddings(image_patch_indices)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: original image with highlighted patch
    ax1.set_title(f'Image {image_index}')
    ax1.axis('off')
    
    # Right plot: heatmap
    ax2.set_title('Patch Similarity Heatmap')
    ax2.axis('off')
    
    # Initialize plots
    im1 = ax1.imshow(resized_image)
    im2 = ax2.imshow(np.zeros((patches_per_col, patches_per_row)), cmap='hot', alpha=0.5, 
                     extent=(0, model_input_size[0], model_input_size[1], 0), vmin=0, vmax=1)
    im2_bg = ax2.imshow(resized_image, alpha=0.5)
    
    # Create colorbar
    cbar = plt.colorbar(im2, ax=ax2)
    cbar.set_label('Cosine Similarity with Selected Patch')
    
    # Rectangle for highlighting current patch
    rect = plt.Rectangle((0, 0), patch_size, patch_size, 
                        edgecolor='red', facecolor='none', linewidth=3)
    ax1.add_patch(rect)
    
    def animate(frame):
        """Update function for animation."""
        patch_index_in_image = frame
        
        # Calculate patch position
        row, col = divmod(patch_index_in_image, patches_per_row)
        left = col * patch_size
        top = row * patch_size
        
        # Update highlighted patch rectangle
        rect.set_xy((left, top))
        
        # Get the selected patch embedding
        selected_patch_embedding = embeddings[patch_index_in_image]
        
        # Compute cosine similarities with all patches
        cos_sims = F.cosine_similarity(
            selected_patch_embedding.unsqueeze(0),
            embeddings
        ).cpu()
        
        # Reshape to grid
        cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)
        
        # Update heatmap
        im2.set_array(cos_sim_grid)
        im2.set_clim(cos_sim_grid.min(), cos_sim_grid.max())
        
        # Update title with current patch index
        ax2.set_title(f'Similarity to Patch {patch_index_in_image} (Row {row}, Col {col})')
        
        return [rect, im2]
    
    # Determine which frames to include
    if max_frames is not None:
        frames = list(range(0, min(patches_per_image, max_frames), skip_patches))
    else:
        frames = list(range(0, patches_per_image, skip_patches))
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=frames, 
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save animation if requested
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_path}")
    
    # Show animation
    if show_animation:
        plt.close()  # Close the static plot
        return HTML(anim.to_jshtml())
    else:
        plt.show()
        return anim


def show_patch_similarities_grid(image_index, loader, dataset_name, model_input_size, 
                                patch_size=14, n_patches_to_show=9, patch_indices=None):
    """
    Shows a grid of patch similarity heatmaps for selected patches in an image.
    
    Args:
        image_index (int): The index of the image in the dataset.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        n_patches_to_show (int): Number of patches to show (if patch_indices not provided).
        patch_indices (list): Specific patch indices to show (optional).
        
    Returns:
        None: Displays the grid of images.
    """
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Calculate patch dimensions
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Calculate indices for all patches in the image
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all embeddings for this image
    embeddings = loader.load_specific_embeddings(image_patch_indices)
    
    # Determine which patches to show
    if patch_indices is None:
        # Sample evenly across the image
        step = max(1, patches_per_image // n_patches_to_show)
        patch_indices = list(range(0, patches_per_image, step))[:n_patches_to_show]
    
    # Calculate grid layout
    n_cols = min(3, len(patch_indices))
    n_rows = (len(patch_indices) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each patch similarity
    for idx, patch_idx in enumerate(patch_indices):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]
        
        # Calculate patch position
        patch_row, patch_col = divmod(patch_idx, patches_per_row)
        
        # Get the selected patch embedding
        selected_patch_embedding = embeddings[patch_idx]
        
        # Compute cosine similarities with all patches
        cos_sims = F.cosine_similarity(
            selected_patch_embedding.unsqueeze(0),
            embeddings
        ).cpu()
        
        # Reshape to grid
        cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)
        
        # Plot the image with heatmap overlay
        ax.imshow(resized_image)
        heatmap = ax.imshow(cos_sim_grid, cmap='hot', alpha=0.5, 
                           extent=(0, model_input_size[0], model_input_size[1], 0),
                           vmin=0, vmax=1)
        
        # Highlight the selected patch
        left = patch_col * patch_size
        top = patch_row * patch_size
        rect = plt.Rectangle((left, top), patch_size, patch_size,
                           edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        
        ax.set_title(f'Patch {patch_idx} (R{patch_row}, C{patch_col})')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(patch_indices), n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].axis('off')
    
    plt.suptitle(f'Patch Similarities for Image {image_index}', fontsize=16)
    plt.tight_layout()
    plt.show()


def show_patch_similarities_simple(image_index, loader, dataset_name, model_input_size, 
                                 patch_size=14, start_patch=None, end_patch=None, delay=0.5):
    """
    Simple version: Shows patch similarity heatmaps one after another.
    
    Args:
        image_index (int): The index of the image in the dataset.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        start_patch (int): Starting patch index (local to image). If None, starts from 0.
        end_patch (int): Ending patch index (local to image, exclusive). If None, goes to last patch.
        delay (float): Delay in seconds between frames.
    """
    from IPython.display import clear_output, display
    import time
    
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Calculate patch dimensions
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Calculate indices for all patches in the image
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all embeddings for this image
    embeddings = loader.load_specific_embeddings(image_patch_indices)
    
    # Determine patch range
    if start_patch is None:
        start_patch = 0
    if end_patch is None:
        end_patch = patches_per_image
    
    # Validate range
    start_patch = max(0, min(start_patch, patches_per_image - 1))
    end_patch = max(start_patch + 1, min(end_patch, patches_per_image))
    
    # Create list of patches to show
    patches_to_show = list(range(start_patch, end_patch))
    
    print(f"Showing patches {start_patch} to {end_patch-1} ({len(patches_to_show)} total patches)...")
    
    # Pre-compute min/max values across all patches for consistent colorbar
    all_similarities = []
    for patch_idx in patches_to_show:
        selected_patch_embedding = embeddings[patch_idx]
        cos_sims = F.cosine_similarity(
            selected_patch_embedding.unsqueeze(0),
            embeddings
        ).cpu()
        all_similarities.append(cos_sims)
    
    # Find global min/max for consistent color scaling
    vmin = min(sims.min().item() for sims in all_similarities)
    vmax = max(sims.max().item() for sims in all_similarities)
    
    # Show each frame
    for i, patch_idx in enumerate(patches_to_show):
        # Clear previous output
        clear_output(wait=True)
        
        # Calculate patch position
        patch_row, patch_col = divmod(patch_idx, patches_per_row)
        
        # Use pre-computed similarities
        cos_sims = all_similarities[i]
        
        # Reshape to grid
        cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)
        
        # Create figure with colorbar space
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Display the image
        ax.imshow(resized_image)
        
        # Overlay the similarity heatmap with consistent scale
        im = ax.imshow(cos_sim_grid, cmap='hot', alpha=0.5, 
                       extent=(0, model_input_size[0], model_input_size[1], 0),
                       vmin=vmin, vmax=vmax)
        
        # Highlight the selected patch with red rectangle
        left = patch_col * patch_size
        top = patch_row * patch_size
        rect = plt.Rectangle((left, top), patch_size, patch_size,
                           edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        
        # Add colorbar with consistent scale
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)
        
        ax.set_title(f'Patch {patch_idx} (Row {patch_row}, Col {patch_col}) - Frame {i+1}/{len(patches_to_show)}')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Wait before next frame
        if i < len(patches_to_show) - 1:
            time.sleep(delay)
    
    print("Animation complete!")


def show_patch_similarities_sequence(image_index, loader, dataset_name, model_input_size, 
                                   patch_size=14, n_patches_to_show=20, delay=0.5, interactive=False):
    """
    Shows patch similarity heatmaps one after another with a delay, creating an animation effect.
    
    Args:
        image_index (int): The index of the image in the dataset.
        loader: ChunkedEmbeddingLoader instance for loading embeddings.
        dataset_name (str): The name of the dataset.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
        patch_size (int): The size of each patch.
        n_patches_to_show (int): Number of patches to show in sequence.
        delay (float): Delay in seconds between frames.
        interactive (bool): If True, use interactive mode for smoother updates.
    """
    from IPython.display import clear_output, display
    import time
    
    if interactive:
        plt.ion()  # Turn on interactive mode
    
    # Retrieve the image
    image = retrieve_image(image_index, dataset_name, test_only=False)
    resized_image = pad_or_resize_img(image, model_input_size)
    
    # Calculate patch dimensions
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Calculate indices for all patches in the image
    image_start_idx = image_index * patches_per_image
    image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
    
    # Load all embeddings for this image
    embeddings = loader.load_specific_embeddings(image_patch_indices)
    
    # Sample patches evenly
    step = max(1, patches_per_image // n_patches_to_show)
    patches_to_show = list(range(0, patches_per_image, step))[:n_patches_to_show]
    
    # Create figure once if using interactive mode
    if interactive:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.show(block=False)
    
    # Create figure that we'll update
    for i, patch_idx in enumerate(patches_to_show):
        if not interactive:
            # Clear previous output and create new figure
            clear_output(wait=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        else:
            # Clear axes for updating
            ax1.clear()
            ax2.clear()
        
        # Calculate patch position
        patch_row, patch_col = divmod(patch_idx, patches_per_row)
        
        # Left: Original image with highlighted patch
        ax1.imshow(resized_image)
        left = patch_col * patch_size
        top = patch_row * patch_size
        rect = plt.Rectangle((left, top), patch_size, patch_size,
                           edgecolor='red', facecolor='none', linewidth=3)
        ax1.add_patch(rect)
        ax1.set_title(f'Image {image_index} - Patch {patch_idx}')
        ax1.axis('off')
        
        # Get the selected patch embedding and compute similarities
        selected_patch_embedding = embeddings[patch_idx]
        cos_sims = F.cosine_similarity(
            selected_patch_embedding.unsqueeze(0),
            embeddings
        ).cpu()
        
        # Reshape to grid
        cos_sim_grid = cos_sims.reshape(patches_per_col, patches_per_row)
        
        # Right: Heatmap overlay
        ax2.imshow(resized_image, alpha=0.5)
        heatmap = ax2.imshow(cos_sim_grid, cmap='hot', alpha=0.5, 
                           extent=(0, model_input_size[0], model_input_size[1], 0),
                           vmin=0, vmax=1)
        ax2.set_title(f'Similarity to Patch {patch_idx} (Row {patch_row}, Col {patch_col})')
        ax2.axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity')
        
        plt.suptitle(f'Frame {i+1}/{len(patches_to_show)}', fontsize=14)
        plt.tight_layout()
        
        if interactive:
            # Update the display
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            # Show the figure
            display(fig)
            plt.close(fig)
        
        # Wait before showing next frame (except for last frame)
        if i < len(patches_to_show) - 1:
            time.sleep(delay)
    
    if interactive:
        plt.ioff()  # Turn off interactive mode
        
    print(f"\nAnimation complete! Showed {len(patches_to_show)} patches.")


def plot_patches_sim_to_vector(cos_sim_grid, resized_image, patch_size, image_index, patch_index_in_image, save_path=None, plot_title=None, bar_title=None, show_plot=True):
    """
    Plot a heatmap of cosine similarities overlayed on the resized image.
    """
    image_width, image_height = resized_image.size
    patches_per_row = image_width // patch_size
    row, col = divmod(patch_index_in_image, patches_per_row)
    left = col * patch_size
    top = row * patch_size
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(resized_image)

    # Overlay the similarity heatmap
    heatmap = ax.imshow(cos_sim_grid, cmap='hot', alpha=0.5, extent=(0, image_width, image_height, 0))

    # Highlight the selected patch
    if patch_index_in_image >= 0:
        rect = plt.Rectangle(
            (left, top), patch_size, patch_size,
            edgecolor='red', facecolor='none', linewidth=2
        )
        ax.add_patch(rect)

    # Add colorbar and labels
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(bar_title)
    ax.set_title(plot_title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')

    if show_plot:
        plt.show()
    plt.clf()
    plt.close(fig)
        
    return torch.tensor(heatmap.get_array())


def binarize_patchsims_for_concept(concept_label, threshold, heatmaps, image_indices, images,
                                   dataset_name, metric_type, model_input_size, top_n=7, 
                                   save_file=None):
    """
    Plots binarized patch similarities for a single concept across multiple images,
    using precomputed heatmaps that are binarized based on the provided threshold.
    The original image is displayed above its corresponding binarized heatmap.

    Args:
        concept_label (str): The name of the concept to be visualized.
        image_indices (list of int): A list of image indices to visualize for the given concept.
        heatmaps (dict): A dictionary where keys are image indices and values are precomputed heatmaps (torch.Tensor).
        images (list of PIL.Image): A list of images.
        model_input_size (tuple): The dimensions (width, height) to which the image is resized for model input.
        dataset_name (str): Dataset name (default is 'CLEVR').
        top_n (int): Number of images to display.
        threshold (float): Threshold value for binarization.
        save_file (str): Where to save the resulting plot. If None, the plot is not saved.
    """
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))
    
    for i, image_index in enumerate(image_indices):
        # Retrieve image and heatmap
        image = retrieve_image(image_index, dataset_name)
        heatmap = heatmaps[image_index]
        
        # Resize image to the model's input size
        resized_image = pad_or_resize_img(image, model_input_size)
        image_width, image_height = resized_image.size
        
        # Convert heatmap to numpy
        heatmap_np = heatmap.detach().cpu().numpy()
        
        # Binarize the heatmap
        mask = heatmap_np >= threshold
        
        # Upsample mask to the image's dimensions
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_resized = mask_img.resize((image_width, image_height), resample=Image.NEAREST)
        mask_resized = np.array(mask_resized) > 127
        
        # Convert image to normalized RGB numpy array
        image_np = np.array(resized_image.convert("RGB")) / 255.0
        H, W, _ = image_np.shape
        
        # Create an RGBA composite image
        rgba_image = np.zeros((H, W, 4))
        rgba_image[mask_resized] = np.concatenate([
            image_np[mask_resized],
            np.full((mask_resized.sum(), 1), 1)
        ], axis=1)
        rgba_image[~mask_resized] = np.array([0, 0, 0, 1])
        
        # Plot original image on top
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f'Image {image_index}')
        axes[0, i].axis('off')
        
        # Plot binarized heatmap below
        axes[1, i].imshow(rgba_image)
        axes[1, i].set_title(f'Binarized Heatmap')
        axes[1, i].axis('off')

    plt.suptitle(f"Concept {concept_label} Binarized {metric_type} (Th={threshold:.5f})", fontsize=16, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 1])

    if save_file:
        save_path = f'../Figs/{dataset_name}/patch_alignment_to_concept/{concept_label}_binarized.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        print(f"Binarized figure saved to {save_path}")

    plt.show()
    
    
def plot_binarized_patchsims_all_concepts(concept_labels, percentile, heatmaps, image_indices, images,
                                          thresholds, metric_type, model_input_size, 
                                          dataset_name, top_n=7, 
                                          save_file=None, nonconcept_thresholds=False):
    """
    Plots binarized patch similarities for multiple concepts across multiple images.

    Args:
        concept_labels (list of str): List of concept names.
        heatmaps (dict): Dictionary where keys are concept names, values are heatmaps per image.
        image_indices (list of int): List of image indices to visualize.
        images (list of PIL.Image): List of images.
        threshold (float): Threshold for binarization.
        model_input_size (tuple): Image resize dimensions for the model.
        dataset_name (str): Name of the dataset.
        save_file (str): Path to save the plot.
    """

    num_concepts = len(concept_labels)
    fig, axes = plt.subplots(num_concepts + 1, top_n, figsize=(top_n * 3, (num_concepts + 1) * 3))

    if top_n == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plot original images in the top row
    for i, image_index in enumerate(image_indices):
        image = retrieve_image(image_index, dataset_name)
        resized_image = pad_or_resize_img(image, model_input_size)
        axes[0, i].imshow(resized_image)
        axes[0, i].set_title(f'Image {image_index}')
        axes[0, i].axis('off')

    # Add concept labels to the left of each row
    axes[0, 0].text(-0.1, 0.5, 'Original\nImage', fontsize=20,
                    verticalalignment='center', horizontalalignment='right', 
                    transform=axes[0, 0].transAxes)

    for j, concept_label in enumerate(concept_labels):
        axes[j + 1, 0].text(-0.1, 0.5, f'{concept_label}\n(thr = {thresholds[concept_label][0]:.2f})', fontsize=20,
                            verticalalignment='center', horizontalalignment='right', 
                            transform=axes[j + 1, 0].transAxes)

    # Loop through each concept and image to plot binarized heatmaps
    for i, image_index in enumerate(image_indices):
        for j, concept_label in enumerate(concept_labels):
            heatmap = heatmaps[concept_label][image_index].detach().cpu().numpy()
            if nonconcept_thresholds:
                mask = heatmap < thresholds[concept_label][0]
            else:
                mask = heatmap >= thresholds[concept_label][0]

            # Resize mask to match the image dimensions
            mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            mask_resized = np.array(mask_resized) > 127
            
            # Convert image to normalized RGB numpy array
            image = retrieve_image(image_index, dataset_name)
            resized_image = pad_or_resize_img(image, model_input_size)
            image_np = np.array(resized_image.convert("RGB")) / 255.0
            
            # Overlay mask with transparency
            rgba_image = np.zeros((*image_np.shape[:2], 4))
            rgba_image[mask_resized] = np.concatenate([image_np[mask_resized], np.full((mask_resized.sum(), 1), 1)], axis=1)
            rgba_image[~mask_resized] = [0, 0, 0, 1]

            # Plot the binarized heatmap
            axes[j + 1, i].imshow(rgba_image)
            axes[j + 1, i].axis('off')
            
    if nonconcept_thresholds:
        plt.suptitle(f"Binarized negative concept {metric_type} at {percentile*100}% percentile for {dataset_name}", fontsize=16, y=1)
    else:
        plt.suptitle(f"Binarized {metric_type} at {percentile*100}% percentile for {dataset_name}", fontsize=16, y=1)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_file:
        plt.savefig(save_file, dpi=300)

    plt.show()
    
    
def plot_binarized_patchsims_single_image_multiple_thresholds(
    concept_labels, heatmaps, image_index, images,
    thresholds_dict, metric_type, model_input_size, 
    dataset_name, save_file=None, nonconcept_thresholds=False):
    """
    Plots binarized patch similarities for multiple concepts at multiple thresholds for a single image.

    Args:
        concept_labels (list of str): List of concept names.
        heatmaps (dict): Dictionary where keys are concept names, values are heatmaps per image.
        image_index (int): Index of the image to visualize.
        images (list of PIL.Image): List of images.
        thresholds_dict (dict): Maps from concept to list of thresholds to visualize.
        model_input_size (tuple): Image resize dimensions for the model.
        metric_type (str): Type of similarity metric used.
        dataset_name (str): Name of the dataset.
        save_file (str, optional): Path to save the plot.
        nonconcept_thresholds (bool): Whether the thresholds indicate non-concept behavior.
    """
    percentiles = list(thresholds_dict.keys())
    num_thresholds = len(thresholds_dict.keys())

    num_concepts = len(concept_labels)
    fig, axes = plt.subplots(num_concepts + 1, num_thresholds, figsize=(num_thresholds * 3, (num_concepts + 1) * 3))
    if num_thresholds == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plot original image in the top row
    image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    for col in range(num_thresholds):
        axes[0, col].imshow(resized_image)
        axes[0, col].set_title(f"Image {image_index}")
        axes[0, col].axis('off')

    # Label for original image row
    axes[0, 0].text(-0.1, 0.5, 'Original\nImage', fontsize=20,
                    verticalalignment='center', horizontalalignment='right', 
                    transform=axes[0, 0].transAxes)

    for row, concept_label in enumerate(concept_labels):
        axes[row + 1, 0].text(-0.1, 0.5, f'{concept_label}', fontsize=20,
                              verticalalignment='center', horizontalalignment='right', 
                              transform=axes[row + 1, 0].transAxes)

        col = 0
        for percentile in percentiles:
            threshold = thresholds_dict[percentile][concept_label][0]
            heatmap = heatmaps[concept_label].detach().cpu().numpy()
            if nonconcept_thresholds:
                mask = heatmap < threshold
            else:
                mask = heatmap >= threshold

            # Resize mask to match the image dimensions
            mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            mask_resized = np.array(mask_resized) > 127

            # Convert image to normalized RGB numpy array
            image_np = np.array(resized_image.convert("RGB")) / 255.0

            # Overlay mask with transparency
            rgba_image = np.zeros((*image_np.shape[:2], 4))
            rgba_image[mask_resized] = np.concatenate([image_np[mask_resized], np.full((mask_resized.sum(), 1), 1)], axis=1)
            rgba_image[~mask_resized] = [0, 0, 0, 1]

            axes[row + 1, col].imshow(rgba_image)
            axes[row + 1, col].axis('off')
            axes[row + 1, col].set_title(f"{percentile*100:.0f}%")
            col += 1

    if nonconcept_thresholds:
        plt.suptitle(f"Binarized negative concept {metric_type} for {dataset_name}", fontsize=16, y=1)
    else:
        plt.suptitle(f"Binarized {metric_type} for {dataset_name}", fontsize=16, y=1)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()
    
    
def plot_binarized_patchsims_pos_neg_single_image(
    image_index, images,
    positive_heatmap, negative_heatmap,
    threshold_pos, threshold_neg,
    model_input_size, dataset_name,
    metric_type="similarity", save_file=None
):
    """
    Plots a single image with binarized patch similarities for positive and negative thresholds.
    Green overlay indicates positive concept activation, red overlay indicates negative concept activation.

    Args:
        image_index (int): Index of the image to visualize.
        images (list of PIL.Image): List of images.
        positive_heatmap (torch.Tensor): Heatmap tensor for positive concept.
        negative_heatmap (torch.Tensor): Heatmap tensor for negative concept.
        threshold_pos (float): Threshold for positive activation.
        threshold_neg (float): Threshold for negative activation.
        model_input_size (tuple): Resize dimensions for model input.
        dataset_name (str): Dataset name for labeling.
        metric_type (str): Similarity or distance type.
        save_file (str): Optional path to save output image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Load and resize image
    image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0

    # Binarize masks
    pos_mask = positive_heatmap >= threshold_pos
    neg_mask = negative_heatmap < threshold_neg

    # Resize masks to match image shape
    pos_mask = Image.fromarray((pos_mask.cpu().numpy() * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
    neg_mask = Image.fromarray((neg_mask.cpu().numpy() * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
    pos_mask = np.array(pos_mask) > 127
    neg_mask = np.array(neg_mask) > 127

    # Compute overlap and exclusive regions
    both_mask = pos_mask & neg_mask
    pos_only = pos_mask & ~both_mask
    neg_only = neg_mask & ~both_mask

    # Create overlay
    rgba_overlay = np.zeros((*image_np.shape[:2], 4))
    rgba_overlay[pos_only] = [0, 1, 0, 0.6]      # Green
    rgba_overlay[neg_only] = [1, 0, 0, 0.6]      # Red
    rgba_overlay[both_mask] = [0.5, 0, 0.5, 0.6] # Purple

    # Composite overlay with image
    overlay_img = image_np.copy()
    out = overlay_img.copy()
    for c in range(3):
        out[..., c] = rgba_overlay[..., 3] * rgba_overlay[..., c] + (1 - rgba_overlay[..., 3]) * overlay_img[..., c]

    # Plot
    ax.imshow(out)
    ax.axis('off')
    ax.set_title(f"Image {image_index}\nGreen: ≥ {threshold_pos:.3f}, Red: < {threshold_neg:.3f}, Purple: Overlap")

    plt.suptitle(f"Binarized Patch Activations - {dataset_name} [{metric_type}]", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()
    
    
    
def plot_binarized_patchsims_pos_neg_grid(
    image_index, images,
    heatmaps_dict, positive_thresholds, negative_thresholds,
    model_input_size, dataset_name,
    metric_type="similarity", save_file=None
):
    """
    Plots a grid of binarized patch activations where columns are positive concepts and rows are negative concepts.

    Args:
        image_index (int): Index of the image to visualize.
        images (list of PIL.Image): List of PIL images.
        heatmaps_dict (dict): Nested dict with heatmaps_dict[concept][image_index] = heatmap (Tensor).
        thresholds_pos (dict): Dict mapping positive concept to threshold.
        thresholds_neg (dict): Dict mapping negative concept to threshold.
        pos_concepts (list of str): Positive concept names (columns).
        neg_concepts (list of str): Negative concept names (rows).
        model_input_size (tuple): Resize dimensions.
        dataset_name (str): Dataset name for title.
        metric_type (str): Similarity or distance.
        save_file (str): Optional path to save.
    """
    concepts = heatmaps_dict.keys()
    n_rows = len(concepts)
    n_cols = len(concepts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0

    for row_idx, neg_concept in enumerate(concepts):
        for col_idx, pos_concept in enumerate(concepts):
            ax = axes[row_idx, col_idx]

            pos_map = heatmaps_dict[pos_concept]
            neg_map = heatmaps_dict[neg_concept]
            pos_thresh = positive_thresholds[pos_concept][0]
            neg_thresh = negative_thresholds[neg_concept][0]

            pos_mask = pos_map >= pos_thresh
            neg_mask = neg_map < neg_thresh

            pos_mask = Image.fromarray((pos_mask.cpu().numpy() * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            neg_mask = Image.fromarray((neg_mask.cpu().numpy() * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            pos_mask = np.array(pos_mask) > 127
            neg_mask = np.array(neg_mask) > 127

            both_mask = pos_mask & neg_mask
            pos_only = pos_mask & ~both_mask
            neg_only = neg_mask & ~both_mask

            rgba_overlay = np.zeros((*image_np.shape[:2], 4))
            rgba_overlay[pos_only] = [0, 1, 0, 0.6]      # Green
            rgba_overlay[neg_only] = [1, 0, 0, 0.6]      # Red
            rgba_overlay[both_mask] = [0.5, 0, 0.5, 0.6] # Purple

            out = image_np.copy()
            for c in range(3):
                out[..., c] = rgba_overlay[..., 3] * rgba_overlay[..., c] + (1 - rgba_overlay[..., 3]) * image_np[..., c]

            ax.imshow(out)
            ax.axis("off")

            if row_idx == 0:
                ax.set_title(f"{pos_concept}\nthr={pos_thresh:.2f}", fontsize=10)

        # Add label inside leftmost axis (first column of each row)
        axes[row_idx, 0].text(
            -0.05, 0.5,
            f"{neg_concept}\nthr={negative_thresholds[neg_concept][0]:.2f}",
            transform=axes[row_idx, 0].transAxes,
            va='center', ha='right',
            fontsize=10
        )

    plt.tight_layout(rect=[0.08, 0.02, 1, 0.92])  # leave space for row labels, less top padding
    plt.suptitle(
        f"Image {image_index} - Positive (Cols) vs Negative (Rows) [{metric_type}]",
        fontsize=14,
        x=0.6, y=0.93  # x centers, y controls vertical placement
    )

    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()
    


def plot_binarized_patchsims_two_concepts(
    percentile,
    images,
    heatmaps1,
    heatmaps2,
    threshold1,
    threshold2,
    model_input_size,
    dataset_name,
    concept1_label="Concept 1",
    concept2_label="Concept 2",
    metric_type="similarity",
    save_file=None
):
    """
    Plots a horizontal strip of images with binarized patch activations from two concept heatmaps.

    Args:
        image_indices (list): List of image indices to visualize.
        images (list of PIL.Image): List of all dataset images.
        heatmaps1 (dict): Mapping from image index to heatmap for Concept 1.
        heatmaps2 (dict): Mapping from image index to heatmap for Concept 2.
        threshold1 (float): Activation threshold for Concept 1.
        threshold2 (float): Activation threshold for Concept 2.
        model_input_size (tuple): Resize dimensions for model input.
        dataset_name (str): Dataset name for title.
        concept1_label (str): Label for Concept 1.
        concept2_label (str): Label for Concept 2.
        metric_type (str): Similarity or distance type.
        save_file (str): Optional path to save output image.
    """
    image_indices = list(heatmaps1.keys())
    n = len(image_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, img_idx in zip(axes, image_indices):
        image = images[img_idx]
        resized_image = pad_or_resize_img(image, model_input_size)
        image_np = np.array(resized_image.convert("RGB")) / 255.0

        # Get and binarize masks
        h1 = heatmaps1[img_idx]
        h2 = heatmaps2[img_idx]
        m1 = (h1 >= threshold1).cpu().numpy()
        m2 = (h2 >= threshold2).cpu().numpy()

        # Resize to image resolution
        m1 = Image.fromarray((m1 * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
        m2 = Image.fromarray((m2 * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
        m1 = np.array(m1) > 127
        m2 = np.array(m2) > 127

        # Color overlays
        both_mask = m1 & m2
        only1 = m1 & ~both_mask
        only2 = m2 & ~both_mask

        rgba = np.zeros((*image_np.shape[:2], 4))
        rgba[only1] = [0, 1, 0, 0.6]       # Green
        rgba[only2] = [0, 0, 1, 0.6]       # Blue
        rgba[both_mask] = [0.5, 0, 0.5, 0.6]  # Purple

        composite = image_np.copy()
        for c in range(3):
            composite[..., c] = rgba[..., 3] * rgba[..., c] + (1 - rgba[..., 3]) * composite[..., c]

        ax.imshow(composite)
        ax.set_title(f"Image {img_idx}", fontsize=12)
        ax.axis("off")

    # Legend below
    green_patch = mpatches.Patch(color='green', label=concept1_label)
    blue_patch = mpatches.Patch(color='blue', label=concept2_label)
    purple_patch = mpatches.Patch(color='purple', label='Overlap')
    fig.legend(handles=[green_patch, blue_patch, purple_patch], loc='lower center', ncol=3, fontsize=12)

    plt.suptitle(f"Binarized Patch Activations at {round(percentile*100)}% percentile", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    if save_file:
        plt.savefig(save_file, dpi=300)
    plt.show()
    
    
    
def plot_concept_evolution_over_iterations(
    image_index, heatmaps_over_time, all_images,
    dataset_name, model_input_size, concept_labels,
    fine_tuning_params, top_n=1, metric_type='Distance to Decision Boundary',
    vmin=None, vmax=None, save_file=None, epochs_to_plot=None):
    """
    Plot the evolution of heatmaps for a single image and multiple concepts over iterations.
    Rows = concepts, Columns = iterations (initial + fine-tuned steps)
    """
    total_epochs = sum(epoch_count for _, epoch_count in fine_tuning_params)
    num_concepts = len(concept_labels)

    # Build list of per-epoch patch percentages (e.g., ['init', 'init', 'init', 0.2, 0.2])
    epoch_patch_labels = []
    for patch_percent, num_epochs in fine_tuning_params:
        epoch_patch_labels.extend([patch_percent] * num_epochs)
        
    # If no specific epoch list provided, use all
    if epochs_to_plot is None:
        epochs_to_plot = list(range(1, total_epochs + 1))
    epochs_to_plot = [ep for ep in epochs_to_plot if 1 <= ep <= total_epochs]
    num_epochs_to_plot = len(epochs_to_plot)
    
    # Compute vmin/vmax if needed
    if vmin is None or vmax is None:
        vals = np.concatenate([
            heatmap.flatten() for heatmap_list in heatmaps_over_time.values() for heatmap in heatmap_list
        ])
        vmin = np.min(vals)
        vmax = np.max(vals)

    # Create figure
    fig, axes = plt.subplots(num_concepts + 1, num_epochs_to_plot, figsize=(num_epochs_to_plot * 2.0, (num_concepts + 1) * 2))

    # Get and prepare image
    image = pad_or_resize_img(all_images[image_index], model_input_size)
    image_gray = image.convert("L")

    # Top row: original image + titles for selected epochs
    for j, epoch_num in enumerate(epochs_to_plot):
        ax = axes[0, j]
        ax.imshow(image)
        ax.axis('off')

        patch_label = epoch_patch_labels[epoch_num - 1]
        label = f"{patch_label}%" if patch_label != 'init' else "All Patches"
        ax.set_title(f"{label}\n(Epoch {epoch_num})", fontsize=9)

    axes[0, 0].text(-0.05, 0.5, f"Img {image_index}", fontsize=12,
                    va='center', ha='right', transform=axes[0, 0].transAxes)

    # Plot heatmaps for each concept
    for i, concept_label in enumerate(concept_labels):
        for j, epoch_num in enumerate(epochs_to_plot):
            ax = axes[i + 1, j]
            heatmap = heatmaps_over_time[concept_label][j]  # 0-based indexing

            ax.imshow(heatmap, cmap='hot', alpha=0.6, vmin=vmin, vmax=vmax,
                      extent=[0, model_input_size[0], model_input_size[1], 0])
            ax.imshow(image_gray, cmap='gray', alpha=0.4)
            ax.set_title(f"Max={round(heatmap.max().item(), 4)}\nMin={round(heatmap.min().item(), 4)}", fontsize=8)
            ax.axis('off')

        # Concept label
        axes[i + 1, 0].text(-0.05, 0.5, concept_label, fontsize=12,
                            va='center', ha='right', transform=axes[i + 1, 0].transAxes)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='hot')
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax).set_label(metric_type)

    plt.suptitle("Concept Activation Evolution Over Epochs", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_file:
        plt.savefig(save_file, dpi=500, bbox_inches='tight')

    plt.show()
    
    
def top_left_crop_to_original_aspect(arr, original_size, padded_size):
    """
    Crops the top-left region of a padded image/array to match the original image's aspect ratio.

    Args:
        arr (np.ndarray): Input image or heatmap array (H, W, ...) to crop.
        original_size (tuple): (W_orig, H_orig) of original image.
        padded_size (tuple): (W_pad, H_pad) of the padded image.

    Returns:
        np.ndarray: Cropped array from top-left corner to original aspect ratio.
    """
    W_orig, H_orig = original_size
    W_pad, H_pad = padded_size

    target_height = int(W_pad * H_orig / W_orig)
    cropped = arr[:target_height, :W_pad]

    return cropped


def plot_binarized_patchsims_with_raw_heatmaps(
    concept_labels, heatmaps, image_index, images,
    thresholds_dict, metric_type, model_input_size, 
    dataset_name, save_file=None, 
    nonconcept_thresholds=False, vmin=None, vmax=None):

    percentiles = list(thresholds_dict.keys())
    num_thresholds = len(percentiles)
    num_concepts = len(concept_labels)

    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(num_concepts, num_thresholds + 2, figsize=(5.5, num_concepts * 0.75), constrained_layout=True)
    if num_concepts == 1:
        axes = np.expand_dims(axes, axis=0)

    image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0
    image_cropped = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)

    if vmin is None or vmax is None:
        all_values = np.concatenate([
            heatmaps[concept].detach().cpu().numpy().flatten()
            for concept in concept_labels
        ])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    colorbar_im = None

    for row, concept_label in enumerate(concept_labels):
        heatmap = heatmaps[concept_label].detach().cpu().numpy()
        heatmap_resized = Image.fromarray(heatmap).resize(resized_image.size, resample=Image.NEAREST)
        heatmap_resized = np.array(heatmap_resized)
        heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)

        # Column 0: Original image only for first row
        if row == 0:
            axes[0, 0].imshow(image_cropped)
            axes[0, 0].axis('off')
            axes[0, 0].set_title("Original")
        else:
            axes[row, 0].axis('off')

        # Column 1: Raw heatmap
        im = axes[row, 1].imshow(heatmap_cropped, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title("Raw Heatmap")
        if colorbar_im is None:
            colorbar_im = im

        # Columns 2+: Threshold overlays
        for col_idx, percentile in enumerate(percentiles):
            threshold = thresholds_dict[percentile][concept_label][0]
            mask = heatmap >= threshold if not nonconcept_thresholds else heatmap < threshold
            mask[np.isnan(heatmap)] = False

            mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(model_input_size, resample=Image.NEAREST)
            mask_resized = np.array(mask_resized) > 127

            rgba_image = np.zeros((*image_np.shape[:2], 4))
            rgba_image[mask_resized] = np.concatenate([image_np[mask_resized], np.ones((mask_resized.sum(), 1))], axis=1)
            rgba_image[~mask_resized] = [0, 0, 0, 1]

            rgba_image_cropped = top_left_crop_to_original_aspect(rgba_image, image.size, resized_image.size)

            axes[row, col_idx + 2].imshow(rgba_image_cropped)
            axes[row, col_idx + 2].axis('off')
            if row == 0:
                axes[row, col_idx + 2].set_title(f"{percentile * 100:.0f}%")

        # Concept label on the right
        axes[row, -1].text(1.1, 0.5, concept_label.capitalize(), va='center', ha='left',
                           fontstyle='italic', transform=axes[row, -1].transAxes)

    # Layout and vertical colorbar to the left under the original image
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.95, bottom=0.05,
        wspace=0.01, hspace=0.1
    )
    cbar_ax = fig.add_axes([axes[0, 0].get_position().x0+0.01, axes[0, 0].get_position().y0-0.6, 0.015, 0.5])
    fig.colorbar(colorbar_im, cax=cbar_ax, orientation='vertical', label='Cos Sim to Concept')
    vmin, vmax = colorbar_im.get_clim()
    tick_values = np.linspace(vmin, vmax, 3)
    cbar_ax.set_yticks(tick_values)
    cbar_ax.set_yticklabels([f"{t:.1f}" for t in tick_values])
    cbar_ax.yaxis.set_ticks_position('left')
    cbar_ax.yaxis.set_label_position('left')
    plt.rcParams.update({'font.size': 8})

    if save_file:
        plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight')
    plt.show()


    
    
# def plot_superpatches_on_heatmaps(
#     concept_labels, heatmaps, image_index, images,
#     thresholds, metric_type, model_input_size,
#     dataset_name, save_file=None,
#     vmin=None, vmax=None):


#     num_concepts = len(concept_labels)
#     concepts_per_row = 3
#     num_rows = int(np.ceil(num_concepts / concepts_per_row))

#     fig, axes = plt.subplots(num_rows, concepts_per_row + 1, figsize=((concepts_per_row + 1) * 4, num_rows * 4))
#     axes = np.array(axes)

#     image = images[image_index]
#     resized_image = pad_or_resize_img(image, model_input_size)
#     image_np = np.array(resized_image.convert("RGB")) / 255.0
#     image_cropped = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)

#     if vmin is None or vmax is None:
#         all_values = np.concatenate([
#             heatmaps[concept].detach().cpu().numpy().flatten()
#             for concept in concept_labels
#         ])
#         vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

#     colorbar_im = None

#     # Turn off all axes initially
#     for i in range(num_rows):
#         for j in range(concepts_per_row + 1):
#             axes[i, j].axis('off')

#     # Manually position original image between heatmap rows
#     orig_ax = fig.add_axes([0.01, 0.65 - (1 / (num_rows * 2)), 0.25, 0.3])
#     orig_ax.imshow(image_cropped)
#     orig_ax.axis('off')
#     orig_ax.set_title("Original")

#     for idx, concept_label in enumerate(concept_labels):
#         row = idx // concepts_per_row
#         col = (idx % concepts_per_row) + 1

#         heatmap_orig = heatmaps[concept_label].detach().cpu().numpy()
#         heatmap_resized = Image.fromarray(heatmap_orig).resize(resized_image.size, resample=Image.NEAREST)
#         heatmap_resized = np.array(heatmap_resized)
#         heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)

#         axes[row, col].imshow(image_cropped, alpha=1.0)
#         im = axes[row, col].imshow(heatmap_cropped, cmap='magma', interpolation='nearest', vmin=vmin, vmax=vmax, alpha=0.8)
#         axes[row, col].axis('off')
#         axes[row, col].set_title(concept_label)

#         if colorbar_im is None:
#             colorbar_im = im

#         grid_size = heatmap_orig.shape[0]
#         patch_h = resized_image.size[1] / grid_size
#         patch_w = resized_image.size[0] / grid_size
#         threshold = thresholds[concept_label][0]

#         for i_patch in range(grid_size):
#             for j_patch in range(grid_size):
#                 if heatmap_orig[i_patch, j_patch] >= threshold:
#                     x = j_patch * patch_w
#                     y = i_patch * patch_h
#                     rect = mpatches.Rectangle((x, y), patch_w, patch_h,
#                                               linewidth=2, edgecolor='deepskyblue', facecolor='none')
#                     axes[row, col].add_patch(rect)

#     # Horizontal colorbar centered under the plots
#     fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.2, wspace=0.1, hspace=0.2)
#     cbar_width = 0.6
#     cbar_center = 0.5 - cbar_width / 2
#     cbar_ax = fig.add_axes([cbar_center, 0.1, cbar_width, 0.02])
#     fig.colorbar(colorbar_im, cax=cbar_ax, orientation='horizontal', label='Cosine Similarity to Concept')

#     if save_file:
#         plt.savefig(save_file, dpi=300)
#     plt.show()
def filter_and_plot_concept_images(
    metadata_path,
    required_concepts,
    chosen_split='train',
    start_idx=0,
    n_images=10,
    plot=True
):
    """
    Filters metadata for rows with specified binary concepts and plots first n image thumbnails.

    Args:
        metadata_path (str): Path to metadata CSV.
        required_concepts (list): Concept column names required to be 1 (e.g., ['has_color_red']).
        chosen_split (str): Split to filter on ('train', 'test', etc.).
        n_images (int): Number of matching images to display.
        plot (bool): Whether to plot the images.

    Returns:
        list: Indices of the filtered rows.
    """
    metadata = pd.read_csv(metadata_path)

    # Validate concept columns
    for concept in required_concepts:
        if concept not in metadata.columns:
            raise ValueError(f"Missing concept column: {concept}")

    # Apply filtering
    mask = metadata['split'] == chosen_split
    for concept in required_concepts:
        mask &= metadata[concept] == 1

    filtered_df = metadata[mask][start_idx:start_idx+n_images]

    if plot:
        fig, axes = plt.subplots(1, len(filtered_df), figsize=(3 * len(filtered_df), 3))
        if len(filtered_df) == 1:
            axes = [axes]  # make iterable

        for ax, (_, row) in zip(axes, filtered_df.iterrows()):
            try:
                img = Image.open(f'../Data/Coco/{row["image_path"]}')
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(row['image_path'].split('/')[-1])
            except Exception as e:
                ax.text(0.5, 0.5, "Image load failed", ha='center')
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    return filtered_df.index.tolist()


def plot_aligned_images_chunked(concept_key, dataset_name, acts_loader, 
                               con_label='', k=5, metric_type='Cosine Similarity', 
                               save_image=False, test_only=True):
    """
    Chunked version: Plot images that align well with a selected concept using pre-initialized loader.
    
    Args:
        concept_key (str): The concept to visualize (e.g., 'color::red').
        dataset_name (str): Name of the dataset.
        acts_loader: Pre-initialized ChunkedActivationLoader instance.
        con_label (str): Label to put in path of saved image.
        k (int): Number of top images to display.
        metric_type (str): Type of metric being visualized.
        save_image (bool): Whether to save the plot.
        test_only (bool): Whether to only consider test samples.
    """
    
    # Load the full dataframe (this handles chunking internally)
    comp_df = acts_loader.load_full_dataframe()
    
    # Filter for test samples if requested
    if test_only:
        metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
        test_indices = metadata[metadata['split'] == 'test'].index
        comp_df = comp_df.loc[comp_df.index.intersection(test_indices)]
    
    # Check if concept exists
    if concept_key not in comp_df.columns:
        print(f"Concept '{concept_key}' not found. Available concepts:")
        print(sorted(comp_df.columns)[:10], "...")
        return
    
    # Get top k samples
    top_k_indices = comp_df.nlargest(k, concept_key).index.tolist()
    
    # Calculate grid layout
    n_cols = min(k, 5)
    n_rows = (k + 4) // 5
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    plt.suptitle(f"Top {k} Images with Highest {metric_type} to: {concept_key}", fontsize=16)
    
    for rank, idx in enumerate(top_k_indices):
        if rank >= len(axes):
            break
        
        # Retrieve image
        img = retrieve_image(idx, dataset_name, test_only=False)
        value = comp_df.loc[idx, concept_key]
        
        axes[rank].imshow(img)
        axes[rank].set_title(f"Rank {rank+1}: Image {idx}\n{metric_type} = {value:.4f}")
        axes[rank].axis('off')
    
    # Hide unused axes
    for rank in range(len(top_k_indices), len(axes)):
        axes[rank].axis('off')
    
    plt.tight_layout()
    
    if save_image:
        save_path = f'../Figs/{dataset_name}/most_aligned_w_concepts/concept_{concept_key}_{k}__{con_label}.jpg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
    
    plt.show()


def plot_most_similar_patches_w_heatmaps_chunked(concept_label, dataset_name, acts_loader, 
                                                embeds_loader, con_label='', 
                                                model_input_size=(224, 224), top_n=5, 
                                                metric_type='Cosine Similarity', test_only=True,
                                                save_path=None):
    """
    Chunked version: Plots the most similar patches with heatmaps using pre-initialized loaders.
    
    Args:
        concept_label (str): The concept to visualize.
        dataset_name (str): Name of the dataset.
        acts_loader: Pre-initialized ChunkedActivationLoader instance.
        embeds_loader: Pre-initialized ChunkedEmbeddingLoader instance.
        con_label (str): Label for saving.
        model_input_size (tuple): Model input size.
        top_n (int): Number of top patches to show.
        metric_type (str): Type of metric.
        test_only (bool): Whether to only use test samples.
        save_path (str): Where to save the figure.
    """
    # Load all images
    all_images, _, _ = load_images(dataset_name)
    
    # Load activations
    cos_sims = acts_loader.load_full_dataframe()
    
    # Filter for test samples if needed
    if test_only:
        split_df = get_patch_split_df(dataset_name, patch_size=14, model_input_size=model_input_size)
        test_indices = split_df[split_df == 'test'].index
        cos_sims = cos_sims.loc[cos_sims.index.intersection(test_indices)]
    
    # Get top patches
    if concept_label not in cos_sims.columns:
        print(f"Concept '{concept_label}' not found.")
        return
    
    concept_cos_sims = cos_sims[concept_label]
    most_similar_patches = concept_cos_sims.nlargest(top_n).index.tolist()
    
    # Create figure
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))
    if top_n == 1:
        axes = axes.reshape(2, 1)
    
    # Process each top patch
    for i, patch_idx in enumerate(most_similar_patches):
        # Get image index and patch info
        patches_per_image = (model_input_size[0] // 14) * (model_input_size[1] // 14)
        image_idx = patch_idx // patches_per_image
        patch_in_image = patch_idx % patches_per_image
        
        image = all_images[image_idx]
        resized_image = pad_or_resize_img(image, model_input_size)
        
        # Plot original image
        axes[0, i].imshow(resized_image)
        axes[0, i].set_title(f'Image {image_idx}')
        axes[0, i].axis('off')
        
        # Calculate patch location and highlight
        patches_per_row = model_input_size[0] // 14
        patch_row, patch_col = divmod(patch_in_image, patches_per_row)
        left = patch_col * 14
        top = patch_row * 14
        
        # Create highlighted version
        image_with_patch = resized_image.copy()
        draw = ImageDraw.Draw(image_with_patch)
        draw.rectangle([left, top, left + 14, top + 14], outline="red", width=3)
        
        # Load embeddings for this image's patches
        image_start_idx = image_idx * patches_per_image
        image_patch_indices = list(range(image_start_idx, image_start_idx + patches_per_image))
        image_embeddings = embeds_loader.load_specific_embeddings(image_patch_indices)
        
        # Get concept vector (you might need to load this separately)
        # For now, using the embedding of the top patch as a proxy
        concept_vector = image_embeddings[patch_in_image]
        
        # Compute similarities for heatmap
        similarities = F.cosine_similarity(
            concept_vector.unsqueeze(0),
            image_embeddings
        ).cpu().reshape(model_input_size[1] // 14, model_input_size[0] // 14)
        
        # Plot heatmap
        axes[1, i].imshow(image_with_patch.convert('L'), cmap='gray', alpha=0.4)
        heatmap = axes[1, i].imshow(similarities, cmap='hot', alpha=0.6,
                                   extent=[0, model_input_size[0], model_input_size[1], 0])
        axes[1, i].set_title(f'Similarity: {concept_cos_sims.iloc[patch_idx]:.3f}')
        axes[1, i].axis('off')
    
    plt.suptitle(f"Top {top_n} Patches for Concept: {concept_label}", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    
    plt.show()


def plot_patchsims_all_concepts_from_loader(img_idx, concept_labels, acts_loader, dataset_name,
                                           model_input_size=(224, 224), patch_size=14,
                                           metric_type='Cosine Similarity', vmin=None, vmax=None, 
                                           sort_by_act=False, save_file=None):
    """
    Wrapper: Uses activation loader to create heatmaps and plot all concepts for one image.
    
    Args:
        img_idx (int): Index of the image.
        concept_labels (list): List of concept names to visualize.
        acts_loader: ChunkedActivationLoader instance.
        dataset_name (str): Dataset name.
        model_input_size (tuple): Model input size.
        patch_size (int): Size of patches.
        Other args same as plot_patchsims_all_concepts.
    """
    # Load activations
    cos_sims = acts_loader.load_full_dataframe()
    
    # Calculate patches per image
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Get patch indices for this image
    start_idx = img_idx * patches_per_image
    end_idx = start_idx + patches_per_image
    
    # Extract activations for this image and reshape into heatmaps
    heatmaps = {}
    for concept in concept_labels:
        if concept in cos_sims.columns:
            # Get activations for all patches in this image
            image_acts = cos_sims[concept].iloc[start_idx:end_idx]
            # Reshape into 2D heatmap
            heatmap = torch.tensor(image_acts.values).reshape(patches_per_col, patches_per_row)
            heatmaps[concept] = heatmap
        else:
            print(f"Warning: Concept '{concept}' not found in activations")
    
    # Call the existing function
    plot_patchsims_all_concepts(img_idx, heatmaps, model_input_size, dataset_name,
                               metric_type, vmin, vmax, sort_by_act, save_file)


def plot_patchsims_heatmaps_all_concepts_from_loader(concept_labels, image_indices, acts_loader, 
                                                    dataset_name, model_input_size=(224, 224), 
                                                    patch_size=14, top_n=7, save_file=None, 
                                                    metric_type='Cosine Similarity', vmin=None, vmax=None):
    """
    Wrapper: Uses activation loader to create heatmaps for multiple concepts across multiple images.
    
    Args:
        concept_labels (list): List of concept names.
        image_indices (list): List of image indices to visualize.
        acts_loader: ChunkedActivationLoader instance.
        dataset_name (str): Dataset name.
        model_input_size (tuple): Model input size.
        patch_size (int): Size of patches.
        Other args same as plot_patchsims_heatmaps_all_concepts.
    """
    # Load all images
    all_images, _, _ = load_images(dataset_name)
    
    # Load activations
    cos_sims = acts_loader.load_full_dataframe()
    
    # Calculate patches per image
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col
    
    # Create heatmaps dictionary structure
    heatmaps = {}
    for concept in concept_labels:
        if concept not in cos_sims.columns:
            print(f"Warning: Concept '{concept}' not found")
            continue
        heatmaps[concept] = {}
        
        for img_idx in image_indices:
            # Get patch indices for this image
            start_idx = img_idx * patches_per_image
            end_idx = start_idx + patches_per_image
            
            # Extract and reshape activations
            image_acts = cos_sims[concept].iloc[start_idx:end_idx]
            heatmap = torch.tensor(image_acts.values).reshape(patches_per_col, patches_per_row)
            heatmaps[concept][img_idx] = heatmap
    
    # Use only the first top_n images
    image_indices = image_indices[:top_n]
    
    # Call the existing function
    plot_patchsims_heatmaps_all_concepts(concept_labels, heatmaps, image_indices, all_images,
                                       model_input_size, dataset_name, top_n, 
                                       save_file, metric_type, vmin, vmax)


def plot_superpatches_on_heatmaps(
    concept_labels, heatmaps, image_index, images,
    thresholds, metric_type, model_input_size,
    dataset_name, save_file=None,
    vmin=None, vmax=None):


    num_concepts = len(concept_labels)
    concepts_per_row = 3
    num_rows = int(np.ceil(num_concepts / concepts_per_row))

    fig, axes = plt.subplots(num_rows, concepts_per_row + 1, figsize=(5.5, 2.9))
    axes = np.array(axes)

    image = images[image_index]
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0
    image_cropped = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)

    if vmin is None or vmax is None:
        all_values = np.concatenate([
            heatmaps[concept].detach().cpu().numpy().flatten()
            for concept in concept_labels
        ])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    colorbar_im = None

    # Turn off all axes initially
    for i in range(num_rows):
        for j in range(concepts_per_row + 1):
            axes[i, j].axis('off')

    # Manually position original image between heatmap rows
    orig_ax = fig.add_axes([0.01, 0.64 - (1 / (num_rows * 2)), 0.25, 0.32])
    orig_ax.imshow(image_cropped)
    orig_ax.axis('off')
    orig_ax.set_title("Original")

    for idx, concept_label in enumerate(concept_labels):
        row = idx // concepts_per_row
        col = (idx % concepts_per_row) + 1

        heatmap_orig = heatmaps[concept_label].detach().cpu().numpy()
        heatmap_resized = Image.fromarray(heatmap_orig).resize(resized_image.size, resample=Image.NEAREST)
        heatmap_resized = np.array(heatmap_resized)
        heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)

        axes[row, col].imshow(image_cropped, alpha=1.0)
        im = axes[row, col].imshow(heatmap_cropped, cmap='magma', interpolation='nearest', vmin=vmin, vmax=vmax, alpha=0.8)
        axes[row, col].axis('off')
        axes[row, col].set_title(concept_label.capitalize(), fontstyle='italic')

        if colorbar_im is None:
            colorbar_im = im

        grid_size = heatmap_orig.shape[0]
        patch_h = resized_image.size[1] / grid_size
        patch_w = resized_image.size[0] / grid_size
        threshold = thresholds[concept_label][0]

        for i_patch in range(grid_size):
            for j_patch in range(grid_size):
                if heatmap_orig[i_patch, j_patch] >= threshold:
                    x = j_patch * patch_w
                    y = i_patch * patch_h
                    rect = mpatches.Rectangle((x, y), patch_w, patch_h,
                                              linewidth=1, edgecolor='deepskyblue', facecolor='none')
                    axes[row, col].add_patch(rect)

    # Horizontal colorbar centered under the plots
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.2, wspace=0.1, hspace=0.2)
    cbar_width = 0.6
    cbar_center = 0.5 - cbar_width / 2
    cbar_ax = fig.add_axes([cbar_center, 0.16, cbar_width, 0.02])
    fig.colorbar(colorbar_im, cax=cbar_ax, orientation='horizontal', label='Cosine Similarity to Concept')

    if save_file:
        plt.savefig(save_file, dpi=500, format='pdf', bbox_inches='tight')
    plt.show()


def plot_best_detecting_clusters_calibrated(dataset_name: str,
                                           model_name: str,
                                           sample_type: str,
                                           n_clusters: int,
                                           acts_loader,
                                           concepts_to_show=None,
                                           top_n_clusters: int = 3,
                                           top_n_samples: int = 5,
                                           metric: str = 'f1',
                                           model_input_size=None,
                                           test_only: bool = True,
                                           percent_thru_model: int = 100,
                                           save_dir=None):
    """
    Plot the best detecting clusters for concepts, using the percentile that performed best on calibration set.
    
    Args:
        dataset_name: Name of dataset
        model_name: Name of model (e.g., 'CLIP', 'Llama')
        sample_type: 'patch' or 'cls'
        n_clusters: Number of clusters in k-means
        acts_loader: Activation loader
        concepts_to_show: List of specific concepts to show (None for all)
        top_n_clusters: Number of best clusters to show per concept
        top_n_samples: Number of samples to show per cluster
        metric: Metric to use for ranking clusters ('f1', 'precision', 'recall')
        model_input_size: Model input size (required for patch)
        test_only: Whether to use only test samples
        percent_thru_model: Percentage through model
        save_dir: Directory to save figures
    """
    from calibration_selection_utils import find_best_percentile_from_calibration_allpairs
    import ast
    
    # Construct concept label
    con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    
    # Find best percentile using calibration results
    print(f"Finding best percentile using calibration set performance...")
    try:
        best_percentile, cal_matches = find_best_percentile_from_calibration_allpairs(
            dataset_name, 
            con_label,
            concepts_to_include=concepts_to_show,
            metric=metric
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure you've run all_detection_stats.py to generate calibration results.")
        return
    
    print(f"Selected percentile {best_percentile} based on calibration {metric}")
    
    # Load test results at the best percentile
    test_path = f"Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile}_{con_label}.csv"
    
    if not os.path.exists(test_path):
        print(f"Test results not found at best percentile: {test_path}")
        return
    
    # Load and parse test results
    test_df = pd.read_csv(test_path)
    
    # Parse concept column and find best clusters for each concept on test set
    test_matches = {}
    
    for concept in cal_matches.keys():
        concept_rows = []
        for _, row in test_df.iterrows():
            concept_tuple = ast.literal_eval(row['concept'])
            if concept_tuple[0] == concept:
                concept_rows.append({
                    'cluster_id': concept_tuple[1],
                    metric: row[metric]
                })
        
        if concept_rows:
            concept_df = pd.DataFrame(concept_rows)
            top_clusters = concept_df.nlargest(top_n_clusters, metric)
            test_matches[concept] = [
                (row['cluster_id'], row[metric]) 
                for _, row in top_clusters.iterrows()
            ]
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot results
    print(f"\nShowing best detecting clusters at percentile {best_percentile}:")
    print("="*60)
    
    for concept_name, cluster_list in test_matches.items():
        print(f"\nConcept: {concept_name}")
        print(f"Calibration best: cluster {cal_matches[concept_name][0]} ({metric}={cal_matches[concept_name][1]:.3f})")
        print(f"Test set top {top_n_clusters} clusters:")
        
        for cluster_id, metric_value in cluster_list:
            print(f"  - Cluster {cluster_id}: {metric}={metric_value:.3f}")
        
        # Plot based on sample type
        if sample_type == 'cls':
            # For CLS, create subplot for all clusters
            n_cols = min(top_n_clusters, len(cluster_list))
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
            if n_cols == 1:
                axes = [axes]
            
            for idx, (cluster_id, metric_value) in enumerate(cluster_list[:n_cols]):
                plt.sca(axes[idx])
                
                # Use existing visualization function
                plot_aligned_images(
                    acts_loader=acts_loader,
                    con_label=f"cluster_{cluster_id}",
                    concept_key=str(cluster_id),
                    k=top_n_samples,
                    dataset_name=dataset_name,
                    metric_type=f'Cluster {cluster_id} Activation',
                    save_image=False,
                    test_only=test_only
                )
                
                axes[idx].set_title(f'Cluster {cluster_id} → {concept_name}\n({metric}={metric_value:.3f}, percentile={best_percentile})')
            
            plt.suptitle(f'Best Detecting Clusters for "{concept_name}" (Cal-selected p={best_percentile})', fontsize=14, y=1.02)
            plt.tight_layout()
            
            if save_dir:
                save_path = os.path.join(save_dir, f"{concept_name.replace('::', '_')}_best_clusters_p{best_percentile}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        elif sample_type == 'patch':
            # For patch embeddings, show heatmaps
            if model_input_size is None:
                raise ValueError("model_input_size required for patch visualization")
            
            for idx, (cluster_id, metric_value) in enumerate(cluster_list):
                plot_most_similar_patches_w_heatmaps_and_corr_images(
                    concept_label=str(cluster_id),
                    acts_loader=acts_loader,
                    con_label=f"{concept_name}_cluster_{cluster_id}_p{best_percentile}",
                    dataset_name=dataset_name,
                    model_input_size=model_input_size,
                    top_n=top_n_samples,
                    metric_type=f'Cluster {cluster_id} → {concept_name} ({metric}={metric_value:.3f}, p={best_percentile})',
                    test_only=test_only,
                    save_path=f"{save_dir}/{concept_name.replace('::', '_')}_cluster_{cluster_id}_p{best_percentile}.png" if save_dir else None
                )


def compare_calibration_vs_fixed_percentiles(dataset_name: str,
                                            model_name: str,
                                            sample_type: str,
                                            n_clusters: int,
                                            fixed_percentiles=None,
                                            metric: str = 'f1',
                                            percent_thru_model: int = 100):
    """
    Compare performance of calibration-selected percentile vs fixed percentiles.
    
    Args:
        dataset_name: Name of dataset
        model_name: Model name
        sample_type: 'patch' or 'cls'
        n_clusters: Number of clusters
        fixed_percentiles: List of fixed percentiles to compare against
        metric: Metric to compare
        percent_thru_model: Percentage through model
    """
    from calibration_selection_utils import find_best_percentile_from_calibration_allpairs
    import ast
    
    if fixed_percentiles is None:
        fixed_percentiles = [0.1, 0.5, 0.9]
    
    con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    
    # Get calibration-selected percentile
    try:
        best_percentile, cal_matches = find_best_percentile_from_calibration_allpairs(
            dataset_name, con_label, metric=metric
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Compare performance
    results = []
    
    # Add calibration-selected result
    test_path = f"Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{best_percentile}_{con_label}.csv"
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        # Calculate average best metric for each concept
        concept_best_metrics = []
        for concept in cal_matches.keys():
            concept_rows = []
            for _, row in test_df.iterrows():
                concept_tuple = ast.literal_eval(row['concept'])
                if concept_tuple[0] == concept:
                    concept_rows.append(row[metric])
            if concept_rows:
                concept_best_metrics.append(max(concept_rows))
        
        results.append({
            'method': f'Calibration-selected (p={best_percentile})',
            'percentile': best_percentile,
            f'test_avg_best_{metric}': np.mean(concept_best_metrics)
        })
    
    # Add fixed percentile results
    for p in fixed_percentiles:
        test_path = f"Quant_Results/{dataset_name}/detectionmetrics_allpairs_per_{p}_{con_label}.csv"
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            concept_best_metrics = []
            for concept in cal_matches.keys():
                concept_rows = []
                for _, row in test_df.iterrows():
                    concept_tuple = ast.literal_eval(row['concept'])
                    if concept_tuple[0] == concept:
                        concept_rows.append(row[metric])
                if concept_rows:
                    concept_best_metrics.append(max(concept_rows))
            
            results.append({
                'method': f'Fixed (p={p})',
                'percentile': p,
                f'test_avg_best_{metric}': np.mean(concept_best_metrics)
            })
    
    # Create comparison plot
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if 'Calibration' in row['method'] else 'blue' for _, row in results_df.iterrows()]
    bars = plt.bar(results_df['method'], results_df[f'test_avg_best_{metric}'], color=colors)
    
    # Add value labels on bars
    for bar, value in zip(bars, results_df[f'test_avg_best_{metric}']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.ylabel(f'Average Best {metric.upper()} on Test Set')
    plt.title(f'Calibration-Selected vs Fixed Percentiles\n{dataset_name} - {model_name} {sample_type} (n_clusters={n_clusters})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print("\nComparison Results:")
    print(results_df.to_string(index=False))
    
    return results_df











