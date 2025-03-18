import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import os
from .compute_concepts_utils import calculate_patch_location, make_image_with_highlighted_patch
from .patch_alignment_utils import compute_patch_similarities_to_vector

### For Computing Correlations and Visualizing them ###
def get_binary_image_concept_activations(cosine_similarities, device, threshold=0.2):
    """
    Computes binary image concept activations based on a threshold applied to cosine similarities.

    Args:
        cosine_similarities (torch.Tensor): Tensor of shape [n_images, n_patches, n_concepts] 
                                             containing cosine similarities.
        threshold (float): Threshold for identifying patch activations for gt concepts.
        device (str): Device to perform the computation on, e.g., 'cuda' for GPU.

    Returns:
        torch.Tensor: Tensor of shape [n_images, n_concepts] representing the image concept activations.
    """
    cosine_similarities = cosine_similarities.to(device)  # Ensure tensor is on the GPU

    # Apply threshold to identify patch activations for gt concepts
    activated = cosine_similarities > threshold  # Shape: [n_images, n_patches, n_concepts]

    # Determine if an image is activated for a concept (logical OR over patches)
    image_concept_activations = activated.any(dim=1).int()  # Shape: [n_images, n_concepts]

    return image_concept_activations


def get_dynamic_binary_image_concept_activations(cosine_similarities, device, percentile=90):
    """
    Computes binary image concept activations based on a dynamic threshold
    determined by the specified percentile of cosine similarities.

    Args:
        cosine_similarities (torch.Tensor): Tensor of shape [n_images, n_patches, n_concepts]
                                             containing cosine similarities.
        percentile (float): Percentile to define the threshold for identifying activations.

    Returns:
        torch.Tensor: Tensor of shape [n_images, n_concepts] representing the image concept activations.
    """
    cosine_similarities = cosine_similarities.to(device)  # Ensure tensor is on the GPU
    
    # Compute the dynamic threshold based on the given percentile
    threshold = torch.quantile(cosine_similarities.flatten()[:3000], percentile / 100)  # Flatten to 1D for quantile calculation

    print("threshold:", threshold)
    
    # Apply the dynamic threshold to identify patch activations for gt concepts
    activated = cosine_similarities > threshold  # Shape: [n_images, n_patches, n_concepts]

    # Determine if an image is activated for a concept (logical OR over patches)
    image_concept_activations = activated.any(dim=1).int()  # Shape: [n_images, n_concepts]

    return image_concept_activations


def compute_jaccard_similarity_matrix(tensor_a, tensor_b, device):
    """
    Compute the Jaccard similarity matrix between two tensors of concept activations.

    Args:
        tensor_a (torch.Tensor): A boolean or integer tensor of shape [n_images, n_concepts_a].
        tensor_b (torch.Tensor): A boolean or integer tensor of shape [n_images, n_concepts_b].
        device (str): Device to perform the computation on, e.g., 'cuda' for GPU.

    Returns:
        torch.Tensor: A Jaccard similarity matrix of shape [n_concepts_a, n_concepts_b].
    """
    tensor_a = tensor_a.to(device).float()
    tensor_b = tensor_b.to(device).float() 

    # Compute the intersection (co-activation)
    intersection = torch.matmul(tensor_a.T, tensor_b)  # Shape: [n_concepts_a, n_concepts_b]

    # Compute the union
    union = (
        tensor_a.sum(dim=0).unsqueeze(1) +  # Sum over images for each concept in tensor_a
        tensor_b.sum(dim=0).unsqueeze(0) -  # Sum over images for each concept in tensor_b
        intersection  # Subtract the intersection to avoid double-counting
    )

    # Avoid division by zero
    union[union == 0] = 1

    # Compute Jaccard similarity
    jaccard_similarity = intersection / union  # Shape: [n_concepts_a, n_concepts_b]

    return jaccard_similarity


def calculate_group_average_similarity(group, concepts):
    """
    Calculate the average similarity for a group of concepts.

    Args:
        group (set): A set of concept indices representing the group.
        similarity_matrix (torch.Tensor): A (n_concepts, n_concepts) Jaccard similarity matrix.

    Returns:
        float: The average similarity for the group.
    """
    if len(group) < 2:
        # If the group has fewer than 2 concepts, the average similarity is undefined (set to 1)
        return 0

    # Extract the vectors for the concepts in the group
    group_vectors = [concepts[concept_name] for concept_name in group]

    # Calculate cosine similarity between all pairs of vectors
    similarities = []
    for i in range(len(group_vectors)):
        for j in range(i + 1, len(group_vectors)):
            cos_sim = F.cosine_similarity(group_vectors[i].unsqueeze(0), group_vectors[j].unsqueeze(0))
            similarities.append(cos_sim.item())

    # Compute the average similarity
    average_similarity = sum(similarities) / len(similarities)
    
    return average_similarity




### For Clustering Concepts by Correlation and Visualizing them ###
def mask_out_diag_and_visited(matrix, visited):
    """
    Mask out the diagonals and the rows/columns corresponding to the indices in `visited`.

    Args:
        matrix (torch.Tensor): A square tensor of shape [n, n].
        visited (set): A set of indices to mask out.

    Returns:
        torch.Tensor: The masked tensor with diagonals and rows/columns from `visited` set to -inf.
    """
    # Mask out diagonals
    matrix.fill_diagonal_(-1)

    # Mask out the rows and columns corresponding to the visited indices
    for idx in visited:
        matrix[idx, :] = -1  # Mask the row
        matrix[:, idx] = -1  # Mask the column

    return matrix


def plot_group_sizes(groups):
    """
    Plots the number of concepts in each group and prints the total number of groups.
    
    Args:
        groups (list of lists): A list of groups where each group is a list of concept indices.
    """
    print(f"{len(groups)} groups created")
    
    # Calculate the size of each group
    group_sizes = [len(group) for group in groups]
    
    # Plot the sizes of the groups
    plt.bar(range(len(groups)), group_sizes, color='skyblue', edgecolor='black')
    plt.xlabel('Group Index', fontsize=14)
    plt.ylabel('Number of Concepts', fontsize=14)
    plt.title('Number of Concepts in Each Group', fontsize=16)
    plt.tight_layout()
    plt.show()
    

def group_concepts_by_similarity(jaccard_similarity, group_variance=0.1):
    """
    Groups concepts based on Jaccard similarity, clustering concepts with similarity values within 
    a specified variance of the highest similarity.

    The algorithm iteratively identifies pairs of concepts with high similarity, groups them together, 
    and continues until all concepts are assigned to a group. Concepts are added to existing groups 
    if they match the similarity criteria, or new groups are created if no match is found.

    Args:
        jaccard_similarity (torch.Tensor): A square matrix (n x n) of Jaccard similarity scores.
        group_variance (float): Allowed variance from the highest similarity value for grouping.

    Returns:
        list of lists: A list of groups, where each group is a list of concept indices.
    """
    groups = []
    visited = set()
    current_corr_matrix = jaccard_similarity.clone()
    num_concepts = jaccard_similarity.size(0)

    while len(visited) < num_concepts:
        if len(visited) == num_concepts - 1:
            groups.append([i for i in range(num_concepts) if i not in visited])
            break

        # Mask diagonal and visited concepts
        current_corr_matrix.fill_diagonal_(float('-inf'))
        current_corr_matrix[list(visited), :] = float('-inf')
        current_corr_matrix[:, list(visited)] = float('-inf')

        # Find the maximum similarity and corresponding pairs
        highest_corr = current_corr_matrix.max().item()
        index_pairs = torch.nonzero(
            (current_corr_matrix >= highest_corr - group_variance)
            & (current_corr_matrix <= highest_corr + group_variance)
        ).tolist()

        for i, j in index_pairs:
            for group in groups:
                if i in group or j in group:
                    group.update({i, j})
                    visited.update({i, j})
                    break
            else:  # Create a new group if no existing group matches
                groups.append({i, j})
                visited.update({i, j})

    # Convert groups from sets to lists for final output
    return [list(group) for group in groups]


def compute_average_vector(group, concepts, percentile=90):
    """
    Computes the average vector for a group after removing outliers based on cosine similarity.

    Args:
        group (list): List of concept names in the group.
        percentile (float): Percentile used to define the threshold for removing outliers.

    Returns:
        torch.Tensor: The average vector for the group after outlier removal.
    """
    # Get the vectors for the concepts in the group
    group_vectors = [concepts[concept_name] for concept_name in group]
    group_vectors = torch.stack(group_vectors)  # Stack into a tensor [n_concepts, embed_dim]
    
    # Compute the pairwise cosine similarity matrix
    normed_vectors = F.normalize(group_vectors, p=2, dim=1)  # Normalize for cosine similarity
    similarity_matrix = torch.matmul(normed_vectors, normed_vectors.T)  # Cosine similarity matrix
    
    # Compute the mean similarity for each vector to all other vectors in the group
    mean_similarity = similarity_matrix.mean(dim=1)
    
    # Compute the threshold for outlier removal based on the given percentile
    threshold = torch.quantile(mean_similarity, percentile / 100)
    
    # Filter out vectors with mean similarity below the threshold (outliers)
    inliers_mask = mean_similarity >= threshold
    filtered_group_vectors = group_vectors[inliers_mask]
    
    # Compute the average of the filtered vectors
    if len(filtered_group_vectors) > 0:
        group_avg = torch.mean(filtered_group_vectors, dim=0)
    else:
        group_avg = torch.zeros(group_vectors[0].shape)  # Return a zero vector if no inliers remain
    
    return group_avg

def get_avg_group_vectors(groups, concepts, percentile=0.9):
    """
    Compute the group vectors for each group by averaging vectors w outlier removal.

    Args:
        groups (list): List of groups of concept names.
        concepts (dict): Dictionary containing vectors for each concept.

    Returns:
        torch.Tensor: A tensor of shape [n_groups, embed_dim] representing the group vectors.
    """
    group_vectors = []
    
    # Iterate over each group
    for group in groups:
        group_vectors.append(compute_average_vector(group, concepts, percentile))
        
    # Stack the vectors into a tensor [n_groups, embed_dim]
    return torch.stack(group_vectors)


def get_sim_to_centroid(concept_vector, group, concepts, percentile=90):
    avg_group_vector = compute_average_vector(group, concepts, percentile=percentile)
    cos_sim = F.cosine_similarity(avg_group_vector, concept_vector, dim=0)
    return cos_sim

def get_most_representative_concept(group, concepts):
    """
    Finds the most representative concept in the group based on cosine similarity to all other concepts.

    Args:
        group (list): List of concept names in the group.
        concepts (dict): Dictionary containing vectors for each concept.

    Returns:
        str: The name of the most representative concept in the group.
    """
    # Get the vectors for the concepts in the group
    group_vectors = [concepts[concept_name] for concept_name in group]
    group_vectors = torch.stack(group_vectors)  # Stack into a tensor [n_concepts, embed_dim]
    
    # Compute the pairwise cosine similarity matrix
    normed_vectors = F.normalize(group_vectors, p=2, dim=1)  # Normalize for cosine similarity
    similarity_matrix = torch.matmul(normed_vectors, normed_vectors.T)  # Cosine similarity matrix
    
    # Compute the average similarity for each vector to all other vectors in the group
    mean_similarity = similarity_matrix.mean(dim=1)
    
    # Find the index of the most representative concept (highest mean similarity)
    most_representative_idx = torch.argmax(mean_similarity)
    
    # Get the name of the most representative concept
    most_representative_concept = most_representative_idx.item()
    
    return most_representative_concept


def get_most_rep_group_vectors(groups, concepts):
    """
    Compute the group vectors for each group, either by averaging the vectors or selecting the most representative concept.

    Args:
        groups (list): List of groups of concept names.
        concepts (dict): Dictionary containing vectors for each concept.

    Returns:
        torch.Tensor: A tensor of shape [n_groups, embed_dim] representing the group vectors.
    """
    group_vectors = []
    group_concept_reps = []
    
    # Iterate over each group
    for group in groups:
        concept = get_most_representative_concept(group, concepts)
        group_concept_reps.append(concept)
        group_vectors.append(concepts[concept])
        
    # Stack the vectors into a tensor [n_groups, embed_dim]
    return torch.stack(group_vectors), group_concept_reps


def compute_group_cosine_sims(group_vectors, embeds, device, batch_size=1024):
    """
    Computes the cosine similarity between each embedding and the average vector of each group,
    and returns the cosine similarity matrix as a DataFrame.

    Args:
        groups (list): List of groups, where each group is a list of concept indices.
        embeds (torch.Tensor): Embedding tensor of shape [n_samples, embed_dim].
        batch_size (int): Batch size for processing embeddings.
        device (str): Device to run the computations on, e.g., 'cuda' or 'cpu'.

    Returns:
        pd.DataFrame: DataFrame of cosine similarities between samples and groups.
    """
    # Compute group vectors
    group_vectors = group_vectors.to(device)
    
    n_samples = embeds.size(0)
    batch_cosine_sim = []

    # Process embeddings in batches and compute cosine similarity
    for i in tqdm(range(0, n_samples, batch_size)):
        batch_embeds = embeds[i:i + batch_size].to(device)  # Shape: [batch_size, embed_dim]

        # Expand group vectors for batch computation
        group_vectors_expanded = group_vectors.unsqueeze(0).expand(batch_embeds.size(0), -1, -1).to(device)  # [batch_size, n_groups, embed_dim]
        
        # Compute cosine similarity for the batch
        batch_cos_sim = F.cosine_similarity(batch_embeds.unsqueeze(1).to(device), group_vectors_expanded, dim=2)  # [batch_size, n_groups]
        
        # Append the results for this batch
        batch_cosine_sim.append(batch_cos_sim.cpu())  # Move to CPU if necessary

    # Concatenate all the batches into a single tensor
    cosine_sim_matrix = torch.cat(batch_cosine_sim, dim=0)  # Shape: [n_samples, n_groups]

    # Convert cosine similarity matrix to DataFrame
    grouped_concept_cos_sims_df = pd.DataFrame(cosine_sim_matrix.numpy())
    grouped_concept_cos_sims_df.columns = [f"Group_{i}" for i in range(cosine_sim_matrix.shape[1])]

    return grouped_concept_cos_sims_df

def calculate_group_similarities(groups, concepts):
    """
    Calculates the average similarity for each group.

    Args:
        groups (list): List of groups, where each group is a list of concept indices.

    Returns:
        list: List of tuples with group index and average similarity.
    """
    group_similarities = [
        (i, calculate_group_average_similarity(groups[i], concepts)) for i in range(len(groups))
    ]
    
    return group_similarities


def plot_aggregated_most_similar_patches_w_heatmaps_and_corr_images(group, images, cos_sims, concepts,
                                                                     embeds, con_label, dataset_name='CLEVR', patch_size=14, 
                                                                    model_input_size=(224, 224), top_n=5, vmin=None, vmax=None):
    """
    Computes an average group vector and then gets the patches most similar to that. Then, gets the heatmap for each concept in the
    group's activation with the images those patches are in, and averages them.
    """
    #get the most similar patches to the average of the vectors in the group
    avg_group_vector = compute_average_vector(group, concepts, percentile=90)
    avg_vector_cos_sims = F.cosine_similarity(embeds, avg_group_vector.unsqueeze(0), dim=1)
    most_similar_patches = torch.topk(avg_vector_cos_sims, top_n)[1]
    
    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))
    
    aggregated_heatmaps = {}
    images_w_patches = []
    for patch_idx in tqdm(most_similar_patches):
        patch_idx = int(patch_idx)
        # Determine the image index
        image_idx = int(patch_idx // ((model_input_size[0] // patch_size) * (model_input_size[1] // patch_size)))
        image = images[image_idx]
        
        #Compute the heatmap for all of the concepts in the group
        heatmaps_across_concepts = []
        for concept_label in group:
            #Compute the heatmap for that image-concept combo
            heatmap_path = f'Heatmaps/{dataset_name}/patchsim_concept_{concept_label}_img_{image_idx}_heatmaptype_avgsim_model__{con_label}'
            try:
                heatmap = compute_patch_similarities_to_vector(image_index=image_idx, concept_label=str(concept_label), 
                                                  images=images, embeddings=embeds,  cossims=cos_sims,
                                                  dataset_name='CLEVR', patch_size=14, model_input_size=(224, 224),
                                                  save_path=None,
                                                  heatmap_path=heatmap_path, show_plot=False)
            except:
                os.remove(heatmap_path) #delete incomplete halfway saved heatmap
                heatmap = compute_patch_similarities_to_vector(image_index=image_idx, concept_label=str(concept_label), 
                                                  images=images, embeddings=embeds,  cossims=cos_sims,
                                                  dataset_name='CLEVR', patch_size=14, model_input_size=(224, 224),
                                                  save_path=None,
                                                  heatmap_path=heatmap_path, show_plot=False)
            heatmaps_across_concepts.append(heatmap)
        
        #keep track of the average heatmap across all of the concepts in the group
        avg_heatmap = torch.stack(heatmaps_across_concepts).mean(dim=0)
        aggregated_heatmaps[image_idx] = avg_heatmap    

        # Get what you need for the image above the heatmap showing where the patch comes from
        left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size, model_input_size)
        image_with_patch = make_image_with_highlighted_patch(image, left, top, right, bottom, model_input_size, grayscale=True)
        images_w_patches.append(image_with_patch)
    
    if not vmin and not vmax:
        all_values = [value.item() for heatmap in aggregated_heatmaps.values() for row in heatmap for value in row]
        vmin, vmax = min(all_values), max(all_values)
    
    for i, patch_idx in enumerate(most_similar_patches):
        patch_idx = int(patch_idx)
        image_idx = patch_idx // ((model_input_size[0] // patch_size) * (model_input_size[1] // patch_size))
        image = images[image_idx].resize(model_input_size)
        
        # Plot the original_image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Image {image_idx}')
        axes[0, i].axis('off')
        
        # Plot the image with highlighted patch and heatmap
        axes[1, i].imshow(images_w_patches[i], alpha=0.8)
        heatmap_overlay = axes[1, i].imshow(aggregated_heatmaps[image_idx], cmap='hot', alpha=0.3, extent=(0, model_input_size[0], model_input_size[1], 0),
                                           vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')


    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label('Cosine Similarity')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    plt.suptitle(f"Most Aligned Patches to Concept {concept_label}", fontsize=16, y=1.05)
    plt.show()
    

def plot_aggregated_heatmaps_and_corr_images(image_indices, group, images, con_label, cos_sims, embeds, 
                                             dataset_name='CLEVR', patch_size=14, model_input_size=(224, 224), vmin=None, vmax=None):
    """
    Computes an average group vector and then gets the patches most similar to that. Then, gets the heatmap for each concept in the
    group's activation with the images those patches are in, and averages them.
    """
    aggregated_heatmaps = {}
    for image_idx in tqdm(image_indices):
        image = images[image_idx]
        
        #Compute the heatmap for all of the concepts in the group
        heatmaps_across_concepts = []
        for concept_label in group:
            #Compute the heatmap for that image-concept combo
            heatmap_path = f'Heatmaps/{dataset_name}/patchsim_concept_{concept_label}_img_{image_idx}_heatmaptype_avgsim_model__{con_label}'
            try:
                heatmap = compute_patch_similarities_to_vector(image_index=image_idx, concept_label=str(concept_label), 
                                                  images=images, embeddings=embeds,  cossims=cos_sims,
                                                  dataset_name='CLEVR', patch_size=patch_size, model_input_size=model_input_size,
                                                  save_path=None,
                                                  heatmap_path=heatmap_path, show_plot=False)
            except:
                os.remove(heatmap_path) #delete incomplete halfway saved heatmap
                heatmap = compute_patch_similarities_to_vector(image_index=image_idx, concept_label=str(concept_label), 
                                                  images=images, embeddings=embeds,  cossims=cos_sims,
                                                  dataset_name='CLEVR', patch_size=patch_size, model_input_size=model_input_size,
                                                  save_path=None,
                                                  heatmap_path=heatmap_path, show_plot=False)
            heatmaps_across_concepts.append(heatmap)
        
        #keep track of the average heatmap across all of the concepts in the group
        avg_heatmap = torch.stack(heatmaps_across_concepts).mean(dim=0)
        aggregated_heatmaps[image_idx] = avg_heatmap    
    
    if not vmin and not vmax:
        all_values = [value.item() for heatmap in aggregated_heatmaps.values() for row in heatmap for value in row]
        vmin, vmax = min(all_values), max(all_values)
    
    fig, axes = plt.subplots(2, len(image_indices), figsize=(len(image_indices) * 3, 6))
    for i, image_idx in enumerate(image_indices):
        image = images[image_idx].resize(model_input_size)
        
        # Plot the original_image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Image {image_idx}')
        axes[0, i].axis('off')
        
        # Plot the image with highlighted patch and heatmap
        axes[1, i].imshow(image, alpha=0.7)
        axes[1, i].set_title(f'Heatmap Max = {round(aggregated_heatmaps[image_idx].max().item(), 2)}')
        heatmap_overlay = axes[1, i].imshow(aggregated_heatmaps[image_idx], cmap='hot', alpha=0.4, extent=(0, model_input_size[0], model_input_size[1], 0),
                                           vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')


    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label('Cosine Similarity')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    plt.suptitle(f"Most Aligned Patches to Concept {concept_label}", fontsize=16, y=1.05)
    plt.show()