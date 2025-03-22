import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

# from visualize_concepts_w_samples_utils import plot_patches_sim_to_vector


############ For Reasoning About Patch Indices #################
def compute_patches_per_image(patch_size=14, model_input_size=(224, 224)):
    num_patches_per_row = model_input_size[0] // patch_size  # 224 // 14 = 16
    num_patches_per_col = model_input_size[1] // patch_size  # 224 // 14 = 16
    patches_per_image = num_patches_per_row * num_patches_per_col  # 16 * 16 = 256 patches per image
    return patches_per_image


def get_patch_range_for_image(image_index, patch_size=14, model_input_size=(224, 224)):
    """
    Computes the starting and ending patch indices for a given image index.

    Args:
        image_index (int): The index of the image.
        patch_size (int): The size of each patch (default is 14).
        model_input_size (tuple): The dimensions (width, height) to which the image is resized (default is (224, 224)).

    Returns:
        tuple: (start_patch_index, end_patch_index) for the given image index.
    """
    # Compute number of patches per row and column
    patches_per_row = model_input_size[0] // patch_size
    patches_per_col = model_input_size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col

    # Compute start and end patch indices
    start_patch_index = image_index * patches_per_image
    end_patch_index = start_patch_index + patches_per_image

    return start_patch_index, end_patch_index


def calculate_patch_location(image, patch_idx, patch_size=14, model_input_size=(224, 224)):
    """
    Helper function to calculate the coordinates of a patch within an image.

    Args:
        image (PIL.Image): The original image.
        patch_idx (int): The index of the patch.
        patch_size (int): Size of the patch.
        model_input_size (tuple): The size to which the image is resized during embedding.

    Returns:
        tuple: (left, top, right, bottom) coordinates of the patch.
    """
    # Resize the image to match the model's embedding input size
    resized_image = image.resize(model_input_size)

    # Calculate the number of patches per row and column in the resized image
    patches_per_row = resized_image.size[0] // patch_size
    patches_per_col = resized_image.size[1] // patch_size
    patches_per_image = patches_per_row * patches_per_col

    # Determine the row and column of the patch
    patch_within_image_idx = patch_idx % patches_per_image
    row_idx = patch_within_image_idx // patches_per_row
    col_idx = patch_within_image_idx % patches_per_row

    # Calculate the coordinates of the patch
    left = col_idx * patch_size
    top = row_idx * patch_size
    right = left + patch_size
    bottom = top + patch_size

    return left, top, right, bottom


def get_image_idx_from_global_patch_idx(patch_index, patch_size=14, image_size=(224, 224)):
    """
    Given a patch index in a 1D vector, returns the image index it belongs to.

    Args:
        patch_index (int): The index of the patch in the flattened representation.
        patch_size (int): The size of each patch (assumed square).
        image_size (int): The size of the image (assumed square).

    Returns:
        int: The index of the image the patch belongs to.
    """
    patches_per_image = compute_patches_per_image(
        patch_size=patch_size, model_input_size=image_size
    )
    return patch_index // patches_per_image


def calculate_patch_indices(image_index, patch_index_in_image, patch_size, model_input_size):
    """Calculate the number of patches and global index for the selected patch."""
    image_width, image_height = model_input_size
    patches_per_row = image_width // patch_size
    patches_per_col = image_height // patch_size

    global_patch_idx = image_index * (patches_per_row * patches_per_col) + patch_index_in_image
    return patches_per_row, patches_per_col, global_patch_idx


def get_patch_split_df(dataset_name, patch_size=14, model_input_size=(224, 224)):
    """
    Expands an image-level metadata DataFrame to a per-patch split DataFrame.

    Args:
        image_metadata_df (pd.DataFrame): DataFrame containing image-level metadata, including a "split" column.
        num_patches (int): Number of patches per image (e.g., 14x14 = 196 patches).

    Returns:
        pd.DataFrame: A new DataFrame where each patch has its own row and inherits the split from the image.
    """
    per_sample_metadata_df = pd.read_csv(f"../Data/{dataset_name}/metadata.csv")
    split_df = per_sample_metadata_df["split"]

    num_patches = compute_patches_per_image(patch_size, model_input_size)

    # Repeat each row num_patches times and reset index
    patch_metadata_df = split_df.loc[split_df.index.repeat(num_patches)].reset_index(drop=True)
    return patch_metadata_df


############# Visualize patch similarities to vectors #############
def compute_patch_similarities_to_vector(
    image_index,
    concept_label,
    images,
    embeddings,
    cossims,
    dataset_name="CLEVR",
    patch_size=14,
    model_input_size=(224, 224),
    save_path=None,
    heatmap_path=None,
    show_plot=False,
):
    """
    Computes a heatmap of cosine similarities between a target vector and all patches in a given image.

    Args:
        image_index (int): The index of the image in the list of images.
        concept_label (str): The name of the concept to be visualized.
        images (list of PIL.Image): A list of images.
        embeddings (torch.tensor): Patch embeddings for the dataset.
        cossims (pd.DataFrame): File containing precomputed cosine similarities for each patch and concept.
        dataset_name (str): The name of the dataset. Defaults to 'CLEVR'.
        patch_size (int): The size of the patches into which the image is divided. Defaults to 14.
        model_input_size (tuple): The dimensions (width, height) to which the image is resized for model input.
                                  Defaults to (224, 224).
        save_path (str, optional): Path to save the heatmap plot. If None, the plot is not saved.
        heatmap_path (str, optional): Path to save the heatmap tensor. If None, the plot is not saved.

    Returns:
        matplotlib.figure.Figure: The heatmap figure showing cosine similarity overlayed on the image.

    Notes:
        - If a heatmap plot already exists at the specified save path, it will be loaded and displayed.
        - The heatmap shows the cosine similarity between the target vector and each patch's embedding
          in a grid corresponding to the patch layout.
    """
    # Return heatmap if it already exists
    if not show_plot and os.path.exists(heatmap_path):
        try:
            heatmap = torch.load(heatmap_path)
            return heatmap
        except:
            os.remove(heatmap_path)  # delete incomplete halfway saved heatmap

    # Process the image
    image = images[image_index]
    resized_image = image.resize(model_input_size)

    # Calculate patch indices
    patches_per_row, patches_per_col, _ = calculate_patch_indices(
        image_index, 0, patch_size, model_input_size
    )

    # Extract embeddings for the entire image
    # start_patch_index = image_index * (patches_per_row * patches_per_col)
    # end_patch_index = start_patch_index + (patches_per_row * patches_per_col)
    start_patch_index, end_patch_index = get_patch_range_for_image(
        image_index, patch_size=patch_size, model_input_size=model_input_size
    )
    # Compute cosine similarities between the target vector and all patches
    concept_cos_sims = cossims[concept_label].to_numpy()
    curr_image_cos_sims = concept_cos_sims[start_patch_index:end_patch_index]

    # Reshape similarities to match the patch grid
    cos_sim_grid = curr_image_cos_sims.reshape(patches_per_col, patches_per_row)

    # Plot the heatmap
    heatmap = torch.tensor(cos_sim_grid)
    if heatmap_path:
        torch.save(heatmap, heatmap_path)

    return heatmap


def compute_heatmaps_for_concept(
    concept_label, my_image_indices, images, embeds, cos_sims, dataset_name, con_label, top_n
):
    """
    Computes heatmaps for a given concept by calculating patch similarities across specified images.

    Args:
        concept_label (str): The label of the concept for which heatmaps are generated.
        my_image_indices (list): A list of indices representing the images to process.
        images (list): A list of images corresponding to the embeddings.
        embeds (torch.Tensor): A tensor of embeddings for the patches of the images.
        cos_sims (torch.Tensor): A tensor of cosine similarity values for the patches.
        dataset_name (str): The name of the dataset (used for saving the heatmaps).
        con_label (str): A label or identifier for the concept, typically used in filenames.

    Returns:
        dict: A dictionary where keys are image indices and values are the corresponding heatmaps.
    """
    heatmaps = {}
    for image_index in my_image_indices:
        heatmap_path = f"Heatmaps/{dataset_name}/patchsim_concept_{concept_label}_img_{image_index}_heatmaptype_avgsim_model__{con_label}"
        heatmap = compute_patch_similarities_to_vector(
            image_index=image_index,
            concept_label=str(concept_label),
            images=images,
            embeddings=embeds,
            cossims=cos_sims,
            dataset_name="CLEVR",
            patch_size=14,
            model_input_size=(224, 224),
            save_path=None,
            heatmap_path=heatmap_path,
        )
        heatmaps[image_index] = heatmap
    return heatmaps


def plot_aggregated_most_similar_patches_w_heatmaps_and_corr_images(
    group,
    images,
    cos_sims,
    concepts,
    embeds,
    con_label,
    dataset_name="CLEVR",
    patch_size=14,
    model_input_size=(224, 224),
    top_n=5,
    vmin=None,
    vmax=None,
):
    """
    Plots the most similar patches with a chosen concept, as well as the heatmaps for that concept and the corresponding image.
    """
    # get the most similar patches to the average of the vectors in the group
    avg_group_vector = compute_average_vector(group, concepts, percentile=90)
    avg_vector_cos_sims = F.cosine_similarity(embeds, avg_group_vector.unsqueeze(0), dim=1)
    most_similar_patches = torch.topk(avg_vector_cos_sims, top_n)[1]

    fig, axes = plt.subplots(2, top_n, figsize=(top_n * 3, 6))

    aggregated_heatmaps = {}
    images_w_patches = []
    for patch_idx in most_similar_patches:
        # Determine the image index
        image_idx = patch_idx // (
            (model_input_size[0] // patch_size) * (model_input_size[1] // patch_size)
        )
        image = images[image_idx]

        # Compute the heatmap for all of the concepts in the group
        heatmaps_across_concepts = []
        for concept_label in group:
            # Compute the heatmap for that image-concept combo
            heatmap_path = f"Heatmaps/{dataset_name}/patchsim_concept_{concept_label}_img_{image_idx}_heatmaptype_avgsim_model__{con_label}"
            try:
                heatmap = compute_patch_similarities_to_vector(
                    image_index=image_idx,
                    concept_label=str(concept_label),
                    images=images,
                    embeddings=embeds,
                    cossims=cos_sims,
                    dataset_name="CLEVR",
                    patch_size=14,
                    model_input_size=(224, 224),
                    save_path=None,
                    heatmap_path=heatmap_path,
                    show_plot=False,
                )
            except:
                os.remove(heatmap_path)  # delete incomplete halfway saved heatmap
                heatmap = compute_patch_similarities_to_vector(
                    image_index=image_idx,
                    concept_label=str(concept_label),
                    images=images,
                    embeddings=embeds,
                    cossims=cos_sims,
                    dataset_name="CLEVR",
                    patch_size=14,
                    model_input_size=(224, 224),
                    save_path=None,
                    heatmap_path=heatmap_path,
                    show_plot=False,
                )
            heatmaps_across_concepts.append(heatmap)

        # keep track of the average heatmap across all of the concepts in the group
        avg_heatmap = torch.stack(heatmaps_across_concepts).mean(dim=0)
        aggregated_heatmaps[image_idx] = avg_heatmap

        # Get what you need for the image above the heatmap showing where the patch comes from
        left, top, right, bottom = calculate_patch_location(
            image, patch_idx, patch_size, model_input_size
        )
        image_with_patch = make_image_with_highlighted_patch(
            image, left, top, right, bottom, model_input_size, grayscale=True
        )
        images_w_patches.append(image_with_patch)

    if not vmin and not vmax:
        all_values = [
            value
            for heatmap in aggregated_heatmaps.values()
            for row in aggregated_heatmaps
            for value in row
        ]
        vmin, vmax = min(all_values), max(all_values)

    for i, patch_idx in enumerate(most_similar_patches):
        image_idx = patch_idx // (
            (model_input_size[0] // patch_size) * (model_input_size[1] // patch_size)
        )
        image = images[image_idx].resize(model_input_size)

        # Plot the original_image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Image {image_idx}")
        axes[0, i].axis("off")

        # Plot the image with highlighted patch and heatmap
        axes[1, i].imshow(images_w_patches[i], alpha=0.8)
        ax_heatmap.set_title(
            f"Heatmap Max = {round(aggregated_heatmaps[image_idx].max().item(), 2)}"
        )
        heatmap_overlay = axes[1, i].imshow(
            aggregated_heatmaps[image_idx],
            cmap="hot",
            alpha=0.3,
            extent=(0, model_input_size[0], model_input_size[1], 0),
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, i].axis("off")

    # Add a color bar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = plt.colorbar(heatmap_overlay, cax=cbar_ax)
    cbar.set_label("Cosine Similarity")

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the color bar
    plt.suptitle(f"Most Aligned Patches to Concept {concept_label}", fontsize=16, y=1.05)
    plt.show()
