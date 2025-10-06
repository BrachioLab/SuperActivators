import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
from typing import List, Dict, Tuple, Optional

from utils.general_utils import retrieve_image, pad_or_resize_img, get_paper_plotting_style
from utils.patch_alignment_utils import compute_patch_similarities_to_vector, get_patch_range_for_image
from utils.visualize_concepts_w_samples_utils import top_left_crop_to_original_aspect
from utils.activation_loader import ChunkedActivationLoader


def get_default_percentiles(model_name: str) -> List[int]:
    """Get default percentile values for a given model."""
    if model_name == 'CLIP':
        return [2, 15, 28, 40, 52, 65, 78, 90, 100]
    elif model_name in ['Llama', 'Llama-Vision']:
        return [2, 15, 28, 40, 52, 65, 78, 90, 100]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_activations_for_ptm(
    dataset_name: str, 
    model_name: str, 
    concept_type: str,
    sample_type: str,
    percentthrumodel: int
) -> ChunkedActivationLoader:
    """Load activations for a specific percent through model."""
    # Construct the concept label
    if concept_type == 'avg':
        con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percentthrumodel}'
    elif concept_type == 'linsep':
        con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percentthrumodel}'
    else:
        raise ValueError(f"Unsupported concept type: {concept_type}")
    
    # Create activation loader
    if 'avg' in con_label:
        acts_path = f'/scratch/cgoldberg/Cosine_Similarities/{dataset_name}'
    else:
        acts_path = f'/scratch/cgoldberg/Distances/{dataset_name}'
    
    acts_loader = ChunkedActivationLoader(
        base_path=acts_path,
        con_label=con_label,
        chunk_size=100000
    )
    
    return acts_loader


def load_thresholds_for_ptm(
    dataset_name: str,
    con_label: str,
    concepts: List[str]
) -> Dict[str, float]:
    """Load best detection thresholds for each concept at a specific PTM."""
    # Load all percentiles
    all_thresholds_path = f'Thresholds/{dataset_name}/all_percentiles_{con_label}.pt'
    all_thresholds = torch.load(all_thresholds_path, weights_only=True)
    
    # Load best detection percentiles
    best_percentiles_path = f'Best_Detection_Percentiles_Cal/{dataset_name}/best_percentiles_{con_label}.pt'
    best_detect_percentiles = torch.load(best_percentiles_path, weights_only=True)
    
    # Get thresholds for each concept
    concept_to_thresholds = {}
    for concept in concepts:
        if concept in best_detect_percentiles:
            best_percentile = best_detect_percentiles[concept]['best_percentile']
            concept_to_thresholds[concept] = all_thresholds[best_percentile][concept][0]
        else:
            print(f"Warning: No threshold found for concept '{concept}' in {con_label}")
    
    return concept_to_thresholds


def compute_heatmap_for_concept(
    concept: str,
    image_index: int,
    acts_loader: ChunkedActivationLoader,
    dataset_name: str,
    patch_size: int,
    model_input_size: Tuple[int, int]
) -> np.ndarray:
    """Compute activation heatmap for a single concept."""
    # Get patch range for the image
    start_patch_index, end_patch_index = get_patch_range_for_image(
        image_index, patch_size=patch_size, model_input_size=model_input_size
    )
    
    # Load activations for this concept
    df = acts_loader.load_concept_range(
        concept_names=[str(concept)], 
        start_idx=start_patch_index, 
        end_idx=end_patch_index
    )
    curr_image_acts = df[str(concept)].values
    
    # Reshape to grid
    patches_per_side = model_input_size[0] // patch_size
    heatmap = curr_image_acts.reshape(patches_per_side, patches_per_side)
    
    return heatmap


def plot_superdetectors_over_layers(
    image_index: int,
    concepts: List[str],
    dataset_name: str,
    model_name: str,
    concept_type: str = 'avg',
    sample_type: str = 'patch',
    model_input_size: Tuple[int, int] = (224, 224),
    percentiles: Optional[List[int]] = None,
    save_file: Optional[str] = None,
    figure_width: Optional[float] = None,
    heatmap_alpha: float = 0.85,
    patch_size: int = 14
):
    """
    Plot superactivators across different layers for multiple concepts.
    
    Parameters:
    -----------
    image_index : int
        Index of the image to visualize
    concepts : List[str]
        List of concept names to visualize
    dataset_name : str
        Name of the dataset (e.g., 'Coco', 'CLEVR')
    model_name : str
        Model name (e.g., 'CLIP', 'Llama')
    concept_type : str
        Type of concepts ('avg' or 'linsep')
    sample_type : str
        Sample type ('cls' or 'patch')
    model_input_size : Tuple[int, int]
        Model input dimensions
    percentiles : Optional[List[int]]
        Percentiles through model to visualize. If None, uses model defaults
    save_file : Optional[str]
        Path to save the figure
    figure_width : Optional[float]
        Figure width in inches. If None, auto-calculates
    heatmap_alpha : float
        Alpha value for heatmap overlay
    patch_size : int
        Size of patches for vision models
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Apply paper plotting style
    plt.rcParams.update(get_paper_plotting_style())
    
    # Get percentiles if not provided
    if percentiles is None:
        percentiles = get_default_percentiles(model_name)
    
    # Load and prepare image
    image = retrieve_image(image_index, dataset_name)
    resized_image = pad_or_resize_img(image, model_input_size)
    image_np = np.array(resized_image.convert("RGB")) / 255.0
    
    # Crop for LLAMA if needed
    if model_input_size == (224, 224):
        image_cropped = image_np
    else:
        image_cropped = top_left_crop_to_original_aspect(image_np, image.size, resized_image.size)
    
    # Create figure
    num_concepts = len(concepts)
    num_ptms = len(percentiles)
    
    if figure_width is None:
        figure_width = 1.5 + num_ptms * 1.8  # Adjust spacing
    figure_height = num_concepts * 1.8
    
    fig = plt.figure(figsize=(figure_width, figure_height))
    
    # Collect all heatmaps to determine global vmin/vmax
    all_heatmaps = {}
    all_thresholds = {}
    
    # First pass: collect all heatmaps and thresholds
    for ptm_idx, ptm in enumerate(percentiles):
        # Load activations for this PTM
        acts_loader = load_activations_for_ptm(
            dataset_name, model_name, concept_type, sample_type, ptm
        )
        
        # Construct concept label for thresholds
        if concept_type == 'avg':
            con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{ptm}'
        else:
            con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{ptm}'
        
        # Load thresholds for this PTM
        ptm_thresholds = load_thresholds_for_ptm(dataset_name, con_label, concepts)
        all_thresholds[ptm] = ptm_thresholds
        
        # Compute heatmaps for all concepts
        for concept in concepts:
            heatmap = compute_heatmap_for_concept(
                concept, image_index, acts_loader, dataset_name, 
                patch_size, model_input_size
            )
            all_heatmaps[(concept, ptm)] = heatmap
    
    # Determine global vmin/vmax
    all_values = np.concatenate([h.flatten() for h in all_heatmaps.values()])
    vmin, vmax = np.percentile(all_values, [5, 95])
    
    # Plot layout
    gs = fig.add_gridspec(num_concepts, num_ptms + 1, 
                         width_ratios=[1.5] + [1] * num_ptms,
                         hspace=0.3, wspace=0.2)
    
    # Plot original image column
    for row_idx, concept in enumerate(concepts):
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(image_cropped)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title('Original', fontsize=plt.rcParams['font.size'])
        ax.text(-0.1, 0.5, concept.capitalize(), 
                transform=ax.transAxes, rotation=90, 
                verticalalignment='center', horizontalalignment='right',
                fontsize=plt.rcParams['font.size'], fontstyle='italic')
    
    # Plot heatmaps
    colorbar_im = None
    for row_idx, concept in enumerate(concepts):
        for col_idx, ptm in enumerate(percentiles):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            
            # Get heatmap and threshold
            heatmap = all_heatmaps[(concept, ptm)]
            threshold = all_thresholds[ptm].get(concept, vmax)  # Use vmax if no threshold
            
            # Resize heatmap to match image size
            heatmap_resized = Image.fromarray(heatmap).resize(resized_image.size, resample=Image.NEAREST)
            heatmap_resized = np.array(heatmap_resized)
            
            # Crop if needed
            if model_input_size == (224, 224):
                heatmap_cropped = heatmap_resized
            else:
                heatmap_cropped = top_left_crop_to_original_aspect(heatmap_resized, image.size, resized_image.size)
            
            # Plot image and heatmap overlay
            ax.imshow(image_cropped, alpha=1.0)
            im = ax.imshow(heatmap_cropped, cmap='magma', interpolation='nearest', 
                          vmin=vmin, vmax=vmax, alpha=heatmap_alpha)
            ax.axis('off')
            
            # Add title for first row
            if row_idx == 0:
                ax.set_title(f'{ptm}%', fontsize=plt.rcParams['font.size'] - 1)
            
            # Store for colorbar
            if colorbar_im is None:
                colorbar_im = im
            
            # Draw superactivator borders
            grid_size = heatmap.shape[0]
            patch_h = resized_image.size[1] / grid_size
            patch_w = resized_image.size[0] / grid_size
            
            for i_patch in range(grid_size):
                for j_patch in range(grid_size):
                    if heatmap[i_patch, j_patch] >= threshold:
                        x = j_patch * patch_w
                        y = i_patch * patch_h
                        rect = mpatches.Rectangle((x, y), patch_w, patch_h,
                                                 linewidth=1, edgecolor='deepskyblue', 
                                                 facecolor='none')
                        ax.add_patch(rect)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(colorbar_im, cax=cbar_ax)
    cbar.set_label('Activation Score', fontsize=plt.rcParams['font.size'])
    
    # Add legend for superactivators
    legend_elements = [mpatches.Rectangle((0, 0), 1, 1, facecolor='none', 
                                        edgecolor='deepskyblue', linewidth=1.5,
                                        label='SuperActivators')]
    fig.legend(handles=legend_elements, loc='lower right', 
              bbox_to_anchor=(0.98, 0.02), frameon=False,
              fontsize=plt.rcParams['font.size'] - 1)
    
    if save_file:
        plt.savefig(save_file, dpi=300, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_file}")
    
    return fig


# Example usage function
def example_usage():
    """Example of how to use the plotting function."""
    # Parameters
    image_index = 806
    concepts = ['person', 'chair', 'food', 'electronic', 'train', 'bus']
    dataset_name = 'Coco'
    model_name = 'CLIP'
    
    # Create the plot
    fig = plot_superdetectors_over_layers(
        image_index=image_index,
        concepts=concepts,
        dataset_name=dataset_name,
        model_name=model_name,
        concept_type='avg',
        sample_type='patch',
        model_input_size=(224, 224),
        percentiles=None,  # Use default percentiles
        save_file='../Figs/superdetectors_over_layers_example.pdf',
        figure_width=None,  # Auto-calculate
        heatmap_alpha=0.85
    )
    
    plt.show()