"""
Parallel visualization utilities for faster layer analysis.
Uses multiprocessing to create visualizations in parallel.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel processing
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from functools import partial


def create_single_visualization(args):
    """
    Worker function to create a single visualization.
    Args should be a tuple of all needed parameters.
    """
    (similarities_dict, concept_names, img_idx, image_data, 
     dataset_name, model_input_size, con_label, per_layer_scale,
     selected_concepts, patch_size, save_dir) = args
    
    from utils.layer_analysis_utils import visualize_patch_concept_heatmaps
    from PIL import Image
    
    # Reconstruct PIL image from numpy array
    original_image = Image.fromarray(image_data)
    
    # Create visualization
    scale_suffix = "_per_layer_scale" if per_layer_scale else ""
    save_path = os.path.join(save_dir, 
                            f'patch_heatmaps_image_{img_idx}_{con_label}{scale_suffix}.png')
    
    visualize_patch_concept_heatmaps(
        similarities_dict, 
        concept_names, 
        img_idx,
        original_image=original_image,
        dataset_name=dataset_name,
        model_input_size=model_input_size,
        con_label=con_label,
        per_layer_scale=per_layer_scale,
        selected_concepts=selected_concepts,
        patch_size=patch_size,
        save_path=save_path,
        show_plot=False
    )
    
    return f"Completed: {save_path}"


def parallel_visualize_all(similarities_by_layer, concept_names, test_images, 
                          dataset_name, model_input_size, con_label,
                          selected_concepts=None, patch_size=14, n_workers=None):
    """
    Create all visualizations in parallel using multiprocessing.
    
    Args:
        similarities_by_layer: Dict of layer similarities
        concept_names: List of concept names
        test_images: List of PIL images
        dataset_name: Dataset name
        model_input_size: Model input size tuple
        con_label: Concept label for naming
        selected_concepts: Optional filtered concepts
        patch_size: Patch size
        n_workers: Number of parallel workers (None = use all CPUs)
    """
    import multiprocessing
    
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(test_images) * 2)
    
    # Prepare save directory
    save_dir = f'../Figs/{dataset_name}/layer_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert similarities to CPU and to simple dict for pickling
    similarities_dict = {}
    for layer_pct, layer_sims in similarities_by_layer.items():
        similarities_dict[layer_pct] = {}
        for concept_name, sims in layer_sims.items():
            similarities_dict[layer_pct][concept_name] = sims.cpu()
    
    # Prepare all tasks
    tasks = []
    for img_idx, image in enumerate(test_images):
        # Convert PIL image to numpy for pickling
        image_data = np.array(image)
        
        # Global scale version
        tasks.append((
            similarities_dict, concept_names, img_idx, image_data,
            dataset_name, model_input_size, con_label, False,
            selected_concepts, patch_size, save_dir
        ))
        
        # Per-layer scale version
        tasks.append((
            similarities_dict, concept_names, img_idx, image_data,
            dataset_name, model_input_size, con_label, True,
            selected_concepts, patch_size, save_dir
        ))
    
    # Process in parallel
    print(f"Creating {len(tasks)} visualizations with {n_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(create_single_visualization, task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"  {result}")
            except Exception as e:
                print(f"  Error in visualization: {e}")
    
    print("All visualizations completed!")


def create_gpu_accelerated_heatmap(similarities, patch_dims, colormap='hot'):
    """
    Create heatmap entirely on GPU using PyTorch.
    
    Args:
        similarities: Tensor of similarities
        patch_dims: Tuple of (rows, cols) for reshaping
        colormap: Name of colormap to use
    
    Returns:
        RGBA image as numpy array
    """
    device = similarities.device
    
    # Reshape similarities
    heatmap = similarities.reshape(patch_dims)
    
    # Normalize to [0, 1]
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
    
    # Apply colormap approximation on GPU
    if colormap == 'hot':
        # Hot colormap approximation
        rgba = torch.zeros((*heatmap_norm.shape, 4), device=device)
        rgba[..., 0] = torch.clamp(heatmap_norm * 3, 0, 1)  # Red
        rgba[..., 1] = torch.clamp(heatmap_norm * 3 - 1, 0, 1)  # Green
        rgba[..., 2] = torch.clamp(heatmap_norm * 3 - 2, 0, 1)  # Blue
        rgba[..., 3] = 0.7  # Alpha
    else:
        # Default to grayscale
        rgba = torch.stack([heatmap_norm] * 3 + [torch.ones_like(heatmap_norm) * 0.7], dim=-1)
    
    return rgba.cpu().numpy()