import torch
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import pandas as pd
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image

from pycocotools.coco import COCO

from  .general_utils import load_images, retrieve_present_concepts
from .patch_alignment_utils import get_image_idx_from_global_patch_idx

### all-dataset purpose ###
def plot_seg_maps(dataset_name, input_image_size=(224, 224), img_idx=-1):
    data_dir = f'../Data/{dataset_name}/'
    metadata = pd.read_csv(f'{data_dir}/metadata.csv')
    n_images = len(metadata)
    
    # Select a random image index
    if img_idx < 0:
        img_idx = random.randint(0, n_images - 1)
    img_path = metadata.iloc[img_idx]['image_path']
    image = Image.open(f'{data_dir}/{img_path}')
                       
    # Plot the original image
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f"Image {img_idx}")
    ax.axis('off')

    # Retrieve the segmentation maps for the selected image
    seg_maps = retrieve_all_concept_segmentations(img_idx, dataset_name, input_image_size)
    
    # Prepare the segmentation maps for plotting
    # We will plot segmentation maps in groups of 3 per row
    num_maps = len(seg_maps)
    num_rows = (num_maps + 2) // 3  # Calculate the number of rows (3 maps per row)

    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    
    # Flatten axs array to make it easier to index
    axs = axs.flatten()
    
    # Resize the image to the input size
    resized_image = np.array(image.resize(input_image_size))
    
    # Loop through each segmentation map and plot it
    for i, (concept_key, seg_map) in enumerate(seg_maps.items()):
        # Create a binary mask where 1 is for the object area, and 0 is for background
        binary_mask = np.array(seg_map).astype(np.uint8)  # Convert to 0 (background) and 1 (object)

        # Create a masked image: set pixels to black where the mask is 0 (background)
        masked_image = np.zeros_like(resized_image)  # Create a black image as the background
        masked_image[binary_mask == 1] = resized_image[binary_mask == 1]  # Keep object pixels

        # Plot the image with the mask applied
        axs[i].imshow(masked_image)
        axs[i].set_title(f"{concept_key}")
        axs[i].axis('off')
    
    # Hide unused axes
    for i in range(num_maps, len(axs)):
        axs[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    
def retrieve_all_concept_segmentations(img_idx, dataset_name, input_image_size):
    if dataset_name == 'CLEVR':
        return retrieve_all_concept_segmentations_clevr(img_idx)
    elif dataset_name == 'Coco':
        return retrieve_all_concept_segmentations_coco(img_idx) 
    elif dataset_name == 'Surgery':
        return retrieve_all_concept_segmentations_surgery(img_idx) 
    


def sort_mapping_by_split(all_samples_per_concept, dataset_name, sample_type):
    # Load metadata
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    
    # Build a dictionary mapping sample indices to their split value.
    # Assuming the DataFrame index corresponds to the sample index.
    sample_to_split = metadata['split']
    
    # Initialize a regular dictionary to map split -> concept -> list of samples
    all_splits_dic = {}

    # Loop through each concept and its samples
    for concept, samples in all_samples_per_concept.items():
        for sample in samples:
            if sample_type == 'patch':  # Have to convert to image index if looking for patches
                img_idx = get_image_idx_from_global_patch_idx(sample)
            elif sample_type == 'image':
                img_idx = sample
            split = sample_to_split.iloc[img_idx]
            
            # Initialize the dictionaries if they do not exist yet
            if split not in all_splits_dic:
                all_splits_dic[split] = {}
            if concept not in all_splits_dic[split]:
                all_splits_dic[split][concept] = []
            
            # Append the sample to the appropriate place
            all_splits_dic[split][concept].append(sample)
    
    return all_splits_dic


def find_closest_to_gt(unsupervised_concepts, metrics, gt_patch_concepts, device):
    gt_metrics = pd.DataFrame()
    # Find the most aligned concept
    alignment_results = {}
    for gt_key, gt_embedding in gt_patch_concepts.items():
        max_similarity = float('-inf')
        best_match = None

        for unsupervised_key, unsupervised_embedding in unsupervised_concepts.items():
            # Compute cosine similarity using PyTorch's cosine_similarity
            similarity = F.cosine_similarity(gt_embedding.unsqueeze(0).to(device), unsupervised_embedding.unsqueeze(0).to(device)).item()

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = unsupervised_key

        alignment_results[gt_key] = (best_match, gt_embedding, max_similarity)
        gt_metrics[gt_key] = metrics[str(best_match)]
    alignment_results = dict(sorted(alignment_results.items(), key=lambda x: x[0]))
    return alignment_results, gt_metrics


### just for patches ###
def sort_patch_embeddings_by_concept(gt_patches_per_concept, embeddings, patch_size=14):
    """
    Maps patch indices to patch embeddings for each concept.

    Args:
        gt_patches_per_concept (dict): A dictionary where keys are concepts and values are lists of (image_idx, patch_idx).
        embeddings (torch.Tensor): A tensor containing the precomputed patch embeddings of shape (N, P, 1024),
            where N is the number of images, P is the number of patches (16x16), and 1024 is the embedding dimension.
        patch_size (int): Size of the patches (default is 14).

    Returns:
        dict: A dictionary where keys are concepts and values are lists of patch embeddings.
    """
    concept_embeddings = defaultdict(list)
    
    for concept, patches in gt_patches_per_concept.items():
        for idx in range(embeddings.shape[0]):
            if idx in patches:
                concept_embeddings[concept].append(embeddings[idx, :])
    return concept_embeddings


def map_concepts_to_image_indices(dataset_name):
    """
    Maps concepts to image indices based on a metadata CSV file.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        defaultdict(list): A dictionary where keys are concepts and values are lists of image indices 
                           that contain the respective concept.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    concepts = [col for col in metadata.columns if col != "image_path"]
    concept_to_images = defaultdict(list)
    for idx, row in metadata.iterrows():
        for concept in concepts:
            if row[concept] == 1:
                concept_to_images[concept].append(idx)
    sorted_concept_to_images = dict(sorted(concept_to_images.items()))
    return sorted_concept_to_images


def map_concepts_to_patch_indices(dataset_name, patch_size=14):
    """
    Maps concepts to patch indices based on object masks and metadata.

    Args:
        dataset_name (String): Name of dataset.
        patch_size (int): Size of the patches used to divide the images (default is 14).

    Returns:
        dict: A dictionary where keys are concepts, and values are lists of (image_idx, patch_idx).
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    
    n_images = len(metadata)
    h, w = 224, 224
    patches_per_row = w // patch_size

    concept_patch_indices = defaultdict(list)

    for idx, info in metadata.iterrows():
        active_concepts = [col for col in metadata.columns if col not in ['split', 'image_filename', 'class'] and info[col] == 1]
        concept_masks = retrieve_all_concept_segmentations(idx, dataset_name, (h, w))
        # object_mask = object_masks[idx, :, :]  # Shape: (224, 224)
        
        for concept in active_concepts:
            # Loop through all patches (16x16 grid for CLEVR)
            concept_mask = concept_masks[concept]
            for patch_idx in range(patches_per_row ** 2):
                row_idx = patch_idx // patches_per_row
                col_idx = patch_idx % patches_per_row
                i_start, j_start = row_idx * patch_size, col_idx * patch_size

                mask_window = concept_mask[i_start:i_start + patch_size, j_start:j_start + patch_size]
                if mask_window.sum() > 0:  # If patch contains part of the object
                    concept_patch_indices[concept].append((idx * (patches_per_row ** 2)) + patch_idx)

    sorted_concept_patch_indices = dict(sorted(concept_patch_indices.items()))
    return sorted_concept_patch_indices


### CLEVR ###
def extract_CLEVR_concept_pixels_batch(images, gray_tolerance=20):
    """
    Extract foreground masks for a batch of images by identifying non-grayscale pixels.

    Args:
        images (torch.Tensor): Batch of images with shape (N, H, W, C), where N is the batch size, 
                               H and W are image height and width, and C is the number of channels (3 for RGB).
        gray_tolerance (int): Tolerance value for determining if a pixel is grayscale. 
                              A pixel is considered grayscale if the absolute difference 
                              between any two color channels (R, G, B) is less than or equal to this value.

    Returns:
        torch.Tensor: Foreground masks with shape (N, H, W), where each mask is a binary image (1 for foreground, 0 for background).
    """
    r, g, b = images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]
    is_gray = (torch.abs(r - g) <= gray_tolerance) & \
              (torch.abs(g - b) <= gray_tolerance) & \
              (torch.abs(r - b) <= gray_tolerance)
    is_foreground = ~is_gray
    return is_foreground.to(dtype=torch.float32)

def compute_all_concept_masks_clevr(batch_size=32):
    """
    Compute foreground concept masks for a dataset of images in batches. Optionally, plot a subset of images with masks.

    Args:
        images (list[PIL.Image.Image]): List of images to process, where each image is a PIL Image.
        n_to_plot (int): Number of example images to plot with their corresponding masks (default: 5).
        batch_size (int): Number of images to process in each batch (default: 32).
        save_path (str): Where to save masks.

    Returns:
        torch.Tensor: A tensor of foreground masks for all images with shape (N, 224, 224), where N is the number of images.
    """
    images, _, _ = load_images(dataset_name='CLEVR')
    
    # Initialize an empty tensor for object masks
    num_images = len(images)
    object_masks = torch.zeros((num_images, 224, 224), dtype=torch.float32)
    
    # Convert PIL images to tensors
    print("Converting images to tensors...")
    image_tensors = [torch.tensor(np.array(img), dtype=torch.float32) for img in images]
    image_tensors = torch.stack(image_tensors)  # Shape: (N, H, W, C)
    
    # Process in batches
    num_batches = (len(images) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="batch"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_images)
        batch_images = image_tensors[batch_start:batch_end]  # Shape: (B, H, W, C)

        # Extract masks
        masks = extract_CLEVR_concept_pixels_batch(batch_images, gray_tolerance=20).cpu()  # Shape: (B, H, W)

        # Resize masks to (224, 224)
        resized_masks = F.interpolate(masks.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
        
        object_masks[batch_start:batch_end] = resized_masks
    
    #save concepts
    torch.save(object_masks, f'../Data/CLEVR/object_segmentations.pt')

    
def retrieve_all_concept_segmentations_clevr(img_idx, n_attributes=2):
    object_masks_all_images =  torch.load(f'../Data/CLEVR/object_segmentations.pt')
    object_mask = object_masks_all_images[img_idx]
    present_concepts = retrieve_present_concepts(img_idx, 'CLEVR')
    object_masks = {}
    for concept in present_concepts:
        object_masks[concept] = object_mask
    return object_masks

    
#### Coco #####
def retrieve_coco_concept_segmentation(img_idx, concept_key):
    """ Returns the segmentation map for a concept in an image"""
    # Path to COCO annotations file
    ann_file = '../Data/Coco/instances_val2017.json'

    # Initialize COCO API
    coco = COCO(ann_file)

    # Get the image ID using the index
    img_ids = coco.getImgIds()
    img_id = img_ids[img_idx]

    # Load the image metadata
    img_metadata = coco.loadImgs(img_id)[0]

    # Get the annotations (masks) for this image
    ann_ids = coco.getAnnIds(imgIds=img_metadata['id'], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # Get all categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    category_name_to_supercategory = {cat['name']: cat['supercategory'] for cat in cats}

    # Initialize an empty list to store masks for the requested concept
    concept_masks = []

    # Check if concept_key is a supercategory or category
    if concept_key in category_name_to_supercategory.values():  # It's a supercategory
        # Loop through annotations and get masks for all categories in the supercategory
        for ann in anns:
            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            supercategory_name = category_name_to_supercategory[category_name]

            if supercategory_name == concept_key:
                # Get the mask for this annotation
                seg_map = coco.annToMask(ann)
                concept_masks.append(seg_map)

    else:  # It's a category
        # Loop through annotations and get masks for the requested category
        for ann in anns:
            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']

            if category_name == concept_key:
                # Get the mask for this annotation
                seg_map = coco.annToMask(ann)
                concept_masks.append(seg_map)

    # If no matching masks are found, return None or handle it as needed
    if not concept_masks:
        return None

    # Combine all masks for the requested concept (if there are multiple)
    combined_mask = concept_masks[0]
    for mask in concept_masks[1:]:
        combined_mask |= mask  # Combine by OR operation

    return combined_mask

def retrieve_all_concept_segmentations_coco(img_idx, input_image_size=(224, 224)):
    # Path to COCO annotations file
    ann_file = '../Data/Coco/instances_val2017.json'

    # Initialize COCO API
    coco = COCO(ann_file)

    # Get the image ID using the index
    img_ids = coco.getImgIds()
    img_id = img_ids[img_idx]

    # Load the image metadata
    img_metadata = coco.loadImgs(img_id)[0]

    # Get the annotations (masks) for this image
    ann_ids = coco.getAnnIds(imgIds=img_metadata['id'], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # Get all categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    category_name_to_supercategory = {cat['name']: cat['supercategory'] for cat in cats}

    # Initialize a dictionary to store segmentation maps by concept
    concept_seg_maps = {}

    # Loop through annotations and retrieve masks for each category/supercategory present in the image
    for ann in anns:
        category_id = ann['category_id']
        category_name = coco.loadCats(category_id)[0]['name']
        supercategory_name = category_name_to_supercategory[category_name]

        # Get the mask for this annotation and add it to both the category and supercategory lists
        seg_map = coco.annToMask(ann)
        
        # Resize the segmentation map to the input image size
        seg_map_pil = Image.fromarray(seg_map)  # Convert to PIL image for resizing
        seg_map_resized = seg_map_pil.resize(input_image_size, Image.NEAREST)  # Resize to input size

        # Convert back to NumPy array after resizing
        resized_mask = np.array(seg_map_resized)

        # Combine masks for the same category
        if category_name in concept_seg_maps:
            concept_seg_maps[category_name] = np.logical_or(concept_seg_maps[category_name], resized_mask)
        else:
            concept_seg_maps[category_name] = resized_mask

        # Combine masks for the same supercategory
        if supercategory_name in concept_seg_maps:
            concept_seg_maps[supercategory_name] = np.logical_or(concept_seg_maps[supercategory_name], resized_mask)
        else:
            concept_seg_maps[supercategory_name] = resized_mask


    # If no masks were found for a concept, it will not be included in the dictionary
    return concept_seg_maps


###### Surgery #######
def retrieve_all_concept_segmentations_surgery(img_idx, input_image_size=(224, 224)):
    organs = {0: 'background', 1: 'liver', 2: 'gallbladder', 3: 'hepatocystic_triangle'}
    all_segmentations = torch.load('../Data/Surgery/segmentations.pt')
    img_segmentations = all_segmentations[img_idx]

    concept_segmentations = {}
    for organ_num, organ in organs.items():
        # Create a binary mask where the organ is present (1s where organ_num is found, else 0s)
        organ_seg = (img_segmentations == organ_num).to(torch.uint8)

        # Resize to input_image_size if needed
        if organ_seg.shape != input_image_size:
            organ_seg = F.interpolate(organ_seg.unsqueeze(0).unsqueeze(0).float(), size=input_image_size, mode="nearest").squeeze(0).squeeze(0).to(torch.uint8)
            
        concept_segmentations[organ] = organ_seg

    return concept_segmentations
    
     
    