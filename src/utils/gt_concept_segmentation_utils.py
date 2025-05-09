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

import importlib
import general_utils
importlib.reload(general_utils)
import patch_alignment_utils
importlib.reload(patch_alignment_utils)
from general_utils import load_images, retrieve_present_concepts, pad_or_resize_img_tensor, filter_coco_concepts, get_split_df
from patch_alignment_utils import get_image_idx_from_global_patch_idx, get_patch_split_df

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
    seg_maps = retrieve_all_concept_segmentations(img_idx, dataset_name)
    
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
    
    
def retrieve_all_concept_segmentations(img_idx, dataset_name):
    if dataset_name == 'CLEVR':
        return retrieve_all_concept_segmentations_clevr(img_idx)
    elif dataset_name == 'Coco':
        return retrieve_all_concept_segmentations_coco(img_idx) 
    elif dataset_name == 'Surgery':
        return retrieve_all_concept_segmentations_surgery(img_idx) 
    


def sort_mapping_by_split(all_samples_per_concept, dataset_name, sample_type, model_input_size):
    if sample_type =='patch':
        split_df = get_patch_split_df(dataset_name, model_input_size)
    else:
        split_df = get_split_df(dataset_name)
    
    # Initialize a regular dictionary to map split -> concept -> list of samples
    all_splits_dic = {}

    # Loop through each concept and its samples
    for concept, samples in all_samples_per_concept.items():
        for sample in samples:
            split = split_df.iloc[sample]
            
            # Initialize the dictionaries if they do not exist yet
            if split not in all_splits_dic:
                all_splits_dic[split] = {}
            if concept not in all_splits_dic[split]:
                all_splits_dic[split][concept] = []
            
            # Append the sample to the appropriate place
            all_splits_dic[split][concept].append(sample)
    
    gt_samples_per_concept_train = all_splits_dic['train']
    gt_samples_per_concept_test = all_splits_dic['test']
    torch.save(gt_samples_per_concept_train, f'GT_Samples/{dataset_name}/gt_{sample_type}_per_concept_train_inputsize_{model_input_size}.pt')
    torch.save(gt_samples_per_concept_test, f'GT_Samples/{dataset_name}/gt_{sample_type}_per_concept_test_inputsize_{model_input_size}.pt')
    print(f'train and test mappings saved :)')
    return gt_samples_per_concept_train, gt_samples_per_concept_test
    


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


def map_concepts_to_image_indices(dataset_name, model_input_size):
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
    if dataset_name == 'Coco':
        concepts = filter_coco_concepts(concepts)
    
    concept_to_images = defaultdict(list)
    for idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        for concept in concepts:
            if row[concept] == 1:
                concept_to_images[concept].append(idx)
    sorted_concept_to_images = dict(sorted(concept_to_images.items()))
    
    torch.save(sorted_concept_to_images, f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt')
    print(f"Concept to image dic saved to'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt :)")
    return sorted_concept_to_images


def map_concepts_to_patch_indices(dataset_name, model_input_size, patch_size=14):
    """
    Maps concepts to patch indices based on object masks and metadata.

    Args:
        dataset_name (String): Name of dataset.
        patch_size (int): Size of the patches used to divide the images (default is 14).

    Returns:
        dict: A dictionary where keys are concepts, and values are lists of (image_idx, patch_idx).
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    if dataset_name == 'Coco':
        curr_concepts = filter_coco_concepts(metadata.columns)
    
    n_images = len(metadata)
    h, w = model_input_size
    patches_per_row = w // patch_size

    concept_patch_indices = defaultdict(list)

    for idx, info in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        active_concepts = [col for col in metadata.columns if col in curr_concepts and info[col] == 1]
        concept_masks = retrieve_all_concept_segmentations(idx, dataset_name)
        for concept in active_concepts:
            # Loop through all patches
            concept_mask = concept_masks[concept]
            resized_concept_mask = pad_or_resize_img_tensor(concept_mask, model_input_size) #rezise to fit model input size w same aspect
            for patch_idx in range(patches_per_row ** 2):
                row_idx = patch_idx // patches_per_row
                col_idx = patch_idx % patches_per_row
                i_start, j_start = row_idx * patch_size, col_idx * patch_size

                mask_window = resized_concept_mask[i_start:i_start + patch_size, j_start:j_start + patch_size]
                if torch.any(mask_window == 1):  # If patch contains part of the object
                    global_patch_index = (idx * (patches_per_row ** 2)) + patch_idx
                    concept_patch_indices[concept].append(global_patch_index)

    sorted_concept_patch_indices = dict(sorted(concept_patch_indices.items()))
    
    torch.save(sorted_concept_patch_indices, f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
    print(f"Concept to patch dic saved to'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt :)")
    return sorted_concept_patch_indices

def compute_attention_masks(all_text_samples, processor, dataset_name, model_input_size):
    tokens_list = []
    relevant_tokens_list = []
    for text_samples in tqdm(all_text_samples):
        inputs = processor(
                    None,  # No image input
                    text_samples,  # Text input
                    add_special_tokens=False,
                    padding=True,
                    token='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',
                    return_tensors="pt"
                )

        tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
        tokens_list.append(tokens)
        for token in tokens:
            if token == '<|begin_of_text|>':
                relevant_tokens_list.append(0)
            else:
                relevant_tokens_list.append(1)
                
    relevant_tokens = torch.tensor(relevant_tokens_list)
    torch.save(tokens_list, f'GT_Samples/{dataset_name}/tokens.pt')
    torch.save(relevant_tokens, f'GT_Samples/{dataset_name}/patches_w_image_mask_inputsize_{model_input_size}.pt')
    return tokens_list, relevant_tokens

def map_sentences_to_concept_gt_jailbreak(dataset_name, model_input_size):
    concept_to_sentences = defaultdict(list)
    concept_to_sentences_train = defaultdict(list)
    concept_to_sentences_test = defaultdict(list)
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    for idx, row in metadata.iterrows():
        class_as_list = row['class'].split()
        if class_as_list[0] == 'benign': #might need to change this if you change how you get concepts
            continue
        concept = " ".join(class_as_list[1:])
        concept_to_sentences[concept].append(idx)
        if row['split'] == 'train':
            concept_to_sentences_train[concept].append(idx)
        elif row['split'] == 'test':
            concept_to_sentences_test[concept].append(idx)
        
    
    sorted_concept_to_sentences = dict(sorted(concept_to_sentences.items()))
    sorted_concept_to_sentences_train = dict(sorted(concept_to_sentences_train.items()))
    sorted_concept_to_sentences_test = dict(sorted(concept_to_sentences_test.items()))
    torch.save(sorted_concept_to_sentences, f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt')
    torch.save(sorted_concept_to_sentences_train, f'GT_Samples/{dataset_name}/gt_samples_per_concept_train_inputsize_{model_input_size}.pt')
    torch.save(sorted_concept_to_sentences_test, f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt')
    print(f"Concept to sentence dic saved to 'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt'")
    return sorted_concept_to_sentences, sorted_concept_to_sentences_train, sorted_concept_to_sentences_test


def map_sentence_to_concept_gt(dataset_name, model_input_size):
    """
    Converts word-level concept labels to token-level indices per concept using pre-tokenized sentences.
    Also maps concepts to sentence indices.
    
    Returns:
        - concept_to_token_indices, concept_to_token_indices_train, concept_to_token_indices_test
        - concept_to_sentence_indices, concept_to_sentence_indices_train, concept_to_sentence_indices_test
    """
    df = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    tokenized_sentences = torch.load(f'GT_Samples/{dataset_name}/tokens.pt')

    label_columns = [col for col in df.columns if col not in ['review_index', 'word', 'sample_filename', 'split']]

    concept_to_token_indices = defaultdict(list)
    concept_to_token_indices_train = defaultdict(list)
    concept_to_token_indices_test = defaultdict(list)

    concept_to_sentence_indices = defaultdict(set)
    concept_to_sentence_indices_train = defaultdict(set)
    concept_to_sentence_indices_test = defaultdict(set)

    current_token_index = 0
    grouped = df.groupby("sample_filename")
    assert len(grouped) == len(tokenized_sentences), "Mismatch between metadata and tokenized sentences"

    for sent_idx, ((_, group), tokens) in enumerate(tqdm(zip(grouped, tokenized_sentences), total=len(tokenized_sentences), desc="Mapping concepts to token indices")):
        split = group["split"].iloc[0]

        # Assume one token per word by default if lengths are not matched
        token_counts = [1 for _ in group["word"]]

        token_idx = current_token_index
        for (_, row), n_tokens in zip(group.iterrows(), token_counts):
            for concept in label_columns:
                if row[concept] == 1:
                    indices = list(range(token_idx, token_idx + n_tokens))
                    concept_to_token_indices[concept].extend(indices)
                    concept_to_sentence_indices[concept].add(sent_idx)

                    if split == "train":
                        concept_to_token_indices_train[concept].extend(indices)
                        concept_to_sentence_indices_train[concept].add(sent_idx)
                    elif split == "test":
                        concept_to_token_indices_test[concept].extend(indices)
                        concept_to_sentence_indices_test[concept].add(sent_idx)
            token_idx += n_tokens

        current_token_index += len(tokens)

    out_dir = f'GT_Samples/{dataset_name}'
    os.makedirs(out_dir, exist_ok=True)

    def save_and_sort(d, name, prefix='gt_tokens_per_concept'):
        sorted_d = dict(sorted(d.items()))
        torch.save(sorted_d, f"{out_dir}/{prefix}{name}_inputsize_{model_input_size}.pt")
        return sorted_d

    # Token index outputs
    d_all = save_and_sort(concept_to_token_indices, "")
    d_train = save_and_sort(concept_to_token_indices_train, "_train")
    d_test = save_and_sort(concept_to_token_indices_test, "_test")

    # Sentence index outputs
    s_all = save_and_sort({k: sorted(v) for k, v in concept_to_sentence_indices.items()}, "", prefix='gt_sentence_samples_per_concept')
    s_train = save_and_sort({k: sorted(v) for k, v in concept_to_sentence_indices_train.items()}, "_train", prefix='gt_sentence_samples_per_concept')
    s_test = save_and_sort({k: sorted(v) for k, v in concept_to_sentence_indices_test.items()}, "_test", prefix='gt_sentence_samples_per_concept')

    print(f"✅ Saved token and sentence-level concept indices to {out_dir}")
    return d_all, d_train, d_test, s_all, s_train, s_test
        

def map_concepts_to_token_indices(dataset_name, tokens_list, relevant_tokens, model_input_size):
    """
    Maps concepts to image indices based on a metadata CSV file.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        defaultdict(list): A dictionary where keys are concepts and values are lists of image indices 
                           that contain the respective concept.
    """
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')
    
    
    concepts = [col for col in metadata.columns if col != "sample_filename"]
    concept_to_tokens = defaultdict(list)
    
    overall_idx = 0
    for sentence_idx, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        for token in tokens_list[sentence_idx]:
            for concept in concepts:
                if row[concept] == 1 and relevant_tokens[overall_idx] == 1: #if concept is present and not special token
                    concept_to_tokens[concept].append(overall_idx)
            overall_idx +=1
            
    sorted_concept_to_tokens = dict(sorted(concept_to_tokens.items()))
    
    torch.save(sorted_concept_to_tokens, f'GT_Samples/{dataset_name}/gt_patch_per_concept_inputsize_{model_input_size}.pt')
    print(f"Concept to token dic saved to 'GT_Samples/{dataset_name}/gt_patch_per_concept_inputsize_{model_input_size}.pt'")
    return sorted_concept_to_tokens
    


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
    object_masks = []
    
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

        # # pad masks to match the model input size
        # padded_masks = pad_tensors_to_size(masks, input_model_size)
        for i in range(masks.shape[0]):
            object_masks.append(masks[i, :, :])
    
    #save concepts
    torch.save(object_masks, f'../Data/CLEVR/object_segmentations.pt')
    print(f'Masks saved to ../Data/CLEVR/object_segmentations.pt')

    
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

def retrieve_all_concept_segmentations_coco(img_idx):
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
        # seg_map_pil = Image.fromarray(seg_map)  # Convert to PIL image for resizing
        # seg_map_resized = seg_map_pil.resize(input_image_size, Image.NEAREST)  # Resize to input size

        # Convert back to NumPy array after resizing
        # resized_mask = np.array(seg_map_resized)

        # Combine masks for the same category
        if category_name in concept_seg_maps:
            concept_seg_maps[category_name] = np.logical_or(concept_seg_maps[category_name], seg_map)
        else:
            concept_seg_maps[category_name] = seg_map

        # Combine masks for the same supercategory
        if supercategory_name in concept_seg_maps:
            concept_seg_maps[supercategory_name] = np.logical_or(concept_seg_maps[supercategory_name], seg_map)
        else:
            concept_seg_maps[supercategory_name] = seg_map


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
    
     
    