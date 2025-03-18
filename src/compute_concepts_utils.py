"""Utils for Computing Concepts"""

from tqdm import tqdm
import os
import pandas as pd
import random
import numpy as np
import math
from collections import defaultdict


import torch
from torchvision import transforms
from sklearn.metrics import mean_squared_error, f1_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
# import cupy as cp 
# from cuml.cluster import KMeans as cuml_kmeans 
from fast_pytorch_kmeans import KMeans

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle
from IPython.display import display, clear_output

from .general_utils import retrieve_image, load_images, get_split_df
from .patch_alignment_utils import get_patch_split_df

### For Computing Embeddings ###
def get_final_cls_embeddings(model, processor, images, device):
    """
    Extracts class token embeddings at final layer of given model for each image.

    Args:
        model: The CLIP model to generate embeddings.
        processor: The processor used for transforming the images and text.
        images: A list of PIL.Image objects.
        device: The device to move the tensors to.

    Returns:
        torch.Tensor: The generated image embeddings.
    """
    # Preprocess the images for the model (passing PIL Images directly)
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = inputs.to(device)

    # Generate image embeddings without computing gradients
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])  # Extract class token embeddings
    
    return image_features

def get_intermediate_representations(model, processor, images, device, percent_thru_model):
    """
    Extracts embeddings from chosen layer of given model for each patch and class token in each image.

    Args:
        model: The CLIP model to generate embeddings.
        processor: The processor used for transforming the images and text.
        images: A list of PIL.Image objects.
        device: The device to move the tensors to.
        percent_thru_model (int) : For patch concepts, percentage through model intermediate rep is extracted from.

    Returns:
        torch.Tensor: The generated embeddings per patch per image.
    """
    # Extracts embeddings from a specific layer of the CLIP image encoder. (will need to change if you use different model)
    layer_index = int(len(model.vision_model.encoder.layers) * (percent_thru_model/100)) - 1 
    layer_output = []
    
    def hook(module, input, output):
        layer_output.append(output)
    
    # Register the hook to the specified layer
    layer = model.vision_model.encoder.layers[layer_index]
    handle = layer.register_forward_hook(hook)
    
    # Preprocess the image and convert it to tensor format
    processed_images = processor(images=images, return_tensors="pt", padding=True).to(device)
    
    # The output of the hook contains the patch embeddings
    with torch.no_grad():
        model.get_image_features(pixel_values=processed_images['pixel_values'])
    all_embeddings = layer_output[0][0]
    handle.remove()
    return all_embeddings

def get_clip_cls_embeddings(model, processor, images, device, percent_thru_model):
    """
    Extracts class token embeddings at final layer of given model for each image.

    Args:
        model: The CLIP model to generate embeddings.
        percent_thru_model (int): Percent through model that embeddings are taken from.
        processor: The processor used for transforming the images and text.
        images: A list of PIL.Image objects.
        device: The device to move the tensors to.
        percent_thru_model (int): Percent through model that embedding is extracted from.

    Returns:
        torch.Tensor: The generated image embeddings.
    """
    if percent_thru_model == 100:
        return get_final_cls_embeddings(model, processor, images, device)
    else:
        all_embeddings = get_intermediate_representations(model, processor, images, device, percent_thru_model)
        #remove class token
        cls_embeddings = all_embeddings[:, 0, :]
    return cls_embeddings

def get_clip_patch_embeddings(model, processor, images, device, percent_thru_model):
    """
    Extracts embeddings from chosen layer of given model for each patch in each image.

    Args:
        model: The CLIP model to generate embeddings.
        processor: The processor used for transforming the images and text.
        images: A list of PIL.Image objects.
        device: The device to move the tensors to.
        percent_thru_model (int) : For patch concepts, percentage through model intermediate rep is extracted from.

    Returns:
        torch.Tensor: The generated embeddings per patch per image.
    """
    all_embeddings = get_intermediate_representations(model, processor, images, device, percent_thru_model)
   
    #remove class token
    patch_embeddings = all_embeddings[:, 1:, :]
    
    #flatten image dimension so its (n_patches, embed_dim)
    patch_embeddings = patch_embeddings.reshape(-1, patch_embeddings.size(-1))
    
    return patch_embeddings


def get_llama_cls_embeddings(model, processor, images, device, percent_thru_model=100):
    """
    Extracts llama cls embeddings for each image.

    Args:
        model: The LLAMA model to generate embeddings.
        processor: The processor used for transforming the images and text.
        images: A list of PIL.Image objects.
        device: The device to move the tensors to.
        percent_thru_model (int) : Not used, for now
    Returns:
        torch.Tensor: The generated embeddings per patch per image.
    """
    inputs = processor(images,
                        add_special_tokens=False,
                        return_tensors="pt").to(device)
    
    #get embeddings from vision model
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"],
            aspect_ratio_ids=inputs["aspect_ratio_ids"],
            aspect_ratio_mask=inputs["aspect_ratio_mask"],
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True
        )
        cross_attention_states = vision_outputs[0]
        cross_attention_states = model.multi_modal_projector(cross_attention_states).reshape(
            -1, cross_attention_states.shape[-2], model.hidden_size
        )
    
    # Initialize storage for embeddings (excluding the class token)
    all_embs = []
    for i in range(0, len(images) * 4, 4):  # Only look at first tile
        curr_emb = cross_attention_states[i, :, :]  
        embs_no_cls = curr_emb[-1, :]  # just get the last token (cls token)
        all_embs.append(embs_no_cls) 

    # Stack collected embeddings into a single tensor
    cls_embeddings = torch.stack(all_embs)
    
    return cls_embeddings


def get_llama_patch_embeddings(model, processor, images, device, percent_thru_model=100):
    """
    Extracts llama patch embeddings for each image.

    Args:
        model: The LLAMA model to generate embeddings.
        processor: The processor used for transforming the images and text.
        images: A list of PIL.Image objects.
        device: The device to move the tensors to.
        percent_thru_model (int) : Not used, for now
    Returns:
        torch.Tensor: The generated embeddings per patch per image.
    """
    inputs = processor(images,
                        add_special_tokens=False,
                        return_tensors="pt").to(device)
    
    #get embeddings from vision model
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"],
            aspect_ratio_ids=inputs["aspect_ratio_ids"],
            aspect_ratio_mask=inputs["aspect_ratio_mask"],
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True
        )
        cross_attention_states = vision_outputs[0]
        cross_attention_states = model.multi_modal_projector(cross_attention_states).reshape(
            -1, cross_attention_states.shape[-2], model.hidden_size
        )
    
    # Initialize storage for embeddings (excluding the class token)
    all_embs = []
    for i in range(0, len(images) * 4, 4):  # Only look at first tile
        curr_emb = cross_attention_states[i, :, :]  
        embs_no_cls = curr_emb[:-1, :]  # Exclude the last token (cls token)
        all_embs.append(embs_no_cls) 

    # Stack collected embeddings into a single tensor
    patch_embeddings = torch.stack(all_embs)
    
    return patch_embeddings



# def get_surgery_embeddings(model, processor, images, device, percent_thru_model):
#     processed_images = processor(images).to(device)

#     with torch.no_
#     output = model(processed_images, mode='video')['img_emb']

#     return output


def compute_raw_batch_embeddings(images, embedding_fxn, model, processor, device, 
                                 percent_thru_model, dataset_name, batch_size=100):
    """
    Compute raw embeddings for images in batches and split into train/test sets.

    Args:
        images (list): List of PIL.Image objects.
        embedding_fxn (function): Function to compute embeddings.
        model: Model for generating embeddings.
        processor: Processor for preprocessing images.
        device: Device to run the model on.
        percent_thru_model (int): Model layer from which to extract representations.
        dataset_name (str): Dataset name (used for loading metadata).
        batch_size (int): Number of images per batch.

    Returns:
        tuple: (train_embeddings, test_embeddings)
    """
    print("Computing embeddings in batches...")
    embeddings = []
    n_batches = (len(images) + batch_size - 1) // batch_size  # Compute number of batches
    for i in tqdm(range(n_batches), desc="Computing embeddings"):
        batch_images = images[i * batch_size:(i + 1) * batch_size]
        batch_embeddings = embedding_fxn(model, processor, batch_images, device, percent_thru_model)
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0).cpu()  # Concatenate all batch embeddings
    print(f"Extracted embeddings of shape: {embeddings.shape}")
    return embeddings


def sort_embeddings_by_split(embeddings, dataset_name):
    """
    Sort embeddings into train and test sets based on metadata split.

    Args:
        embeddings (torch.Tensor): The embeddings to split.
        dataset_name (str): The name of the dataset.

    Returns:
        tuple: (train_embeddings, test_embeddings)
    """
    # Load metadata
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')

    # Compute number of embeddings per image
    n_embeddings_per_sample = embeddings.shape[0] // len(metadata)  # Ensure integer division

    # Create boolean masks for train/test
    train_mask = metadata["split"] == "train"
    test_mask = metadata["split"] == "test"

    # Repeat each split mask for all patches per image
    train_mask = train_mask.repeat(n_embeddings_per_sample).to_numpy()
    test_mask = test_mask.repeat(n_embeddings_per_sample).to_numpy()

    # Apply masks to embeddings
    train_embeddings = embeddings[train_mask]
    test_embeddings = embeddings[test_mask]

    return train_embeddings, test_embeddings

def compute_train_avg_and_norm(embeddings, dataset_name):
    # Load metadata
    metadata = pd.read_csv(f'../Data/{dataset_name}/metadata.csv')

    # Compute number of embeddings per image
    n_embeddings_per_sample = embeddings.shape[0] // len(metadata) 

    # Create boolean masks for train/test
    train_mask = metadata["split"] == "train"

    # Repeat each split mask for all patches per image
    train_mask = train_mask.repeat(n_embeddings_per_sample).to_numpy()

    # Apply masks to embeddings
    train_embeddings = embeddings[train_mask]
    
    mean_train_embedding = train_embeddings.mean(dim=0)
    train_norm = train_embeddings.norm(dim=1, keepdim=True).mean()
    return mean_train_embedding, train_norm
    

def center_and_normalize_embeddings(embeddings, dataset_name):
    """
    Center and normalize embeddings using statistics from the training set.

    Args:
        train_embeddings (torch.Tensor): Tensor of training set embeddings.
        test_embeddings (torch.Tensor): Tensor of test set embeddings.

    Returns:
        tuple: (normalized_train_embeddings, normalized_test_embeddings)
    """
    mean_train_embedding, train_norm = compute_train_avg_and_norm(embeddings, dataset_name)
    centered_embeddings = embeddings - mean_train_embedding
    norm_embeddings = centered_embeddings / train_norm

    return norm_embeddings, mean_train_embedding, train_norm


def compute_batch_embeddings(images, embedding_fxn, model, processor, device, 
                             percent_thru_model, dataset_name,
                             embeddings_file=None, batch_size=100):
    """
    Compute, center, and normalize embeddings for images.

    Args:
        images (list): List of PIL.Image objects.
        embedding_fxn (function): Function to compute embeddings.
        model: Model for generating embeddings.
        processor: Processor for preprocessing images.
        device: Device to run the model on.
        percent_thru_model (int): Model layer from which to extract representations.
        dataset_name (str): Dataset name.
        embeddings_file (str): Path to save embeddings.
        batch_size (int): Number of images per batch.

    Returns:
        tuple: (normalized_train_embeddings, normalized_test_embeddings)
    """
    embeddings = compute_raw_batch_embeddings(
        images, embedding_fxn, model, processor, device, percent_thru_model, dataset_name, batch_size
    )
    
    norm_embeddings, mean_train_embedding, train_norm = center_and_normalize_embeddings(embeddings, dataset_name)

    embeds_dic = {
            'normalized_embeddings': norm_embeddings,
            'mean_train_embedding': mean_train_embedding,
            'train_norm': train_norm
        }
    if embeddings_file:
        output_file = f'Embeddings/{dataset_name}/{embeddings_file}'
        # Save as a dictionary
        torch.save(embeds_dic, output_file)
        print(f"Embeddings saved to {output_file} :)")

    return embeds_dic
    

### For Computing Concept Vectors ###
def run_fast_pytorch_kmeans(n_clusters, train_embeddings, test_embeddings, device):
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=True, max_iter=10000, tol=1e-6)
    
    # Fit KMeans on training embeddings and predict cluster labels
    print(f"Fitting KMeans for {n_clusters} clusters...")
    train_labels = kmeans.fit_predict(train_embeddings.to(device))
    test_labels = kmeans.predict(test_embeddings.to(device))
    
    # Retrieve cluster centers
    cluster_centers = kmeans.centroids
    return train_labels, test_labels, cluster_centers
    

def gpu_kmeans(n_clusters, embeddings, dataset_name, device, concepts_filename=None, sample_type ='patch'):
    """
    Performs GPU-accelerated KMeans clustering on embeddings and saves the cluster centers.

    Args:
        n_clusters (int): Number of clusters for KMeans.
        dataset_name (str): Name of the dataset.
        train_embeddings (torch.Tensor): Training embeddings for clustering.
        test_embeddings (torch.Tensor): Test embeddings for cluster assignment.
        concepts_filename (str, optional): Filename to save the concepts.

    Returns:
        dict: Mapping of cluster labels to cluster centers.
        dict: Mapping of cluster labels to sample indices from training embeddings.
        dict: Mapping of cluster labels to sample indices from test embeddings.
    """
    #separate embeddings into test and train
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name)
    elif sample_type == 'image':
        split_df = get_split_df(dataset_name)
    train_image_indices = split_df[split_df == 'train'].index
    test_image_indices = split_df[split_df == 'test'].index
    train_embeddings = embeddings[train_image_indices]
    test_embeddings = embeddings[test_image_indices]
    
    train_labels, test_labels, cluster_centers = run_fast_pytorch_kmeans(n_clusters, 
                                                                         train_embeddings, 
                                                                         test_embeddings, 
                                                                         device)
    # Map cluster labels to cluster centers
    label_to_center = {str(label): center.cpu() for label, center in enumerate(cluster_centers)}

    # Store dictionary mapping each concept to all of the training samples that are in that cluster
    train_cluster_to_samples = defaultdict(list)
    for i, train_idx in enumerate(train_image_indices):
        cluster_label = str(train_labels[i].item())
        train_cluster_to_samples[cluster_label].append(train_idx)
    train_cluster_to_samples = dict(sorted(train_cluster_to_samples.items()))

    # for idx, label in enumerate(train_labels):
    #     train_cluster_to_samples[label.item()].append(idx)
    # train_cluster_to_samples = dict(sorted(train_cluster_to_samples.items()))

    # Store dictionary mapping each concept to all of the testing samples that are in that cluster
    test_cluster_to_samples = defaultdict(list)
    # for idx, label in enumerate(test_labels):
    #     test_cluster_to_samples[label.item()].append(idx)
    # test_cluster_to_samples = dict(sorted(test_cluster_to_samples.items()))
    for i, test_idx in enumerate(test_image_indices):
        cluster_label = str(test_labels[i].item())
        test_cluster_to_samples[cluster_label].append(test_idx)
    test_cluster_to_samples = dict(sorted(test_cluster_to_samples.items()))

    # Save results if filename is provided
    if concepts_filename:
        base_path = f'Concepts/{dataset_name}/'
        torch.save(label_to_center, f'{base_path}{concepts_filename}')
        torch.save(train_cluster_to_samples, f'{base_path}train_samples_{concepts_filename}')
        torch.save(test_cluster_to_samples, f'{base_path}test_samples_{concepts_filename}')
        print(f"Saved cluster centers and sample indices to {base_path} :)")

    print("Loaded concepts :)")
    return label_to_center, train_cluster_to_samples, test_cluster_to_samples


def aggregate_concept_vectors(concept_embeddings, dataset_name, save_file=None):
    """
    Aggregates concept embeddings by computing their mean, centers, and normalizes the concept vectors.

    Args:
        concept_embeddings (dict): A dictionary where keys are concept names and values are lists 
            of embeddings corresponding to those concepts.
        dataset_name (str): The name of the dataset. 
        save_file (str, optional): File name to save the aggregated concept vectors. If None, the vectors 
            will not be saved. Defaults to None.

    Returns:
        dict: A dictionary of processed concept vectors, where each vector is mean-centered and normalized.
    """
    # Compute mean vectors for each concept
    concept_vectors = {concept: torch.stack(embeddings).mean(dim=0) 
                       for concept, embeddings in concept_embeddings.items()}
    
    # Stack all mean vectors to compute global statistics
    all_vectors = torch.stack(list(concept_vectors.values()))
    
    # Compute global mean and L2 norm
    global_mean = all_vectors.mean(dim=0)
    global_l2_norm = all_vectors.norm(dim=1, keepdim=True).mean()  # Mean L2 norm of all concept vectors
    
    # Normalize all concept vectors based on global L2 norm
    for concept, vector in concept_vectors.items():
        # Normalize by the global L2 norm 
        normalized_vector = vector / global_l2_norm
        concept_vectors[concept] = normalized_vector

    # Save to file if specified
    if save_file:
        save_path = f'Concepts/{dataset_name}/{save_file}'
        torch.save(concept_vectors, save_path)
        print(f"Concept vectors saved to {save_path}")
    
    return concept_vectors


def compute_avg_concept_vectors(gt_samples_per_concept_train, embeddings, dataset_name=None, output_file=None):
    """
    Computes the average concept vectors by aggregating the embeddings of samples 
    belonging to each concept. Optionally normalizes the vectors.

    Args:
        gt_samples_per_concept_train (dict): A dictionary where keys are concept names 
                                             and values are lists of sample indices.
        embeddings (torch.Tensor): A tensor containing embeddings where rows correspond 
                                   to samples and columns correspond to feature dimensions.
        normalize (bool, optional): If True, normalizes the computed concept vectors to 
                                    have unit norm. Defaults to True.

    Returns:
        dict: A dictionary where keys are concept names and values are the computed 
              mean (and optionally normalized) concept vectors.
    """
    concepts = {}
    
    for concept, samples in gt_samples_per_concept_train.items():
        concept_embeddings = [embeddings[sample, :] for sample in samples]  # Collect embeddings
        
        concept_tensor = torch.stack(concept_embeddings)  # Convert to tensor
        avg_vector = torch.mean(concept_tensor, dim=0)  # Compute mean

        avg_vector = avg_vector / avg_vector.norm()  # Normalize to unit norm

        concepts[concept] = avg_vector
    
    if output_file:
        torch.save(concepts, f'Concepts/{dataset_name}/{output_file}')
    return concepts


def compute_a_linear_separator(concept, embeds, concept_gt_patches, split_df, lr=0.01, epochs=100, patience=15, tolerance=3, batch_size=32, device='cuda'):
    """
    Compute a linear separator for a given concept with early stopping, W&B logging, and test metrics tracking.

    Args:
        concept (str): Concept name.
        embeds (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
        concept_gt_patches (set): Set of indices where the concept is present.
        split_df (pd.DataFrame): DataFrame containing split information ('train'/'test').
        lr (float): Learning rate.
        epochs (int): Number of epochs.
        patience (int): Early stopping patience.
        device (str): Device to use.

    Returns:
        torch.Tensor: Learned concept representation (weight vector).
        dict: Combined logs for train and test metrics.
    """ 
    print(f"Training linear classifier for concept {concept}")

    # Create binary labels (1 if concept present, 0 otherwise)
    concept_labels = torch.zeros(embeds.shape[0], device=device)
    concept_labels[list(concept_gt_patches)] = 1

    # Get training samples
    train_mask = split_df == 'train'
    train_embeds = embeds[train_mask.to_numpy()].to(device)
    train_labels = concept_labels[train_mask.to_numpy()].to(device)

    # Get test samples
    test_mask = split_df == 'test'
    test_embeds = embeds[test_mask.to_numpy()].to(device)
    test_labels = concept_labels[test_mask.to_numpy()].to(device)

    N, D = train_embeds.shape

    # Define linear model
    model = nn.Linear(D, 1, bias=False).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(train_embeds, train_labels.float())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_embeds, test_labels.float())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Track metrics
    logs = {'train_loss': [], 'train_accuracy': [], 'train_f1': [], 'test_loss': [], 'test_accuracy': [], 'test_f1': []}
    best_loss = float("inf")
    patience_counter = 0  

    # Training loop
    for epoch in range(epochs):
        # Train phase
        model.train()
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0
        num_batches_train = 0
        all_train_preds = []
        all_train_labels = []

        for batch_features, batch_labels in train_dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()

            outputs = model(batch_features).view(-1)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == batch_labels).sum().item()
            total_train += batch_labels.size(0)

            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(batch_labels.cpu().numpy())

            epoch_train_loss += loss.item()
            num_batches_train += 1

        avg_train_loss = epoch_train_loss / num_batches_train
        train_accuracy = correct_train / total_train
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

        logs['train_loss'].append(avg_train_loss)
        logs['train_accuracy'].append(train_accuracy)
        logs['train_f1'].append(train_f1)

        ### eval mode
        with torch.no_grad():
            test_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            for batch_features, batch_labels in test_dataloader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

                # Forward pass
                outputs = model(batch_features).view(-1)
                loss = criterion(outputs, batch_labels)

                # Accumulate loss
                test_loss += loss.item()

                # Predictions
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

                # Collect all labels and predictions for F1 score calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

            # Compute metrics after processing all batches
            avg_test_loss = test_loss / len(test_dataloader)
            test_accuracy = correct / total
            test_f1 = f1_score(all_labels, all_preds, zero_division=0)

            # Append results to logs
            logs['test_loss'].append(avg_test_loss)
            logs['test_accuracy'].append(test_accuracy)
            logs['test_f1'].append(test_f1)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f} - Train Acc: {train_accuracy:.4f} - Train F1: {train_f1:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}", end="\r")

        # Early stopping
        avg_past_loss = round(sum(logs['train_loss']) / len(logs['train_loss']), tolerance)
        if round(avg_train_loss, tolerance) >= avg_past_loss:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} with loss {avg_train_loss:.4f}, accuracy {train_accuracy:.4f}, and F1-score {train_f1:.4f}")
                break
        else:
            patience_counter = 0  

        best_loss = min(best_loss, avg_train_loss)

    return model.weight.detach().squeeze(0).cpu(), logs

    
def compute_linear_separators(embeds, gt_patches_per_concept, dataset_name, sample_type, device='cuda', output_file=None, lr=0.01, epochs=100, batch_size=32, patience=15, tolerance=3):
    """
    Compute linear separator concept representations for all concepts in a dataset.
    
    Args:
        embeds (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
        dataset_name (str): Name of the dataset (used to locate metadata file).
        lr (float): Learning rate for optimization.
        epochs (int): Number of training epochs.
    
    Returns:
        dict: Dictionary mapping each concept name to its learned concept representation.
    """
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name)
    elif sample_type == 'image':
        split_df = get_split_df(dataset_name)
    
    concept_names = gt_patches_per_concept.keys()
    
    concept_representations = {}
    logs = {}
    
    for concept_name in tqdm(concept_names):
        concept_gt_patches = gt_patches_per_concept[concept_name]
        linear_separator, concept_logs = compute_a_linear_separator(concept_name, embeds, concept_gt_patches, split_df, lr, epochs, patience, tolerance, batch_size, device)
        concept_representations[concept_name] = linear_separator
        logs[concept_name] = concept_logs
    
    if output_file:
        torch.save(concept_representations, f'Concepts/{dataset_name}/{output_file}')
        print(f"Concepts saved to Concepts/{dataset_name}/{output_file} :)")
        torch.save(logs, f'Concepts/{dataset_name}/logs_{output_file}')
        print(f"Logs saved to Concepts/{dataset_name}/logs_{output_file}")
    
    return concept_representations, logs
  
    
    
###For Computing Similarity Metrics###
# def compute_cosine_sims(embeddings, concepts, output_file, dataset_name, batch_size=32):
#     """
#     Compute cosine similarity between each image embedding and each concept vector.

#     Args:
#         embeddings_file (str): File with data embeddings.
#         concepts_file (str): File with concept embeddings.
#         output_file (str): File to save the cosine similarities.
#         dataset_name (str): The name of the dataset. 
#         batch_size (int): The number of images to process per batch.

#     Returns:
#         None
#     """
#     # Move embeddings and concept embeddings to GPU (if not already on GPU)
#     embeddings = embeddings.to('cuda') if not embeddings.is_cuda else embeddings
#     all_concept_embeddings = {k: v.to('cuda') if not v.is_cuda else v for k, v in concepts.items()}
    
#     # Convert concept embeddings to a tensor (batch of all concept embeddings)
#     all_concept_embeddings_tensor = torch.stack(list(all_concept_embeddings.values())).to('cuda')

#     # Initialize a list to store cosine similarity rows
#     cosine_similarity_rows = []

#     # Compute cosine similarity in batches
#     n_images = embeddings.shape[0]
#     for i in tqdm(range(0, n_images, batch_size)):
#         # Get a batch of image embeddings
#         batch_embeddings = embeddings[i:i+batch_size]
        
#         # Compute cosine similarity for the batch
#         cosine_similarities = F.cosine_similarity(batch_embeddings.unsqueeze(1), all_concept_embeddings_tensor.unsqueeze(0), dim=2)

#         # Convert each batch's similarity results into rows
#         for image_cosine_similarities in cosine_similarities:
#             cosine_similarity_row = {
#                 concept_value: image_cosine_similarities[i].item()
#                 for i, concept_value in enumerate(all_concept_embeddings.keys())
#             }
#             cosine_similarity_rows.append(cosine_similarity_row)
        
#         # Free memory for the next batch
#         del batch_embeddings, cosine_similarities
#         torch.cuda.empty_cache()

#     # Create a DataFrame where each column corresponds to a concept-value combination
#     cosine_similarity_df = pd.DataFrame(cosine_similarity_rows)

#     if output_file:
#         # Save the cosine similarity results to a CSV file
#         output_path = f'Cosine_Similarities/{dataset_name}/{output_file}'
#         cosine_similarity_df.to_csv(output_path, index=False)

#         print(f"Cosine similarity results for all concepts saved at {output_file} :)")
#     return cosine_similarity_df

def compute_cosine_sims(embeddings, concepts, output_file, dataset_name, device, batch_size=32):
    """
    Compute cosine similarity between each image embedding and each concept vector in batches,
    and save the resulting DataFrame to a CSV file.

    Args:
        embeddings (torch.Tensor): Tensor of image embeddings of shape (n_samples, n_features).
        concepts (dict): Mapping from concept names to their embedding tensors.
        output_file (str): Filename to save the cosine similarities.
        dataset_name (str): The name of the dataset.
        device (torch.device or str): Device on which to perform computations.
        batch_size (int): Number of images to process per batch.

    Returns:
        pd.DataFrame: DataFrame with one row per image and one column per concept.
    """
    # Move embeddings and concept embeddings to the specified device
    embeddings = embeddings.to(device)
    all_concept_embeddings = {k: v.to(device) for k, v in concepts.items()}
    
    # Create a tensor for all concept embeddings in a fixed order
    concept_keys = list(all_concept_embeddings.keys())
    all_concept_embeddings_tensor = torch.stack([all_concept_embeddings[k] for k in concept_keys])
    
    cosine_similarity_rows = []
    n_images = embeddings.shape[0]
    
    for i in tqdm(range(0, n_images, batch_size), desc="Processing batches"):
        # Get batch embeddings
        batch_embeddings = embeddings[i:i+batch_size]
        # Compute cosine similarity in a vectorized way:
        # batch_embeddings: (batch_size, n_features)
        # all_concept_embeddings_tensor: (n_concepts, n_features)
        # After unsqueezing and computing similarity, result shape: (batch_size, n_concepts)
        cosine_similarities = F.cosine_similarity(
            batch_embeddings.unsqueeze(1), 
            all_concept_embeddings_tensor.unsqueeze(0), 
            dim=2
        )
        # Move the result to CPU and convert to list of rows
        batch_sims = cosine_similarities.cpu().tolist()
        # Create dictionary rows where keys are concept names
        batch_rows = [dict(zip(concept_keys, row)) for row in batch_sims]
        cosine_similarity_rows.extend(batch_rows)
    
    # Create the DataFrame from all similarity rows
    cosine_similarity_df = pd.DataFrame(cosine_similarity_rows)
    
    # Save the DataFrame if an output filename is provided
    if output_file:
        base_path = f'Cosine_Similarities/{dataset_name}/'
        os.makedirs(base_path, exist_ok=True)
        output_path = os.path.join(base_path, output_file)
        cosine_similarity_df.to_csv(output_path, index=False)
        print(f"Cosine similarity results saved at {output_path}")
    
    return cosine_similarity_df


def compute_signed_distances(embeds, concept_weights, dataset_name, device, output_file=None):
    """
    Compute signed distances for each test sample and save as a DataFrame.

    Args:
        embeds (torch.Tensor): Tensor of shape (N, D), test embeddings.
        concept_representations (dict): Dict mapping concept names to learned weight vectors.
        output_file (str): File path to save the DataFrame.

    Returns:
        pd.DataFrame: DataFrame where rows are samples and columns are concepts.
    """
    # Move embeddings and concept weights to GPU
    embeds = embeds.to(device)

    # Move concept weights to the same device
    concept_weights = {k: v.to(device) for k, v in concept_weights.items()}

    num_samples = embeds.shape[0]
    concept_names = list(concept_weights.keys())

    # Initialize a dictionary to store signed distances
    signed_distances = {}

    # Compute signed distances in parallel on GPU
    for concept_name, weight_vector in tqdm(concept_weights.items()):
        norm_weight = torch.norm(weight_vector, p=2)
        signed_distances[concept_name] = (embeds @ weight_vector) / norm_weight

    # Convert to CPU for DataFrame creation
    signed_dist_df = pd.DataFrame({k: v.cpu().numpy() for k, v in signed_distances.items()})

    if output_file is not None:
        # Save DataFrame to CSV
        signed_dist_df.to_csv(f"Distances/{dataset_name}/{output_file}", index=False)
        print(f"Signed distances saved to Distances/{dataset_name}/{output_file}")

    return signed_dist_df
    

### Functions for Visualizing Patch Methods ###

def plot_similar_patches_to_given_patch(image_index, patch_index_in_image, embeddings, images, save_path, 
                                        patch_size=14, top_k=5, model_input_size=(224, 224)):
    """
    Given a patch index, this function plots the given patch and the most similar patches based on cosine similarity.
    
    Args:
        image_index (int): The index of the selected image.
        patch_index_in_image (int): The image of the selected patch in the image.
        embeddings (torch.Tensor): A tensor containing the embeddings for each patch (shape: n_patches x embed_dim).
        images (list of PIL.Image): A list of images corresponding to the patches.
        patch_size (int): Size of each patch.
        top_k (int): Number of top similar patches to display.
        model_input_size (tuple): The input size to which the image should be resized (width, height).
    
    Returns:
        None: Displays the patches and their respective images with highlighted locations.
    """
    image = images[image_index]
    patch_idx = get_global_patch_idx(image_index, patch_index_in_image, images, 
                           patch_size=patch_size, model_input_size=model_input_size)
    
    left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size)
    
    make_image_with_highlighted_patch(image, left, top, right, bottom, plot_image_title=f'Image {image_index}: Patch {patch_index_in_image}')
    
    # Get the embedding of the selected patch
    patch_embedding = embeddings[patch_idx]

    # Compute cosine similarities between the patch embedding and all patch embeddings
    cos_sims = cosine_similarity(patch_embedding.unsqueeze(0).cpu().numpy(), 
                                 embeddings.cpu().numpy()).flatten()

    # Sort by similarity and get the top k similar patches
    top_k_patch_indices = cos_sims.argsort()[::-1][:top_k]
    
    overall_title = f'{top_k} Patches Most Similar to Image {image_index}, Patch {patch_index_in_image}'
    plot_patches_w_corr_images(top_k_patch_indices, cos_sims, images, overall_title, save_path=save_path, patch_size=patch_size, model_input_size=model_input_size)
    
    
def find_top_k_concepts_for_patch(patch_idx, embeddings, concepts, top_k=5):
    """
    Find the top k concepts that are most similar to the embedding of the given patch.
    
    Args:
        patch_idx (int): The index of the selected patch in the embeddings tensor.
        embeddings (torch.Tensor): A tensor containing the embeddings for each patch (shape: n_patches x embed_dim).
        concepts (dict): A dictionary where the key is the concept label and the value is the concept embedding tensor 
                         (shape: n_concepts x embed_dim).
        top_k (int): The number of top concepts to return based on similarity (default is 5).
    
    Returns:
        top_k_concepts (list): List of the concept labels corresponding to the top k most similar concepts.
        top_k_sims (list): List of cosine similarity values corresponding to the top k most similar concepts.
    """
    # Get the embedding of the selected patch
    patch_embedding = embeddings[patch_idx]

    # Compute cosine similarities between the patch embedding and all concept embeddings
    cos_sims = []
    for concept_label, concept_tensor in concepts.items():
        sim = cosine_similarity(patch_embedding.unsqueeze(0).cpu().numpy(), 
                                concept_tensor.unsqueeze(0).cpu().numpy()).flatten()
        cos_sims.append((concept_label, sim))

    # Sort by similarity and get the top k concepts
    cos_sims.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity value (descending order)
    top_k_concepts = [x[0] for x in cos_sims[:top_k]]
    top_k_sims = [x[1][0] for x in cos_sims[:top_k]]
    
    return top_k_concepts, top_k_sims


def plot_concepts_most_aligned_w_chosen_patch(image_index, patch_index_in_image, images, embeddings, concepts, 
                                              cos_sims, save_dir, overall_label, patch_size=14, k_concepts=5,
                                              n_examples_per_concept=5, model_input_size=(224, 224)):
    """
    Allow the user to manually select a patch from an image, find the top k most aligned concepts to the patch, 
    and display the corresponding patches of these top k concepts.

    Args:
        image_index (int): The index of the selected image.
        patch_index_in_image (int): The index of the selected patch within the image.
        images (list of PIL.Image): A list of images to choose from.
        embeddings (torch.Tensor): A tensor containing the embeddings for each patch (shape: n_patches x embed_dim).
        concepts (dict): A dictionary where the key is the concept label and the value is the concept embedding tensor 
        cos_sims (pd.Dataframe) : cosine similarities between each patch 
        save_dir (str) : Dir to see if the plot is already saved.
        overall_label (str): Label to help find correct images.
        patch_size (int): The size of each patch (default is 14).
        k_concepts (int): The number of top concepts to return based on similarity (default is 5).
        n_examples_per_concept (int): The number of aligning patches per concept to display.
    
    Returns:
        None: Displays the chosen patch and the patches for the top aligned concepts.
    """
    # Plot the selected patch with the corresponding image
    image = images[image_index]
    patch_idx = get_global_patch_idx(image_index, patch_index_in_image, images, 
                           patch_size=patch_size, model_input_size=model_input_size)
    
    left, top, right, bottom = calculate_patch_location(image, patch_idx, patch_size)
    make_image_with_highlighted_patch(image, left, top, right, bottom, plot_image_title=f'Image {image_index}: Patch {patch_index_in_image}')

    # Find the top k concepts most similar to the selected patch
    top_k_concepts, top_k_sims = find_top_k_concepts_for_patch(patch_idx, embeddings, concepts, k_concepts)

    # Plot the patches for each of the top k concepts
    for i, concept_label in enumerate(top_k_concepts):
        print(f'Rank {i+1}: Concept {concept_label} (Sim: {top_k_sims[i]:.2f})')
        print(f"Plotting top patches for concept {concept_label}")
        
        save_path = f'{save_dir}/{n_examples_per_concept}_patches_simto_concept_{concept_label}__{overall_label}'
        plot_top_patches_for_concept(str(concept_label), cos_sims, images, save_path, top_n=n_examples_per_concept, patch_size=patch_size, model_input_size=model_input_size)

    
### Other ###
def calculate_wcss(embeddings, labels, cluster_centers):
    """
    Calculates the Within-Cluster Sum of Squares (WCSS).
    
    Args:
        embeddings (torch.Tensor): Tensor of shape (n_samples, n_features) containing the data points.
        labels (torch.Tensor or array-like): Cluster labels for each sample.
        cluster_centers (torch.Tensor): Tensor of shape (n_clusters, n_features) representing cluster centroids.
    
    Returns:
        float: The WCSS value.
    """
    # Ensure labels are a tensor on the same device as embeddings
    embeddings = embeddings.to(labels.device)
    
    # For each data point, select its corresponding cluster center.
    # This uses advanced indexing: cluster_centers[labels] returns a tensor of shape (n_samples, n_features)
    centers_for_samples = cluster_centers[labels]
    
    # Compute the squared Euclidean distance between each sample and its assigned center
    squared_diffs = (embeddings - centers_for_samples) ** 2
    squared_distances = squared_diffs.sum(dim=1)
    
    # WCSS is the sum of these squared distances over all samples
    wcss = squared_distances.sum().item()
    
    # Average WCSS over the number of samples
    avg_wcss = wcss / embeddings.shape[0]
    return avg_wcss

def calculate_davies_bouldin(embeds, cluster_labels):
    """
    Compute the Davies-Bouldin Index.
    
    Args:
        embeds (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
        cluster_labels (torch.Tensor): Tensor of shape (N,) where each element is the cluster label assigned to the corresponding sample.
    
    Returns:
        float: The Davies-Bouldin Index score.
    """
    # Convert embeddings and labels to numpy arrays (required by scikit-learn)
    embeds_np = embeds.cpu().numpy()
    cluster_labels_np = cluster_labels.cpu().numpy()

    # Compute the Davies-Bouldin Index using scikit-learn's function
    db_index = davies_bouldin_score(embeds_np, cluster_labels_np)
    return db_index


def compute_calinski_score(embeds, cluster_labels):
    """
    Compute the Calinski-Harabasz Index.
    
    Args:
        embeds (torch.Tensor): Tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
        cluster_labels (torch.Tensor): Tensor of shape (N,) where each element is the cluster label assigned to the corresponding sample.
    
    Returns:
        float: The Calinski-Harabasz Index score.
    """
    # Convert embeddings and labels to numpy arrays (required by scikit-learn)
    embeds_np = embeds.cpu().numpy()
    cluster_labels_np = cluster_labels.cpu().numpy()

    # Compute the Calinski-Harabasz Index using scikit-learn's function
    ch_index = calinski_harabasz_score(embeds_np, cluster_labels_np)
    return ch_index

def evaluate_clustering_metrics(n_clusters_list, embeddings, dataset_name, device):
    """
    Evaluates clustering performance using multiple metrics and updates a live plot.
    
    Args:
        n_clusters_list (list): List of cluster numbers to evaluate.
        embeddings (torch.Tensor or np.array): Embeddings to be clustered.
        dataset_name (str): Name of the dataset.
        device (str): Compute device (e.g., 'cuda' or 'cpu').
    
    Returns:
        dict: Dictionary containing metric values.
    """
    split_df = get_patch_split_df(dataset_name)
    train_image_indices = split_df[split_df == 'train'].index
    test_image_indices = split_df[split_df == 'test'].index
    train_embeddings = embeddings[train_image_indices]
    test_embeddings = embeddings[test_image_indices]
    
    metrics = {
        'Train WCSS': [],
        'Test WCSS': [],
        'Train Davies-Bouldin Index': [],
        'Test Davies-Bouldin Index': [],
        'Train Calinski Score': [],
        'Test Calinski Score': []
    }

    # Initialize plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    metric_names = ['WCSS', 'Davies-Bouldin Index', 'Calinski Score']
    colors = ['blue', 'orange']
    lines = {}

    for i, metric in enumerate(metric_names):
        for j, split in enumerate(['Train', 'Test']):
            label = f"{split} {metric}"
            line, = axes[i].plot([], [], label=label, color=colors[j], marker='o' if j == 0 else 'x')
            lines[label] = line
        axes[i].set_title(f'{metric} vs Number of Clusters')
        axes[i].set_xlabel('Number of Clusters')
        axes[i].legend()

    plt.tight_layout()
    display(fig)

    for n_clusters in n_clusters_list:
        # Get the clustering results
        train_labels, test_labels, cluster_centers = run_fast_pytorch_kmeans(n_clusters, train_embeddings, test_embeddings, device)

        # Compute metrics
        train_wcss = calculate_wcss(train_embeddings, train_labels, cluster_centers)
        test_wcss = calculate_wcss(test_embeddings, test_labels, cluster_centers)
        train_db = calculate_davies_bouldin(train_embeddings, train_labels)
        test_db = calculate_davies_bouldin(test_embeddings, test_labels)
        train_ch = compute_calinski_score(train_embeddings, train_labels)
        test_ch = compute_calinski_score(test_embeddings, test_labels)

        # Append metrics
        metrics['Train WCSS'].append(train_wcss)
        metrics['Test WCSS'].append(test_wcss)
        metrics['Train Davies-Bouldin Index'].append(train_db)
        metrics['Test Davies-Bouldin Index'].append(test_db)
        metrics['Train Calinski Score'].append(train_ch)
        metrics['Test Calinski Score'].append(test_ch)

        # Update plot data
        for i, metric in enumerate(metric_names):
            for split in ['Train', 'Test']:
                label = f"{split} {metric}"
                lines[label].set_xdata(n_clusters_list[:len(metrics[label])])
                lines[label].set_ydata(metrics[label])

            axes[i].relim()  # Recalculate limits
            axes[i].autoscale_view()  # Autoscale

        # Refresh the plot
        clear_output(wait=True)
        display(fig)

    # Ensure the final plot remains visible
    plt.close(fig) 
    
    
def plot_train_history(train_history, metric_type, concepts=None):
    """
    Plots the train and test metrics over epochs for multiple concepts with different colors.

    Args:
        train_history (dict): Dictionary where keys are concept names and values are dictionaries 
                               containing 'train_*' and 'test_*' metric lists for each concept over epochs.
        metric_type (str): The metric to plot, e.g., 'loss', 'accuracy', or 'f1'.
    """
    if concepts is None:
        num_concepts = len(train_history)
    else:
        num_concepts = len(concepts)
        train_history = {k:v for k, v in train_history.items() if k in concepts}
        
    num_cols = 3  # 3 plots per row
    num_rows = math.ceil(num_concepts / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    if num_rows == 1:
        axes = np.array(axes)  # Ensure it's an array for consistent indexing

    axes = axes.flatten()  # Flatten in case of fewer concepts than grid slots

    for i, (concept, metrics) in enumerate(train_history.items()):
        # Retrieve the relevant metrics
        train_metric = metrics[f'train_{metric_type}']
        test_metric = metrics[f'test_{metric_type}']
        
        axes[i].plot(train_metric, label=f"Train {metric_type}", color='blue')
        axes[i].plot(test_metric, label=f"Test {metric_type}", color='red')
        
        axes[i].set_title(f"{metric_type} for {concept}")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(metric_type)
        axes[i].legend()

    # Hide empty subplots if the number of concepts isn't a multiple of 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    
def plot_train_history_justtrain(train_history, metric_type, concepts=None):
    """
    Plots the train and test metrics over epochs for multiple concepts with different colors.

    Args:
        train_history (dict): Dictionary where keys are concept names and values are dictionaries 
                               containing 'train_*' and 'test_*' metric lists for each concept over epochs.
        metric_type (str): The metric to plot, e.g., 'loss', 'accuracy', or 'f1'.
    """
    if concepts is None:
        num_concepts = len(train_history)
    else:
        num_concepts = len(concepts)
        train_history = {k:v for k, v in train_history.items() if k in concepts}
        
    num_cols = 3  # 3 plots per row
    num_rows = math.ceil(num_concepts / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    if num_rows == 1:
        axes = np.array(axes)  # Ensure it's an array for consistent indexing

    axes = axes.flatten()  # Flatten in case of fewer concepts than grid slots

    for i, (concept, metrics) in enumerate(train_history.items()):
        # Retrieve the relevant metrics
        train_metric = metrics[f'{metric_type}']
        
        axes[i].plot(train_metric, color='blue')
        
        axes[i].set_title(f"{metric_type} for {concept}")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(metric_type)

    # Hide empty subplots if the number of concepts isn't a multiple of 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()