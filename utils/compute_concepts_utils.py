"""Utils for Computing Concepts"""

from tqdm import tqdm
import os
import csv
import pandas as pd
import random
import numpy as np
import math
from collections import defaultdict
import copy
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import gc


import torch
from torchvision import transforms
from sklearn.metrics import mean_squared_error, f1_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
# import cupy as cp 
# from cuml.cluster import KMeans as cuml_kmeans 
from fast_pytorch_kmeans import KMeans
# import faiss


import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle
from IPython.display import display, clear_output

import importlib
import general_utils
import patch_alignment_utils
importlib.reload(general_utils)
importlib.reload(patch_alignment_utils)

from general_utils import retrieve_image, load_images, get_split_df, create_binary_labels, filter_coco_concepts
from patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence

### For Computing Embeddings ###
def get_final_cls_embeddings(model, processor, images, device):
    """
    Extracts class token embeddings at final layer of given model for each image.

    Args:
        model: The CLIP model to generate embeddings.
        processor: The processor used for transformifng the images and text.
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
        cross_attention_states = model.model.multi_modal_projector(cross_attention_states).reshape(
            -1, cross_attention_states.shape[-2], model.model.hidden_size
        )
    
    # Initialize storage for embeddings (excluding the class token)
    all_embs = []
    for i in range(0, len(images) * 4, 4):  # Only look at first tile
        curr_emb = cross_attention_states[i, :, :]  
        embs_no_cls = curr_emb[0, :]  # just get the first token (cls token)
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
        cross_attention_states = vision_outputs[0] #get last hidden state
        cross_attention_states = model.model.multi_modal_projector(cross_attention_states).reshape(
            -1, cross_attention_states.shape[-2], model.model.hidden_size
        )
    
    # Initialize storage for embeddings (excluding the class token)
    all_embs = []
    for i in range(0, len(images) * 4, 4):  # Only look at first tile
        curr_emb = cross_attention_states[i, :, :]  
        embs_no_cls = curr_emb[1:, :]  # Exclude the first token (cls token)
        all_embs.append(embs_no_cls) 

    # Stack collected embeddings into a single tensor
    patch_embeddings = torch.stack(all_embs)
    patch_embeddings = patch_embeddings.reshape(-1, patch_embeddings.shape[2]) #flatten so each patch is own dim
    
    return patch_embeddings


def get_llama_text_patch_embeddings(model, processor, text_samples, device, percent_thru_model=100):
    """
    Extracts embeddings from chosen layer of given model for each patch in each image.

    Args:
        model: The LLAMA model to generate embeddings.
        processor: The processor used for transforming the images and text.
        text_samples: A list of text strings.
        device: The device to move the tensors to.
        percent_thru_model (int) : Not used, for now
    Returns:
        torch.Tensor: The generated embeddings per patch per image.
    """
    hidden_states_list = []
    with torch.no_grad():
        for text_sample in text_samples: #do one by one so special tokens consistent w different batch sizes
            inputs = processor.tokenizer(
                text_sample,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt"
            ).to(model.device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_states = outputs.hidden_states
            hidden = hidden_states[-1].squeeze() #just get hidden state from last layer  
            hidden_states_list.append(hidden)

    hidden_states = torch.cat(hidden_states_list, dim=0)
    return hidden_states


def get_llama_text_cls_embeddings(model, processor, text_samples, device, percent_thru_model=100):
    """
    Extracts CLS-style embeddings (first token) from LLaMA model for each text sample.

    Args:
        model: The LLaMA model (e.g., LlamaForCausalLM).
        processor: The processor used for tokenization.
        text_samples: A list of text strings.
        device: Device for model execution.
    
    Returns:
        torch.Tensor: CLS-style embeddings per text sample.
    """
    cls_embeddings = []
    with torch.no_grad():
        for text_sample in text_samples:
            # Tokenize with special tokens
            inputs = processor(
                text=text_sample,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt"
            ).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            cls_emb = last_hidden[:, -1, :]  # Take the last token as pseudo-CLS
            cls_embeddings.append(cls_emb)

    return torch.cat(cls_embeddings, dim=0)


# def get_clip_text_cls_embeddings(model, processor, text_samples, device, percent_thru_model=100):
#     """
#     Extracts CLS embeddings for text using CLIP.
    
#     Args:
#         model: The CLIP model to generate embeddings.
#         processor: The processor used for transforming text.
#         text_samples: A list of strings.
#         device: The device to move the tensors to.
#         percent_thru_model (int): Percent through model that embedding is extracted from.

#     Returns:
#         torch.Tensor: The generated text embeddings.
#     """
#     model.eval()
    
#     # Preprocess the text for the model
#     inputs = processor(text=text_samples, return_tensors="pt", padding=True, truncation=True)
#     inputs = inputs.to(device)

#     # Generate text embeddings without computing gradients
#     with torch.no_grad():
#         text_features = model.get_text_features(input_ids=inputs['input_ids'], 
#                                                attention_mask=inputs.get('attention_mask', None))
    
#     return text_features


# def get_clip_text_patch_embeddings(model, processor, text_samples, device, percent_thru_model=100):
#     """
#     Extracts token-level embeddings for text using CLIP.
    
#     Args:
#         model: The CLIP model to generate embeddings.
#         processor: The processor used for transforming text.
#         text_samples: A list of strings.
#         device: The device to move the tensors to.
#         percent_thru_model (int): Percent through model that embedding is extracted from.

#     Returns:
#         torch.Tensor: The generated token embeddings.
#     """
#     model.eval()
#     all_token_embeddings = []
    
#     with torch.no_grad():
#         inputs = processor(text=text_samples, return_tensors="pt", padding=True, truncation=True)
#         inputs = inputs.to(device)
        
#         # Get the text encoder outputs
#         text_outputs = model.text_model(**inputs, output_hidden_states=True)
        
#         # Get the last hidden states
#         last_hidden_states = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
#         # Flatten to get all tokens
#         for i, text in enumerate(text_samples):
#             # Get the actual length (excluding padding)
#             attention_mask = inputs['attention_mask'][i]
#             actual_length = attention_mask.sum().item()
            
#             # Extract non-padded tokens
#             token_embeddings = last_hidden_states[i, :actual_length, :]
#             all_token_embeddings.append(token_embeddings)
    
#     # Concatenate all token embeddings
#     return torch.cat(all_token_embeddings, dim=0)


# def get_simple_text_cls_embeddings(model, tokenizer, text_samples, device, percent_thru_model=100):
#     """
#     Simple text embeddings using a basic transformer model.
#     """
#     model.eval()
#     model = model.to(device)
    
#     # Tokenize and get embeddings
#     inputs = tokenizer(text_samples, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     inputs = inputs.to(device)
    
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Use mean pooling of last hidden states
#         embeddings = outputs.last_hidden_state.mean(dim=1)
    
#     return embeddings


# def get_simple_text_patch_embeddings(model, tokenizer, text_samples, device, percent_thru_model=100):
#     """
#     Simple token-level embeddings using a basic transformer model.
#     """
#     model.eval()
#     model = model.to(device)
#     all_token_embeddings = []
    
#     with torch.no_grad():
#         inputs = tokenizer(text_samples, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         inputs = inputs.to(device)
        
#         outputs = model(**inputs)
#         last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
#         # Flatten to get all tokens (excluding padding)
#         for i, text in enumerate(text_samples):
#             attention_mask = inputs['attention_mask'][i]
#             actual_length = attention_mask.sum().item()
            
#             # Extract non-padded tokens
#             token_embeddings = last_hidden_states[i, :actual_length, :]
#             all_token_embeddings.append(token_embeddings)
    
#     # Concatenate all token embeddings
#     return torch.cat(all_token_embeddings, dim=0)



# def get_llama_text_patch_embeddings(model, processor, text_samples, device, percent_thru_model=100):
#     """
#     Returns a flat tensor of token embeddings for all text samples in a batch.
#     Each token gets a 4096-dim vector, excluding padding.
    
#     Args:
#         model: The LLAMA model to generate embeddings.
#         processor: The HuggingFace processor.
#         text_samples: A list of strings (batch).
#         device: torch device.

#     Returns:
#         torch.Tensor: [n_total_tokens, hidden_dim]
#     """
#     model.eval()
#     all_token_embeddings = []

#     with torch.no_grad():
#         inputs = processor(
#             text=text_samples,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             add_special_tokens=False,
#             return_attention_mask=True,
#             token='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'
#         ).to(device)

#         outputs = model(
#             **inputs,
#             output_hidden_states=True,
#             return_dict=True
#         )

#         last_hidden = outputs.hidden_states[-1]  # [B, T, D]
#         attention_mask = inputs["attention_mask"]  # [B, T]

#         for j in range(last_hidden.size(0)):
#             valid_token_mask = attention_mask[j].bool()
#             token_embeddings = last_hidden[j][valid_token_mask]  # [n_valid_tokens, D]
#             all_token_embeddings.append(token_embeddings.cpu())

#     return torch.cat(all_token_embeddings, dim=0)


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
        # print(f"batch embeddings: {batch_embeddings.shape}")
        embeddings.append(batch_embeddings.cpu())
        del batch_embeddings
        torch.cuda.empty_cache()
    embeddings = torch.cat(embeddings, dim=0)  # Concatenate all batch embeddings
    # print(f"Extracted embeddings of shape: {embeddings.shape}")
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

def compute_train_avg_and_norm(embeddings, dataset_name, model_input_size, sample_type):
    if sample_type == 'patch':
        #Load split_df
        split_df = get_patch_split_df(dataset_name, model_input_size)

        # Create boolean masks for train/test
        train_mask = split_df[split_df == 'train']

        #Filter out the embeddings that are 'padding'
        relevant_indices = filter_patches_by_image_presence(train_mask.index, dataset_name, model_input_size).tolist()
        final_mask = train_mask.loc[train_mask.index.intersection(relevant_indices)]
    else:
        split_df = get_split_df(dataset_name)
        final_mask = split_df[split_df == 'train']
    
    # Apply masks to embeddings
    train_embeddings = embeddings[final_mask.index.to_list()].float()
    
    mean_train_embedding = train_embeddings.mean(dim=0)
    train_norm = train_embeddings.norm(dim=1, keepdim=True).mean()
    return mean_train_embedding, train_norm
    

def center_and_normalize_embeddings(embeddings, dataset_name, model_input_size, sample_type):
    """
    Center and normalize embeddings using statistics from the training set.

    Args:
        train_embeddings (torch.Tensor): Tensor of training set embeddings.
        test_embeddings (torch.Tensor): Tensor of test set embeddings.

    Returns:
        tuple: (normalized_train_embeddings, normalized_test_embeddings)
    """
    mean_train_embedding, train_norm = compute_train_avg_and_norm(embeddings, dataset_name, model_input_size, sample_type)
    centered_embeddings = embeddings - mean_train_embedding
    norm_embeddings = centered_embeddings / train_norm

    return norm_embeddings, mean_train_embedding, train_norm


def compute_batch_embeddings(images, embedding_fxn, model, processor, device, 
                             percent_thru_model, dataset_name, model_input_size,
                             embeddings_file=None, batch_size=100, scratch_dir=""):
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
    
    # if 'Cal' in dataset_name:
    #     train_embeds_dic = torch.load(f'{scratch_dir}Embeddings/{dataset_name.split("-")[0]}/{embeddings_file}')
    #     mean_train_embedding, train_norm = train_embeds_dic['mean_train_embedding'], train_embeds_dic['train_norm']
    #     centered_embeddings = embeddings - mean_train_embedding
    #     norm_embeddings = centered_embeddings / train_norm
    # else:
    #     if 'cls' in embeddings_file:
    #         sample_type = 'cls'
    #     else:
    #         sample_type = 'patch'
    #     norm_embeddings, mean_train_embedding, train_norm = center_and_normalize_embeddings(embeddings, 
    #                                                                                         dataset_name, 
    #                                                                                         model_input_size,
    #                                                                                         sample_type)
    if 'cls' in embeddings_file:
        sample_type = 'cls'
    else:
        sample_type = 'patch'
    norm_embeddings, mean_train_embedding, train_norm = center_and_normalize_embeddings(embeddings, 
                                                                                        dataset_name, 
                                                                                        model_input_size,
                                                                                        sample_type)
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

# import os
# import torch
# import pandas as pd
# from tqdm import tqdm
# from compute_concepts_utils import get_patch_split_df, filter_patches_by_image_presence


# def get_llama_text_patch_embeddings(model, processor, text_samples, device, output_dir, batch_index):
#     """
#     Returns a flat tensor of token embeddings for all text samples in a batch and writes them to disk.
#     """
#     model.eval()
#     all_token_embeddings = []

#     with torch.no_grad():
#         inputs = processor(
#             text=text_samples,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             add_special_tokens=False,
#             return_attention_mask=True,
#             token='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'
#         ).to(device)

#         outputs = model(
#             **inputs,
#             output_hidden_states=True,
#             return_dict=True
#         )

#         last_hidden = outputs.hidden_states[-1]  # [B, T, D]
#         attention_mask = inputs["attention_mask"]  # [B, T]

#         for j in range(last_hidden.size(0)):
#             valid_token_mask = attention_mask[j].bool()
#             token_embeddings = last_hidden[j][valid_token_mask]  # [n_valid_tokens, D]
#             all_token_embeddings.append(token_embeddings.cpu())

#     embeddings = torch.cat(all_token_embeddings, dim=0)
#     os.makedirs(output_dir, exist_ok=True)
#     torch.save(embeddings, os.path.join(output_dir, f"raw_batch_{batch_index}.pt"))
#     torch.cuda.empty_cache()


# def compute_raw_batch_embeddings_to_disk(images, embedding_fxn, model, processor, device, 
#                                           percent_thru_model, dataset_name, batch_size=100):
#     print("Computing and saving raw embeddings batch-wise...")
#     output_dir = f"Embeddings/{dataset_name}/raw_batches"
#     n_batches = (len(images) + batch_size - 1) // batch_size

#     for i in tqdm(range(n_batches), desc="Computing embeddings"):
#         batch_images = images[i * batch_size:(i + 1) * batch_size]
#         embedding_fxn(model, processor, batch_images, device, output_dir, batch_index=i)

#     return output_dir


# def load_all_raw_embeddings(raw_dir):
#     print("Loading all saved raw embeddings...")
#     tensors = []
#     for file in sorted(os.listdir(raw_dir)):
#         if file.endswith(".pt"):
#             tensors.append(torch.load(os.path.join(raw_dir, file)))
#     return torch.cat(tensors, dim=0)


# def compute_batch_embeddings(images, embedding_fxn, model, processor, device, 
#                              percent_thru_model, dataset_name, model_input_size,
#                              embeddings_file=None, batch_size=100):
#     raw_dir = compute_raw_batch_embeddings_to_disk(
#         images, embedding_fxn, model, processor, device, 
#         percent_thru_model, dataset_name, batch_size
#     )

#     embeddings = load_all_raw_embeddings(raw_dir)

#     norm_embeddings, mean_train_embedding, train_norm = center_and_normalize_embeddings(
#         embeddings, dataset_name, model_input_size
#     )

#     embeds_dic = {
#         'normalized_embeddings': norm_embeddings,
#         'mean_train_embedding': mean_train_embedding,
#         'train_norm': train_norm
#     }

#     if embeddings_file:
#         os.makedirs(f'Embeddings/{dataset_name}', exist_ok=True)
#         output_file = f'Embeddings/{dataset_name}/{embeddings_file}'
#         torch.save(embeds_dic, output_file)
#         print(f"Embeddings saved to {output_file} :)")

#     return embeds_dic
    

### For Computing Concept Vectors ###
def assign_labels_to_centers(embeddings, cluster_centers, device):
    embeddings = embeddings.to(device)
    cluster_centers = cluster_centers.to(device)
    dists = torch.cdist(embeddings, cluster_centers, p=2)
    labels = torch.argmin(dists, dim=1)
    return labels.cpu()

def run_fast_pytorch_kmeans(n_clusters, train_embeddings, test_embeddings, cal_embeddings, device):   
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=True, max_iter=10000, tol=1e-6)
    
    # Fit k-means
    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=1)
    print(f"Fitting KMeans for {n_clusters} clusters...")
    train_labels = kmeans.fit_predict(train_embeddings.to(device))
    cluster_centers = kmeans.centroids.detach().cpu()
    
    # Fit KMeans on training embeddings and predict cluster labels
    test_labels = assign_labels_to_centers(test_embeddings, cluster_centers, device)
    cal_labels = assign_labels_to_centers(cal_embeddings, cluster_centers, device)
    
    # Retrieve cluster centers
    cluster_centers = kmeans.centroids
    return train_labels, test_labels, cal_labels, cluster_centers


# def run_fast_pytorch_kmeans(n_clusters, train_embeddings, test_embeddings, cal_embeddings, device='cuda', max_iter=300):
#     """
#     Memory-efficient FAISS-based KMeans that avoids CPU copies.

#     Args:
#         n_clusters (int): Number of clusters.
#         train_embeddings (torch.Tensor): [N, D], float32, on CPU.
#         test_embeddings (torch.Tensor): [M, D], float32, on CPU.
#         device (str): 'cuda' or 'cpu'.
#         max_iter (int): Max iterations for KMeans.

#     Returns:
#         train_labels (torch.LongTensor)
#         test_labels (torch.LongTensor)
#         cluster_centers (torch.Tensor)  # [k, d]
#     """
#     # Ensure float32 and CPU, no graph
#     assert not train_embeddings.requires_grad and not test_embeddings.requires_grad
#     assert train_embeddings.dtype == torch.float32 and test_embeddings.dtype == torch.float32
#     assert train_embeddings.device.type == 'cpu' and test_embeddings.device.type == 'cpu'

#     train_np = train_embeddings.numpy()
#     test_np = test_embeddings.numpy()
#     cal_np = cal_embeddings.numpy()
#     d = train_np.shape[1]

#     kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=max_iter, verbose=True, gpu=(device == 'cuda'))
#     print(f"Fitting FAISS KMeans with {n_clusters} clusters on {len(train_np)} samples...")
#     kmeans.train(train_np)

#     print("Assigning Labels to Kmeans Clusters...")
#     train_labels = kmeans.index.search(train_np, 1)[1].squeeze()
#     test_labels = kmeans.index.search(test_np, 1)[1].squeeze()
#     cal_labels = kmeans.index.search(cal_np, 1)[1].squeeze()

#     # Convert results back to torch
#     train_labels = torch.from_numpy(train_labels).long()
#     test_labels = torch.from_numpy(test_labels).long()
#     cal_labels = torch.from_numpy(cal_labels).long()
#     cluster_centers = torch.from_numpy(kmeans.centroids)

#     return train_labels, test_labels, cal_labels, cluster_centers



def map_samples_to_clusters(train_image_indices, train_labels, test_image_indices, test_labels, 
                            cal_image_indices, cal_labels, dataset_name, concepts_filename=None):
    # Initialize cluster mappings
    train_cluster_to_samples = defaultdict(list)
    test_cluster_to_samples = defaultdict(list)
    cal_cluster_to_samples = defaultdict(list)

    # Map training samples if available
    if len(train_labels) > 0:
        for i, train_idx in enumerate(train_image_indices):
            cluster_label = str(train_labels[i].item())
            train_cluster_to_samples[cluster_label].append(train_idx)
    else:
        print("No training labels provided — skipping train mapping.")

    # Map test samples if available
    if len(test_labels) > 0:
        for i, test_idx in enumerate(test_image_indices):
            cluster_label = str(test_labels[i].item())
            test_cluster_to_samples[cluster_label].append(test_idx)
    else:
        print("No test labels provided — skipping test mapping.")
        
    # Map cal samples if available
    if len(cal_labels) > 0:
        for i, cal_idx in enumerate(cal_image_indices):
            cluster_label = str(cal_labels[i].item())
            cal_cluster_to_samples[cluster_label].append(cal_idx)
    else:
        print("No cal labels provided — skipping test mapping.")

    # Convert to regular dicts and sort keys
    train_cluster_to_samples = dict(sorted(train_cluster_to_samples.items()))
    test_cluster_to_samples = dict(sorted(test_cluster_to_samples.items()))
    cal_cluster_to_samples = dict(sorted(cal_cluster_to_samples.items()))

    # Optionally save to file
    if concepts_filename:
        torch.save(train_cluster_to_samples, f'Concepts/{dataset_name}/train_samples_{concepts_filename}')
        torch.save(test_cluster_to_samples, f'Concepts/{dataset_name}/test_samples_{concepts_filename}')
        torch.save(cal_cluster_to_samples, f'Concepts/{dataset_name}/cal_samples_{concepts_filename}')
        print(f"Saved mapped cluster indices to Concepts/{dataset_name}/train_samples_{concepts_filename} :)")

    return train_cluster_to_samples, test_cluster_to_samples, cal_cluster_to_samples


def gpu_kmeans(n_clusters, embeddings, dataset_name, device, model_input_size, concepts_filename=None, sample_type ='patch', map_samples=True):
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
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    #separate embeddings into test and train
    relevant_indices = torch.arange(embeddings.shape[0])
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, model_input_size)
        # Filter patches that are 'padding' given the preprocessing schemes
        relevant_indices = filter_patches_by_image_presence(relevant_indices, dataset_name, model_input_size).tolist()
        
    elif sample_type == 'cls':
        split_df = get_split_df(dataset_name)
        relevant_indices = split_df.index
    
    
    # Get train and test image indices from split_df
    train_image_indices = split_df[split_df == 'train'].index
    test_image_indices = split_df[split_df == 'test'].index
    cal_image_indices = split_df[split_df == 'cal'].index
    train_relevant_indices = [idx for idx in relevant_indices if idx in train_image_indices]
    test_relevant_indices = [idx for idx in relevant_indices if idx in test_image_indices]
    cal_relevant_indices = [idx for idx in relevant_indices if idx in cal_image_indices]

    train_embeddings = embeddings[train_relevant_indices]
    test_embeddings = embeddings[test_relevant_indices]
    cal_embeddings = embeddings[cal_relevant_indices]
    

    # if concepts_filename and os.path.exists(f'Concepts/{dataset_name}/{concepts_filename}'):
    #     label_to_center = torch.load(f'Concepts/{dataset_name}/{concepts_filename}')
    #     train_labels = torch.load(f'Concepts/{dataset_name}/train_labels_{concepts_filename}')
    #     test_labels = torch.load(f'Concepts/{dataset_name}/test_labels_{concepts_filename}')
    # else:
    train_labels, test_labels, cal_labels, cluster_centers = run_fast_pytorch_kmeans(n_clusters, 
                                                                         train_embeddings, 
                                                                         test_embeddings, 
                                                                         cal_embeddings,
                                                                         device)
    # Map cluster labels to cluster centers
    label_to_center = {label: center.cpu() for label, center in enumerate(cluster_centers)}
    label_to_center = dict(sorted(label_to_center.items()))
    label_to_center = {str(label): center for label, center in label_to_center.items()}
    if concepts_filename:
        torch.save(label_to_center, f'Concepts/{dataset_name}/{concepts_filename}')
        torch.save(cluster_centers, f'Concepts/{dataset_name}/cluster_centers_{concepts_filename}')
        torch.save(train_labels, f'Concepts/{dataset_name}/train_labels_{concepts_filename}')
        torch.save(test_labels, f'Concepts/{dataset_name}/test_labels_{concepts_filename}')
        torch.save(cal_labels, f'Concepts/{dataset_name}/cal_labels_{concepts_filename}')
        print(f"Saved cluster centers and labels to Concepts/{dataset_name}/{concepts_filename} :)")
    
    train_cluster_to_samples, test_cluster_to_samples, cal_cluster_to_samples = [], [], []
    if map_samples:
        train_cluster_to_samples, test_cluster_to_samples, cal_cluster_to_samples = map_samples_to_clusters(
                                                                            train_relevant_indices, train_labels, 
                                                                            test_relevant_indices, test_labels,
                                                                            cal_relevant_indices, cal_labels,
                                                                            dataset_name, concepts_filename)

    return label_to_center, train_cluster_to_samples, test_cluster_to_samples, cal_cluster_to_samples



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

        # avg_vector = avg_vector / avg_vector.norm()  # Normalize to unit norm

        concepts[concept] = avg_vector
    
    if output_file:
        torch.save(concepts, f'Concepts/{dataset_name}/{output_file}')
        print(f'Concepts saved to Concepts/{dataset_name}/{output_file} :)') 
    return concepts



def sort_data_by_split(embeds, all_concept_labels, split, dataset_name, model_input_size, sample_type):
    """
    Extract embeddings and labels corresponding to a specified split. Doesn't include padding.
    
    Args:
        embeds (torch.Tensor): Tensor containing embeddings.
        labels (torch.Tensor): Tensor containing binary labels.
        split_df (pd.DataFrame): DataFrame with split information ('train'/'test').
        split (str): The split to extract (e.g., 'train' or 'test').
    
    Returns:
        tuple: A tuple (split_embeds, split_labels) corresponding to the specified split.
    """
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size)
        relevant_indices = filter_patches_by_image_presence(split_df.index, dataset_name, model_input_size).tolist()
    elif sample_type == 'cls':
        split_df = get_split_df(dataset_name)
        relevant_indices = split_df.index
    
    split_indices = (split_df[split_df == split]).index
    overlapping_indices = list(set(split_indices).intersection(relevant_indices))
    
    split_embeds = embeds[overlapping_indices]
    split_all_concept_labels = {}
    for concept, labels in all_concept_labels.items():
        split_labels = labels[overlapping_indices]
        split_all_concept_labels[concept] = split_labels
    return split_embeds, split_all_concept_labels


def balance_dataset(embeds, labels, seed=42, max_samples=100000):
    """
    Balance the dataset for positive and negative examples, optionally capping the number of samples per class.

    Args:
        embeds (torch.Tensor): Tensor of embeddings (N, D).
        labels (torch.Tensor): Binary labels (N,).
        seed (int): Random seed.
        max_samples (int or None): Optional cap for the number of pos/neg samples to include.

    Returns:
        tuple: (balanced_embeds, balanced_labels) or (None, None) if not enough data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        print(f"Skipping due to insufficient data: {len(pos_indices)} pos, {len(neg_indices)} neg")
        return None, None

    # Choose the number of samples per class
    num_samples = min(len(pos_indices), len(neg_indices))
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    # Resample to balance
    pos_indices = resample(pos_indices.cpu().numpy(), n_samples=num_samples, replace=False, random_state=seed)
    neg_indices = resample(neg_indices.cpu().numpy(), n_samples=num_samples, replace=False, random_state=seed)

    balanced_indices = torch.tensor(np.concatenate([pos_indices, neg_indices]), dtype=torch.long)

    balanced_embeds = embeds[balanced_indices]
    balanced_labels = labels[balanced_indices]

    return balanced_embeds, balanced_labels


def balance_dataset_evenly_for_ooc(embeds, all_concept_labels, target_concept, seed=42):
    """
    Balance the dataset so that 50% of the samples are in-concept (target_concept) and 50% are out-of-concept. Out of
    the 50% that are out of concept, they are distributed evenly across other concepts.

    Args:
        embeds (torch.Tensor): Tensor of embeddings (N x D).
        all_concept_labels (dict): Dictionary mapping each concept to a binary tensor (N,) indicating concept presence.
        target_concept (str): The concept for which we are training a classifier.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (balanced_embeds, balanced_labels) after undersampling.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    labels = all_concept_labels[target_concept]
    pos_indices = labels.nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
    num_pos = len(pos_indices)

    non_target_concepts = [c for c in all_concept_labels if c != target_concept]

    neg_indices_np = neg_indices.cpu().numpy()
    candidate_matrix = np.stack([
        all_concept_labels[c][neg_indices].cpu().numpy() for c in non_target_concepts
    ], axis=0)

    def choose_candidate(col, seed=42):
        np.random.seed(seed)  # Set seed for reproducibility
        candidates = np.nonzero(col)[0]
        if candidates.size == 0:
            return -1
        else:
            return np.random.choice(candidates)

    chosen_candidates = np.apply_along_axis(choose_candidate, 0, candidate_matrix)

    group_assignments = {c: [] for c in non_target_concepts}
    group_assignments["none"] = []

    for i, candidate in enumerate(chosen_candidates):
        idx = int(neg_indices_np[i])
        if candidate == -1:
            group_assignments["none"].append(idx)
        else:
            chosen_concept = non_target_concepts[int(candidate)]
            group_assignments[chosen_concept].append(idx)

    if len(group_assignments["none"]) == 0:
        del group_assignments["none"]

    target_total_negatives = num_pos
    num_groups = len(group_assignments)
    equal_share = target_total_negatives // num_groups

    selected_negatives = {}
    total_selected = 0

    for group, indices in group_assignments.items():
        if len(indices) <= equal_share:
            selected_negatives[group] = indices[:]
        else:
            sampled = resample(indices, n_samples=equal_share, replace=False, random_state=seed)
            selected_negatives[group] = sampled
        total_selected += len(selected_negatives[group])

    remaining_needed = target_total_negatives - total_selected

    extra_counts = {group: len(set(group_assignments[group]) - set(selected_negatives.get(group, [])))
                    for group in group_assignments}

    total_extras = sum(extra_counts.values())

    if remaining_needed > 0 and total_extras > 0:
        allocated = {group: int(round((extra_counts[group] / total_extras) * remaining_needed))
                     for group in extra_counts}

        allocated_total = sum(allocated.values())
        diff = remaining_needed - allocated_total

        if diff:
            sorted_groups = sorted(extra_counts.items(), key=lambda x: x[1], reverse=True)
            idx = 0
            while diff:
                grp = sorted_groups[idx % len(sorted_groups)][0]
                allocated[grp] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
                idx += 1

        for group, extra_alloc in allocated.items():
            if extra_alloc > 0:
                available = list(set(group_assignments[group]) - set(selected_negatives.get(group, [])))
                if len(available) >= extra_alloc:
                    sampled = resample(available, n_samples=extra_alloc, replace=False, random_state=seed)
                else:
                    sampled = available
                selected_negatives[group].extend(sampled)

    all_selected_negatives = []
    for group in selected_negatives:
        all_selected_negatives.extend(selected_negatives[group])

    if len(pos_indices) > len(all_selected_negatives):
        pos_indices = pos_indices[torch.randperm(len(pos_indices), generator=torch.Generator().manual_seed(seed))[:len(all_selected_negatives)]]

    balanced_indices = torch.cat([pos_indices, torch.tensor(all_selected_negatives, dtype=torch.long)])
    balanced_embeds = embeds[balanced_indices]
    balanced_labels = labels[balanced_indices]

    return balanced_embeds, balanced_labels


def print_balancing_results(train_labels, test_labels):
    """
    Print the count of positive and negative samples for both training and test sets.
    
    Args:
        train_labels (torch.Tensor): Tensor of training labels.
        test_labels (torch.Tensor): Tensor of test labels.
    
    Returns:
        None
    """
    num_train_pos = (train_labels == 1).sum().item()
    num_train_neg = (train_labels == 0).sum().item()

    num_test_pos = (test_labels == 1).sum().item()
    num_test_neg = (test_labels == 0).sum().item()

    print(f"Resampled to {len(train_labels)} train samples "
          f"({num_train_pos} positive, {num_train_neg} negative); "
          f"{len(test_labels)} test samples "
          f"({num_test_pos} positive, {num_test_neg} negative)")
    
    
def create_dataloader(embeds, labels, batch_size, shuffle=True):
    """
    Create a DataLoader from embeddings and labels.
    
    Args:
        embeds (torch.Tensor): Tensor of embeddings.
        labels (torch.Tensor): Tensor of labels.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = TensorDataset(embeds, labels.float())
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dl
    
    
def evaluate_model(model, dl, criterion, device):
    """
    Evaluate the model on data provided by the dataloader.
    
    Args:
        model (nn.Module): The model to evaluate.
        dl (DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module): Loss function.
        device (str): The device on which to perform computations.
    
    Returns:
        tuple: A tuple (avg_loss, accuracy, f1) containing the average loss, accuracy, and F1 score.
    """
    model.eval()
    with torch.no_grad():
        sum_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        for batch_features, batch_labels in dl:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features.float()).view(-1)
            loss = criterion(outputs, batch_labels)

            sum_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        avg_loss = sum_loss / len(dl)
        acc = correct / total
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1

def log_progress(logs, avg_train_loss, train_acc, train_f1, avg_test_loss, test_acc, test_f1, epoch, epochs):
    """
    Log and print training and testing metrics for the current epoch.
    
    Args:
        logs (dict): Dictionary to store metrics over epochs.
        avg_train_loss (float): Average training loss for the epoch.
        train_acc (float): Training accuracy for the epoch.
        train_f1 (float): Training F1 score for the epoch.
        avg_test_loss (float): Average test loss for the epoch.
        test_acc (float): Test accuracy for the epoch.
        test_f1 (float): Test F1 score for the epoch.
        epoch (int): Current epoch index.
        epochs (int): Total number of epochs.
    
    Returns:
        dict: The updated logs dictionary.
    """
    logs['train_loss'].append(avg_train_loss)
    logs['train_accuracy'].append(train_acc)
    logs['train_f1'].append(train_f1)
    logs['test_loss'].append(avg_test_loss)
    logs['test_accuracy'].append(test_acc)
    logs['test_f1'].append(test_f1)
    
    msg = (f"Epoch [{epoch+1}/{epochs}] - "
           f"Train Loss: {avg_train_loss:.6f} - Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f} | "
           f"Test Loss: {avg_test_loss:.4f} - Test Acc: {test_acc:.4f} - Test F1: {test_f1:.4f}")
    
    if epoch+1 == epochs:
        print(msg)
    else:
        print(msg, end="\r")
        
    return logs
    
def create_linear_model(D, device, weights=None):
    """
    Creates a single-layer linear model (no bias), optionally initializing with given weights.

    Args:
        D (int): Input dimensionality (feature size).
        device (str or torch.device): Device to move the model to.
        weights (torch.Tensor, optional): Tensor of shape (1, D) or (D,) to initialize model weights.

    Returns:
        nn.Linear: A linear model with weights optionally preloaded.
    """
    model = nn.Linear(D, 1, bias=False).to(device)

    if weights is not None:
        # Ensure weights shape is (1, D)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        elif weights.shape != (1, D):
            raise ValueError(f"Expected weights of shape (1, {D}) or ({D},), got {weights.shape}")
        
        model.weight.data.copy_(weights.to(device))

    return model
    
def train_model(train_dl, test_dl, epochs, lr, weight_decay, lr_step_size, lr_gamma, patience, tolerance, device, model=None):
    """
    Train a linear model using the provided training and test dataloaders.
    
    Args:
        train_dl (DataLoader): DataLoader for the training dataset.
        test_dl (DataLoader): DataLoader for the test dataset.
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay coefficient for the optimizer.
        lr_step_size (int): Number of epochs between each learning rate decay step.
        lr_gamma (float): Factor by which the learning rate is decayed.
        patience (int): Number of epochs with insufficient improvement before early stopping.
        tolerance (float): Minimum improvement required to reset the early stopping counter.
        device (str): The device on which to train the model.
    
    Returns:
        tuple: A tuple (model_weights, logs) where model_weights is the learned weight vector,
               and logs is a dictionary containing the training metrics.
    """
    if model is None:
        D = len(train_dl.dataset[0][0])
        model = create_linear_model(D, device)
    
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Tracking metrics
    logs = {'train_loss': [], 'train_accuracy': [], 'train_f1': [], 'test_loss': [], 'test_accuracy': [], 'test_f1': []}
    best_loss = float("inf")
    patience_counter = 0  
    
    for epoch in range(epochs):
        #training 
        model.train()
        for batch_features, batch_labels in train_dl:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()

            outputs = model(batch_features.float()).view(-1)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
        #evaluation
        train_avg_loss, train_acc, train_f1 = evaluate_model(model, train_dl, criterion, device)
        test_avg_loss, test_acc, test_f1 = evaluate_model(model, test_dl, criterion, device)
        logs = log_progress(logs, train_avg_loss, train_acc, train_f1, test_avg_loss, test_acc, test_f1, epoch, epochs)
        
        #Potential early stopping
        if logs['train_f1'][-1] >= 0.99:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        if epoch > 0 and (best_loss - logs['train_loss'][-1]) < tolerance:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        else:
            patience_counter = 0  # Reset if improvement is sufficient

        best_loss = min(best_loss, logs['train_loss'][-1])
        
        scheduler.step()
        
    return model.weight.detach().squeeze(0).cpu(), logs


def create_dataloaders(concept, train_embeds, train_all_concept_labels, test_embeds, test_all_concept_labels, 
                       batch_size, balance_data, balance_negatives):
    # Balance dataset potentially (undersampling)
    if balance_data:
        if balance_negatives:
            print("Balancing Overall Data and Negatives")
            train_embeds, train_labels = balance_dataset_evenly_for_ooc(train_embeds, train_all_concept_labels, concept)
            test_embeds, test_labels = balance_dataset_evenly_for_ooc(test_embeds, test_all_concept_labels, concept)
        else:
            print("Balancing Just Overall Data")
            train_embeds, train_labels = balance_dataset(train_embeds, train_all_concept_labels[concept])
            test_embeds, test_labels = balance_dataset(test_embeds, test_all_concept_labels[concept])
    else:
        train_labels = train_all_concept_labels[concept]
        test_labels = test_all_concept_labels[concept]
   
    if train_embeds is None or test_embeds is None:
        return None, None
        
    print_balancing_results(train_labels, test_labels)
    
    #Create Dataloaders
    train_dl = create_dataloader(train_embeds, train_labels, batch_size, shuffle=True)
    test_dl = create_dataloader(test_embeds, test_labels, batch_size, shuffle=False)
    return train_dl, test_dl


def compute_a_linear_separator(
    concept, train_embeds, test_embeds, train_all_concept_labels, test_all_concept_labels,
    lr=0.01, epochs=100, patience=15, tolerance=0.001, batch_size=32, weight_decay=1e-2, 
    lr_step_size=10, lr_gamma=0.5, device='cuda', balance_data=True, balance_negatives=False
):
    """
    Compute a linear separator for a given concept with dataset balancing, weight decay, and LR scheduling.

    Args:
        concept (str): Concept name.
        embeds (torch.Tensor): (N, D) tensor of embeddings.
        concept_gt_patches (set): Indices where the concept is present.
        lr (float): Initial learning rate.
        epochs (int): Max training epochs.
        patience (int): Early stopping patience.
        batch_size (int): Training batch size.
        weight_decay (float): Weight decay for Adam optimizer.
        lr_step_size (int): Step size for LR scheduler.
        lr_gamma (float): Decay factor for LR scheduler.
        device (str): Compute device.
        undersampling_ratio (float): Ratio of in-concept to out-of-concept samples for undersampling in training.

    Returns:
        torch.Tensor: Learned concept weight vector.
        dict: Training & test metrics.
    """
    print(f"Training linear classifier for concept {concept}")
    train_dl, test_dl = create_dataloaders(concept, train_embeds, train_all_concept_labels, test_embeds, test_all_concept_labels, 
                       batch_size, balance_data, balance_negatives)
    
    if train_dl is not None and test_dl is not None:
        #Train Model
        model_weights, logs = train_model(train_dl, test_dl, epochs, lr, weight_decay, lr_step_size, lr_gamma, patience, tolerance, device)
    else:
        #no patches in cluster, just make random model
        model = create_linear_model(train_embeds.shape[1], device)
        model_weights = model.weight.detach().squeeze(0).cpu()
        logs = []
    
    return model_weights, logs


def compute_linear_separators(embeds, gt_samples_per_concept, dataset_name, sample_type, model_input_size, 
                              device='cuda', output_file=None, lr=0.01, epochs=100, batch_size=32, patience=15, 
                              tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5, balance_data=True, 
                              balance_negatives=False):
    """
    Computes linear separators for concepts.
    
    Args:
        embeds: Dictionary mapping concept names to embedding tensors.
        gt_samples_per_concept: Dictionary mapping concept names to lists of positive sample indices.
        dataset_name: Name of the dataset.
        sample_type: Type of sampling method.
        model_input_size: Input size for the model.
        device: Compute device (default: 'cuda').
        output_file: Path to save results (default: None).
        lr, epochs, batch_size, patience, tolerance, weight_decay, lr_step_size, lr_gamma: Training hyperparameters.
        balance_data: Whether to balance positive and negative samples.
        balance_negatives: Whether to balance negative samples across concepts.

    Returns:
        Dictionary containing learned linear separators and logs.
    """
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name, model_input_size=model_input_size)
    elif sample_type == 'image':
        split_df = get_split_df(dataset_name)
    
    concept_names = gt_samples_per_concept.keys()
    
    concept_representations = {}
    logs = {}
    
    #compute labels
    print("Computing labels")
    all_concept_labels = create_binary_labels(embeds.shape[0], gt_samples_per_concept)
    
    print("Sorting by train/test")
    # Separate train and test_data (filtering out patches that don't correspond to any image locations)
    train_embeds, train_all_concept_labels = sort_data_by_split(embeds, all_concept_labels, 'train', dataset_name, model_input_size, sample_type)
    test_embeds, test_all_concept_labels = sort_data_by_split(embeds, all_concept_labels, 'test', dataset_name, model_input_size, sample_type)
    
    for concept_name in tqdm(concept_names):
        linear_separator, concept_logs = compute_a_linear_separator(concept=concept_name, 
                                                                    train_embeds=train_embeds, test_embeds=test_embeds,
                                                                    train_all_concept_labels=train_all_concept_labels,
                                                                    test_all_concept_labels=test_all_concept_labels,
                                                                    lr=lr, epochs=epochs, patience=patience, 
                                                                    tolerance=tolerance, batch_size=batch_size,
                                                                    weight_decay=weight_decay, lr_step_size=lr_step_size,
                                                                    lr_gamma=lr_gamma, device=device,
                                                                    balance_data=balance_data,
                                                                    balance_negatives=balance_negatives)
        concept_representations[concept_name] = linear_separator
        logs[concept_name] = concept_logs
    
    if output_file:
        torch.save(concept_representations, f'Concepts/{dataset_name}/{output_file}')
        print(f"Concepts saved to Concepts/{dataset_name}/{output_file} :)")
        torch.save(logs, f'Concepts/{dataset_name}/logs_{output_file}')
        print(f"Logs saved to Concepts/{dataset_name}/logs_{output_file}")
    
    return concept_representations, logs
  


# def filter_concept_by_patch_activations(split, embeddings, concept_activations, top_percent,
#                                         concept_labels, dataset_name, model_input_size, impose_negatives):
#     """
#     Filters and labels embeddings for a single concept by selecting the top and negative activations.

#     Args:
#         embeddings (torch.Tensor): Subset of all patch embeddings (shape: [n_samples, hidden_dim]).
#         concept_activations (pd.Series): Activation scores for one concept (shape: [n_samples]).
#         top_percent (float): Percentile of top activations to select.
#         impose_negatives (bool): If True, select most negative activations instead of random negatives.

#     Returns:
#         (torch.Tensor, torch.Tensor): Filtered embeddings and corresponding binary labels.
#     """
#     # Get intersection of relevant and training/test indices
#     split_df = get_patch_split_df(dataset_name, model_input_size)
#     nonpadding_indices = filter_patches_by_image_presence(split_df.index, dataset_name, model_input_size).tolist()
#     # print("non-padding indices:", nonpadding_indices[:10])
#     split_indices = split_df[split_df == split].index
#     # print("split indices:", split_indices[:10])
#     # print("nonpadding indices:", nonpadding_indices[:10])
#     relevant_indices = sorted(list(set(split_indices).intersection(nonpadding_indices)))

#     # filter concept activations 
#     concept_activations = concept_activations.loc[relevant_indices]
    
#     # print(concept_activations)
#     # print("embeddings shape:", embeddings.shape)
#     # print("concept labels shape", concept_labels.shape)

#     if concept_labels is not None:
#         # Get GT-positive and GT-negative index sets
#         pos_gt = sorted((concept_labels == 1).nonzero(as_tuple=True)[0].tolist())
#         neg_gt = sorted((concept_labels == 0).nonzero(as_tuple=True)[0].tolist())
        
#         # Keep only relevant ones
#         pos_gt = sorted(list(set(pos_gt).intersection(set(relevant_indices))))
#         neg_gt = sorted(list(set(neg_gt).intersection(set(relevant_indices))))
        
#         n_total = min(len(pos_gt), len(neg_gt))
#         n_top = int(np.ceil(top_percent * n_total))

#         # Restrict activations to only positive and negative GTs
#         pos_activations = concept_activations.loc[relevant_indices]
#         neg_activations = concept_activations.loc[relevant_indices]
        
#         # Take top activations *within* GT-positive and GT-negative
#         pos_indices = pos_activations.nlargest(n_top).index.to_numpy()

#         if impose_negatives:
#             neg_indices = neg_activations.nsmallest(n_top).index.to_numpy()
#         else:
#             neg_indices = np.random.choice(neg_activations.index, size=n_top, replace=False)

#         n_to_sample = min(len(pos_indices), len(neg_indices))
#         pos_indices = np.random.choice(pos_indices, size=n_to_sample, replace=False)
#         neg_indices = np.random.choice(neg_indices, size=n_to_sample, replace=False)
#         # print(pos_indices[:5])

#         # from general_utils import load_images
#         # from visualize_concepts_w_samples_utils import plot_patches_w_corr_images
#         # import visualize_concepts_w_samples_utils
#         # importlib.reload(visualize_concepts_w_samples_utils)
#         # all_images, train_images, test_images = load_images(dataset_name='CLEVR')
#         # print("pos samples", pos_indices.tolist()[:7])
#         # plot_patches_w_corr_images(pos_indices.tolist()[:7], concept_activations, all_images, 'Blah', (224, 224),
#         #                        save_path=None, patch_size=14, metric_type='CosSim')
#         # print("neg samples", neg_indices.tolist()[:7])
#         # plot_patches_w_corr_images(neg_indices.tolist()[:7], concept_activations, all_images, 'Blah', (224, 224),
#         #                        save_path=None, patch_size=14, metric_type='CosSim')
        
#         all_indices = np.concatenate([pos_indices, neg_indices])
#         labels = concept_labels[all_indices]

#     else:
#         n_total = len(concept_activations)
#         n_top = int(np.ceil(top_percent * n_total))
#         top_indices = concept_activations.nlargest(n_top).index.to_numpy()
#         if impose_negatives:
#             bot_indices = concept_activations.nsmallest(n_top).index.to_numpy()
#         else:
#             remaining_indices = concept_activations.drop(index=top_indices).index.to_numpy()
#             bot_indices = np.random.choice(remaining_indices, size=n_top, replace=False)
        
#         labels = np.concatenate([np.ones(n_top), np.zeros(n_top)])
#         all_indices = np.concatenate([top_indices, bot_indices])
#         labels = torch.tensor(labels, dtype=torch.float32)

#     # print("first 5 indices:", all_indices)
#     filtered_embeddings = embeddings[all_indices]
#     return filtered_embeddings, labels

# def filter_concept_by_patch_activations(split, embeddings, concept_activations, top_percent,
#                                         concept_labels, dataset_name, model_input_size, impose_negatives):
#     """
#     Filters and labels embeddings for a single concept by selecting the top and negative activations.

#     Args:
#         embeddings (torch.Tensor): Subset of all patch embeddings (shape: [n_samples, hidden_dim]).
#         concept_activations (pd.Series): Activation scores for one concept (shape: [n_samples]).
#         top_percent (float): Percentile of top activations to select.
#         impose_negatives (bool): If True, select most negative activations instead of random negatives.

#     Returns:
#         (torch.Tensor, torch.Tensor): Filtered embeddings and corresponding binary labels.
#     """
#     # Get intersection of relevant and training/test indices
#     split_df = get_patch_split_df(dataset_name, model_input_size)
#     nonpadding_indices = filter_patches_by_image_presence(split_df.index, dataset_name, model_input_size).tolist()
#     # print("non-padding indices:", nonpadding_indices[:10])
#     split_indices = split_df[split_df == split].index
#     relevant_indices = list(set(split_indices).intersection(nonpadding_indices))
#     # print("relevant indices:", len(relevant_indices))

#     #filter concept activations 
#     concept_activations = concept_activations.iloc[relevant_indices]
    
#     # print(concept_activations)
#     # print("embeddings shape:", embeddings.shape)
#     # print("concept labels shape", concept_labels.shape)
    
#     n_total = len(concept_activations)
#     n_top = int(np.ceil(top_percent * n_total))

#     top_indices = concept_activations.nlargest(n_top).index.to_numpy()

#     if impose_negatives:
#         bot_indices = concept_activations.nsmallest(n_top).index.to_numpy()
#     else:
#         remaining_indices = concept_activations.drop(index=top_indices).index.to_numpy()
#         bot_indices = np.random.choice(remaining_indices, size=n_top, replace=False)
    
#     if concept_labels is not None:
#         pos_gt = (concept_labels == 1).nonzero(as_tuple=True)[0]
#         neg_gt = (concept_labels == 0).nonzero(as_tuple=True)[0]
        
#         pos_candidates = list(set(pos_gt.tolist()).intersection(set(top_indices.tolist())))
#         neg_candidates = list(set(neg_gt.tolist()).intersection(set(bot_indices.tolist())))
        
#         n_to_sample = min(len(pos_candidates), len(neg_candidates))
#         # Sample equally from both
#         pos_indices = np.random.choice(pos_candidates, size=n_to_sample, replace=False)
#         neg_indices = np.random.choice(neg_candidates, size=n_to_sample, replace=False)
#         # print(pos_indices[:5])
        
#         # from general_utils import load_images
#         # from visualize_concepts_w_samples_utils import plot_patches_w_corr_images
#         # import visualize_concepts_w_samples_utils
#         # importlib.reload(visualize_concepts_w_samples_utils)
#         # all_images, train_images, test_images = load_images(dataset_name='CLEVR')
#         # print("pos samples", pos_indices.tolist()[:7])
#         # plot_patches_w_corr_images(pos_indices.tolist()[:7], concept_activations, all_images, 'Blah', (224, 224),
#         #                        save_path=None, patch_size=14, metric_type='CosSim')
#         # print("neg samples", neg_indices.tolist()[:7])
#         # plot_patches_w_corr_images(neg_indices.tolist()[:7], concept_activations, all_images, 'Blah', (224, 224),
#         #                        save_path=None, patch_size=14, metric_type='CosSim')
        
#         all_indices = np.concatenate([pos_indices, neg_indices])
#         labels = concept_labels[all_indices]

#     else:
#         labels = np.concatenate([np.ones(n_top), np.zeros(n_top)])
#         all_indices = np.concatenate([top_indices, bot_indices])
#         labels = torch.tensor(labels, dtype=torch.float32)

#     # print("first 5 indices:", all_indices)
#     filtered_embeddings = embeddings[all_indices]
#     return filtered_embeddings, labels


def filter_embeddings_by_patch_activations(embeddings, act_metrics, gt_samples_per_concept, top_percent, split, dataset_name,
                                           model_input_size, use_gt_labels=True,
                                           impose_negatives=False):
    """
    For each concept, select the top n% of *split* samples based on cosine similarity and
    randomly sample the same number from the rest. Return filtered embeddings and binary labels.

    Args:
        embeddings (torch.Tensor): Tensor of shape (n_samples, hidden_dim)
        act_metrics (pd.DataFrame): DataFrame of shape (n_samples, n_concepts) with cosine similarities
        top_percent (float): Percentage (0 < top_percent < 1) of top samples to select
        split (str): 'train' or 'test'
        dataset_name (str): Name of dataset (used to get train/test split)
        model_input_size (tuple): Needed for indexing the split
        impose_negatives (bool): If True, selects most negative activations instead of random negatives.

    Returns:
        dict: Dictionary mapping each concept to a tuple (filtered_embeddings, labels)
    """
    assert 0 < top_percent < 1, "top_percent must be between 0 and 1"
    np.random.seed(42)

    concept_names = act_metrics.columns
    selected_embeddings, selected_labels = {}, {}
    
    if use_gt_labels: #use actual labels
        all_concept_labels = create_binary_labels(embeddings.shape[0], gt_samples_per_concept)
    else: #consider only superpatches as 'positive' examples
        all_concept_labels = {concept:None for concept in concept_names}
        
    for concept in concept_names:
        # concept_embeds, labels = filter_concept_by_patch_activations(
        #     embeddings, act_metrics[concept].to_numpy(), top_percent, all_concept_labels[concept], impose_negatives
        # )
        concept_embeds, labels = filter_concept_by_patch_activations(split, embeddings, act_metrics[concept], top_percent,
                                        all_concept_labels[concept], dataset_name, model_input_size, impose_negatives)
        selected_embeddings[concept] = concept_embeds
        selected_labels[concept] = labels

    print(f"Selecting top {top_percent*100}% {split} patches ({len(labels) // 2})")

    return selected_embeddings, selected_labels



def compute_linear_separators_w_superpatches(top_per, embeds, original_dists, gt_samples_per_concept, dataset_name,
                                             model_input_size, 
                                  device='cuda', output_file=None, lr=0.01, epochs=100, batch_size=32, patience=15, 
                                  tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5, balance_data=True, 
                                  balance_negatives=False, impose_negatives=False):
    """
    Computes linear separators for concepts.
    
    Args:
        embeds: Dictionary mapping concept names to embedding tensors.
        gt_samples_per_concept: Dictionary mapping concept names to lists of positive sample indices.
        dataset_name: Name of the dataset.
        sample_type: Type of sampling method.
        model_input_size: Input size for the model.
        device: Compute device (default: 'cuda').
        output_file: Path to save results (default: None).
        lr, epochs, batch_size, patience, tolerance, weight_decay, lr_step_size, lr_gamma: Training hyperparameters.
        balance_data: Whether to balance positive and negative samples.
        balance_negatives: Whether to balance negative samples across concepts.

    Returns:
        Dictionary containing learned linear separators and logs.
    """  
    # Might have removeed concepts originally
    curr_concepts = list(original_dists.columns)
    gt_samples_per_concept = {c: samples for c, samples in gt_samples_per_concept.items() if c in curr_concepts}
    
    concept_representations = {}
    logs = {}
    
    # Separate train and test_data (filtering out patches that don't correspond to any image locations)
    train_all_concept_embeds, train_all_concept_labels = filter_embeddings_by_patch_activations(embeds, original_dists, top_per, 
                                                                         'train', dataset_name, model_input_size,
                                                                        impose_negatives=impose_negatives)
    test_all_concept_embeds, test_all_concept_labels = filter_embeddings_by_patch_activations(embeds, original_dists, top_per, 
                                                                         'test', dataset_name, model_input_size,
                                                                        impose_negatives=impose_negatives)
    
    for concept_name in tqdm(curr_concepts):
        train_embeds = train_all_concept_embeds[concept_name]
        test_embeds = test_all_concept_embeds[concept_name]
        linear_separator, concept_logs = compute_a_linear_separator(concept=concept_name, 
                                                                    train_embeds=train_embeds, test_embeds=test_embeds,
                                                                    train_all_concept_labels=train_all_concept_labels,
                                                                    test_all_concept_labels=test_all_concept_labels,
                                                                    lr=lr, epochs=epochs, patience=patience, 
                                                                    tolerance=tolerance, batch_size=batch_size,
                                                                    weight_decay=weight_decay, lr_step_size=lr_step_size,
                                                                    lr_gamma=lr_gamma, device=device,
                                                                    balance_data=balance_data,
                                                                    balance_negatives=balance_negatives)
        concept_representations[concept_name] = linear_separator
        logs[concept_name] = concept_logs
    
    if output_file:
        torch.save(concept_representations, f'Concepts/{dataset_name}/{output_file}')
        print(f"Concepts saved to Concepts/{dataset_name}/{output_file} :)")
        torch.save(logs, f'Concepts/{dataset_name}/logs_{output_file}')
        print(f"Logs saved to Concepts/{dataset_name}/logs_{output_file}")
    
    return concept_representations, logs


def compute_linear_separators_w_superpatches_across_pers(top_pers, embeds, original_dists, gt_samples_per_concept, 
                                                         dataset_name, model_input_size, device='cuda', output_file=None,
                                                         lr=0.01, epochs=100, batch_size=32, patience=15, 
                                  tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5, balance_data=True, 
                                  balance_negatives=False, impose_negatives=False):
    for top_per in top_pers:
        if impose_negatives:
            per_output_file = f'imposeneg_per_{top_per}_{output_file}'
        else:
            per_output_file = f'per_{top_per}_{output_file}'
        print(f"Computing classifiers using {top_per* 100}% superpatches")
        # if per_output_file in os.listdir(f'Concepts/{dataset_name}'): #skip if already computed
        #     continue
        # else:
        compute_linear_separators_w_superpatches(top_per, embeds, original_dists, gt_samples_per_concept, dataset_name, 
                                                 model_input_size, device=device, output_file=per_output_file, lr=lr, 
                                                 epochs=epochs, batch_size=batch_size, patience=patience, 
                                                  tolerance=tolerance, weight_decay=weight_decay, 
                                                 lr_step_size=lr_step_size, lr_gamma=lr_gamma,
                                                  balance_data=balance_data, 
                                                  balance_negatives=balance_negatives,
                                                  impose_negatives=impose_negatives)
        
        
# def finetune_linear_separators_w_superpatches(fine_tuning_params, concepts_across_iterations, all_logs, embeds, dataset_name, 
#                                               model_input_size, device, batch_size, lr, weight_decay,
#                                               lr_step_size, lr_gamma, patience, tolerance, all_concept_labels, impose_negatives):
#     """
#     Fine-tunes a set of linear classifiers for each concept using top-k% superpatches selected
#     based on current signed distances from the decision boundary.

#     For each concept, the function filters patches whose activations fall in the top percentile range, 
#     retrains the classifier on those patches, and updates the weight vector and training logs.

#     Args:
#         fine_tuning_params (list): List of (per_superpatches, n_epochs) for training.
#         curr_weights (dict): Current concept -> weight vector.
#         all_logs (dict): Current concept -> list of logs from previous training rounds.
#         embeds (torch.Tensor): Patch-level embeddings (n_patches, embed_dim).
#         curr_dists (pd.DataFrame): Concept-wise signed distances of patches to the decision boundary.
#         dataset_name (str): Name of the dataset (used for patch filtering).
#         model_input_size (tuple): Image input size for indexing patch location.
#         device (str): Device to train on (e.g., 'cuda').
#         batch_size (int): Mini-batch size for training.
#         lr (float): Learning rate.
#         weight_decay (float): L2 regularization.
#         lr_step_size (int): Step size for learning rate decay.
#         lr_gamma (float): Multiplicative decay factor for learning rate.
#         patience (int): Early stopping patience.
#         tolerance (float): Minimum improvement to continue training.
#         impose_negatives (bool): If True, selects most negative patches as counterexamples.

#     Returns:
#         all_logs (dict): Updated logs per concept after all rounds.
#         curr_weights (dict): Updated weights per concept after fine-tuning.
#     """
#     curr_weights = copy.deepcopy(concepts_across_iterations[0])
#     for i, (per, epochs) in enumerate(fine_tuning_params):
#         if per == 'init':
#             continue
            
#         print(f"Fine tuning model with top {per*100}% of superpatches")
#         per_logs = {}
#         #compute distances using last round of training
#         curr_dists = compute_signed_distances(embeds, curr_weights, dataset_name, 
#                                               device, output_file=None, batch_size=512)
#         for concept, concept_weights in curr_weights.items():
#             #get the training embeds/labels for the next round of fine-tuning based on per% superpatches
#             train_embeds, train_labels = filter_concept_by_patch_activations('train', embeds, curr_dists[concept], per,
#                                         all_concept_labels[concept], dataset_name, model_input_size, impose_negatives)
#             test_embeds, test_labels = filter_concept_by_patch_activations('test', embeds, curr_dists[concept], per,
#                                         all_concept_labels[concept], dataset_name, model_input_size, impose_negatives)
            
#             # train_embeds, train_all_concept_labels = sort_data_by_split(embeds, all_concept_labels, 'train', dataset_name, model_input_size, 'patch')
#             # test_embeds, test_all_concept_labels = sort_data_by_split(embeds, all_concept_labels, 'test', dataset_name, model_input_size, 'patch')
            
#             print(f"Fine-tuning concept {concept} {len(train_embeds)} training samples, {len(test_embeds)} test samples")
#             if train_embeds.shape[0] > 0 and test_embeds.shape[0] > 0:                                                       
#                 train_dl = create_dataloader(train_embeds, train_labels, batch_size, shuffle=True)
#                 test_dl = create_dataloader(test_embeds, test_labels, batch_size, shuffle=False)
#                 # train_dl = create_dataloader(train_embeds, train_all_concept_labels[concept], batch_size, shuffle=False)
#                 # test_dl = create_dataloader(test_embeds, test_all_concept_labels[concept], batch_size, shuffle=False)
#                 # train_dl, test_dl = create_dataloaders(concept, train_embeds, train_all_concept_labels, test_embeds, test_all_concept_labels, 
#                 #        batch_size, balance_data=True, balance_negatives=False)
                
                
#                 model = create_linear_model(train_embeds.shape[1], device, weights=concept_weights)
#                 concept_curr_weights, concept_curr_logs = train_model(train_dl, test_dl, epochs, lr, weight_decay, 
#                                                     lr_step_size, lr_gamma, patience, tolerance, device, model=model)
            
#             per_logs[concept] = concept_curr_logs
#             curr_weights[concept] = concept_curr_weights
#         concepts_across_iterations.append(copy.deepcopy(curr_weights))
#         all_logs.append(per_logs)
            
#     return all_logs, concepts_across_iterations
                
    
# def compute_linear_separators_finetuned_w_superpatches(fine_tuning_params, embeds, gt_samples_per_concept, 
#                                              dataset_name, model_input_size, device='cuda', output_file=None,
#                                              lr=0.01, batch_size=32, patience=15, 
#                                              tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5,
#                                               balance_data=True, balance_negatives=False, use_gt_labels=True,
#                                              impose_negatives=False):
#     """
#     Trains and fine-tunes linear classifiers for each concept using increasing percentages of 
#     superpatch activations.

#     The function first trains an initial set of linear classifiers using balanced data, then iteratively
#     fine-tunes each classifier using the top-k% of patches most activated by the previous round's weights.

#     Args:
#         fine_tuning_params (list): List of (per_superpatches, n_epochs) for training.
#         embeds (torch.Tensor): Patch-level embeddings of shape (n_patches, embed_dim).
#         gt_samples_per_concept (dict): Mapping from concept name to indices of patches with that concept.
#         dataset_name (str): Name of the dataset (used for patch filtering).
#         model_input_size (tuple): Size of input images (used for patch indexing).
#         device (str): Device identifier (e.g., 'cuda').
#         output_file (str or None): If provided, used to save logs or weights.
#         lr (float): Learning rate.
#         batch_size (int): Training batch size.
#         patience (int): Early stopping patience.
#         tolerance (float): Minimum improvement threshold for early stopping.
#         weight_decay (float): Weight decay for regularization.
#         lr_step_size (int): Step size for LR scheduler.
#         lr_gamma (float): Decay factor for LR scheduler.
#         balance_data (bool): Whether to balance classes during initial training.
#         balance_negatives (bool): Whether to balance negatives specifically during training.
#         impose_negatives (bool): If True, use most negative activations as negatives during fine-tuning.

#     Returns:
#         final_logs (dict): Mapping from concept -> list of training logs per fine-tuning round.
#         final_concept_weights (dict): Mapping from concept -> final learned weight tensor.
#     """
#     #create an init model and rn it for a couple epochs
#     print("Initial Training")
#     init_weights, init_logs = compute_linear_separators(embeds, gt_samples_per_concept, dataset_name, 'patch', model_input_size, 
#                               device=device, output_file=None, lr=lr, epochs=fine_tuning_params[0][1], batch_size=batch_size,
#                               patience=patience, 
#                               tolerance=tolerance, weight_decay=weight_decay, lr_step_size=lr_step_size, lr_gamma=lr_gamma,
#                               balance_data=balance_data, balance_negatives=balance_negatives) 
    
#     if use_gt_labels: #use actual labels
#         all_concept_labels = create_binary_labels(embeds.shape[0], gt_samples_per_concept)
#     else: #consider only superpatches as 'positive' examples
#         all_concept_labels = {concept:None for concept in concept_names}
        
#     final_logs, concepts_across_iterations = finetune_linear_separators_w_superpatches(fine_tuning_params, [init_weights],
#                                                                                        [init_logs], 
#                                                                                   embeds, dataset_name, 
#                                                                                   model_input_size, device, batch_size, lr,
#                                                                                   weight_decay,
#                                                                                   lr_step_size, lr_gamma, patience, tolerance,
#                                                                                   all_concept_labels, impose_negatives)
#     if output_file:
#         out = f'finetuned_{fine_tuning_params}_{output_file}'
#         if impose_negatives:
#             out = 'impose_neg_' + out
#         if use_gt_labels:
#             out = 'gtlabels_'+ out
            
#         torch.save(concepts_across_iterations, f'Concepts/{dataset_name}/{out}')
#         print(f"Concepts saved to Concepts/{dataset_name}/{out} :)")
#         torch.save(final_logs, f'Concepts/{dataset_name}/logs_{out}')
#         print(f"Logs saved to Concepts/{dataset_name}/logs_{out}")
        
#     return concepts_across_iterations, final_logs
    
        
###For Computing Similarity Metrics###
# def compute_cosine_sims(embeddings, concepts, output_file, dataset_name, device, batch_size=32):
#     """
#     Compute cosine similarity between each image embedding and each concept vector in batches,
#     and save the resulting DataFrame to a CSV file.

#     Args:
#         embeddings (torch.Tensor): Tensor of image embeddings of shape (n_samples, n_features).
#         concepts (dict): Mapping from concept names to their embedding tensors.
#         output_file (str): Filename to save the cosine similarities.
#         dataset_name (str): The name of the dataset.
#         device (torch.device or str): Device on which to perform computations.
#         batch_size (int): Number of images to process per batch.

#     Returns:
#         pd.DataFrame: DataFrame with one row per image and one column per concept.
#     """
#     if dataset_name == 'Coco' and 'kmeans' not in output_file:
#         concept_keys = filter_coco_concepts(list(concepts.keys()))
#     else:
#         concept_keys = list(concepts.keys())
               
#     # Move embeddings and concept embeddings to the specified device
#     all_concept_embeddings = {k: v.to(device) for k, v in concepts.items() if k in concept_keys}
    
#     # Create a tensor for all concept embeddings in a fixed order
#     all_concept_embeddings_tensor = torch.stack([all_concept_embeddings[k] for k in concept_keys])
    
#     cosine_similarity_rows = []
#     n_images = embeddings.shape[0]
    
#     for i in tqdm(range(0, n_images, batch_size), desc="Processing batches"):
#         # Get batch embeddings
#         batch_embeddings = embeddings[i:i+batch_size]
#         batch_embeddings = batch_embeddings.to(device)
#         # Compute cosine similarity in a vectorized way:
#         # batch_embeddings: (batch_size, n_features)
#         # all_concept_embeddings_tensor: (n_concepts, n_features)
#         # After unsqueezing and computing similarity, result shape: (batch_size, n_concepts)
#         cosine_similarities = F.cosine_similarity(
#             batch_embeddings.unsqueeze(1), 
#             all_concept_embeddings_tensor.unsqueeze(0), 
#             dim=2
#         )
#         # Move the result to CPU and convert to list of rows
#         batch_sims = cosine_similarities.cpu().tolist()
#         # Create dictionary rows where keys are concept names
#         batch_rows = [dict(zip(concept_keys, row)) for row in batch_sims]
#         cosine_similarity_rows.extend(batch_rows)
    
#     # Create the DataFrame from all similarity rows
#     cosine_similarity_df = pd.DataFrame(cosine_similarity_rows)
    
#     # Save the DataFrame if an output filename is provided
#     if output_file and output_file != 'kmeans':
#         base_path = f'Cosine_Similarities/{dataset_name}/'
#         output_path = os.path.join(base_path, output_file)
#         cosine_similarity_df.to_csv(output_path, index=False)
#         print(f"Cosine similarity results saved at {output_path}")
    
#     return cosine_similarity_df

def write_batch_cosine_sims(writer, embeddings, i, batch_size, device, 
                            all_concept_embeddings_tensor, concept_keys):
    """
    Writes a batch of cosine similarities to CSV using a given writer.

    Args:
        writer (csv.DictWriter): Open CSV writer
        concept_keys (list of str): Concept names in the correct order
        cosine_sim_tensor (torch.Tensor): Tensor of shape [batch_size, num_concepts]
    """
    batch_embeddings = embeddings[i:i+batch_size].to(device)

    cosine_similarities = F.cosine_similarity(
        batch_embeddings.unsqueeze(1),
        all_concept_embeddings_tensor.unsqueeze(0),
        dim=2
    )
    batch_sims = cosine_similarities.cpu().tolist()
    for row in batch_sims:
        writer.writerow(dict(zip(concept_keys, row)))
    del batch_embeddings, cosine_similarities, batch_sims
        

def compute_cosine_sims(embeddings, concepts, output_file, dataset_name, device, scratch_dir='', batch_size=32):
    if dataset_name == 'Coco' and 'kmeans' not in output_file:
        concept_keys = filter_coco_concepts(list(concepts.keys()))
    else:
        concept_keys = list(concepts.keys())

    all_concept_embeddings = {k: v.to(device) for k, v in concepts.items() if k in concept_keys}
    all_concept_embeddings_tensor = torch.stack([all_concept_embeddings[k] for k in concept_keys])

    base_path = f'{scratch_dir}Cosine_Similarities/{dataset_name}/'
    os.makedirs(base_path, exist_ok=True)
    output_path = os.path.join(base_path, output_file)

    with open(output_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=concept_keys)
        writer.writeheader()

        with torch.no_grad():
            for i in tqdm(range(0, embeddings.shape[0], batch_size), desc="Computing cosine similarities"):
                write_batch_cosine_sims(writer, embeddings, i, batch_size, device, 
                                       all_concept_embeddings_tensor, concept_keys)
    print(f"Cosine similarity results saved at {output_path}")


def write_signed_distance_batch_to_csv(writer, cluster_ids, batch_embeds, cluster_weights, device):
    """
    Vectorized version: Computes and writes signed distances for a batch to a CSV writer.

    Args:
        writer (csv.DictWriter): CSV writer object
        cluster_ids (list): Cluster IDs as strings (column names)
        batch_embeds (Tensor): [B, D] embeddings
        cluster_weights (dict): cluster_id (str) -> weight tensor (on CPU)
        device (str): 'cuda' or 'cpu'
    """
    # Stack all weight vectors: [C, D]
    weight_matrix = torch.stack([cluster_weights[cid] for cid in cluster_ids]).to(device).to(batch_embeds.dtype)

    # Normalize weight vectors: [C, D]
    weight_matrix = torch.nn.functional.normalize(weight_matrix, dim=1)

    # Normalize embeddings: [B, D]
    batch_embeds = torch.nn.functional.normalize(batch_embeds, dim=1)

    # Compute cosine similarities in one go: [B, C]
    sims = batch_embeds @ weight_matrix.T

    # Convert to list of dicts (one row per sample)
    for row in sims.cpu().tolist():
        writer.writerow(dict(zip(cluster_ids, row)))
        
def write_dist_row(writer, embeds, i, batch_size, device, weight_matrix, concept_ids):
    batch = embeds[i:i+batch_size].to(device)
    sims = batch @ weight_matrix.T
    for row in sims.cpu().tolist():
        writer.writerow(dict(zip(concept_ids, row)))
    del batch
    torch.cuda.empty_cache()
        
def compute_signed_distances(embeds, concepts, dataset_name, device, output_file, scratch_dir, batch_size=100):
    """
    Computes signed distances between embeddings and cluster directions, resuming if partially written.

    Args:
        embeds (Tensor): [N, D] embeddings
        cluster_weights (dict): cluster_id -> weight tensor (1D)
        dataset_name (str): Used for output folder
        device (str): 'cuda' or 'cpu'
        output_file (str): Filename to write to
        batch_size (int): Batch size for processing
    """
    concept_ids = [str(k) for k in concepts.keys()]
    output_path = os.path.join(f"{scratch_dir}Distances", dataset_name, output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check how many rows already written
    start_idx = 0
    # if os.path.exists(output_path):
    #     try:
    #         with open(output_path, 'r') as f:
    #             row_count = sum(1 for _ in f) - 1  # subtract header
    #             if row_count > 0:
    #                 start_idx = row_count
    #                 print(f"Resuming from row {start_idx}")
    #     except Exception as e:
    #         print(f"Error checking existing file: {e}")

    # Prepare weight matrix outside loop
    weight_matrix = torch.stack([concepts[cid] for cid in concept_ids]).to(device).to(embeds.dtype)
    # weight_matrix = torch.nn.functional.normalize(weight_matrix, dim=1)

    mode = 'a' if start_idx > 0 else 'w'
    with open(output_path, mode=mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=concept_ids)
        if start_idx == 0:
            writer.writeheader()

        with torch.no_grad():
            for i in tqdm(range(start_idx, embeds.shape[0], batch_size), desc="Writing signed distances"):
                write_dist_row(writer, embeds, i, batch_size, device, weight_matrix, concept_ids)

    print(f"Saved signed distances to: {output_path}")
    
# def compute_signed_distances(embeds, cluster_weights, dataset_name, device, output_file, scratch_dir, batch_size=100):
#     """
#     Computes signed distances between embeddings and cluster directions.

#     Args:
#         embeds (Tensor): [N, D] embeddings
#         cluster_weights (dict): cluster_id -> weight tensor (1D)
#         dataset_name (str): Used for output folder
#         device (str): 'cuda' or 'cpu'
#         output_file (str): Filename to write to
#         batch_size (int): Batch size for processing
#     """
#     cluster_ids = [str(k) for k in cluster_weights.keys()]
#     output_path = os.path.join(f"{scratch_dir}Distances", dataset_name, output_file)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     with open(output_path, mode='w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=cluster_ids)
#         writer.writeheader()

#         with torch.no_grad():
#             for i in tqdm(range(0, embeds.shape[0], batch_size), desc="Writing signed distances"):
#                 batch = embeds[i:i+batch_size].to(device)
#                 write_signed_distance_batch_to_csv(writer, cluster_ids, batch, cluster_weights, device)
#                 del batch
#                 torch.cuda.empty_cache()

#     print(f"Saved signed distances to: {output_path}")
    

# def compute_signed_distances(embeds, concept_weights, dataset_name, device, output_file=None, batch_size=100):
#     """
#     Compute signed distances for each test sample in batches and save as a DataFrame.

#     Args:
#         embeds (torch.Tensor): Tensor of shape (N, D), test embeddings.
#         concept_weights (dict): Dict mapping concept names to learned weight vectors.
#         dataset_name (str): Name of the dataset.
#         device (str): Device to run computations on ('cuda' or 'cpu').
#         output_file (str, optional): File path to save the DataFrame.
#         batch_size (int, optional): Number of samples per batch. Default is 100.

#     Returns:
#         pd.DataFrame: DataFrame where rows are samples and columns are concepts.
#     """
#     num_samples = embeds.shape[0]
#     concept_names = list(concept_weights.keys())
#     if dataset_name == 'Coco' and output_file is not None and 'kmeans' not in output_file:
#         concept_names = filter_coco_concepts(concept_names)

#     # Storage for signed distances (kept on CPU)
#     signed_distances = {name: [] for name in concept_names}

#     num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches

#     with torch.no_grad():  # Disable autograd for memory efficiency
#         for i in tqdm(range(num_batches), desc="Computing signed distances"):
#             batch_embeds = embeds[i * batch_size : (i + 1) * batch_size].to(device)  # Move only batch to GPU

#             for concept_name, weight_vector in concept_weights.items():
#                 if concept_name in concept_names:
#                     weight_vector = weight_vector.to(device).to(batch_embeds.dtype)  # Move to GPU per batch
#                     norm_weight = torch.norm(weight_vector, p=2)
#                     batch_distances = (batch_embeds @ weight_vector) / norm_weight
#                     signed_distances[concept_name].append(batch_distances.cpu())  # Move back to CPU

#                     del weight_vector  # Free memory
#                     torch.cuda.empty_cache()  # Clear unused GPU memory

#             del batch_embeds  # Free batch memory
#             torch.cuda.empty_cache()  # Extra safety

#     # Concatenate and create DataFrame
#     signed_dist_df = pd.DataFrame({k: torch.cat(v, dim=0).numpy() for k, v in signed_distances.items()})

#     if output_file is not None:
#         output_path = f"Distances/{dataset_name}/{output_file}"
#         signed_dist_df.to_csv(output_path, index=False)
#         print(f"Signed distances saved to {output_path}")

#     return signed_dist_df


def compute_zscore_stats(dataset_name, dists_file, con_label, scratch_dir=""):
    """
    Computes mean and std per concept from a CSV of cosine similarities or distances.

    Args:
        calibration_csv_path (str): Path to CSV with shape [N_samples, N_concepts]

    Returns:
        pd.Series, pd.Series: (mean_per_concept, std_per_concept)
    """
    zscore_path = f'{scratch_dir}Distances/{dataset_name}/zscores_{con_label}.pt'
    if os.path.exists(zscore_path):
        print(f"Already computed z scores for {dataset_name}")
        return
    raw_dists = pd.read_csv(f'{scratch_dir}Distances/{dataset_name}/{dists_file}')
    mean = raw_dists.mean(axis=0)
    std = raw_dists.std(axis=0).replace(0, 1e-6)  # avoid divide-by-zero
    torch.save({'mean':mean, 'std':std},
               f'{scratch_dir}Distances/{dataset_name}/zscores_{con_label}.pt')
    print(f'Z scores saved to {scratch_dir}Distances/{dataset_name}/zscores_{con_label}.pt')


def apply_zscore_normalization(dists_file, dataset_name, con_label, scratch_dir=""):
    """
    Applies z-score normalization to activations using precomputed stats.

    Args:
        activation_csv_path (str): Path to CSV of activations to normalize
        mean (pd.Series): Mean per concept from calibration set
        std (pd.Series): Std per concept from calibration set

    Returns:
        pd.DataFrame: Z-scored activations, same shape as input
    """
    if 'Cal' in dataset_name:
        zscore_path = f'{scratch_dir}Distances/{dataset_name}/zscores_{con_label}.pt'
    else:
        zscore_path = f'{scratch_dir}Distances/{dataset_name}-Cal/zscores_{con_label}.pt'
           
    zscore_stats = torch.load(zscore_path, weights_only=False)
    mean, std = zscore_stats['mean'], zscore_stats['std']
    
    raw_dists = pd.read_csv(f'{scratch_dir}Distances/{dataset_name}/{dists_file}')
    zscored_dists = (raw_dists - mean) / std
        
    output_path = f'{scratch_dir}Distances/{dataset_name}/zscored_{dists_file}'
    zscored_dists.to_csv(output_path, index=False)
    print(f'Z-scored distances saved to {output_path}')
    return zscored_dists
    

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

def evaluate_clustering_metrics(n_clusters_list, embeddings, dataset_name, device, model_input_size, concepts_filenames=None, sample_type='patch'):
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
    #separate embeddings into test and train
    relevant_indices = torch.arange(embeddings.shape[0])
    if sample_type == 'patch':
        split_df = get_patch_split_df(dataset_name)
        # Filter patches that are 'padding' given the preprocessing schemes
        relevant_indices = filter_patches_by_image_presence(relevant_indices, dataset_name, model_input_size).tolist()
        
    elif sample_type == 'cls':
        split_df = get_split_df(dataset_name)
        relevant_indices = split_df.index
    
    
    # Get train and test image indices from split_df
    train_image_indices = split_df[split_df == 'train'].index
    test_image_indices = split_df[split_df == 'test'].index
    train_relevant_indices = [idx for idx in relevant_indices if idx in train_image_indices]
    test_relevant_indices = [idx for idx in relevant_indices if idx in test_image_indices]

    train_embeddings = embeddings[train_relevant_indices]
    test_embeddings = embeddings[test_relevant_indices]
    
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

    for i, n_clusters in enumerate(n_clusters_list):
        # Get the clustering results
        
        #no saving
        # train_labels, test_labels, cluster_centers = run_fast_pytorch_kmeans(n_clusters, train_embeddings, test_embeddings, device)
        
        #saving
        concepts_filename = concepts_filenames[i]
        if not os.path.exists(f'Concepts/{dataset_name}/cluster_centers_{concepts_filename}'):
            gpu_kmeans(n_clusters, embeddings, dataset_name, device, model_input_size, concepts_filename, sample_type=sample_type, map_samples=False)
        cluster_centers = torch.load(f'Concepts/{dataset_name}/cluster_centers_{concepts_filename}', weights_only=False)
        train_labels = torch.load(f'Concepts/{dataset_name}/train_labels_{concepts_filename}', weights_only=False)
        test_labels = torch.load(f'Concepts/{dataset_name}/test_labels_{concepts_filename}', weights_only=False)
        

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
        
        axes[i].plot(train_metric, label=f"Train {metric_type}", color='blue', marker='o')
        axes[i].plot(test_metric, label=f"Test {metric_type}", color='red', marker='o')
        
        axes[i].set_title(f"{metric_type} for {concept}")
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(metric_type)
        axes[i].legend()

    # Hide empty subplots if the number of concepts isn't a multiple of 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    

def plot_avg_train_history(train_history, metric_type, concepts=None):
    """
    Plots the average train and test metrics over epochs across all (or selected) concepts.
    Shorter concept histories are padded by repeating the final value.
    
    Args:
        train_history (dict): concept -> {'train_*': [...], 'test_*': [...]}
        metric_type (str): Metric name like 'loss', 'accuracy', or 'f1'.
        concepts (list of str, optional): Which concepts to include. Defaults to all.
    """
    if concepts is not None:
        train_history = {k: v for k, v in train_history.items() if k in concepts}

    # Get the max number of epochs across all concepts
    max_epochs = max(len(v[f'train_{metric_type}']) for v in train_history.values())

    def pad_to_length(arr, length):
        """Pad array with last value to a fixed length"""
        if len(arr) == length:
            return arr
        return arr + [arr[-1]] * (length - len(arr))

    # Pad each metric list to max_epochs
    train_metrics = np.array([
        pad_to_length(v[f'train_{metric_type}'], max_epochs)
        for v in train_history.values()
    ])
    test_metrics = np.array([
        pad_to_length(v[f'test_{metric_type}'], max_epochs)
        for v in train_history.values()
    ])

    # Compute means and stds
    avg_train = train_metrics.mean(axis=0)
    std_train = train_metrics.std(axis=0)
    avg_test = test_metrics.mean(axis=0)
    std_test = test_metrics.std(axis=0)

    # Plot
    epochs = np.arange(max_epochs)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, avg_train, label=f"Train {metric_type}", color='blue')
    plt.fill_between(epochs, avg_train - std_train, avg_train + std_train, color='blue', alpha=0.2)
    plt.plot(epochs, avg_test, label=f"Test {metric_type}", color='red')
    plt.fill_between(epochs, avg_test - std_test, avg_test + std_test, color='red', alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel(f"Average {metric_type}")
    plt.title(f"Average {metric_type} Over Epochs (across concepts)")
    plt.legend()
    plt.grid(True)
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