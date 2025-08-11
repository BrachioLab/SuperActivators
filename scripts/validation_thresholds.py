import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.compute_concepts_utils import gpu_kmeans, compute_linear_separators
from utils.activation_utils import compute_cosine_sims, compute_signed_distances
from utils.memory_management_utils import ChunkedActivationLoader
from utils.unsupervised_utils import compute_detection_metrics_over_percentiles_allpairs, find_best_clusters_per_concept_from_detectionmetrics, filter_and_save_best_clusters, get_matched_concepts_and_data, \
compute_concept_thresholds_over_percentiles_all_pairs
from utils.superdetector_inversion_utils import find_all_superdetector_patches, all_superdetector_inversions_across_percentiles
from utils.quant_concept_evals_utils import detect_then_invert_metrics_over_percentiles, compute_concept_thresholds_over_percentiles
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices, map_concepts_to_image_indices
from utils.filter_datasets_utils import filter_concept_dict


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Qwen', ('text', 'text3'))]
MODELS = [('Llama', ('text', 'text'))]  # Focus on Llama text model

DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
DATASETS = ['Sarcasm']  # Focus on Sarcasm dataset
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]
SAMPLE_TYPES = [('patch', 1000)]  # Focus on patch (token) level

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100
SCRATCH_DIR = '/scratch/cgoldberg/'
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 300


def get_files_for_avg(model_name, n_clusters, sample_type, percent_thru_model):
    con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep(model_name, n_clusters, sample_type, percent_thru_model):
    con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"dists_{concepts_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, cossim_file

    
def get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model):
    """
    Constructs filenames and labels for regular k-means concept pipeline.

    Args:
        model_name (str): Name of the model (e.g., 'vit_b_32')
        n_clusters (int): Number of clusters used in k-means
        sample_type (str): Type of embedding source (e.g., 'patch', 'cls')
        percent_thru_model (int): Percentage through model (e.g., 100)

    Returns:
        tuple: (con_label, embeddings_file, concepts_file, cossim_file)
    """
    con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f"kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model):
    """
    Constructs filenames and labels for linear separator k-means concept pipeline.

    Args:
        model_name (str): Name of the model (e.g., 'vit_b_32')
        n_clusters (int): Number of clusters used in k-means
        sample_type (str): Type of embedding source (e.g., 'patch', 'cls')
        percent_thru_model (int): Percentage through model (e.g., 100)

    Returns:
        tuple: (con_label, embeddings_file, dists_file, dists_path)
    """
    con_label = f"{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_{percent_thru_model}"
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f"kmeans_{n_clusters}_linsep_concepts_{embeddings_file}"
    dists_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{embeddings_file[:-3]}.pt"
    return con_label, embeddings_file, concepts_file, dists_file


def get_files_for_sae(model_name, sample_type, percent_thru_model):
    """
    Constructs filenames and labels for SAE (Sparse Autoencoder) concept pipeline.
    Note: SAE is only available for CLIP model with patch embeddings.

    Args:
        model_name (str): Name of the model (must be 'CLIP')
        sample_type (str): Type of embedding source (must be 'patch')
        percent_thru_model (int): Percentage through model (e.g., 100)

    Returns:
        tuple: (con_label, embeddings_file, concepts_file, acts_file) or None if not applicable
    """
    if model_name != 'CLIP' or sample_type != 'patch':
        return None
    
    # SAE uses patchsae for CLIP patch embeddings
    sae_name = 'patchsae'
    con_label = f"{model_name}_sae_{sae_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}"
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    # SAE doesn't have a traditional concepts file, but we'll use a placeholder
    concepts_file = f"sae_{sae_name}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    acts_file = f"sae_acts_{sae_name}_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    return con_label, embeddings_file, concepts_file, acts_file


def get_all_files(model_name, n_clusters, sample_type, percent_thru_model):
    all_files = []
    all_files.append(get_files_for_avg(model_name, n_clusters, sample_type, percent_thru_model))
    all_files.append(get_files_for_linsep(model_name, n_clusters, sample_type, percent_thru_model))
    all_files.append(get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    all_files.append(get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    
    # Add SAE if applicable (CLIP patch only)
    sae_files = get_files_for_sae(model_name, sample_type, percent_thru_model)
    if sae_files is not None:
        all_files.append(sae_files)
    
    return all_files
    

def get_cluster_labels(dataset_name, kmeans_concept_file):
    print("loading gt clusters from kmeans")
    train_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/train_samples_{kmeans_concept_file}')
    test_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/test_samples_{kmeans_concept_file}')
    cluster_to_samples = defaultdict(list)
    for cluster, samples in train_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster, samples in test_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster in cluster_to_samples:
        cluster_to_samples[cluster] = sorted(cluster_to_samples[cluster])
    cluster_to_samples = dict(cluster_to_samples)
    return cluster_to_samples




def get_act_metrics(dataset_name, acts_file):
    # Use ChunkedActivationLoader to handle both chunked and non-chunked files
    loader = ChunkedActivationLoader(dataset_name, acts_file, scratch_dir=SCRATCH_DIR)
    
    # Get activation info
    info = loader.get_activation_info()
    if info['is_chunked']:
        print(f"   Loading chunked activation file ({info['num_chunks']} chunks, {info['total_samples']:,} samples)...")
    else:
        print(f"   Loading single activation file...")
    
    # Return loader instead of loading full dataframe
    return loader
      


if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
        # Skip invalid dataset-input size combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        
        
        print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
        #get gt values from calibration dataset
        if sample_type == 'patch':   
            gt_samples_per_concept_cal = torch.load(f"GT_Samples/{dataset_name}/gt_patch_per_concept_cal_inputsize_{model_input_size}.pt")   
        else:
            gt_samples_per_concept_cal = torch.load(f"GT_Samples/{dataset_name}/gt_samples_per_concept_cal_inputsize_{model_input_size}.pt")
        
        # Filter to only relevant concepts for this dataset
        gt_samples_per_concept_cal = filter_concept_dict(gt_samples_per_concept_cal, dataset_name)
        print(f"  Filtered to {len(gt_samples_per_concept_cal)} concepts for {dataset_name}")
  
        all_files = get_all_files(model_name, n_clusters, sample_type, PERCENT_THRU_MODEL)
        for con_label, embeddings_file, concepts_file, acts_file in all_files:
            print(con_label)
            
            # Let ChunkedActivationLoader handle finding the file (chunked or not)
            # It will check for both regular and chunked versions
            
            #load concepts
            if 'sae' in con_label:
                # For SAE, we don't have a separate concepts file
                # The concepts are implicitly the SAE units themselves
                print("   SAE detected - concepts are the SAE units")
                concepts = None  # Will be handled differently in threshold computation
            else:
                concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
            
            #compute raw act metrics with cal embeds and concept
            try:
                act_loader = get_act_metrics(dataset_name, acts_file)
                info = act_loader.get_activation_info()
                # Print concise info without the full concept list
                print(f"Activation info: {info['total_samples']:,} samples, {info['num_concepts']} concepts, {info['num_chunks']} chunks")
            except FileNotFoundError as e:
                print(f"   ⚠️  Activation file not found: {acts_file}, skipping...")
                print(f"   Error: {e}")
                continue
      
            #compute threshold for those concepts
            if 'kmeans' in con_label or 'sae' in con_label:
                print("computing thresholds over percentiles (chunked)")
                # Use chunked version that only loads calibration data
                compute_concept_thresholds_over_percentiles_all_pairs(act_loader, 
                                                                      gt_samples_per_concept_cal, 
                                                                      PERCENTILES, DEVICE,
                                                                      dataset_name, con_label)  
            else:
                print("computing thresholds over percentiles")
                # Use the original function which we'll make compatible with loaders
                compute_concept_thresholds_over_percentiles(gt_samples_per_concept_cal, 
                                                            act_loader, PERCENTILES, DEVICE, 
                                                            dataset_name, con_label, n_vectors=1,
                                                            n_concepts_to_print=0)