import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.compute_concepts_utils import gpu_kmeans, compute_linear_separators, compute_avg_concept_vectors
from utils.activation_utils import compute_cosine_sims, compute_signed_distances
from utils.memory_management_utils import ChunkedEmbeddingLoader
from utils.unsupervised_utils import compute_detection_metrics_over_percentiles_allpairs, \
find_best_clusters_per_concept_from_detectionmetrics, filter_and_save_best_clusters, get_matched_concepts_and_data, \
compute_concept_thresholds_over_percentiles_all_pairs
from utils.superdetector_inversion_utils import find_all_superdetector_patches, all_superdetector_inversions_across_percentiles, \
     detect_then_invert_locally_metrics_over_percentiles
from utils.quant_concept_evals_utils import detect_then_invert_metrics_over_percentiles, compute_concept_thresholds_over_percentiles
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices, map_concepts_to_image_indices
from utils.filter_datasets_utils import filter_concept_dict


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
DATASETS = ['Broden-Pascal']
MODELS = [('Llama', (560, 560))]
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]
SAMPLE_TYPES = [('patch', 1000)]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100
SCRATCH_DIR = '/scratch/cgoldberg/'
BATCH_SIZE = 1000  # Increased from 500 for better GPU utilization
PRELOAD_ALL_CHUNKS = True  # Load all chunks into memory for faster training

def get_gt(sample_type, dataset_name, model_input_size):
    if sample_type == 'patch':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
        gt_samples_per_concept_train = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_train_inputsize_{model_input_size}.pt')
    else:
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt')
        gt_samples_per_concept_train = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_train_inputsize_{model_input_size}.pt')
    return gt_samples_per_concept, gt_samples_per_concept_train

    
def get_files_for_avg(model_name, sample_type, percent_thru_model):
    con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep(model_name, sample_type, percent_thru_model):
    con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_{percent_thru_model}'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt"
    concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_{percent_thru_model}.pt'
    cossim_file = f"dists_{concepts_file[:-3]}.csv"
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
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.csv"
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
    dists_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{embeddings_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, dists_file


def get_all_files(model_name, sample_type, n_clusters, percent_thru_model):
    all_files = []
    all_files.append(get_files_for_avg(model_name, sample_type, percent_thru_model))
    all_files.append(get_files_for_linsep(model_name, sample_type, percent_thru_model))
    all_files.append(get_files_for_reg_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    all_files.append(get_files_for_linsep_kmeans(model_name, n_clusters, sample_type, percent_thru_model))
    return all_files

def get_cluster_labels(dataset_name, kmeans_concept_file):
    print("loading gt clusters from kmeans")
    train_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/train_samples_{kmeans_concept_file}')
    test_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/test_samples_{kmeans_concept_file}')
    cal_cluster_to_samples = torch.load(f'Concepts/{dataset_name}/cal_samples_{kmeans_concept_file}')
    
    cluster_to_samples = defaultdict(list)
    for cluster, samples in train_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster, samples in test_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster, samples in cal_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
        
    for cluster in cluster_to_samples:
        cluster_to_samples[cluster] = sorted(cluster_to_samples[cluster])
    cluster_to_samples = dict(cluster_to_samples)
    return cluster_to_samples

def get_unsupervised_concepts(embeddings_file, n_clusters, dataset_name, concepts_file, model_input_size, sample_type, model_name, loader):
    # Construct full path for backward compatibility with functions that expect it
    embeddings_path = f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}"
    if 'linsep' in concepts_file:
        kmeans_concepts_file = "_".join(part for part in concepts_file.split("_") if "linsep" not in part)
        cluster_to_samples = get_cluster_labels(dataset_name, kmeans_concepts_file)
        
        # Always use chunked version for memory efficiency
        print("   Using chunked linear separators for unsupervised concepts...")
        concepts, _ = compute_linear_separators(embeddings_path, cluster_to_samples, dataset_name, sample_type, model_input_size, 
                                  device=DEVICE, output_file=concepts_file, lr=0.001, epochs=1000, batch_size=256, patience=20, 
                                  tolerance=0.001, weight_decay=1e-4, lr_step_size=5, lr_gamma=0.8, balance_data=True, 
                                  balance_negatives=False)
    else:
        # Always use chunked K-means for memory efficiency
        print("   Using chunked K-means clustering...")
        concepts, _, _, _ = gpu_kmeans(n_clusters=n_clusters, embeddings_path=embeddings_path, dataset_name=dataset_name,
                                      device=DEVICE, model_input_size=model_input_size,
                                      concepts_filename=concepts_file, sample_type=sample_type,
                                      map_samples=True)
        
    return concepts


def get_supervised_concepts(embeddings_file, dataset_name, concepts_file, model_input_size, sample_type, loader):
    # Construct full path for backward compatibility with functions that expect it
    embeddings_path = f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}"
    gt_samples_per_concept, gt_samples_per_concept_train = get_gt(sample_type, dataset_name, model_input_size)
    
    # Filter concepts to only those relevant for this dataset
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    gt_samples_per_concept_train = filter_concept_dict(gt_samples_per_concept_train, dataset_name)
    
    print(f"computing concepts for {len(gt_samples_per_concept)} concepts (filtered from original)")
    
    if 'linsep' in concepts_file:
        # Always use chunked version for memory efficiency
        print("   Using chunked linear separators for supervised concepts...")
        concepts, logs = compute_linear_separators(embeddings_path, gt_samples_per_concept, dataset_name, 
                                                 sample_type=sample_type, device=DEVICE,
                                                 model_input_size=model_input_size,
                                                 output_file=concepts_file, batch_size=64,  # Reduced batch size
                                                 lr=0.001, epochs=1000, patience=20, tolerance=0.001,
                                                 weight_decay=0.0001, lr_step_size=5, lr_gamma=0.8,
                                                 balance_data=True,
                                                 balance_negatives=False)

        
    else:
        # Always use chunked version for memory efficiency
        print("   Using chunked average concept computation...")
        concepts = compute_avg_concept_vectors(gt_samples_per_concept_train, loader, 
                                             dataset_name=dataset_name, output_file=concepts_file)
    return concepts

        
if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        
        print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
        #get gt values
        if sample_type == 'patch':
            gt_samples_per_concept = torch.load(f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt")   
        else:
            gt_samples_per_concept = torch.load(f"GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt")
        
        # Filter to only relevant concepts for this dataset
        gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
        print(f"  Filtered to {len(gt_samples_per_concept)} concepts for {dataset_name}")


        all_files = get_all_files(model_name, sample_type, n_clusters, PERCENT_THRU_MODEL)
        for con_label, embeddings_file, concepts_file, acts_file in all_files:
            print(con_label)
                
            #load embeddings (handles both chunked and non-chunked)
            loader = ChunkedEmbeddingLoader(dataset_name, embeddings_file, SCRATCH_DIR, device=DEVICE)
            
            # Check if embeddings are chunked
            info = loader.get_embedding_info()
            if info['is_chunked']:
                print(f"   Loading chunked embeddings ({info['num_chunks']} chunks, {info['total_samples']:,} samples)...")
            
            # Pass loader to concept computation functions
            # They will decide whether to use chunked processing based on data size
            
            if 'kmeans' in con_label: #unsupervised concepts
                get_unsupervised_concepts(embeddings_file, n_clusters, dataset_name, concepts_file, model_input_size, sample_type, model_name, loader)
            else: #supervised concepts
                #compute concepts
                get_supervised_concepts(embeddings_file, dataset_name, concepts_file, model_input_size, sample_type, loader)
