import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath("utils"))

from compute_concepts_utils import gpu_kmeans, compute_cosine_sims, compute_signed_distances, compute_linear_separators, compute_avg_concept_vectors
from unsupervised_utils import compute_detection_metrics_over_percentiles_allpairs, \
find_best_clusters_per_concept_from_detectionmetrics, filter_and_save_best_clusters, get_matched_concepts_and_data, \
compute_concept_thresholds_over_percentiles_all_pairs
from superdetector_inversion_utils import find_all_superdetector_patches, all_superdetector_inversions_across_percentiles, \
     detect_then_invert_locally_metrics_over_percentiles
from quant_concept_evals_utils import detect_then_invert_metrics_over_percentiles, compute_concept_thresholds_over_percentiles
from gt_concept_segmentation_utils import map_concepts_to_patch_indices, map_concepts_to_image_indices


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']
SAMPLE_TYPES = [('cls', 50), ('patch', 1000)]

MODELS = [('Llama', ('text', 'text'))]
DATASETS = ['iSarcasm']


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 500

def get_gt(sample_type, dataset_name, model_input_size):
    if sample_type == 'patch':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
        gt_samples_per_concept_train = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_train_inputsize_{model_input_size}.pt')
    else:
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt')
        gt_samples_per_concept_train = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_train_inputsize_{model_input_size}.pt')
    return gt_samples_per_concept, gt_samples_per_concept_train

    
def get_files_for_avg(model_name, sample_type):
    con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt"
    concepts_file = f'avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt'
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep(model_name, sample_type):
    con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt"
    concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt'
    cossim_file = f"dists_{concepts_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, cossim_file

def get_files_for_reg_kmeans(model_name, n_clusters, sample_type):
    """
    Constructs filenames and labels for regular k-means concept pipeline.

    Args:
        model_name (str): Name of the model (e.g., 'vit_b_32')
        n_clusters (int): Number of clusters used in k-means
        sample_type (str): Type of embedding source (e.g., 'patch', 'cls')

    Returns:
        tuple: (con_label, embeddings_file, concepts_file, cossim_file)
    """
    con_label = f"{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_100"
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt"
    concepts_file = f"kmeans_{n_clusters}_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt"
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep_kmeans(model_name, n_clusters, sample_type):
    """
    Constructs filenames and labels for linear separator k-means concept pipeline.

    Args:
        model_name (str): Name of the model (e.g., 'vit_b_32')
        n_clusters (int): Number of clusters used in k-means
        sample_type (str): Type of embedding source (e.g., 'patch', 'cls')
        dataset_name (str): Name of the dataset (e.g., 'CLEVR')

    Returns:
        tuple: (con_label, embeddings_file, dists_file, dists_path)
    """
    con_label = f"{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_100"
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt"
    concepts_file = f"kmeans_{n_clusters}_linsep_concepts_{embeddings_file}"
    dists_file = f"dists_kmeans_{n_clusters}_linsep_concepts_{embeddings_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, dists_file


def get_all_files(model_name, sample_type, n_clusters):
    all_files = []
    all_files.append(get_files_for_avg(model_name, sample_type))
    all_files.append(get_files_for_linsep(model_name, sample_type))
    all_files.append(get_files_for_reg_kmeans(model_name, n_clusters, sample_type))
    all_files.append(get_files_for_linsep_kmeans(model_name, n_clusters, sample_type))
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

def get_unsupervised_concepts(embeds, n_clusters, dataset_name, concepts_file, model_input_size, sample_type, model_name):
    if 'linsep' in concepts_file:
        kmeans_concepts_file = "_".join(part for part in concepts_file.split("_") if "linsep" not in part)
        cluster_to_samples = get_cluster_labels(dataset_name, kmeans_concepts_file)
        concepts, _ = compute_linear_separators(embeds, cluster_to_samples, dataset_name, sample_type, model_input_size, 
                                  device=DEVICE, output_file=concepts_file, lr=0.001, epochs=1000, batch_size=64, patience=20, 
                                  tolerance=0.001, weight_decay=1e-4, lr_step_size=5, lr_gamma=0.8, balance_data=True, 
                                  balance_negatives=False)
    else:
        concepts, _, _, _ = gpu_kmeans(n_clusters=n_clusters, embeddings=embeds, dataset_name=dataset_name,
                                      device=DEVICE, model_input_size=model_input_size,
                                      concepts_filename=concepts_file, sample_type=sample_type,
                                      map_samples=True)
        
    return concepts, embeds


def get_supervised_concepts(embeds, dataset_name, concepts_file, model_input_size, sample_type):
    gt_samples_per_concept, gt_samples_per_concept_train = get_gt(sample_type, dataset_name, model_input_size)
    print("computing concepts")
    if 'linsep' in concepts_file:
        concepts, logs = compute_linear_separators(embeds, gt_samples_per_concept, dataset_name, 
                                                 sample_type=sample_type, device=DEVICE,
                                                 model_input_size=model_input_size,
                                                 output_file=concepts_file, batch_size=64,
                                                 lr=0.001, epochs=1000, patience=20, tolerance=0.001,
                                                 weight_decay=0.0001, lr_step_size=5, lr_gamma=0.8,
                                                 balance_data=True,
                                                 balance_negatives=False)
    else:
        concepts = compute_avg_concept_vectors(gt_samples_per_concept_train, embeds, 
                                       dataset_name=dataset_name, output_file=concepts_file)
    return concepts

        
if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue

        
        print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
        #get gt values
        if sample_type == 'patch':
            gt_samples_per_concept = torch.load(f"GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt")   
        else:
            gt_samples_per_concept = torch.load(f"GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt")

  
        all_files = get_all_files(model_name, sample_type, n_clusters)
        for con_label, embeddings_file, concepts_file, acts_file in all_files:
            print(con_label)
            #load embeddings
            embeds_dic = torch.load(f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}")
            embeds = embeds_dic['normalized_embeddings']  
            
            if 'kmeans' in con_label: #unsupervised concepts
                get_unsupervised_concepts(embeds, n_clusters, dataset_name, concepts_file, model_input_size, sample_type, model_name)
            else: #supervised concepts
                #compute concepts
                get_supervised_concepts(embeds, dataset_name, concepts_file, model_input_size, sample_type)
