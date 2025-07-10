import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import os
import gc
from collections import defaultdict
from itertools import product
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.compute_concepts_utils import gpu_kmeans, compute_cosine_sims, compute_signed_distances, compute_linear_separators, compute_avg_concept_vectors
from utils.unsupervised_utils import compute_detection_metrics_over_percentiles_allpairs, find_best_clusters_per_concept_from_detectionmetrics, filter_and_save_best_clusters, get_matched_concepts_and_data, compute_concept_thresholds_over_percentiles_all_pairs
from utils.superdetector_inversion_utils import find_all_superdetector_patches, all_superdetector_inversions_across_percentiles, detect_then_invert_locally_metrics_over_percentiles
from utils.quant_concept_evals_utils import detect_then_invert_metrics_over_percentiles, compute_concept_thresholds_over_percentiles, compute_detection_metrics_over_percentiles
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices, map_concepts_to_image_indices


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm', 'GoEmotions']
SAMPLE_TYPES = [('cls', 50), ('patch', 1000)]
DATASETS = ['GoEmotions']


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 500

    
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


def get_act_metrics(dataset_name, acts_file):
    if 'linsep' in acts_file:
        print("Loading distances...")
        out_path = f"{SCRATCH_DIR}Distances/{dataset_name}/{acts_file}"
        dists = pd.read_csv(f"{SCRATCH_DIR}Distances/{dataset_name}/{acts_file}") 
        return dists
    
    else:
        print("Loading cosine similarities...")
        out_path = f"{SCRATCH_DIR}Cosine_Similarities/{dataset_name}/{acts_file}"
        cos_sims = pd.read_csv(f"{SCRATCH_DIR}Cosine_Similarities/{dataset_name}/{acts_file}")
        return cos_sims

        
if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
        # Skip invalid dataset-input size combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
            
        
        print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
        #get gt
        gt_samples_per_concept_test = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt')
  
        all_files = get_all_files(model_name, sample_type, n_clusters)
        for con_label, embeddings_file, concepts_file, acts_file in all_files:
            print(con_label)
            
            #get act metrics
            act_metrics = get_act_metrics(dataset_name, acts_file)
            
            print(act_metrics.shape)
            
            if 'kmeans' in con_label: #unsupervised concepts
                # Step 1: Compute detection metrics
                print("Computing detection metrics over all pairs")
                compute_detection_metrics_over_percentiles_allpairs(
                    PERCENTILES,
                    gt_samples_per_concept_test,
                    dataset_name,
                    model_input_size,
                    DEVICE,
                    con_label,
                    act_metrics,
                    scratch_dir=SCRATCH_DIR, 
                    sample_type=sample_type,
                    patch_size=14
                )

                # Step 2: Find best clusters per concept
                print("Matching concepts/clusters by detection rates")
                best_clusters_by_detect = find_best_clusters_per_concept_from_detectionmetrics(
                    dataset_name,
                    model_name,
                    sample_type,
                    metric_type='f1',
                    percentiles=PERCENTILES, 
                    con_label=con_label
                )
                filter_and_save_best_clusters(dataset_name, con_label) #sort them in plotting-compatible files


                # Step 3: Write superdetectors to file
                matched_acts, _, _, _, _ = get_matched_concepts_and_data(dataset_name,
                                                                        con_label,
                                                                        act_metrics,
                                                                        gt_samples_per_concept_test=None,
                                                                        gt_samples_per_concept=None,
                                                                        concepts=None
                                                                        )
                print("Writing superdetectors")
                for percentile in tqdm(PERCENTILES):
                        find_all_superdetector_patches(percentile, matched_acts, dataset_name, 
                                                       model_input_size, con_label, DEVICE)
                
                
            else: #supervised concepts
                    
                #compute detection metrics
                print("Computing detection metrics")
                compute_detection_metrics_over_percentiles(PERCENTILES, 
                                                           gt_samples_per_concept_test, 
                                                           act_metrics, dataset_name, model_input_size, DEVICE, 
                                                           con_label, sample_type=sample_type, patch_size=14)
                
                # Write superdetectors to file
                print("Writing superdetectors")
                for percentile in tqdm(PERCENTILES):
                        find_all_superdetector_patches(percentile, act_metrics, dataset_name, 
                                                       model_input_size, con_label, DEVICE)
                        
            del act_metrics
            gc.collect()

