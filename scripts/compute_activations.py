import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.compute_concepts_utils import gpu_kmeans, compute_cosine_sims, compute_signed_distances, compute_linear_separators
from utils.unsupervised_utils import compute_detection_metrics_over_percentiles_allpairs, \
find_best_clusters_per_concept_from_detectionmetrics, filter_and_save_best_clusters, get_matched_concepts_and_data, \
compute_concept_thresholds_over_percentiles_all_pairs
from utils.superdetector_inversion_utils import find_all_superdetector_patches, all_superdetector_inversions_across_percentiles, \
     detect_then_invert_locally_metrics_over_percentiles
from utils.quant_concept_evals_utils import detect_then_invert_metrics_over_percentiles, compute_concept_thresholds_over_percentiles
from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices, map_concepts_to_image_indices


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']
SAMPLE_TYPES = [('cls', 50), ('patch', 1000)]

DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
SAMPLE_TYPES = [('cls', 50)]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 200


def get_files_for_avg(model_name, n_clusters, sample_type):
    con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
    embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt"
    concepts_file = f'avg_concepts_{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt'
    cossim_file = f"cosine_similarities_{concepts_file[:-3]}.csv"
    return con_label, embeddings_file, concepts_file, cossim_file


def get_files_for_linsep(model_name, n_clusters, sample_type):
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


def get_all_files(model_name, n_clusters, sample_type):
    all_files = []
    all_files.append(get_files_for_avg(model_name, n_clusters, sample_type))
    all_files.append(get_files_for_linsep(model_name, n_clusters, sample_type))
    all_files.append(get_files_for_reg_kmeans(model_name, n_clusters, sample_type))
    all_files.append(get_files_for_linsep_kmeans(model_name, n_clusters, sample_type))
    return all_files
    
     


if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
        all_files = get_all_files(model_name, n_clusters, sample_type)
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
        
        for con_label, embeddings_file, concepts_file, acts_file in all_files:
            print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
            print(con_label)
            
            print("Loading embeddings...")
            embeds_dic = torch.load(f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}")
            embeds = embeds_dic['normalized_embeddings'] 
            
            
            concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')

            if 'linsep' in acts_file:
                compute_signed_distances(embeds, concepts, dataset_name, DEVICE,
                                            acts_file, SCRATCH_DIR, BATCH_SIZE)
                torch.cuda.empty_cache()            
                torch.cuda.ipc_collect()           

            else:
                compute_cosine_sims(embeddings = embeds, 
                                            concepts = concepts, 
                                            output_file = acts_file,
                                            dataset_name = dataset_name, device=DEVICE,
                                            batch_size=BATCH_SIZE, scratch_dir=SCRATCH_DIR)
                torch.cuda.empty_cache()            
                torch.cuda.ipc_collect()
                