import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gc

# Removed unused imports from non-chunked compute_concepts_utils
# Only use chunked versions to avoid fallbacks
from utils.superdetector_inversion_utils_chunked import find_all_superdetector_patches_chunked, all_superdetector_inversions_across_percentiles_chunked, \
     detect_then_invert_locally_metrics_over_percentiles_chunked
from utils.quant_concept_evals_utils_chunked import detect_then_invert_metrics_over_percentiles_chunked, compute_concept_thresholds_over_percentiles_chunked, compute_detection_metrics_over_percentiles_chunked, get_matched_concepts_and_data_fully_chunked
# Removed unused imports from non-chunked gt_concept_segmentation_utils
from utils.memory_management_utils import ChunkedEmbeddingLoader


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Mistral', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm']
SAMPLE_TYPES = [('patch', 1000)]

MODELS = [('Llama', (560, 560))]
DATASETS = ['CLEVR']

sample_type = 'patch'
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
    cluster_to_samples = defaultdict(list)
    for cluster, samples in train_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster, samples in test_cluster_to_samples.items():
        cluster_to_samples[cluster].extend(samples)
    for cluster in cluster_to_samples:
        cluster_to_samples[cluster] = sorted(cluster_to_samples[cluster])
    cluster_to_samples = dict(cluster_to_samples)
    return cluster_to_samples


def get_act_metrics_chunked(dataset_name, acts_file):
    """
    Load activation metrics using ChunkedActivationLoader for memory efficiency.
    """
    from utils.memory_management_utils import ChunkedActivationLoader
    
    print(f"Loading activation metrics: {acts_file}")
    activation_loader = ChunkedActivationLoader(dataset_name, acts_file, SCRATCH_DIR)
    
    # For now, load full dataframe but using efficient chunked loading
    # Future optimization: modify utility functions to work with chunked data
    return activation_loader.load_full_dataframe()

        
if __name__ == "__main__":
    print("🚀 Starting chunked inversion stats")
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)
    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:
        print(f"🔄 Config: {model_name}, {dataset_name}, {sample_type}")
        # Skip invalid dataset-input size combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue

        # Removed skip condition to allow Llama patch processing
        
        print(f"Processing model {model_name} dataset {dataset_name} sample type {sample_type}")
        print("   Loading GT data...")
        #get gt
        gt_patches_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
        gt_patches_per_concept_test = torch.load(f'GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt')

        #load embeds using chunked loader
        print("Setting up chunked embedding loader...")
        embeddings_file = f"{model_name}_{sample_type}_embeddings_percentthrumodel_100.pt"
        embeddings_path = f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}"
        embedding_loader = ChunkedEmbeddingLoader(embeddings_path, DEVICE)
        
        print(f"📊 Embedding info: {embedding_loader.get_embedding_info()}")
    
        all_files = get_all_files(model_name, sample_type, n_clusters)
        for con_label, _, concepts_file, acts_file in all_files:  
            print(con_label)
            #get act metrics
            act_metrics = get_act_metrics_chunked(dataset_name, acts_file)
            
            if 'kmeans' in con_label: #unsupervised concepts
                continue
                concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
                
                matched_acts, matched_gt_patches_per_concept_test, \
                matched_gt_patches_per_concept, matched_concepts = get_matched_concepts_and_data_fully_chunked(dataset_name,
                                                                                con_label,
                                                                                act_metrics,
                                                                                gt_patches_per_concept_test,
                                                                                gt_patches_per_concept,
                                                                                concepts=concepts
                                                                                )
                # Detect then invert using plain cossim stats
                print("Computing inversion metrics")
                detect_then_invert_metrics_over_percentiles_chunked(PERCENTILES, PERCENTILES, 
                                                    matched_acts, matched_concepts, matched_gt_patches_per_concept, 
                                                    matched_gt_patches_per_concept_test,
                                                    DEVICE, dataset_name, model_input_size, con_label,
                                                    embedding_loader, all_object_patches=None, patch_size=14) 


                # Detect then invert using local superpatches
                print("Computing inversion metrics using superpatches") 
                all_superdetector_inversions_across_percentiles_chunked(PERCENTILES, 'avg', embedding_loader, matched_acts,
                               matched_gt_patches_per_concept_test, dataset_name, model_input_size, con_label, 
                                            DEVICE, patch_size=14, local=True)

                detect_then_invert_locally_metrics_over_percentiles_chunked(PERCENTILES, PERCENTILES, matched_acts, 
                                                    matched_concepts, matched_gt_patches_per_concept, 
                                                    matched_gt_patches_per_concept_test,
                                                    DEVICE, dataset_name, model_input_size, con_label,
                                                    embedding_loader, all_object_patches=None, patch_size=14,
                                                    agglomerate_type='avg')
                
                
            else: #supervised concepts
                concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
                detect_then_invert_metrics_over_percentiles_chunked(PERCENTILES, PERCENTILES, 
                                                act_metrics, concepts, gt_patches_per_concept, gt_patches_per_concept_test,
                                                DEVICE, dataset_name, model_input_size, con_label, embedding_loader,
                                                all_object_patches=None,
                                                patch_size=14)

                all_superdetector_inversions_across_percentiles_chunked(PERCENTILES, 'avg', embedding_loader, act_metrics,
                                   gt_patches_per_concept_test, dataset_name, model_input_size, con_label, 
                                                DEVICE, patch_size=14, local=True)
                detect_then_invert_locally_metrics_over_percentiles_chunked(PERCENTILES, PERCENTILES, act_metrics, 
                                                        concepts, gt_patches_per_concept, gt_patches_per_concept_test,
                                                        DEVICE, dataset_name, model_input_size, con_label, embedding_loader,
                                                        all_object_patches=None, patch_size=14,
                                                        agglomerate_type='avg')
            
            del act_metrics
            gc.collect()
            
        # Clean up embedding loader
        del embedding_loader
        gc.collect()