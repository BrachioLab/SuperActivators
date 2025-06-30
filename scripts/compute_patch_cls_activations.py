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
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm']
DATASETS = ['CLEVR']


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 200


def get_files_for_avg_patch_cls(model_name):
    """Get files for patch-CLS cosine similarity computation with avg concepts"""
    con_label = f'{model_name}_avg_cls_embeddings_percentthrumodel_100'
    patch_embeddings_file = f"{model_name}_patch_embeddings_percentthrumodel_100.pt"
    cls_concepts_file = f'avg_concepts_{model_name}_cls_embeddings_percentthrumodel_100.pt'
    cossim_file = f"patch_cls_cosine_similarities_{cls_concepts_file[:-3]}.csv"
    return con_label, patch_embeddings_file, cls_concepts_file, cossim_file


def get_files_for_linsep_patch_cls(model_name):
    """Get files for patch-CLS signed distance computation with linsep concepts"""
    con_label = f'{model_name}_linsep_cls_embeddings_BD_True_BN_False_percentthrumodel_100'
    patch_embeddings_file = f"{model_name}_patch_embeddings_percentthrumodel_100.pt"
    cls_concepts_file = f'linsep_concepts_BD_True_BN_False_{model_name}_cls_embeddings_percentthrumodel_100.pt'
    cossim_file = f"patch_cls_dists_{cls_concepts_file[:-3]}.csv"
    return con_label, patch_embeddings_file, cls_concepts_file, cossim_file

    
# def get_files_for_reg_kmeans_patch_cls(model_name, n_clusters):
#     """Get files for patch-CLS cosine similarity computation with regular k-means concepts"""
#     con_label = f"{model_name}_kmeans_{n_clusters}_cls_embeddings_kmeans_percentthrumodel_100"
#     patch_embeddings_file = f"{model_name}_patch_embeddings_percentthrumodel_100.pt"
#     cls_concepts_file = f"kmeans_{n_clusters}_concepts_{model_name}_cls_embeddings_percentthrumodel_100.pt"
#     cossim_file = f"patch_cls_cosine_similarities_{cls_concepts_file[:-3]}.csv"
#     return con_label, patch_embeddings_file, cls_concepts_file, cossim_file


# def get_files_for_linsep_kmeans_patch_cls(model_name, n_clusters):
#     """Get files for patch-CLS signed distance computation with linear separator k-means concepts"""
#     con_label = f"{model_name}_kmeans_{n_clusters}_linsep_cls_embeddings_kmeans_percentthrumodel_100"
#     patch_embeddings_file = f"{model_name}_patch_embeddings_percentthrumodel_100.pt"
#     cls_concepts_file = f"kmeans_{n_clusters}_linsep_concepts_{model_name}_cls_embeddings_percentthrumodel_100.pt"
#     dists_file = f"patch_cls_dists_kmeans_{n_clusters}_linsep_concepts_{model_name}_cls_embeddings_percentthrumodel_100.csv"
#     return con_label, patch_embeddings_file, cls_concepts_file, dists_file


def get_all_patch_cls_files(model_name):
    """Get all file combinations for patch-CLS activation computation"""
    all_files = []
    all_files.append(get_files_for_avg_patch_cls(model_name))
    all_files.append(get_files_for_linsep_patch_cls(model_name))
    # all_files.append(get_files_for_reg_kmeans_patch_cls(model_name, n_clusters)) #for now just look at supervised
    # all_files.append(get_files_for_linsep_kmeans_patch_cls(model_name, n_clusters))
    return all_files
    
     


if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS)
    for (model_name, model_input_size), dataset_name in experiment_configs:
        # Skip invalid dataset-input size combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
            
        all_files = get_all_patch_cls_files(model_name)
        
        for con_label, patch_embeddings_file, cls_concepts_file, acts_file in all_files:
            print(f"Processing patch-CLS activations for model {model_name} dataset {dataset_name}")
            print(f"Concept label: {con_label}")
            print(f"Using patch embeddings: {patch_embeddings_file}")
            print(f"Using CLS concepts: {cls_concepts_file}")
            print(f"Output file: {acts_file}")
            
            # Check if patch embeddings file exists
            patch_embeddings_path = f"{SCRATCH_DIR}Embeddings/{dataset_name}/{patch_embeddings_file}"
            if not os.path.exists(patch_embeddings_path):
                print(f"  WARNING: Patch embeddings file not found: {patch_embeddings_path}")
                continue
                
            # Check if CLS concepts file exists 
            cls_concepts_path = f'Concepts/{dataset_name}/{cls_concepts_file}'
            if not os.path.exists(cls_concepts_path):
                print(f"  WARNING: CLS concepts file not found: {cls_concepts_path}")
                continue
            
            print("  Loading patch embeddings...")
            patch_embeds_dic = torch.load(patch_embeddings_path)
            patch_embeds = patch_embeds_dic['normalized_embeddings'] 
            
            print("  Loading CLS concepts...")
            cls_concepts = torch.load(cls_concepts_path)

            if 'linsep' in acts_file or 'dists' in acts_file:
                print("  Computing signed distances between patches and CLS concepts...")
                compute_signed_distances(patch_embeds, cls_concepts, dataset_name, DEVICE,
                                            acts_file, SCRATCH_DIR, BATCH_SIZE)
                torch.cuda.empty_cache()            
                torch.cuda.ipc_collect()           

            else:
                print("  Computing cosine similarities between patches and CLS concepts...")
                compute_cosine_sims(embeddings = patch_embeds, 
                                            concepts = cls_concepts, 
                                            output_file = acts_file,
                                            dataset_name = dataset_name, device=DEVICE,
                                            batch_size=BATCH_SIZE, scratch_dir=SCRATCH_DIR)
                torch.cuda.empty_cache()            
                torch.cuda.ipc_collect()
                
            print(f"  ✓ Completed {acts_file}")