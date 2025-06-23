import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath("utils"))

from compute_concepts_utils import gpu_kmeans, compute_cosine_sims, compute_signed_distances, compute_linear_separators
from unsupervised_utils import compute_detection_metrics_over_percentiles_allpairs, find_best_clusters_per_concept_from_detectionmetrics, filter_and_save_best_clusters, get_matched_concepts_and_data
from superdetector_inversion_utils import find_all_superdetector_patches, all_superdetector_inversions_across_percentiles, \
     detect_then_invert_locally_metrics_over_percentiles
from quant_concept_evals_utils import detect_then_invert_metrics_over_percentiles


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560))]
DATASETS = ['CLEVR', 'Coco']
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]

DATASETS = ['CLEVR']



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100
SCRATCH_DIR = '/scratch/'
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 500


def get_relevant_gt_info(dataset_name, model_input_size, sample_type):
    """
    Loads and returns ground truth patch/image indices for training and test splits
    based on the selected sample type (patch or cls).

    Returns:
        tuple: (gt_samples_per_concept, gt_samples_per_concept_test,
            gt_images_per_concept, gt_images_per_concept_test)
    """
    gt_patches_per_concept = torch.load(
        f"{SCRATCH_DIR}GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt"
    )
    gt_patches_per_concept_test = torch.load(
        f"{SCRATCH_DIR}GT_Samples/{dataset_name}/gt_patch_per_concept_test_inputsize_{model_input_size}.pt"
    )

    gt_images_per_concept = torch.load(
        f"{SCRATCH_DIR}GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt"
    )
    gt_images_per_concept_test = torch.load(
        f"{SCRATCH_DIR}GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_{model_input_size}.pt"
    )

    if sample_type == 'patch':
        gt_samples_per_concept = gt_patches_per_concept
        gt_samples_per_concept_test = gt_patches_per_concept_test
    elif sample_type == 'cls':
        gt_samples_per_concept = gt_images_per_concept
        gt_samples_per_concept_test = gt_images_per_concept_test
    else:
        raise ValueError(f"Invalid sample_type: {sample_type}")

    return gt_samples_per_concept, gt_samples_per_concept_test, gt_images_per_concept, gt_images_per_concept_test
    
    
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


def get_concepts(n_clusters, dataset_name, concepts_file, embeddings_file, model_input_size, sample_type, model_name):
    # print("Loading concepts...")
    # embeds = None
    # try:
    #     concepts = torch.load(f'Concepts/{dataset_name}/{concepts_file}')
    # except:
    #     print("No concepts, computing them")
    print("Loading embeddings...")
    embeds_dic = torch.load(f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}")
    embeds = embeds_dic['normalized_embeddings']
    if 'linsep' in concepts_file:
        _, _, kmeans_concepts_file, _ = get_files_for_reg_kmeans(model_name, n_clusters, sample_type)
        cluster_to_samples = get_cluster_labels(dataset_name, kmeans_concepts_file)
        concepts, _ = compute_linear_separators(embeds, cluster_to_samples, dataset_name, sample_type, model_input_size, 
                                  device=DEVICE, output_file=concepts_file, lr=0.001, epochs=1000, batch_size=64, patience=20, 
                                  tolerance=0.001, weight_decay=1e-4, lr_step_size=5, lr_gamma=0.8, balance_data=True, 
                                  balance_negatives=False)
    else:
        concepts, _, _ = gpu_kmeans(n_clusters=n_clusters, dataset_name=dataset_name,
                                    embeddings=embeds, device=DEVICE, sample_type=sample_type,
                                    model_input_size=model_input_size, concepts_filename=concepts_file)
    return concepts, embeds



def get_act_metrics(concepts, dataset_name, acts_file, batch_size, embeds, embeddings_file):
    if 'linsep' in acts_file:
        print("Loading distances...")
        out_path = f"{SCRATCH_DIR}Distances/{dataset_name}/{acts_file}"
        if not os.path.exists(out_path):
            print("No distances found, computing them")
            if embeds is None:
                print("Loading embeddings...")
                embeds_dic = torch.load(f"Embeddings/{dataset_name}/{embeddings_file}")
                embeds = embeds_dic['normalized_embeddings'] 
            compute_signed_distances(embeds, concepts, dataset_name, DEVICE,
                                            acts_file, SCRATCH_DIR, batch_size)

        dists = pd.read_csv(f"{SCRATCH_DIR}Distances/{dataset_name}/{acts_file}") 
        return dists
    
    else:
        print("Loading cosine similarities...")
        out_path = f"{SCRATCH_DIR}Cosine_Similarities/{dataset_name}/{acts_file}"
        # if not os.path.exists(out_path):
        #     print("No cosine similarities found, computing them")
        if embeds is None:
            print("Loading embeddings...")
            embeds_dic = torch.load(f"Embeddings/{dataset_name}/{embeddings_file}")
            embeds = embeds_dic['normalized_embeddings']  
        cos_sims = compute_cosine_sims(embeddings = embeds, 
                                        concepts = concepts, 
                                        output_file = acts_file,
                                        dataset_name = dataset_name, device=DEVICE,
                                        batch_size=batch_size, scratch_dir=SCRATCH_DIR)
        
        cos_sims = pd.read_csv(f"{SCRATCH_DIR}Cosine_Similarities/{dataset_name}/{acts_file}")
        return cos_sims


def write_superdetectors(dataset_name, con_label, matched_acts, 
                         matched_gt_samples_per_concept_test, model_input_size):
    """ for chae """
    for percentile in tqdm(PERCENTILES):
        find_all_superdetector_patches(percentile, matched_acts, matched_gt_samples_per_concept_test, 
                                        dataset_name, model_input_size, con_label, DEVICE)
        

def unsupervised_task_pipeline(
    gt_samples_per_concept,
    gt_samples_per_concept_test,
    gt_images_per_concept_test,
    dataset_name,
    model_name,
    model_input_size,
    con_label,
    act_metrics,
    embeds,
    embeddings_file,
    sample_type,
    patch_size=14,
):
    """
    Runs the full unsupervised detection pipeline:
    1. Computes detection metrics across percentiles.
    2. Finds best clusters per concept using detection metrics.
    3. Writes superdetectors to disk.
    """
    # Step 1: Compute detection metrics
    print("Computing detection metrics over all pairs")
    compute_detection_metrics_over_percentiles_allpairs(
        PERCENTILES,
        gt_samples_per_concept_test,
        gt_images_per_concept_test,
        dataset_name,
        model_input_size,
        DEVICE,
        con_label,
        act_metrics,
        scratch_dir=SCRATCH_DIR, 
        sample_type=sample_type,
        patch_size=patch_size
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

    if sample_type == 'patch':
        # try:
        #     matched_acts = pd.read_csv(f'Unsupervised_Matches/{dataset_name}/actmetrics_{con_label}.csv')
        #     matched_gt_patches_per_concept_test = torch.load(f'Unsupervised_Matches/{dataset_name}/gt_samples_per_concept_test_{con_label}.pt')
        #     matched_gt_patches_per_concept = torch.load(f'Unsupervised_Matches/{dataset_name}/gt_samples_per_concept_{con_label}.pt')
        #     matched_concepts = torch.load(f'Unsupervised_Matches/{dataset_name}/concepts_{con_label}.pt')
        # except:
        #     act_metrics = get_act_metrics(concepts, dataset_name, acts_file, BATCH_SIZE, embeds, embeddings_file)
            
        matched_acts, matched_gt_patches_per_concept_test, \
        matched_gt_patches_per_concept, matched_concepts = get_matched_concepts_and_data(
                                                                                dataset_name,
                                                                                con_label,
                                                                                act_metrics,
                                                                                gt_samples_per_concept_test,
                                                                                gt_samples_per_concept,
                                                                                concepts
                                                                                )

              
        # Step 3: Write superdetectors to file
        print("Saving superdetectors")
        write_superdetectors(
            dataset_name,
            con_label,
            matched_acts,
            matched_gt_patches_per_concept_test,
            model_input_size
        )
        
#         # Detect then invert using plain cossim stats
#         print("Computing inversion metrics")
#         detect_then_invert_metrics_over_percentiles(PERCENTILES, PERCENTILES, 
#                                             matched_acts, matched_concepts, matched_gt_patches_per_concept, 
#                                             matched_gt_patches_per_concept_test,
#                                             DEVICE, dataset_name, model_input_size, con_label,
#                                             all_object_patches=None, patch_size=14) 
        
        
#         # Detect then invert using local superpatches
#         if 'linsep' in con_label:
#             print("Computing inversion metrics using superpatches")
#             if embeds is None:
#                     print("Loading embeddings...")
#                     embeds_dic = torch.load(f"{SCRATCH_DIR}Embeddings/{dataset_name}/{embeddings_file}")
#                     embeds = embeds_dic['normalized_embeddings'] 
#             all_superdetector_inversions_across_percentiles(PERCENTILES, 'avg', embeds, matched_acts,
#                            matched_gt_patches_per_concept_test, dataset_name, model_input_size, con_label, 
#                                         DEVICE, patch_size=14, local=True)

#             detect_then_invert_locally_metrics_over_percentiles(PERCENTILES, PERCENTILES, matched_acts, 
#                                                 matched_concepts, matched_gt_patches_per_concept, 
#                                                 matched_gt_patches_per_concept_test,
#                                                 DEVICE, dataset_name, model_input_size, con_label,
#                                                 all_object_patches=None, patch_size=14,
#                                                 agglomerate_type='avg')
        
        
    print("All done :)")



if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)

    for (model_name, model_input_size), dataset_name, (sample_type, n_clusters) in experiment_configs:      
        #get relevant gt info
        gt_samples_per_concept, gt_samples_per_concept_test, \
        gt_images_per_concept, gt_images_per_concept_test = get_relevant_gt_info(dataset_name, model_input_size, sample_type)

#         ##### KMEANS ####
        print(f"Running analysis for dataset {dataset_name}, Model {model_name} kmeans {sample_type} concepts")

        con_label, embeddings_file, concepts_file, acts_file = get_files_for_reg_kmeans(model_name, n_clusters, sample_type)
        concepts, embeds = get_concepts(n_clusters, dataset_name, concepts_file, embeddings_file, model_input_size,
                                        sample_type, model_name)
        act_metrics = get_act_metrics(concepts, dataset_name, acts_file, BATCH_SIZE, embeds, embeddings_file)


        # Compute detections, find best clusters by detection f1, and write superdetectors to file
        unsupervised_task_pipeline(gt_samples_per_concept,
                                    gt_samples_per_concept_test,
                                    gt_images_per_concept_test,
                                    dataset_name,
                                    model_name,
                                    model_input_size,
                                    con_label,
                                    act_metrics,
                                    embeds,
                                    embeddings_file,
                                    sample_type,
                                    patch_size=14
                                 )


        ##### Linsep KMEANS ####
        print(f"Running analysis for dataset {dataset_name}, Model {model_name} linsep kmeans {sample_type} concepts")
        con_label, embeddings_file, concepts_file, acts_file = get_files_for_linsep_kmeans(model_name, n_clusters, sample_type)
        concepts, embeds = get_concepts(n_clusters, dataset_name, concepts_file, embeddings_file, model_input_size,
                                        sample_type, model_name)
        act_metrics = get_act_metrics(concepts, dataset_name, acts_file, BATCH_SIZE, embeds, embeddings_file)

        # Compute detections, find best clusters by detection f1, and write superdetectors to file
        unsupervised_task_pipeline(gt_samples_per_concept,
                                    gt_samples_per_concept_test,
                                    gt_images_per_concept_test,
                                    dataset_name,
                                    model_name,
                                    model_input_size,
                                    con_label,
                                    act_metrics,
                                    embeds,
                                    embeddings_file,
                                    sample_type,
                                    patch_size=14
                                )
    