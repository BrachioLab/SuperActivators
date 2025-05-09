import torch
import pandas as pd
import numpy as np

import sys
import os

sys.path.append(os.path.abspath("utils"))

import importlib
import compute_concepts_utils 
import general_utils
import patch_alignment_utils
import visualize_concepts_w_samples_utils
import quant_concept_evals_utils
import gt_concept_segmentation_utils
import unsupervised_utils

importlib.reload(compute_concepts_utils)
importlib.reload(general_utils)
importlib.reload(patch_alignment_utils)
importlib.reload(visualize_concepts_w_samples_utils)
importlib.reload(quant_concept_evals_utils)
importlib.reload(gt_concept_segmentation_utils)
importlib.reload(unsupervised_utils)

from general_utils import load_images, plot_image_with_attributes, retrieve_topn_samples
from compute_concepts_utils import get_clip_patch_embeddings, compute_batch_embeddings, \
     gpu_kmeans, evaluate_clustering_metrics,  \
     plot_concepts_most_aligned_w_chosen_patch

from patch_alignment_utils import compute_patch_similarities_to_vector, \
     compute_heatmaps_for_concept

from quant_concept_evals_utils import compute_concept_cosine_stats, compute_concept_thresholds, \
    evaluate_thresholds_across_dataset, compute_concept_metrics, print_threshold_eval_results, plot_metric, plot_metric_distribution,\
    concept_heatmap, concept_heatmap_random_samples, compute_cossim_hist_stats, plot_cosine_similarity_histograms

from visualize_concepts_w_samples_utils import neighboring_patch_comparisons, plot_top_patches_for_concept, \
    plot_patchsims_all_concepts, plot_patchsims_for_concept, plot_most_similar_patches_w_heatmaps_and_corr_images, \
    plot_patchsims_for_concept, plot_patchsims_heatmaps_all_concepts#, plot_similar_patches_to_given_patch

from gt_concept_segmentation_utils import map_concepts_to_patch_indices, sort_mapping_by_split, find_closest_to_gt

from unsupervised_utils import get_topn_aligning_clusters_per_concept, get_most_aligning_cluster_per_concept, \
    find_best_clusters_per_concept, plot_best_cluster_heatmap_per_concept, \
    detect_then_invert_metrics_over_percentiles_all_pairs, plot_cluster_heatmap_per_concept, \
    compute_detection_metrics_over_percentiles_allpairs, compute_cosine_sims_allpairs


scratch_dir = '/scratch/'


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [('CLIP', (224, 224), 'patch'), ('Llama', (560, 560), 'patch')]
DATASETS = ['CLEVR', 'Coco']

MODELS = [('Llama', (560, 560), 'patch')]
DATASETS = ['Coco']

if __name__ == "__main__":
    for DATASET_NAME in DATASETS:
        for (MODEL_NAME, INPUT_IMAGE_SIZE, SAMPLE_TYPE) in MODELS:
                
            PERCENT_THRU_MODEL = 100
            N_CLUSTERS = 1000

            CON_LABEL= f"{MODEL_NAME}_kmeans_{N_CLUSTERS}_patch_embeddings_kmeans_percentthrumodel_{PERCENT_THRU_MODEL}"

            EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
            CONCEPTS_FILE = f'kmeans_{N_CLUSTERS}_concepts_{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
            COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'
            
#             print("Loading embeddings...")
#             embeds_dic = torch.load(f"{scratch_dir}/Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}")
#             embeds = embeds_dic['normalized_embeddings']
            
#             print("Loading concepts...")
#             concepts = torch.load(f'Concepts/{DATASET_NAME}/{CONCEPTS_FILE}')

            print("Loading gt info...")
            gt_patches_per_concept = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_patches_per_concept_train = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_patch_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_patches_per_concept_test = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')

            gt_images_per_concept = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_samples_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_images_per_concept_train = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_samples_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_images_per_concept_test = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')
            
            
            
            # print("Loading cos sims...")
            # compute_cosine_sims_per_concept(embeddings = embeds, 
            #                                 concepts = concepts, 
            #                                 output_file = COSSIM_FILE,
            #                                 scratch_dir=scratch_dir,
            #                                 dataset_name = DATASET_NAME, device=DEVICE,
            #                                 batch_size = 500000) #1 batch would be 8000000
            # compute_cosine_sims_allpairs(embeds, concepts, DATASET_NAME, 
            #                              DEVICE, COSSIM_FILE, scratch_dir, batch_size=32)
            
            print(f"Detection analysis on dataset {DATASET_NAME}, model {MODEL_NAME}")
            percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
            compute_detection_metrics_over_percentiles_allpairs(percentiles, gt_patches_per_concept_test, 
                                                        gt_images_per_concept_test, 
                                                       DATASET_NAME, INPUT_IMAGE_SIZE, DEVICE, 
                                                       CON_LABEL, COSSIM_FILE, scratch_dir=scratch_dir,
                                                       cluster_batch_size=200,
                                                       sample_type='patch', patch_size=14)

# detect_then_invert_metrics_over_percentiles_all_pairs(detect_percentiles=percentiles, 
#             #                                                     invert_percentiles=percentiles, 
#             #                                                     act_metrics=cos_sims, 
#             #                                                     concepts=concepts, 
#             #                                                     gt_samples_per_concept=gt_patches_per_concept,
#             #                                                   gt_samples_per_concept_test=gt_patches_per_concept_test, 
#             #                                                     device=DEVICE, dataset_name=DATASET_NAME, 
#             #                                                     model_input_size=INPUT_IMAGE_SIZE, 
#             #                                                     con_label=CON_LABEL)
            

            
            
            
