import torch
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath("utils"))

import importlib
import compute_concepts_utils
import patch_alignment_utils
import visualize_concepts_w_samples_utils
import general_utils
import quant_concept_evals_utils 
import superdetector_finetuning_utils
importlib.reload(compute_concepts_utils)
importlib.reload(patch_alignment_utils)
importlib.reload(visualize_concepts_w_samples_utils)
importlib.reload(general_utils)
importlib.reload(quant_concept_evals_utils)
importlib.reload(superdetector_finetuning_utils)

from compute_concepts_utils import plot_train_history, \
     compute_signed_distances
from general_utils import retrieve_topn_samples, load_images
from patch_alignment_utils import compute_heatmaps_for_concept
from visualize_concepts_w_samples_utils import plot_patchsims_heatmaps_all_concepts, plot_concept_evolution_over_iterations
from quant_concept_evals_utils import get_patch_detection_tensor, compute_concept_thresholds_over_percentiles
from superdetector_finetuning_utils import compute_f1_over_iterations, plot_f1_over_iterations, \
    compute_linear_separators_finetuned_w_superpatches, plot_all_concepts_metric, \
    plot_weighted_average_metric, plot_best_f1_per_epoch_per_concept, plot_weighted_avg_best_f1_per_epoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
scratch_dir = '/scratch'

MODELS = [('Llama', (560, 560), 'patch'), ('CLIP', (224, 224), 'patch')]
DATASETS = ['Coco', 'CLEVR']

BALANCE_DATA = True
BALANCE_NEGATIVES = False
RESET_OPTIMIZER = False
USE_GT_LABELS = True
IMPOSE_NEGATIVES = False
ADD_NOISE = 0
HIGHEST_NEGATIVES = True

# FINE_TUNING_PARAMS = [[('init', 50)],
#                       [('init', 2), (.9, 2), (.8, 2), (.7, 2), (.6, 2), (.5, 2), (.4, 2), (.3, 2), (.2, 2), (.1, 2), (.05, 2)],
#                       [('init', 5), (.75, 5), (.5, 5), (.25, 5), (.05, 5)],
#                       [('init', 5), (.45, 5), (.4, 5), (.35, 5), (.3, 5), (.25, 5), (.2, 5), (.15, 5), (.1, 5), (.05, 5)],
#                       [('init', 5), (.2, 10), (.1, 10), (.05, 10), (.02, 10)],
#                       [('init', 5), (.2, 5), (.1, 5), (.05, 5), (.02, 5)],
#                       [('init', 10), (.2, 10), (.1, 5), (.05, 5), (.02, 5)],
#                       [('init', 1), (.2, 5), (.1, 5), (.05, 5), (.02, 5)],
#                       [('init', 1), (.45, 1), (.4, 1), (.35, 1), (.3, 1), (.25, 1), (.2, 1), (.15, 1), (.1, 1), (.05, 1)],
#                       [('init', 1), (.1, 20)],
#                       [('init', 5), (.2, 10), (.1, 10), (.05, 10), (.02, 10)],
#                       [('init', 2), (.5, 2)]
#                      ]
FINE_TUNING_PARAMS = [
                      [('init', 2), (.9, 2), (.8, 2), (.7, 2), (.6, 2), (.5, 2), (.4, 2), (.3, 2), (.2, 2), (.1, 2), (.05, 2)],
                      [('init', 5), (.75, 5), (.5, 5), (.25, 5), (.05, 5)],
                      [('init', 5), (.3, 10), (.2, 10), (.1, 10), (.05, 10)],
                      [('init', 1), (.75, 5), (.5, 5), (.25, 5), (.05, 5)]
                     ]

    

if __name__ == "__main__":
    for (MODEL_NAME, MODEL_INPUT_SIZE, SAMPLE_TYPE) in MODELS:
        for DATASET_NAME in DATASETS:
            EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_100.pt'
            CONCEPTS_FILE = f'iterative_superpatch_linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_{EMBEDDINGS_FILE}'
            DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'
            ORIGINAL_CONCEPTS_FILE = f'linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_{EMBEDDINGS_FILE}'
            ORIGINAL_DISTS_FILE = f'dists_{ORIGINAL_CONCEPTS_FILE[:-3]}.csv'
            original_dists = pd.read_csv(f"Distances/{DATASET_NAME}/{ORIGINAL_DISTS_FILE}")
            CON_LABEL = f"iterative_superpatch_{MODEL_NAME}_linsep_{SAMPLE_TYPE}_embeddings_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_percentthrumodel_100"
            
            embeds_dic = torch.load(f"{scratch_dir}/Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}")
            embeds = embeds_dic['normalized_embeddings']
            gt_patches_per_concept = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{MODEL_INPUT_SIZE}.pt')
            gt_patches_per_concept_train = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_patch_per_concept_train_inputsize_{MODEL_INPUT_SIZE}.pt')
            gt_patches_per_concept_test = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')

            gt_samples_per_concept = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_samples_per_concept_inputsize_{MODEL_INPUT_SIZE}.pt')
            gt_samples_per_concept_train = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_samples_per_concept_train_inputsize_{MODEL_INPUT_SIZE}.pt')
            gt_samples_per_concept_test = torch.load(f'{scratch_dir}/GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')
            
            for fine_tuning_params in FINE_TUNING_PARAMS:
                for HIGHEST_NEGATIVES in ['True', 'False']:
                    print(f"Running model {MODEL_NAME} on dataset {DATASET_NAME} with the following parameters:")
                    print(fine_tuning_params)
                    out = f'finetuned_{fine_tuning_params}_{CONCEPTS_FILE}'
                    if ADD_NOISE > 0:
                        out = f'noise_{ADD_NOISE}_' + out
                    if IMPOSE_NEGATIVES:
                        out = 'impose_neg_' + out
                    if HIGHEST_NEGATIVES:
                        out = 'highest_neg_' + out
                    if USE_GT_LABELS:
                        out = 'gtlabels_'+ out
                    if not RESET_OPTIMIZER:
                        out = 'noreset_'+ out
                    try:
                        model_trainers = torch.load(f'Model_Trainers/{DATASET_NAME}/{out}', map_location=torch.device(DEVICE))
                    except:
                        model_trainers = compute_linear_separators_finetuned_w_superpatches(fine_tuning_params, embeds, 
                                                                         gt_patches_per_concept, gt_samples_per_concept_train,
                                                                         gt_patches_per_concept_test, gt_samples_per_concept_test,
                                                                         DATASET_NAME, MODEL_INPUT_SIZE, device=DEVICE, 
                                                                         output_file=CONCEPTS_FILE, lr=0.001, train_batch_size=64, 
                                                                         dist_batch_size= 10000, patience=15, tolerance=0.001, 
                                                                         weight_decay=0.0001, lr_step_size=5, lr_gamma=0.8,
                                                                        balance_data=True, balance_negatives=False,
                                                                                            use_gt_labels=USE_GT_LABELS,
                                                                         impose_negatives=IMPOSE_NEGATIVES, 
                                                                                            reset_optimizer=RESET_OPTIMIZER, 
                                                                         highest_negatives=HIGHEST_NEGATIVES, add_noise=ADD_NOISE,
                                                                         temp_file=f'/scratch/temp/temp.csv')
                
                
                      