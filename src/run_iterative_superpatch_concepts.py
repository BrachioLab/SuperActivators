import torch
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath("utils"))

import importlib
import compute_concepts_utils
importlib.reload(compute_concepts_utils)

from compute_concepts_utils import compute_linear_separators_w_superpatches_across_pers

MODELS = [('Llama', ('text', 'text'), 'token'), ('CLIP', (224, 224), 'patch'), ('Llama', (560, 560), 'patch')]
DATASETS = ['Jailbreak', 'CLEVR', 'Coco']

BALANCE_NEGATIVES = [True, False]
BALANCE_DATA = True
PERCENT_THRU_MODEL = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_PERS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]

if __name__ == "__main__":
    for MODEL_NAME, MODEL_INPUT_SIZE, SAMPLE_TYPE in MODELS:
        for DATASET_NAME in DATASETS:
            if (DATASET_NAME == 'Jailbreak' and MODEL_INPUT_SIZE[0] != 'text') or (MODEL_INPUT_SIZE[0] == 'text' and DATASET_NAME != 'Jailbreak'):
                continue
            if (DATASET_NAME == 'CLEVR' and MODEL_NAME == 'CLIP'):
                continue
                    
            EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
            embeds_dic = torch.load(f"Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}")
            embeds = embeds_dic['normalized_embeddings']

            for BALANCE_NEGATIVE in BALANCE_NEGATIVES:
                CONCEPTS_FILE = f'iterative_superpatch_linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVE}_{EMBEDDINGS_FILE}'
                DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'

                ORIGINAL_CONCEPTS_FILE = f'linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVE}_{EMBEDDINGS_FILE}'
                ORIGINAL_DISTS_FILE = f'dists_{ORIGINAL_CONCEPTS_FILE[:-3]}.csv'
                original_dists = pd.read_csv(f"Distances/{DATASET_NAME}/{ORIGINAL_DISTS_FILE}")


                CON_LABEL = f"iterative_superpatch_{MODEL_NAME}_linsep_{SAMPLE_TYPE}_embeddings_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVE}_percentthrumodel_{PERCENT_THRU_MODEL}"
                    
                gt_patches_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{MODEL_INPUT_SIZE}.pt')
                gt_patches_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_train_inputsize_{MODEL_INPUT_SIZE}.pt')
                gt_patches_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')

                gt_samples_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_inputsize_{MODEL_INPUT_SIZE}.pt')
                gt_samples_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_train_inputsize_{MODEL_INPUT_SIZE}.pt')
                gt_samples_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')

                #make sure only considering concepts with samples in each split
                # concept_keys = set(gt_patches_per_concept_train.keys()) & set(gt_patches_per_concept_test.keys())
                # gt_patches_per_concept = {k: v for k, v in gt_patches_per_concept.items() if k in concept_keys}
                # gt_patches_per_concept_train = {k: v for k, v in gt_patches_per_concept_train.items() if k in concept_keys}
                # gt_patches_per_concept_test = {k: v for k, v in gt_patches_per_concept_test.items() if k in concept_keys}
                # gt_samples_per_concept = {k: v for k, v in gt_samples_per_concept.items() if k in concept_keys}
                # gt_samples_per_concept_train = {k: v for k, v in gt_samples_per_concept_train.items() if k in concept_keys}
                # gt_samples_per_concept_test = {k: v for k, v in gt_samples_per_concept_test.items() if k in concept_keys} 
                
                original_dists = pd.read_csv(f"Distances/{DATASET_NAME}/{ORIGINAL_DISTS_FILE}")
                #negatives random
                compute_linear_separators_w_superpatches_across_pers(TOP_PERS, embeds, original_dists, gt_samples_per_concept, DATASET_NAME, 
                                         MODEL_INPUT_SIZE, device=DEVICE, output_file=CONCEPTS_FILE, lr=0.01, 
                                         epochs=100, batch_size=32, patience=15, 
                                          tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5,
                                          balance_data=BALANCE_DATA, 
                                          balance_negatives=BALANCE_NEGATIVES,
                                          impose_negatives=False)
                #negatives most negative
                compute_linear_separators_w_superpatches_across_pers(TOP_PERS, embeds, original_dists, gt_samples_per_concept, DATASET_NAME, 
                                         MODEL_INPUT_SIZE, device=DEVICE, output_file=CONCEPTS_FILE, lr=0.01, 
                                         epochs=100, batch_size=32, patience=15, 
                                          tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5,
                                          balance_data=BALANCE_DATA, 
                                          balance_negatives=BALANCE_NEGATIVES,
                                          impose_negatives=True)