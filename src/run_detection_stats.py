import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("utils"))

from quant_concept_evals_utils import compute_detection_metrics_over_percentiles
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560))]  
DATASETS = ['CLEVR', 'Coco']
SAMPLE_TYPES = ['cls', 'patch']
percentiles = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

if __name__ == "__main__":
    for DATASET_NAME in DATASETS:
        for (MODEL_NAME, MODEL_INPUT_SIZE) in MODELS:  
            for SAMPLE_TYPE in SAMPLE_TYPES:
                EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_100.pt'

                #avg concepts
                print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME} avg concepts")
                CONCEPTS_FILE = f'avg_concepts_{EMBEDDINGS_FILE}'
                CON_LABEL = f"{MODEL_NAME}_avg_{SAMPLE_TYPE}_embeddings_percentthrumodel_100"

                COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'
                cos_sims = pd.read_csv(f"Cosine_Similarities/{DATASET_NAME}/{COSSIM_FILE}")

                gt_patches_per_concept_test= \
                    torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')
                gt_images_per_concept_test= \
                    torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')
                if SAMPLE_TYPE == 'patch':
                    gt_samples_per_concept_test = gt_patches_per_concept_test
                elif SAMPLE_TYPE == 'cls':
                    gt_samples_per_concept_test = gt_images_per_concept_test
                    
                compute_detection_metrics_over_percentiles(percentiles, gt_samples_per_concept_test, 
                                                           gt_images_per_concept_test, 
                                                           cos_sims, DATASET_NAME, MODEL_INPUT_SIZE, DEVICE, 
                                                           CON_LABEL, sample_type=SAMPLE_TYPE, patch_size=14)
                
            
            
                #linsep concepts
                for (BALANCE_DATA, BALANCE_NEGATIVES) in [(True, True), (True, False)]:
                    print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME}, linsep balance overall {BALANCE_DATA} balance negative {BALANCE_NEGATIVES} concepts")
                    CONCEPTS_FILE = f'linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_{EMBEDDINGS_FILE}'
                    CON_LABEL = f"{MODEL_NAME}_linsep_{SAMPLE_TYPE}_embeddings_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_percentthrumodel_100"
                    DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'

                    gt_patches_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')
                    gt_images_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')


                    dists = pd.read_csv(f"Distances/{DATASET_NAME}/{DISTS_FILE}")
                    if SAMPLE_TYPE == 'patch':
                        gt_samples_per_concept_test = gt_patches_per_concept_test
                    elif SAMPLE_TYPE == 'cls':
                        gt_samples_per_concept_test = gt_images_per_concept_test                                        
                    compute_detection_metrics_over_percentiles(percentiles, gt_samples_per_concept_test, 
                                                               gt_images_per_concept_test, 
                                                               dists, DATASET_NAME, MODEL_INPUT_SIZE, DEVICE, 
                                                               CON_LABEL, sample_type=SAMPLE_TYPE, patch_size=14)
                
             