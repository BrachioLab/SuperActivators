from tqdm import tqdm
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("utils"))
from superdetector_inversion_utils import find_all_superdetector_patches

MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560))]
DATASETS = ['CLEVR', 'Coco']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    for MODEL_NAME, MODEL_INPUT_SIZE in MODELS:
        for DATASET_NAME in DATASETS:
            
            #avg concepts
            print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME} avg concepts")
            EMBEDDINGS_FILE = f'{MODEL_NAME}_patch_embeddings_percentthrumodel_100.pt'
            CONCEPTS_FILE = f'avg_concepts_{EMBEDDINGS_FILE}'
            CON_LABEL = f"{MODEL_NAME}_avg_patch_embeddings_percentthrumodel_100"

            COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'
            cos_sims = pd.read_csv(f"Cosine_Similarities/{DATASET_NAME}/{COSSIM_FILE}")

            gt_samples_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')

            for percentile in tqdm([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
                find_all_superdetector_patches(percentile, cos_sims, gt_samples_per_concept_test, 
                                                DATASET_NAME, MODEL_INPUT_SIZE, CON_LABEL, DEVICE)
                
            for (BALANCE_DATA, BALANCE_NEGATIVES) in [(True, True), (True, False)]:
                print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME}, linsep balance overall {BALANCE_DATA} balance negative {BALANCE_NEGATIVES} concepts")
                CONCEPTS_FILE = f'linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_{EMBEDDINGS_FILE}'
                CON_LABEL = f"{MODEL_NAME}_linsep_patch_embeddings_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_percentthrumodel_100"
                DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'
                
                gt_samples_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')
                
                
                dists = pd.read_csv(f"Distances/{DATASET_NAME}/{DISTS_FILE}")
                for percentile in tqdm([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
                    find_all_superdetector_patches(percentile, dists, gt_samples_per_concept_test, 
                                                DATASET_NAME, MODEL_INPUT_SIZE, CON_LABEL, DEVICE)