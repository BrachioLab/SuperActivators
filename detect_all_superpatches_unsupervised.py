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
N_CLUSTERS = 1000
SAMPLE_TYPE = 'patch'
PERCENT_THRU_MODEL = 100
scratch_dir = '/scratch'

if __name__ == "__main__":
    for MODEL_NAME, MODEL_INPUT_SIZE in MODELS:
        for DATASET_NAME in DATASETS:
            gt_samples_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{MODEL_INPUT_SIZE}.pt')
            
            #kmeans
            print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME} kmeans concepts")
            CON_LABEL= f"{MODEL_NAME}_kmeans_{N_CLUSTERS}_{SAMPLE_TYPE}_embeddings_kmeans_percentthrumodel_{PERCENT_THRU_MODEL}"

            CONCEPTS_FILE = f'kmeans_{N_CLUSTERS}_concepts_{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
            COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'

            cos_sims = pd.read_csv(f'{scratch_dir}/Cosine_Similarities/{DATASET_NAME}/kmeans/{COSSIM_FILE}')
            
            
            alignment_results = torch.load(f'Unsupervised_Matches/{DATASET_NAME}/bestdetects_{CON_LABEL}.pt')
            matching_cluster_ids = [info['best_cluster'] for info in alignment_results.values()]
            matching_cos_sims = cos_sims[[col for col in cos_sims.columns if col in matching_cluster_ids]] 
            
            matched_gt_samples_per_concept_test = {}
            for c, info in alignment_results:
                matched_gt_samples_per_concept_test[info['best_cluster']] = gt_samples_per_concept_test[c]

            for percentile in tqdm([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
                find_all_superdetector_patches(percentile, matching_cos_sims, matched_gt_samples_per_concept_test, 
                                                DATASET_NAME, MODEL_INPUT_SIZE, CON_LABEL, DEVICE)
                
            
            #linsep concepts
            print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME}, linsep concepts")
            CON_LABEL= f"{MODEL_NAME}_kmeans_{N_CLUSTERS}_linsep_{SAMPLE_TYPE}_embeddings_kmeans_percentthrumodel_{PERCENT_THRU_MODEL}"
            EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
            DISTS_FILE = f'dists_kmeans_{N_CLUSTERS}_linsep_concepts_{EMBEDDINGS_FILE[:-3]}.csv'

            dists = pd.read_csv(f'Distances/{DATASET_NAME}/kmeans/matched_{DISTS_FILE}')
            
            alignment_results = torch.load(f'Unsupervised_Matches/{DATASET_NAME}/bestdetects_{CON_LABEL}.pt')
            matching_cluster_ids = [info['best_cluster'] for info in alignment_results.values()]
            matched_dists = dists[[col for col in dists.columns if col in matching_cluster_ids]] 
            matched_gt_samples_per_concept_test = {}
            for c, info in alignment_results:
                matched_gt_samples_per_concept_test[info['best_cluster']] = gt_samples_per_concept_test[c]
                
            for percentile in tqdm([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
                find_all_superdetector_patches(percentile, matched_dists, matched_gt_samples_per_concept_test, 
                                            DATASET_NAME, MODEL_INPUT_SIZE, CON_LABEL, DEVICE)