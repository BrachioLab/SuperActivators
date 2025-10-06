import sys
import os
sys.path.append(os.path.abspath("../utils"))
import torch

from compute_concepts_utils import compute_signed_distances


DATASET_NAMES = ['CLEVR', 'Llama']
MODELS = ['CLIP', 'CLEVR']
SAMPLES_TYPES = ['patch', 'cls']
PERCENT_THRU_MODEL = 100
MODEL_NAME = 'Llama'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CON_LABEL = f"{MODEL_NAME}_linsep_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}"

EMBEDDINGS_FILE = f'{MODEL_NAME}_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
CONCEPTS_FILE = f'linsep_concepts_{MODEL_NAME}_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'

if __name__ == "__main__":
    #kmeans
    print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME} kmeans concepts")
    CON_LABEL= f"{MODEL_NAME}_kmeans_{N_CLUSTERS}_{SAMPLE_TYPE}_embeddings_kmeans_percentthrumodel_{PERCENT_THRU_MODEL}"

    CONCEPTS_FILE = f'kmeans_{N_CLUSTERS}_concepts_{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
    COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'

    cos_sims = pd.read_csv(f'Cosine_Similarities/{DATASET_NAME}/kmeans/matched_{COSSIM_FILE}')
    alignment_results = torch.load(f'Unsupervised_Matches/{DATASET_NAME}/{CON_LABEL}.pt')

    matched_gt_samples_per_concept_test = {alignment_results[k][0]: v for k, v in gt_samples_per_concept_test.items()}

    #linsep concepts
    print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME}, linsep concepts")
    CON_LABEL= f"{MODEL_NAME}_kmeans_{N_CLUSTERS}_linsep_{SAMPLE_TYPE}_embeddings_kmeans_percentthrumodel_{PERCENT_THRU_MODEL}"
    EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
    DISTS_FILE = f'dists_kmeans_{N_CLUSTERS}_linsep_concepts_{EMBEDDINGS_FILE[:-3]}.csv'

    dists = pd.read_csv(f'Distances/{DATASET_NAME}/kmeans/matched_{DISTS_FILE}')
    alignment_results = torch.load(f'Unsupervised_Matches/{DATASET_NAME}/{CON_LABEL}.pt')

    matched_gt_samples_per_concept_test = {alignment_results[k][0]: v for k, v in gt_samples_per_concept_test.items()}
    for percentile in tqdm([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
        find_all_superdetector_patches(percentile, dists, matched_gt_samples_per_concept_test, 
                                    DATASET_NAME, MODEL_INPUT_SIZE, CON_LABEL, DEVICE)