import torch
from collections import defaultdict

import sys
import os
sys.path.append(os.path.abspath("utils"))

from general_utils import load_images
from compute_concepts_utils import compute_linear_separators

DATASET_NAMES = ['Coco', 'CLEVR']
MODELS = [('Llama', (560, 560)), ('CLIP', (224, 224))]
SAMPLE_TYPES = [('patch', 1000), ('cls', 50)]
SAMPLE_TYPES = [('cls', 50)]
PERCENT_THRU_MODEL = 100
N_CLUSTERS = 1000

scratch_dir = '/scratch/cgoldberg'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    for DATASET_NAME in DATASET_NAMES:
        for MODEL_NAME, MODEL_INPUT_SIZE in MODELS:
            for ST, N_CLUSTERS in SAMPLE_TYPES:
                print(f"Computing {ST} concepts for model {MODEL_NAME} on dataset {DATASET_NAME}")


                EMBEDDINGS_FILE = f'{MODEL_NAME}_{ST}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'

                print("Loading embeddings")
                embeds_dic = torch.load(f"{scratch_dir}/Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}")
                embeds = embeds_dic['normalized_embeddings']

                CON_LABEL= f"{MODEL_NAME}_kmeans_{N_CLUSTERS}_linsep_{ST}_embeddings_kmeans_percentthrumodel_{PERCENT_THRU_MODEL}"
                CONCEPTS_FILE = f'kmeans_{N_CLUSTERS}_linsep_concepts_{MODEL_NAME}_{ST}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'

                KMEANS_CONCEPTS_FILE = f'kmeans_{N_CLUSTERS}_concepts_{MODEL_NAME}_{ST}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'

                print("loading gt clusters from kmeans")
                train_cluster_to_samples = torch.load(f'Concepts/{DATASET_NAME}/train_samples_{KMEANS_CONCEPTS_FILE}')
                test_cluster_to_samples = torch.load(f'Concepts/{DATASET_NAME}/test_samples_{KMEANS_CONCEPTS_FILE}')
                cluster_to_samples = defaultdict(list)
                for cluster, samples in train_cluster_to_samples.items():
                    cluster_to_samples[cluster].extend(samples)
                for cluster, samples in test_cluster_to_samples.items():
                    cluster_to_samples[cluster].extend(samples)
                for cluster in cluster_to_samples:
                    cluster_to_samples[cluster] = sorted(cluster_to_samples[cluster])
                cluster_to_samples = dict(cluster_to_samples)

                print("training separators")
                compute_linear_separators(embeds, cluster_to_samples, DATASET_NAME, ST, MODEL_INPUT_SIZE, 
                                  device=DEVICE, output_file=CONCEPTS_FILE, lr=0.001, epochs=1000, batch_size=64, patience=20, 
                                  tolerance=0.001, weight_decay=1e-4, lr_step_size=5, lr_gamma=0.8, balance_data=True, 
                                  balance_negatives=False)
    
    