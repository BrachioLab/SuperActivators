import torch
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("../utils"))

from gt_concept_segmentation_utils import sort_mapping_by_split, map_concepts_to_image_indices, map_concepts_to_patch_indices
from compute_concepts_utils import compute_avg_concept_vectors, compute_cosine_sims, compute_linear_separators, compute_signed_distances

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560))]
DATASETS = ['CLEVR', 'Coco']
SAMPLE_TYPES = [('cls', 'image', map_concepts_to_image_indices), ('patch', 'patch', map_concepts_to_patch_indices)]


if __name__ == "__main__":
    for DATASET_NAME in DATASETS:
        for (MODEL_NAME, INPUT_IMAGE_SIZE) in MODELS:  
            for ST1, ST2, MAP_FXN in SAMPLE_TYPES:
                print("Loading emebeddings")
                EMBEDDINGS_FILE = f'{MODEL_NAME}_{ST1}_embeddings_percentthrumodel_100.pt'
                embeds = torch.load(f"../Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}")['normalized_embeddings']
                
                # get the ground truth mappings
                # gt_samples_per_concept = MAP_FXN(dataset_name=DATASET_NAME, model_input_size=INPUT_IMAGE_SIZE)
                # gt_samples_per_concept_train, gt_samples_per_concept_test = sort_mapping_by_split(gt_samples_per_concept, DATASET_NAME,
                #                                                                               sample_type=ST2, 
                #                                                                               model_input_size=INPUT_IMAGE_SIZE)
                if ST1 == 'patch':
                    gt_samples_per_concept = torch.load(f'../GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
                    gt_samples_per_concept_train = torch.load(f'../GT_Samples/{DATASET_NAME}/gt_patch_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
                    gt_samples_per_concept_test = torch.load(f'../GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')
                else:
                    gt_samples_per_concept = torch.load(f'../GT_Samples/{DATASET_NAME}/gt_samples_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
                    gt_samples_per_concept_train = torch.load(f'../GT_Samples/{DATASET_NAME}/gt_samples_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
                    gt_samples_per_concept_test = torch.load(f'../GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')


                # avg concepts and cos sims
#                 print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME} avg concepts sample type {ST1}\n")
#                 CONCEPTS_FILE = f'avg_concepts_{EMBEDDINGS_FILE}'
# #                 CON_LABEL = f"{MODEL_NAME}_avg_{ST1}_embeddings_percentthrumodel_100"
#                 COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'
                
#                 concepts = compute_avg_concept_vectors(gt_samples_per_concept_train, embeds, 
#                                        dataset_name=DATASET_NAME, output_file=CONCEPTS_FILE)
#                 concepts = torch.load(f'../Concepts/{DATASET_NAME}/{CONCEPTS_FILE}')
#                 compute_cosine_sims(embeddings = embeds, 
#                     concepts = concepts, 
#                     output_file = COSSIM_FILE,
#                     dataset_name = DATASET_NAME,
#                     device = DEVICE,
#                     batch_size=64)
                 
            
                #linsep concepts and dists
                for (BALANCE_DATA, BALANCE_NEGATIVES) in [(True, True), (True, False)]:
                    print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME}, linsep balance overall {BALANCE_DATA} balance negative {BALANCE_NEGATIVES} concepts sample type {ST1}\n")
                    CONCEPTS_FILE = f'linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_{EMBEDDINGS_FILE}'
                    CON_LABEL = f"{MODEL_NAME}_linsep_{ST1}_embeddings_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_percentthrumodel_100"
                    DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'

                    concepts, logs = compute_linear_separators(embeds, gt_samples_per_concept, DATASET_NAME, 
                                                 sample_type=ST2, device=DEVICE, model_input_size=INPUT_IMAGE_SIZE,
                                                 output_file=CONCEPTS_FILE, batch_size=64,
                                                 lr=0.001, epochs=1000, patience=20, tolerance=0.001,
                                                 weight_decay=0.0001, lr_step_size=5, lr_gamma=0.8,
                                                 balance_data=BALANCE_DATA,
                                                 balance_negatives=BALANCE_NEGATIVES)
                    concepts = torch.load(f'../Concepts/{DATASET_NAME}/{CONCEPTS_FILE}')
                    compute_signed_distances(embeds, concepts, DATASET_NAME, device=DEVICE, batch_size=5000,
                                                     output_file=DISTS_FILE)
                