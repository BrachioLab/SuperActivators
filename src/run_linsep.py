import torch
import sys
import os
sys.path.append(os.path.abspath("utils"))

from compute_concepts_utils import compute_linear_separators, compute_signed_distances


PERCENT_THRU_MODEL = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # for DATASET_NAME in ['CLEVR', 'Coco']:
    for DATASET_NAME in ['Coco']:
        for (MODEL_NAME, INPUT_IMAGE_SIZE) in [('CLIP', (224, 224)), ('Llama', (560, 560))]:
        # for (MODEL_NAME, INPUT_IMAGE_SIZE) in [('Llama', (560, 560))]:
        # for (MODEL_NAME, INPUT_IMAGE_SIZE) in [('CLIP', (224, 224))]:
            # for SAMPLE_TYPE in ['cls', 'patch']:
            for SAMPLE_TYPE in ['patch']:
                # if DATASET_NAME == 'CLEVR' and SAMPLE_TYPE == 'cls':
                #     continue
                EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
                embeds_dic = torch.load(f"Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}")
                embeds = embeds_dic['normalized_embeddings']
                
                #for (BALANCE_DATA, BALANCE_NEGATIVES) in [(True, True), (True, False), (False, False)]:
                # for (BALANCE_DATA, BALANCE_NEGATIVES) in [(True, True), (True, False)]:
                for (BALANCE_DATA, BALANCE_NEGATIVES) in [(False, False)]:
                    print(f"\n\n\nStarting to compute concepts for dataset {DATASET_NAME}, model {MODEL_NAME}, sample type {SAMPLE_TYPE}, balance data {BALANCE_DATA}, balance negatives {BALANCE_NEGATIVES}\n\n\n")
                
                    CONCEPTS_FILE = f'linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_{EMBEDDINGS_FILE}'
                    CON_LABEL = f"{MODEL_NAME}_linsep_{SAMPLE_TYPE}_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}"
                    DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'
        
                    if SAMPLE_TYPE == 'patch':
                        gt_patches_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
                        concepts, logs = compute_linear_separators(embeds, gt_patches_per_concept, DATASET_NAME, 
                                             sample_type='patch', device=DEVICE, model_input_size=INPUT_IMAGE_SIZE,
                                             output_file=CONCEPTS_FILE, batch_size=1024,
                                             lr=0.001, epochs=1000, patience=20, tolerance=0.001,
                                             weight_decay=0.01, lr_step_size=5, lr_gamma=0.8,
                                             balance_data=BALANCE_DATA,
                                             balance_negatives=BALANCE_NEGATIVES,
                                             remove_lowest_50pct=(DATASET_NAME=='Coco'))
                    else:
                        gt_images_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_images_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
                        concepts, logs = compute_linear_separators(embeds, gt_images_per_concept, DATASET_NAME, 
                                             sample_type='image', device=DEVICE, model_input_size=INPUT_IMAGE_SIZE,
                                             output_file=CONCEPTS_FILE, batch_size=64,
                                             lr=0.001, epochs=1000, patience=20, tolerance=0.001,
                                             weight_decay=0.01, lr_step_size=5, lr_gamma=0.8,
                                             balance_data=BALANCE_DATA,
                                             balance_negatives=BALANCE_NEGATIVES,
                                             remove_lowest_50pct=(DATASET_NAME=='Coco'))
                    
                    dists = compute_signed_distances(embeds, concepts, DATASET_NAME, output_file=DISTS_FILE, device=DEVICE)
