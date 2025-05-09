import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("utils"))
from quant_concept_evals_utils import plot_activation_percentages_over_thresholds, plot_activation_count_distributions, compute_multiple_activation_analyses, compute_metrics_across_percentiles, detect_then_invert_metrics_over_percentiles

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAMES = ['CLEVR', 'Coco']
MODELS = [('Llama', (560, 560)), ('CLIP', (224, 224))]

if __name__ == "__main__":
    for DATASET_NAME in DATASET_NAMES:
        for (MODEL_NAME, INPUT_IMAGE_SIZE) in MODELS:
            #avg concepts
            print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME} avg concepts")
            EMBEDDINGS_FILE = f'{MODEL_NAME}_patch_embeddings_percentthrumodel_100.pt'
            CONCEPTS_FILE = f'avg_concepts_{EMBEDDINGS_FILE}'
            CON_LABEL = f"{MODEL_NAME}_avg_patch_embeddings_percentthrumodel_100"

            COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'
            cos_sims = pd.read_csv(f"Cosine_Similarities/{DATASET_NAME}/{COSSIM_FILE}")

            gt_patches_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_patches_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_patches_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')

            gt_images_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_images_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_images_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_image_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_images_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_image_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')

            concepts = torch.load(f'Concepts/{DATASET_NAME}/{CONCEPTS_FILE}')

            # make sure only considering concepts with samples in each split
            concept_keys = set(gt_patches_per_concept_train.keys()) & set(gt_patches_per_concept_test.keys())
            gt_patches_per_concept = {k: v for k, v in gt_patches_per_concept.items() if k in concept_keys}
            gt_patches_per_concept_train = {k: v for k, v in gt_patches_per_concept_train.items() if k in concept_keys}
            gt_patches_per_concept_test = {k: v for k, v in gt_patches_per_concept_test.items() if k in concept_keys}
            gt_images_per_concept = {k: v for k, v in gt_images_per_concept.items() if k in concept_keys}
            gt_images_per_concept_train = {k: v for k, v in gt_images_per_concept_train.items() if k in concept_keys}
            gt_images_per_concept_test = {k: v for k, v in gt_images_per_concept_test.items() if k in concept_keys}
            concepts = {k: v for k, v in concepts.items() if k in concept_keys}
            
            all_object_patches =  set()
            for concept, samples in gt_patches_per_concept.items():
                all_object_patches.update(samples)


#           #averge in-concept/out-of-concept percentages over different in-concept patch thresholds
            # plot_activation_percentages_over_thresholds(cos_sims, gt_patches_per_concept_train, 
            #                                                 gt_patches_per_concept_test, 
            #                                                 gt_images_per_concept_train,
            #                                                 gt_images_per_concept_test,
            #                                                 DATASET_NAME, 
            #                                                 INPUT_IMAGE_SIZE, DEVICE, CON_LABEL,
            #                                                 sample_type='patch')
            # plot_activation_count_distributions(cos_sims, gt_patches_per_concept_train, 
            #                               gt_patches_per_concept_test, gt_images_per_concept_train,
            #                               gt_images_per_concept_test, DATASET_NAME, 
            #                               INPUT_IMAGE_SIZE, DEVICE, CON_LABEL)
            # compute_multiple_activation_analyses(cos_sims, gt_patches_per_concept_train, gt_patches_per_concept_test, gt_images_per_concept_train, gt_images_per_concept_test, INPUT_IMAGE_SIZE, 
            #                              DATASET_NAME, CON_LABEL, DEVICE)
            # compute_metrics_across_percentiles(gt_patches_per_concept_test, concepts, cos_sims, INPUT_IMAGE_SIZE, DATASET_NAME, 
            #                            DEVICE, CON_LABEL, sample_type='patch', all_object_patches=all_object_patches,)
            percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
            detect_then_invert_metrics_over_percentiles(percentiles, percentiles, 
                                                cos_sims, concepts, gt_patches_per_concept, gt_patches_per_concept_test,
                                                DEVICE, DATASET_NAME, INPUT_IMAGE_SIZE, CON_LABEL,
                                                all_object_patches=all_object_patches,
                                                n_trials=10, balance_dataset=False, patch_size=14)
             
            
            
            #linsep concepts
            for (BALANCE_DATA, BALANCE_NEGATIVES) in [(True, True), (True, False)]:
                print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME}, linsep balance overall {BALANCE_DATA} balance negative {BALANCE_NEGATIVES} concepts")
                CONCEPTS_FILE = f'linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_{EMBEDDINGS_FILE}'
                CON_LABEL = f"{MODEL_NAME}_linsep_patch_embeddings_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_percentthrumodel_100"
                DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'
                
                gt_patches_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
                gt_patches_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
                gt_patches_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')

                gt_images_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_images_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
                gt_images_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_image_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
                gt_images_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_image_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')
                
                concepts = torch.load(f'Concepts/{DATASET_NAME}/{CONCEPTS_FILE}')
                                                        
                                                       

                #make sure only considering concepts with samples in each split + wasn't removed
                concept_keys = set(gt_patches_per_concept_train.keys()) & set(gt_patches_per_concept_test.keys()) & set(concepts.keys())
                gt_patches_per_concept = {k: v for k,v in gt_patches_per_concept.items() if k in concepts.keys()}
                gt_patches_per_concept_train = {k: v for k,v in gt_patches_per_concept_train.items() if k in concepts.keys()}
                gt_patches_per_concept_test = {k: v for k,v in gt_patches_per_concept_test.items() if k in concepts.keys()}
                                                        
                all_object_patches =  set()
                for concept, samples in gt_patches_per_concept.items():
                    all_object_patches.update(samples)
            
                dists = pd.read_csv(f"Distances/{DATASET_NAME}/{DISTS_FILE}")
                # plot_activation_percentages_over_thresholds(dists, gt_patches_per_concept_train, 
                #                                 gt_patches_per_concept_test, 
                #                                 gt_images_per_concept_train,
                #                                 gt_images_per_concept_test,
                #                                 DATASET_NAME, 
                #                                 INPUT_IMAGE_SIZE, DEVICE, CON_LABEL,
                #                                 sample_type='patch')
                # plot_activation_count_distributions(dists, gt_patches_per_concept_train, 
                #                           gt_patches_per_concept_test, gt_images_per_concept_train,
                #                           gt_images_per_concept_test, DATASET_NAME, 
                #                           INPUT_IMAGE_SIZE, DEVICE, CON_LABEL)
                # compute_multiple_activation_analyses(dists, gt_patches_per_concept_train, gt_patches_per_concept_test, gt_images_per_concept_train, gt_images_per_concept_test, INPUT_IMAGE_SIZE, DATASET_NAME, CON_LABEL, DEVICE)
                # compute_metrics_across_percentiles(gt_patches_per_concept_test, concepts, dists, INPUT_IMAGE_SIZE, DATASET_NAME, 
                #                        DEVICE, CON_LABEL, sample_type='patch', all_object_patches=all_object_patches)
                percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
                detect_then_invert_metrics_over_percentiles(percentiles, percentiles, 
                                                cos_sims, concepts, gt_patches_per_concept, gt_patches_per_concept_test,
                                                DEVICE, DATASET_NAME, INPUT_IMAGE_SIZE, CON_LABEL, all_object_patches=all_object_patches,
                                                n_trials=10, balance_dataset=False, patch_size=14)