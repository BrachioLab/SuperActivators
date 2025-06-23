import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("utils"))

from quant_concept_evals_utils import plot_activation_percentages_over_thresholds, plot_activation_count_distributions, compute_multiple_activation_analyses, compute_metrics_across_percentiles, detect_then_invert_metrics_over_percentiles, compute_cossim_hist_stats
from superdetector_inversion_utils import detect_then_invert_locally_metrics_over_percentiles, detect_then_invert_locally_performance_heatmap, all_superdetector_inversions_across_percentiles
from general_utils import get_coco_concepts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [('CLIP', (224, 224), 'patch', 'patch'), ('Llama', (560, 560), 'patch', 'patch'), ('Llama', ('text', 'text'), 'token', 'patch')]
DATASETS = ['CLEVR', 'Coco', 'Jailbreak']

if __name__ == "__main__":
    for DATASET_NAME in DATASETS:
        for (MODEL_NAME, INPUT_IMAGE_SIZE, SAMPLE_TYPE, ST2) in MODELS:  
            if (DATASET_NAME == 'Jailbreak' and INPUT_IMAGE_SIZE[0] != 'text') or (INPUT_IMAGE_SIZE[0] == 'text' and DATASET_NAME != 'Jailbreak'):
                continue
                
                
            EMBEDDINGS_FILE = f'{MODEL_NAME}_{SAMPLE_TYPE}_embeddings_percentthrumodel_100.pt'
            embeds_dic = torch.load(f"Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}")
            embeds = embeds_dic['normalized_embeddings']
    
            #avg concepts
            print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME} avg concepts")
            CONCEPTS_FILE = f'avg_concepts_{EMBEDDINGS_FILE}'
            CON_LABEL = f"{MODEL_NAME}_avg_{SAMPLE_TYPE}_embeddings_percentthrumodel_100"

            COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'
            cos_sims = pd.read_csv(f"Cosine_Similarities/{DATASET_NAME}/{COSSIM_FILE}")

            gt_patches_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_patches_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_patches_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')

            gt_images_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_images_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
            gt_images_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')

            # concepts = torch.load(f'Concepts/{DATASET_NAME}/{CONCEPTS_FILE}')
             percentiles = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
            compute_detection_metrics_over_percentiles(percentiles, gt_samples_per_concept_test, gt_images_per_concept_test, 
                                               sim_metrics, dataset_name, model_input_size, device, 
                                               con_label, sample_type='patch', patch_size=14)
#             all_object_patches =  set()
#             for concept, samples in gt_patches_per_concept.items():
#                 all_object_patches.update(samples)


#           #averge in-concept/out-of-concept percentages over different in-concept patch thresholds
#             plot_activation_percentages_over_thresholds(cos_sims, gt_patches_per_concept_train, 
#                                                             gt_patches_per_concept_test, 
#                                                             gt_images_per_concept_train,
#                                                             gt_images_per_concept_test,
#                                                             DATASET_NAME, 
#                                                             INPUT_IMAGE_SIZE, DEVICE, CON_LABEL,
#                                                             sample_type='patch')
#             plot_activation_count_distributions(cos_sims, gt_patches_per_concept_train, 
#                                           gt_patches_per_concept_test, gt_images_per_concept_train,
#                                           gt_images_per_concept_test, DATASET_NAME, 
#                                           INPUT_IMAGE_SIZE, DEVICE, CON_LABEL)
#             compute_multiple_activation_analyses(cos_sims, gt_patches_per_concept_train, gt_patches_per_concept_test, gt_images_per_concept_train, gt_images_per_concept_test, INPUT_IMAGE_SIZE, 
#                                          DATASET_NAME, CON_LABEL, DEVICE)
#             compute_metrics_across_percentiles(gt_patches_per_concept_test, concepts, cos_sims, INPUT_IMAGE_SIZE, DATASET_NAME, 
#                                        DEVICE, CON_LABEL, sample_type='patch', all_object_patches=all_object_patches,)
#                 percentiles = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
#                 if not (DATASET_NAME == 'CLEVR' and MODEL_NAME == 'CLIP'):
#                     detect_then_invert_metrics_over_percentiles(percentiles, percentiles, 
#                                                         cos_sims, concepts, gt_patches_per_concept, gt_patches_per_concept_test,
#                                                         DEVICE, DATASET_NAME, INPUT_IMAGE_SIZE, CON_LABEL,
#                                                         all_object_patches=all_object_patches,
#                                                         n_trials=2, balance_dataset=False, patch_size=14)

#                 all_superdetector_inversions_across_percentiles(percentiles, 'avg', embeds, cos_sims,
#                                        gt_patches_per_concept_test, DATASET_NAME, INPUT_IMAGE_SIZE, CON_LABEL, 
#                                                     DEVICE, patch_size=14, local=True)
#                 detect_then_invert_locally_metrics_over_percentiles(percentiles, percentiles, cos_sims, 
#                                                             concepts, gt_patches_per_concept, gt_patches_per_concept_test,
#                                                             DEVICE, DATASET_NAME, INPUT_IMAGE_SIZE, CON_LABEL,
#                                                             all_object_patches=all_object_patches, patch_size=14,
#                                                             agglomerate_type='avg')
            
            
            
            #linsep concepts
            for (BALANCE_DATA, BALANCE_NEGATIVES) in [(True, True), (True, False)]:
                print(f"Running analysis for dataset {DATASET_NAME}, Model {MODEL_NAME}, linsep balance overall {BALANCE_DATA} balance negative {BALANCE_NEGATIVES} concepts")
                CONCEPTS_FILE = f'linsep_concepts_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_{EMBEDDINGS_FILE}'
                CON_LABEL = f"{MODEL_NAME}_linsep_{SAMPLE_TYPE}_embeddings_BD_{BALANCE_DATA}_BN_{BALANCE_NEGATIVES}_percentthrumodel_100"
                DISTS_FILE = f'dists_{CONCEPTS_FILE[:-3]}.csv'
                
                gt_patches_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patches_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
                gt_patches_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
                gt_patches_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_patch_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')

                gt_images_per_concept = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_inputsize_{INPUT_IMAGE_SIZE}.pt')
                gt_images_per_concept_train = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_train_inputsize_{INPUT_IMAGE_SIZE}.pt')
                gt_images_per_concept_test = torch.load(f'GT_Samples/{DATASET_NAME}/gt_samples_per_concept_test_inputsize_{INPUT_IMAGE_SIZE}.pt')
                
                concepts = torch.load(f'Concepts/{DATASET_NAME}/{CONCEPTS_FILE}')
                
                dists = pd.read_csv(f"Distances/{DATASET_NAME}/{DISTS_FILE}")
                                                        
                # if DATASET_NAME == 'Coco':
                #     coco_concepts = get_coco_concepts()
                #     concepts = {k:v for k,v in concepts.items() if k in coco_concepts}
                #     gt_patches_per_concept = {k:v for k,v in  gt_patches_per_concept.items() if k in coco_concepts}
                #     gt_patches_per_concept_test = {k:v for k,v in  gt_patches_per_concept_test.items() if k in coco_concepts}
                #     dists = dists[coco_concepts]
                    
                all_object_patches =  set()
                for concept, samples in gt_patches_per_concept.items():
                    all_object_patches.update(samples)
            
 
                stats = compute_cossim_hist_stats(gt_patches_per_concept, dists, 
                                      DATASET_NAME, None, sample_type=ST2, 
                                      model_input_size=INPUT_IMAGE_SIZE, con_label=CON_LABEL,
                                                  all_object_patches=None)
                stats = compute_cossim_hist_stats(gt_patches_per_concept, dists, 
                                      DATASET_NAME, None, sample_type=ST2,
                                      model_input_size=INPUT_IMAGE_SIZE, con_label=CON_LABEL,
                                                  all_object_patches=all_object_patches)
        
                plot_activation_percentages_over_thresholds(dists, gt_patches_per_concept_train, 
                                                gt_patches_per_concept_test, 
                                                gt_images_per_concept_train,
                                                gt_images_per_concept_test,
                                                DATASET_NAME, 
                                                INPUT_IMAGE_SIZE, DEVICE, CON_LABEL,
                                                sample_type='patch')
                plot_activation_count_distributions(dists, gt_patches_per_concept_train, 
                                          gt_patches_per_concept_test, gt_images_per_concept_train,
                                          gt_images_per_concept_test, DATASET_NAME, 
                                          INPUT_IMAGE_SIZE, DEVICE, CON_LABEL)
                compute_multiple_activation_analyses(dists, gt_patches_per_concept_train, gt_patches_per_concept_test, gt_images_per_concept_train, gt_images_per_concept_test, INPUT_IMAGE_SIZE, DATASET_NAME, CON_LABEL, DEVICE)
                compute_metrics_across_percentiles(gt_patches_per_concept_test, concepts, dists, INPUT_IMAGE_SIZE, DATASET_NAME, 
                                       DEVICE, CON_LABEL, sample_type='patch', all_object_patches=all_object_patches)
                percentiles = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
                detect_then_invert_metrics_over_percentiles(percentiles, percentiles, 
                                                dists, concepts, gt_patches_per_concept, gt_patches_per_concept_test,
                                                DEVICE, DATASET_NAME, INPUT_IMAGE_SIZE, CON_LABEL, all_object_patches=all_object_patches,
                                                n_trials=2, balance_dataset=False, patch_size=14)
                all_superdetector_inversions_across_percentiles(percentiles, 'avg', embeds, dists,
                                   gt_patches_per_concept_test, DATASET_NAME, INPUT_IMAGE_SIZE, CON_LABEL, 
                                                DEVICE, patch_size=14, local=True)
                detect_then_invert_locally_metrics_over_percentiles(percentiles, percentiles, dists, 
                                                        concepts, gt_patches_per_concept, gt_patches_per_concept_test,
                                                        DEVICE, DATASET_NAME, INPUT_IMAGE_SIZE, CON_LABEL,
                                                        all_object_patches=all_object_patches, patch_size=14,
                                                        agglomerate_type='avg')
                
             