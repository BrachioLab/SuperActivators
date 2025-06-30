import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import os
import gc
from collections import defaultdict
from itertools import product
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.correlation_w_labels_utils import compute_label_correlations, save_correlation_results, summarize_correlations


MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text'))]
DATASETS = ['CLEVR']


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENT_THRU_MODEL = 100
SCRATCH_DIR = ''
PERCENTILES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCH_SIZE = 500

def get_act_paths(dataset_name, model_name):
    avg_acts_patch_concepts = f'Cosine_Similarities/{dataset_name}/cosine_similarities_avg_concepts_{model_name}_patch_embeddings_percentthrumodel_100.csv'
    linsep_acts_patch_concepts = f'Distances/{dataset_name}/dists_linsep_concepts_BD_True_BN_False_{model_name}_patch_embeddings_percentthrumodel_100.csv'
    avg_acts_cls_concepts = f'Cosine_Similarities/{dataset_name}/patch_cls_cosine_similarities_avg_concepts_{model_name}_cls_embeddings_percentthrumodel_100.csv'
    linsep_acts_cls_concepts = f'Distances/{dataset_name}/patch_cls_dists_linsep_concepts_BD_True_BN_False_{model_name}_cls_embeddings_percentthrumodel_100.csv'
    return (('patchconcepts_avg', avg_acts_patch_concepts), 
           ('patchconcepts_linsep',linsep_acts_patch_concepts), 
           ('clsconcepts_avg', avg_acts_cls_concepts), 
           ('clsconcepts_linsep', linsep_acts_cls_concepts))
    

        
if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS)
    for (model_name, model_input_size), dataset_name in experiment_configs:
        # Skip invalid dataset-input size combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
            
        
        print(f"Processing model {model_name} dataset {dataset_name}")

        #get gt and labels
        gt_patches_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')

        #get activations
        all_acts_paths = get_act_paths(dataset_name,model_name)
        for acts_scheme, acts_path in all_acts_paths:
            # Compute correlations
            print(f"\nComputing correlations for {acts_scheme}")
            correlation_results = compute_label_correlations(gt_patches_per_concept,
                dataset_name=dataset_name,
                model_input_size=model_input_size,
                acts_path=acts_path,
                device=DEVICE,
                patch_size=14,
                scratch_dir=SCRATCH_DIR
            )
            
            # Save results
            output_path = save_correlation_results(
                correlation_results=correlation_results,
                dataset_name=dataset_name,
                scheme=acts_scheme,
                model_name=model_name,
                scratch_dir=SCRATCH_DIR
            )
            
            # Print summary
            summarize_correlations(correlation_results)
            
            
        
        
            
            
            
            
            

