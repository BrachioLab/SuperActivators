import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath("utils"))

from quant_concept_evals_utils import evaluate_baseline_models_across_dataset, compute_concept_metrics

MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']
SAMPLE_TYPES = ['cls', 'patch']
DATASETS = ['iSarcasm']

N_TRIALS = 2

if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)
    for (model_name, model_input_size), dataset_name, sample_type in experiment_configs:
        # Skip invalid dataset-input size combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
            
        print(f"Running model {model_name} dataset {dataset_name} sample type {sample_type}")
            
        if sample_type == 'patch':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
        else:
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt') 
            
        baseline_results = evaluate_baseline_models_across_dataset(gt_samples_per_concept, dataset_name, 
                                                                   sample_type, model_input_size, 
                                                                   patch_size=14, n_trials=N_TRIALS)
        
        con_label = f"{model_name}_{sample_type}_baseline"
        for baseline_type, (fp, fn, tp, tn) in baseline_results.items():
            compute_concept_metrics(fp, fn, tp, tn, gt_samples_per_concept.keys(), 
                                    dataset_name, con_label, just_obj=False, baseline_type=baseline_type)