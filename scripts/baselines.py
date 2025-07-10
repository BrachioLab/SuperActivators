import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.quant_concept_evals_utils import evaluate_baseline_models_across_dataset, compute_concept_metrics, inversion_baselines

MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Qwen', ('text', 'text3'))]

DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm']
DATASETS = ['GoEmotions']
SAMPLE_TYPES = ['cls', 'patch']

N_TRIALS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    experiment_configs = product(MODELS, DATASETS, SAMPLE_TYPES)
    for (model_name, model_input_size), dataset_name, sample_type in experiment_configs:
        # Skip invalid dataset-input size combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm', 'GoEmotions']:
            continue
            
        print(f"Running model {model_name} dataset {dataset_name} sample type {sample_type}")
            
        if sample_type == 'patch':
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt')
        else:
            gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_inputsize_{model_input_size}.pt') 

        # === DETECTION BASELINES ===
        baseline_results = evaluate_baseline_models_across_dataset(gt_samples_per_concept, dataset_name, 
                                                                   sample_type, model_input_size, 
                                                                   patch_size=14, n_trials=N_TRIALS)
        
        con_label = f"{model_name}_{sample_type}_baseline"
        for baseline_type, (fp, fn, tp, tn) in baseline_results.items():
            compute_concept_metrics(fp, fn, tp, tn, gt_samples_per_concept.keys(), 
                                    dataset_name, con_label, just_obj=False, baseline_type=baseline_type)
        
        # === INVERSION BASELINES ===
        # Only compute for patch-level data (inversion is patch-specific)
        if sample_type == 'patch':
            print(f"Computing inversion baselines...")
            
            # Call inversion baselines function
            inversion_baselines(
                dataset_name=dataset_name,
                model_input_size=model_input_size,
                con_label=f"{model_name}_{sample_type}",
                device=DEVICE,
                patch_size=14
            )
            
            print(f"✓ Completed inversion baselines for {model_name} {dataset_name}")