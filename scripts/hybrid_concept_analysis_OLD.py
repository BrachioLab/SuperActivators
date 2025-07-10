import torch
import numpy as np
import json
import sys
import os
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.hybrid_concept_utils import run_hybrid_analysis_for_config

# Configuration
MODELS = [('CLIP', (224, 224)), ('Llama', (560, 560)), ('Llama', ('text', 'text')), ('Mistral', ('text', 'text2')), ('Qwen', ('text', 'text3'))]
DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces', 'Sarcasm', 'iSarcasm']
METHODS = ['avg', 'linsep']  # Supervised methods only

MODELS = [('Llama', (560, 560))]
DATASETS = ['CLEVR']
METHODS = ['avg']  # Supervised methods only

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRATCH_DIR = ''

# Alpha values to sweep
ALPHA_VALUES = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Best detect and invert percentiles for each dataset/model combo (supervised methods only)
BEST_PERCENTILES = {
    "CLEVR": {
        "CLIP_avg": {"detect": 0.20, "invert": 0.6},
        "CLIP_linsep": {"detect": 0.20, "invert": 0.7},
        "Llama_avg": {"detect": 0.10, "invert": 0.5},
        "Llama_linsep": {"detect": 0.20, "invert": 0.7}
    },
    "Coco": {
        "CLIP_avg": {"detect": 0.20, "invert": 0.4},
        "CLIP_linsep": {"detect": 0.30, "invert": 0.5},
        "Llama_avg": {"detect": 0.05, "invert": 0.4},
        "Llama_linsep": {"detect": 0.10, "invert": 0.5}
    },
    "Broden-OpenSurfaces": {
        "CLIP_avg": {"detect": 0.10, "invert": 0.3},
        "CLIP_linsep": {"detect": 0.20, "invert": 0.4},
        "Llama_avg": {"detect": 0.05, "invert": 0.2},
        "Llama_linsep": {"detect": 0.05, "invert": 0.3}
    },
    "Broden-Pascal": {
        "CLIP_avg": {"detect": 0.20, "invert": 0.4},
        "CLIP_linsep": {"detect": 0.40, "invert": 0.6},
        "Llama_avg": {"detect": 0.10, "invert": 0.4},
        "Llama_linsep": {"detect": 0.10, "invert": 0.5}
    },
    "Sarcasm": {
        "Llama_avg": {"detect": 0.20, "invert": 0.5},
        "Llama_linsep": {"detect": 0.30, "invert": 0.6},
        "Qwen_avg": {"detect": 0.20, "invert": 0.4},  
        "Qwen_linsep": {"detect": 0.20, "invert": 0.5}
    },
    "iSarcasm": {
        "Llama_avg": {"detect": 0.20, "invert": 0.8},
        "Llama_linsep": {"detect": 0.40, "invert": 0.8},
        "Qwen_avg": {"detect": 0.20, "invert": 0.8}, 
        "Qwen_linsep": {"detect": 0.30, "invert": 0.8}
    }
}

if __name__ == "__main__":
    results = {}
    
    experiment_configs = product(MODELS, DATASETS, METHODS)
    
    for (model_name, model_input_size), dataset_name, method in experiment_configs:
        # Skip invalid dataset-model combinations
        if model_input_size[0] == 'text' and dataset_name not in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
        if model_input_size[0] != 'text' and dataset_name in ['Stanford-Tree-Bank', 'Sarcasm', 'iSarcasm']:
            continue
            
        print(f"\n🔄 Processing {model_name} on {dataset_name} with {method}")
        
        # Check if we have best percentiles for this combination
        dataset_key = dataset_name
        method_key = f"{model_name}_{method}"
        
        if dataset_key not in BEST_PERCENTILES:
            print(f"   ⚠️  No best percentiles found for {dataset_name}, skipping...")
            continue
            
        if method_key not in BEST_PERCENTILES[dataset_key]:
            print(f"   ⚠️  No best percentiles for {method_key}, skipping...")
            continue
            
        detect_percentile = BEST_PERCENTILES[dataset_key][method_key]["detect"]
        invert_percentile = BEST_PERCENTILES[dataset_key][method_key]["invert"]
        
        if invert_percentile is None:
            print(f"   ⚠️  No invert percentile for {method_key}, skipping...")
            continue
        
        # Run hybrid analysis
        alpha_results = run_hybrid_analysis_for_config(
            model_name=model_name,
            model_input_size=model_input_size,
            dataset_name=dataset_name,
            method=method,
            detect_percentile=detect_percentile,
            invert_percentile=invert_percentile,
            alpha_values=ALPHA_VALUES,
            device=DEVICE,
            scratch_dir=SCRATCH_DIR
        )
        
        if alpha_results is not None:
            # Store results
            result_key = f"{model_name}_{dataset_name}_{method}"
            individual_result = {
                'dataset': dataset_name,
                'model': model_name,
                'method': method,
                'detect_percentile': detect_percentile,
                'invert_percentile': invert_percentile,
                'alpha_sweep': alpha_results
            }
            results[result_key] = individual_result
            
            # Save individual result immediately
            os.makedirs(f'Hybrid_Results/{dataset_name}', exist_ok=True)
            individual_file = f'Hybrid_Results/{dataset_name}/hybrid_analysis_{model_name}_{method}.json'
            with open(individual_file, 'w') as f:
                json.dump(individual_result, f, indent=2)
            print(f"   💾 Saved individual result to: {individual_file}")
            
            # Print best result for this config
            best_alpha_result = max(alpha_results, key=lambda x: x['weighted_f1'])
            print(f"   ✅ Best F1 = {best_alpha_result['weighted_f1']:.4f} at alpha = {best_alpha_result['alpha']:.2f}")
        else:
            print(f"   ❌ Failed to process {model_name}_{dataset_name}_{method}")
    
    # Save results organized by dataset
    print(f"\n💾 Saving results...")
    
    # Create main Hybrid_Results directory
    os.makedirs('Hybrid_Results', exist_ok=True)
    
    # Group results by dataset
    results_by_dataset = {}
    for result_key, result_data in results.items():
        dataset = result_data['dataset']
        if dataset not in results_by_dataset:
            results_by_dataset[dataset] = {}
        results_by_dataset[dataset][result_key] = result_data
    
    # Save results for each dataset in separate subdirectories
    for dataset_name, dataset_results in results_by_dataset.items():
        # Create dataset subdirectory
        dataset_dir = f'Hybrid_Results/{dataset_name}'
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save dataset-specific results
        output_file = f'{dataset_dir}/hybrid_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(dataset_results, f, indent=2)
        
        print(f"   📁 {dataset_name}: {len(dataset_results)} results → {output_file}")
    
    print(f"\n✅ Analysis complete! Results saved to Hybrid_Results/ subdirectories")
    
    # Print summary
    print(f"\n📈 Summary:")
    for result_key, result_data in results.items():
        best_alpha_result = max(result_data['alpha_sweep'], key=lambda x: x['weighted_f1'])
        print(f"   {result_key}: Best F1 = {best_alpha_result['weighted_f1']:.4f} at alpha = {best_alpha_result['alpha']:.2f}")