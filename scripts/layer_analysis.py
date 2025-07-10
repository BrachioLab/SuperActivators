#!/usr/bin/env python3
"""
Unified layer analysis script for analyzing concept emergence across model layers.
Supports both CLIP and Llama models with flexible configuration.
"""

import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import gc
from PIL import Image
from transformers import CLIPModel, AutoProcessor, MllamaForConditionalGeneration
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.general_utils import load_images, retrieve_topn_samples
from utils.layer_analysis_utils import (
    extract_clip_embeddings_all_layers,
    extract_llama_embeddings_all_layers,
    load_final_layer_stats,
    load_concept_vectors,
    compute_layer_concept_similarities,
    visualize_patch_concept_heatmaps,
    save_layer_embeddings
)

# Configuration presets
CONFIG_PRESETS = {
    'CLEVR_CLIP': {
        'dataset_name': 'CLEVR',
        'model_name': 'CLIP',
        'model_input_size': (224, 224),
        'concept_types': ['avg', 'linsep_bd_true_bn_false'],
        'patch_size': 14,
        'model_path': "openai/clip-vit-large-patch14",
        'selected_concepts': None,  # Use all concepts
        'num_test_images': 3
    },
    'CLEVR_LLAMA': {
        'dataset_name': 'CLEVR',
        'model_name': 'Llama',
        'model_input_size': (560, 560),
        'concept_types': ['avg', 'linsep_bd_true_bn_false'],
        'patch_size': 14,
        'model_path': "meta-llama/Llama-3.2-11B-Vision-Instruct",
        'selected_concepts': None,  # Use all concepts
        'num_test_images': 3
    },
    'COCO_CLIP': {
        'dataset_name': 'Coco',
        'model_name': 'CLIP',
        'model_input_size': (224, 224),
        'concept_types': ['avg', 'linsep_bd_true_bn_false'],
        'patch_size': 14,
        'model_path': "openai/clip-vit-large-patch14",
        'selected_concepts': ['animal', 'bus', 'electronic', 'food', 'person', 'train'],
        'num_test_images': 3
    },
    'COCO_LLAMA': {
        'dataset_name': 'Coco',
        'model_name': 'Llama',
        'model_input_size': (560, 560),
        'concept_types': ['avg', 'linsep_bd_true_bn_false'],
        'patch_size': 14,
        'model_path': "meta-llama/Llama-3.2-11B-Vision-Instruct",
        'selected_concepts': ['animal', 'bus', 'electronic', 'food', 'person', 'train'],
        'num_test_images': 3
    }
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_processor(model_name, model_path, device):
    """Load model and processor for the specified model type."""
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'clip':
        print("Loading CLIP model...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = CLIPModel.from_pretrained(model_path).to(device)
        model.eval()
        return model, processor
        
    elif model_name_lower == 'llama':
        print("Loading Llama model with memory optimization...")
        # Clear GPU memory first
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Load processor first
            print("Loading processor...")
            processor = AutoProcessor.from_pretrained(model_path)
            
            # Load model with optimizations
            print("Loading model...")
            model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Use half precision
                device_map="auto",  # Automatic device mapping
                low_cpu_mem_usage=True,  # Reduce CPU memory usage
            )
            
            model.eval()
            print(f"Model loaded successfully on device: {next(model.parameters()).device}")
            return model, processor
            
        except Exception as e:
            print(f"Error loading model: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            raise e
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def extract_embeddings_all_layers(model, processor, images, device, model_name):
    """Extract embeddings from all layers based on model type."""
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'clip':
        return extract_clip_embeddings_all_layers(model, processor, images, device)
    elif model_name_lower == 'llama':
        return extract_llama_embeddings_all_layers(model, processor, images, device)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def run_layer_analysis(config):
    """Run layer analysis for a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Running layer analysis")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Model: {config['model_name']}")
    print(f"Model input size: {config['model_input_size']}")
    print(f"Concept types: {config['concept_types']}")
    print(f"Device: {DEVICE}")
    print(f"Number of test images: {config['num_test_images']}")
    
    # Load model and processor
    print(f"\n1. Loading {config['model_name']} model and processor...")
    try:
        model, processor = load_model_and_processor(
            config['model_name'], 
            config['model_path'], 
            DEVICE
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        if config['model_name'].lower() == 'llama':
            print("This likely requires GPU with sufficient memory (>20GB)")
        return False
    
    # Load images
    print(f"\n2. Loading images...")
    all_images, train_images, test_images = load_images(
        config['dataset_name'], 
        config['model_input_size']
    )
    
    # Get first N test images
    test_sample_indices = retrieve_topn_samples(
        config['dataset_name'], 
        config['num_test_images'], 
        start_idx=0, 
        split='test'
    )
    selected_test_images = [all_images[i] for i in test_sample_indices]
    
    print(f"Selected {len(selected_test_images)} test images with indices: {test_sample_indices}")
    
    # Extract embeddings from all layers
    print(f"\n3. Extracting embeddings from all {config['model_name']} layers...")
    try:
        embeddings_by_layer = extract_embeddings_all_layers(
            model, processor, selected_test_images, DEVICE, config['model_name']
        )
        
        layer_percentages = sorted(embeddings_by_layer.keys())
        print(f"Extracted embeddings from {len(layer_percentages)} layers: {layer_percentages}")
        
        # For Llama, we no longer use the projector
        projector = None
        if config['model_name'].lower() == 'llama':
            print("Note: Llama embeddings are now extracted WITHOUT using the projector")
            print("Embeddings will be 1280-dimensional instead of 4096-dimensional")
            
            # Clear model from memory after extraction
            del model
            del processor
            torch.cuda.empty_cache()
            gc.collect()
            print("Model cleared from memory")
            
    except Exception as e:
        print(f"Failed to extract layer embeddings: {e}")
        return False
    
    # Load final layer normalization statistics
    print(f"\n4. Loading final layer normalization statistics...")
    try:
        final_mean, final_norm = load_final_layer_stats(
            config['dataset_name'], 
            config['model_name'], 
            device=DEVICE
        )
        print(f"Loaded final layer stats - Mean shape: {final_mean.shape}, Norm: {final_norm}")
    except Exception as e:
        print(f"Error loading final layer stats: {e}")
        return False
    
    # Process each concept type
    for concept_type in config['concept_types']:
        print(f"\n5. Processing concept type: {concept_type}")
        
        # Load concept vectors
        print(f"   Loading concept vectors...")
        try:
            concept_vectors = load_concept_vectors(
                config['dataset_name'], 
                concept_type, 
                device=DEVICE, 
                model_name=config['model_name']
            )
            concept_names = list(concept_vectors.keys())
            print(f"   Loaded {len(concept_names)} concepts: {concept_names}")
        except Exception as e:
            print(f"   Error loading concept vectors: {e}")
            continue
        
        # Filter to selected concepts if specified
        if config['selected_concepts'] is not None:
            filtered_concepts = {name: vec for name, vec in concept_vectors.items() 
                               if name in config['selected_concepts']}
            concept_vectors = filtered_concepts
            concept_names = list(concept_vectors.keys())
            print(f"   Filtered to {len(concept_names)} concepts: {concept_names}")
        
        # Compute similarities across layers
        print(f"   Computing concept similarities across layers...")
        kwargs = {'projector': projector} if projector is not None else {}
        similarities_by_layer = compute_layer_concept_similarities(
            embeddings_by_layer, concept_vectors, final_mean, final_norm, **kwargs
        )
        
        # Create visualizations for each test image
        print(f"   Creating visualizations...")
        for img_idx in range(config['num_test_images']):
            print(f"     Visualizing patch-level concept heatmaps for image {img_idx}...")
            
            # Create concept label for file naming
            con_label = f"{config['model_name']}_{concept_type}"
            
            # Global scale version
            print(f"       Creating global scale visualization...")
            visualize_patch_concept_heatmaps(
                similarities_by_layer, 
                concept_names, 
                img_idx,
                original_image=selected_test_images[img_idx],
                dataset_name=config['dataset_name'],
                model_input_size=config['model_input_size'],
                con_label=con_label,
                per_layer_scale=False,
                selected_concepts=config['selected_concepts'],
                patch_size=config['patch_size'],
                show_plot=False  # Faster when not displaying
            )
            
            # Per-layer scale version
            print(f"       Creating per-layer scale visualization...")
            visualize_patch_concept_heatmaps(
                similarities_by_layer, 
                concept_names, 
                img_idx,
                original_image=selected_test_images[img_idx],
                dataset_name=config['dataset_name'],
                model_input_size=config['model_input_size'],
                con_label=con_label,
                per_layer_scale=True,
                selected_concepts=config['selected_concepts'],
                patch_size=config['patch_size'],
                show_plot=False  # Faster when not displaying
            )
    
    # Skip saving layer embeddings to save time and disk space
    # print(f"\n6. Saving layer embeddings...")
    # save_layer_embeddings(
    #     embeddings_by_layer, 
    #     config['dataset_name'], 
    #     config['model_name'], 
    #     'test_images'
    # )
    
    # Show layer-wise concept emergence statistics
    print("\n=== CONCEPT EMERGENCE STATISTICS ===")
    for concept_name in concept_names[:5]:  # Show first 5 concepts
        print(f"\n{concept_name}:")
        for layer_pct in layer_percentages[::3]:  # Show every 3rd layer
            similarities = similarities_by_layer[layer_pct][concept_name]
            max_sim = torch.max(similarities).item()
            mean_sim = torch.mean(similarities).item()
            print(f"  Layer {layer_pct:3d}%: Max={max_sim:.3f}, Mean={mean_sim:.3f}")
    
    print(f"\n✓ Layer analysis complete!")
    print(f"Visualizations saved to: ../Figs/{config['dataset_name']}/layer_analysis/")
    print(f"Note: Test embeddings were not saved to reduce disk usage.")
    return True


def main():
    """Main function to run layer analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run layer analysis for concept emergence')
    parser.add_argument('--preset', type=str, choices=list(CONFIG_PRESETS.keys()),
                        help='Use a configuration preset')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, choices=['CLIP', 'Llama'], help='Model name')
    parser.add_argument('--num-images', type=int, default=3, help='Number of test images')
    parser.add_argument('--concepts', type=str, nargs='+', help='Specific concepts to analyze')
    parser.add_argument('--concept-types', type=str, nargs='+', 
                        default=['avg', 'linsep_bd_true_bn_false'],
                        help='Concept types to analyze')
    
    args = parser.parse_args()
    
    # Build configuration
    if args.preset:
        config = CONFIG_PRESETS[args.preset].copy()
        print(f"Using preset configuration: {args.preset}")
    else:
        # Build custom configuration
        if not args.dataset or not args.model:
            parser.error("Either --preset or both --dataset and --model must be specified")
        
        # Determine model path and input size based on model
        if args.model.upper() == 'CLIP':
            model_path = "openai/clip-vit-large-patch14"
            model_input_size = (224, 224)
        else:  # Llama
            model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            model_input_size = (560, 560)
        
        config = {
            'dataset_name': args.dataset,
            'model_name': args.model,
            'model_input_size': model_input_size,
            'concept_types': args.concept_types,
            'patch_size': 14,
            'model_path': model_path,
            'selected_concepts': args.concepts,
            'num_test_images': args.num_images
        }
    
    # Override with command line arguments if provided
    if args.num_images:
        config['num_test_images'] = args.num_images
    if args.concepts:
        config['selected_concepts'] = args.concepts
    if args.concept_types:
        config['concept_types'] = args.concept_types
    
    print("Starting layer analysis...")
    print(f"Configuration: {config}")
    
    try:
        success = run_layer_analysis(config)
        if success:
            print(f"\n✓ SUCCESS: Layer analysis completed successfully!")
        else:
            print(f"\n✗ FAILED: Layer analysis encountered errors")
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()