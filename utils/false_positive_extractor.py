import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict
from tqdm import tqdm

# Import the required utility functions
from .general_utils import get_split_df, create_binary_labels
from .patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence


def get_false_positive_indices(dataset_name, model_name, sample_type, concept, 
                              percentile, split='test', method='avg', 
                              model_input_size=None, patch_size=14):
    """
    Extracts the actual indices of false positive samples by running the detection logic.
    
    Args:
        dataset_name (str): Dataset name (e.g., 'Coco', 'CLEVR')
        model_name (str): Model name (e.g., 'CLIP', 'Llama')
        sample_type (str): 'patch' or 'cls'
        concept (str): The concept to analyze
        percentile (float): Detection percentile (e.g., 0.1, 0.2)
        split (str): 'train' or 'test'
        method (str): 'avg' or 'linsep'
        model_input_size (tuple): Model input size, will be inferred if None
        patch_size (int): Patch size for patch-level analysis
        
    Returns:
        dict: Contains false positive information including:
            - 'false_positive_indices': List of indices that are false positives
            - 'activations': Activation values for all indices
            - 'threshold': Threshold used
            - 'ground_truth': Ground truth labels
            - 'predictions': Model predictions
    """
    
    # Infer model input size if not provided
    if model_input_size is None:
        # Check if it's a text dataset
        if dataset_name in ['iSarcasm', 'Sarcasm', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak', 'GoEmotions']:
            model_input_size = ('text', 'text')
        elif model_name == 'CLIP':
            model_input_size = (224, 224)
        elif model_name == 'Llama':
            model_input_size = (560, 560)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    # Construct the concept label
    if method == 'avg':
        con_label = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
    elif method == 'linsep':
        con_label = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Analyzing false positives for concept '{concept}'")
    print(f"Method: {method}, Sample type: {sample_type}, Percentile: {percentile}")
    print(f"Con label: {con_label}")
    
    # Load activation values - different folders for avg vs linsep methods
    if method == 'avg':
        act_metrics_path = f'/shared_data0/cgoldberg/Concept_Inversion/Experiments/Cosine_Similarities/{dataset_name}/{con_label}.csv'
    else:  # linsep
        act_metrics_path = f'/shared_data0/cgoldberg/Concept_Inversion/Experiments/Distances/{dataset_name}/dists_{con_label[con_label.find("_")+1:]}.csv'
    
    try:
        # For CLS methods, don't use index_col=0 since there's no index column
        if sample_type == 'cls':
            act_metrics = pd.read_csv(act_metrics_path, index_col=None)
        else:
            act_metrics = pd.read_csv(act_metrics_path, index_col=0)
        print(f"Loaded activation metrics: {act_metrics.shape}")
            
    except FileNotFoundError:
        print(f"Activation metrics not found at {act_metrics_path}")
        # Try alternative path for linsep
        if method == 'linsep':
            alt_path = f'/shared_data0/cgoldberg/Concept_Inversion/Experiments/Distances/{dataset_name}/dists_linsep_concepts_BD_True_BN_False_{model_name}_{sample_type}_embeddings_percentthrumodel_100.csv'
            try:
                # For CLS methods, don't use index_col=0 since there's no index column
                if sample_type == 'cls':
                    act_metrics = pd.read_csv(alt_path, index_col=None)
                else:
                    act_metrics = pd.read_csv(alt_path, index_col=0)
                print(f"Loaded activation metrics from alternative path: {act_metrics.shape}")
                act_metrics_path = alt_path
            except FileNotFoundError:
                print(f"Alternative path also not found: {alt_path}")
                return None
        else:
            return None
    
    if concept not in act_metrics.columns:
        print(f"Concept '{concept}' not found in activation metrics")
        print(f"Available concepts: {list(act_metrics.columns)}")
        return None
    
    # Load thresholds from all_percentiles file
    threshold_path = f'/shared_data0/cgoldberg/Concept_Inversion/Experiments/Thresholds/{dataset_name}/all_percentiles_{con_label}.pt'
    try:
        all_thresholds = torch.load(threshold_path, weights_only=False)
        print(f"Loaded all thresholds from: {threshold_path}")
        
        if percentile not in all_thresholds:
            print(f"Percentile {percentile} not found in thresholds")
            print(f"Available percentiles: {list(all_thresholds.keys())}")
            return None
            
        percentile_thresholds = all_thresholds[percentile]
        if concept not in percentile_thresholds:
            print(f"Concept '{concept}' not found in thresholds for percentile {percentile}")
            print(f"Available concepts: {list(percentile_thresholds.keys())}")
            return None
            
        threshold = percentile_thresholds[concept][0]
        print(f"Loaded threshold for percentile {percentile}: {threshold}")
    except FileNotFoundError:
        print(f"Threshold file not found at {threshold_path}")
        return None
    
    # Load ground truth samples - always use image/sentence level GT for detection metrics
    gt_path = f'/shared_data0/cgoldberg/Concept_Inversion/Experiments/GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_{model_input_size}.pt'
    
    try:
        gt_samples = torch.load(gt_path, weights_only=False)
        print(f"Loaded ground truth samples for {len(gt_samples)} concepts")
    except FileNotFoundError:
        print(f"Ground truth file not found at {gt_path}")
        return None
    
    # Now we need to do IMAGE-LEVEL detection, not patch-level
    # Use the same logic as find_activated_images_bypatch and compute_detection_metrics_for_per
    
    # For text datasets, use sentence-level detection
    if model_input_size == ('text', 'text'):
        if sample_type == 'cls':
            # For CLS, use the proper function that handles sentence-level activations
            from .quant_concept_evals_utils import find_activated_images_byimage
            # Need to create thresholds for all concepts, but we only care about our target concept
            all_concept_thresholds = {}
            for col in act_metrics.columns:
                if col == concept:
                    all_concept_thresholds[col] = (threshold, float('nan'))
                else:
                    all_concept_thresholds[col] = (999999.0, float('nan'))  # Very high threshold so they don't activate
            
            _, activated_images_test = find_activated_images_byimage(
                act_metrics, all_concept_thresholds, model_input_size, dataset_name
            )
        else:
            # For patch/token level, use the existing token aggregation function
            from .quant_concept_evals_utils import find_activated_sentences_bytoken
            # Need to create thresholds for all concepts, but we only care about our target concept
            all_concept_thresholds = {}
            for col in act_metrics.columns:
                if col == concept:
                    all_concept_thresholds[col] = (threshold, float('nan'))
                else:
                    all_concept_thresholds[col] = (999999.0, float('nan'))  # Very high threshold so they don't activate
            
            _, activated_images_test = find_activated_sentences_bytoken(
                act_metrics, all_concept_thresholds, model_input_size, dataset_name
            )
    else:
        # For image datasets, use different functions based on sample_type
        if sample_type == 'cls':
            # For CLS, use image-level detection directly
            from .quant_concept_evals_utils import find_activated_images_byimage
            # Need to create thresholds for all concepts, but we only care about our target concept
            all_concept_thresholds = {}
            for col in act_metrics.columns:
                if col == concept:
                    all_concept_thresholds[col] = (threshold, float('nan'))
                else:
                    all_concept_thresholds[col] = (999999.0, float('nan'))  # Very high threshold so they don't activate
            
            _, activated_images_test = find_activated_images_byimage(
                act_metrics, all_concept_thresholds, model_input_size, dataset_name
            )
        else:
            # For patch level, use patch max-pooling to get image-level detection
            from .quant_concept_evals_utils import find_activated_images_bypatch
            # Need to create thresholds for all concepts, but we only care about our target concept
            all_concept_thresholds = {}
            for col in act_metrics.columns:
                if col == concept:
                    all_concept_thresholds[col] = (threshold, float('nan'))
                else:
                    all_concept_thresholds[col] = (999999.0, float('nan'))  # Very high threshold so they don't activate
            
            _, activated_images_test = find_activated_images_bypatch(
                act_metrics, all_concept_thresholds, model_input_size, dataset_name, patch_size=patch_size
            )
    
    # Get all test images
    split_df = get_split_df(dataset_name)
    all_test_images = set(split_df[split_df == split].index)
    print(f"Total {split} images: {len(all_test_images)}")
    
    # Get ground truth images for this concept  
    gt_positive_images = set(gt_samples.get(concept, []))
    print(f"Ground truth positive images: {len(gt_positive_images)}")
    
    # Get detected images for this concept
    detected_images = activated_images_test.get(concept, set())
    print(f"Detected images: {len(detected_images)}")
    
    # Compute confusion matrix at IMAGE level
    tp_images = gt_positive_images & detected_images
    fp_images = detected_images - gt_positive_images  # Detected but not in GT
    fn_images = gt_positive_images - detected_images  # In GT but not detected
    tn_images = all_test_images - (tp_images | fp_images | fn_images)
    
    tp_count = len(tp_images)
    fp_count = len(fp_images)
    fn_count = len(fn_images)
    tn_count = len(tn_images)
    
    # Convert false positive images to a list for easier handling
    false_positive_indices = list(fp_images)
    
    print(f"\\nConfusion Matrix:")
    print(f"True Positives: {tp_count}")
    print(f"False Positives: {fp_count}")
    print(f"False Negatives: {fn_count}")
    print(f"True Negatives: {tn_count}")
    print(f"FPR: {fp_count/(fp_count+tn_count):.3f}")
    print(f"FNR: {fn_count/(fn_count+tp_count):.3f}")
    
    # For false positive images, find the maximum activation value that caused the detection
    fp_details = []
    
    if sample_type == 'patch':
        # For patch methods, find max activation within each false positive image
        patches_per_image = (model_input_size[0] // patch_size) ** 2 if model_input_size != ('text', 'text') else None
        
        for fp_image_idx in false_positive_indices:
            if model_input_size == ('text', 'text'):
                # For text, we'd need to map sentence index to token indices
                # For now, just use a placeholder activation
                max_activation = threshold + 0.1  # Placeholder
            else:
                # For images, find the max activation among patches in this image
                start_patch = fp_image_idx * patches_per_image
                end_patch = start_patch + patches_per_image
                
                # Get activations for all patches in this image
                try:
                    image_patch_activations = act_metrics[concept].iloc[start_patch:end_patch]
                    max_activation = image_patch_activations.max()
                except:
                    max_activation = threshold + 0.1  # Fallback
            
            fp_details.append({
                'index': fp_image_idx,
                'activation': max_activation,
                'threshold': threshold
            })
    else:
        # For cls methods, the activation is already at image level
        for fp_image_idx in false_positive_indices:
            try:
                activation = act_metrics[concept].iloc[fp_image_idx]
            except:
                activation = threshold + 0.1  # Fallback
            
            fp_details.append({
                'index': fp_image_idx,
                'activation': activation,
                'threshold': threshold
            })
    
    # Sort by activation value (highest first)
    fp_details.sort(key=lambda x: x['activation'], reverse=True)
    
    results = {
        'concept': concept,
        'method': method,
        'sample_type': sample_type,
        'percentile': percentile,
        'threshold': threshold,
        'false_positive_images': false_positive_indices,
        'false_positive_details': fp_details,
        'total_fp': fp_count,
        'total_tp': tp_count,
        'total_fn': fn_count,
        'total_tn': tn_count,
        'detected_images': detected_images,
        'gt_positive_images': gt_positive_images,
        'all_test_images': all_test_images
    }
    
    return results


def visualize_false_positive_patches(fp_results, dataset_name, n_examples=6):
    """
    Visualizes false positive patches by showing the original images with highlighted patches.
    """
    if fp_results['sample_type'] != 'patch':
        print("Visualization only available for patch-level analysis")
        return
    
    false_positives = fp_results['false_positive_details'][:n_examples]
    
    # Load dataset metadata
    if dataset_name == 'Coco':
        image_dir = '/shared_data0/cgoldberg/Concept_Inversion/Data/Coco/val2017'
        metadata_path = '/shared_data0/cgoldberg/Concept_Inversion/Data/Coco/metadata.csv'
    elif dataset_name == 'CLEVR':
        image_dir = '/shared_data0/cgoldberg/Concept_Inversion/Data/CLEVR/images/val'
        metadata_path = '/shared_data0/cgoldberg/Concept_Inversion/Data/CLEVR/metadata.csv'
    else:
        print(f"Visualization not implemented for dataset: {dataset_name}")
        return
    
    try:
        metadata = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Metadata not found at {metadata_path}")
        return
    
    # Get model input size from results
    model_name = fp_results['method'].split('_')[0] if '_' in str(fp_results) else 'CLIP'
    if 'CLIP' in str(fp_results) or model_name == 'CLIP':
        input_size = (224, 224)
    else:
        input_size = (560, 560)
    
    patch_size = 14
    patches_per_row = input_size[0] // patch_size
    patches_per_image = patches_per_row ** 2
    
    n_cols = min(3, len(false_positives))
    n_rows = (len(false_positives) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if len(false_positives) == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for idx, fp in enumerate(false_positives):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        patch_idx = fp['index']
        
        # Get image index and patch position
        image_idx = patch_idx // patches_per_image
        patch_in_image = patch_idx % patches_per_image
        patch_row = patch_in_image // patches_per_row
        patch_col = patch_in_image % patches_per_row
        
        try:
            # Load and display image
            if image_idx < len(metadata):
                image_name = metadata.iloc[image_idx]['file_name']
                image_path = os.path.join(image_dir, image_name)
                
                if os.path.exists(image_path):
                    img = Image.open(image_path).convert('RGB')
                    img = img.resize(input_size)
                    ax.imshow(img)
                    
                    # Highlight the false positive patch
                    import matplotlib.patches as patches
                    rect = patches.Rectangle(
                        (patch_col * patch_size, patch_row * patch_size),
                        patch_size, patch_size,
                        linewidth=3, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add title with activation info
                    ax.set_title(f"FP #{idx+1}\\nAct: {fp['activation']:.3f} > Thresh: {fp['threshold']:.3f}",
                                fontsize=10)
                else:
                    ax.text(0.5, 0.5, f"Image not found:\\n{image_name}", 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f"Image index {image_idx}\\nout of range", 
                       ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            print(f"Error loading image for patch {patch_idx}: {e}")
            ax.text(0.5, 0.5, f"Error loading\\nimage {image_idx}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
    
    # Remove empty subplots
    for idx in range(len(false_positives), len(axes)):
        fig.delaxes(axes[idx])
    
    concept = fp_results['concept']
    method = fp_results['method']
    sample_type = fp_results['sample_type']
    plt.suptitle(f"False Positives for '{concept}' ({method} {sample_type})", fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_patch_vs_cls_false_positives(dataset_name, model_name, concept, percentile, method='linsep', split='test'):
    """
    Find paragraphs where patch method has false positives but CLS method does not.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Model name
        concept: Concept to analyze
        percentile: Detection percentile
        method: Method to use ('avg' or 'linsep')
        split: Data split to analyze
    """
    print(f"\\n{'='*80}")
    print(f"COMPARING PATCH vs CLS FALSE POSITIVES FOR '{concept.upper()}'")
    print(f"Dataset: {dataset_name}, Model: {model_name}, Method: {method}, Percentile: {percentile}")
    print(f"{'='*80}")
    
    # Get false positives for patch method
    print("\\nAnalyzing PATCH method...")
    patch_results = get_false_positive_indices(
        dataset_name, model_name, 'patch', concept,
        percentile, split=split, method=method
    )
    
    if not patch_results:
        print("Failed to get patch results")
        return
    
    # Get false positives for CLS method
    print("\\nAnalyzing CLS method...")
    cls_results = get_false_positive_indices(
        dataset_name, model_name, 'cls', concept,
        percentile, split=split, method=method
    )
    
    if not cls_results:
        print("Failed to get CLS results")
        return
    
    # Find paragraphs that are false positives in patch but not in CLS
    patch_fp_set = set(patch_results['false_positive_images'])
    cls_fp_set = set(cls_results['false_positive_images'])
    
    patch_only_fp = patch_fp_set - cls_fp_set
    
    print(f"\\nRESULTS SUMMARY:")
    print(f"Patch false positives: {len(patch_fp_set)}")
    print(f"CLS false positives: {len(cls_fp_set)}")
    print(f"Patch-only false positives (what we want): {len(patch_only_fp)}")
    print(f"Overlap (both methods): {len(patch_fp_set & cls_fp_set)}")
    
    if not patch_only_fp:
        print("\\nNo paragraphs found where patch has false positives but CLS does not.")
        return
    
    # Load metadata for displaying paragraphs
    metadata_path = f'/shared_data0/cgoldberg/Concept_Inversion/Data/{dataset_name}/metadata.csv'
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"\\nLoaded metadata with {len(metadata)} rows")
    except FileNotFoundError:
        print(f"Metadata not found at {metadata_path}")
        return
    
    # Get activation details for patch-only false positives
    patch_only_details = []
    for fp_detail in patch_results['false_positive_details']:
        if fp_detail['index'] in patch_only_fp:
            patch_only_details.append(fp_detail)
    
    # Sort by activation (highest first)
    patch_only_details.sort(key=lambda x: x['activation'], reverse=True)
    
    print(f"\\n{'='*80}")
    print(f"PARAGRAPHS WHERE PATCH HAS FALSE POSITIVE BUT CLS DOES NOT")
    print(f"Total examples: {len(patch_only_details)}")
    print(f"{'='*80}")
    
    for i, fp in enumerate(patch_only_details):
        paragraph_idx = fp['index']
        activation = fp['activation']
        threshold = fp['threshold']
        
        print(f"\\n{'-'*60}")
        print(f"PATCH-ONLY FALSE POSITIVE #{i+1}")
        print(f"Paragraph Index: {paragraph_idx}")
        print(f"Patch Activation: {activation:.4f} (threshold: {threshold:.4f})")
        print(f"{'-'*60}")
        
        # Get metadata for this paragraph
        if paragraph_idx < len(metadata):
            row = metadata.iloc[paragraph_idx]
            
            # Print metadata fields
            print("METADATA:")
            for col in metadata.columns:
                if col != 'text_path':  # We'll handle text separately
                    print(f"  {col}: {row[col]}")
            
            # Load and print the actual text
            if 'text_path' in row and pd.notna(row['text_path']):
                text_path = f'/shared_data0/cgoldberg/Concept_Inversion/Data/{dataset_name}/{row["text_path"]}'
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                    
                    print(f"\\nTEXT CONTENT:")
                    print(f'"{text_content}"')
                    
                except FileNotFoundError:
                    print(f"\\nTEXT FILE NOT FOUND: {text_path}")
                except Exception as e:
                    print(f"\\nERROR READING TEXT: {e}")
            else:
                print(f"\\nNO TEXT PATH FOUND IN METADATA")
        else:
            print(f"ERROR: Paragraph index {paragraph_idx} out of range (metadata has {len(metadata)} rows)")
    
    return {
        'patch_results': patch_results,
        'cls_results': cls_results,
        'patch_only_fp': patch_only_fp,
        'patch_only_details': patch_only_details
    }


def print_false_positive_paragraphs(fp_results, dataset_name, max_examples=None):
    """
    Print the actual paragraph content and metadata for false positive examples.
    
    Args:
        fp_results: Results from get_false_positive_indices()
        dataset_name: Name of the dataset
        max_examples: Maximum number of examples to show (None for all)
    """
    if not fp_results or not fp_results['false_positive_details']:
        print("No false positive examples found.")
        return
    
    # Load metadata
    metadata_path = f'/shared_data0/cgoldberg/Concept_Inversion/Data/{dataset_name}/metadata.csv'
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"Loaded metadata with {len(metadata)} rows")
    except FileNotFoundError:
        print(f"Metadata not found at {metadata_path}")
        return
    
    false_positives = fp_results['false_positive_details']
    if max_examples:
        false_positives = false_positives[:max_examples]
    
    print(f"\\n{'='*80}")
    print(f"FALSE POSITIVE PARAGRAPHS FOR '{fp_results['concept'].upper()}'")
    print(f"Method: {fp_results['method']} {fp_results['sample_type']}")
    print(f"Total false positives: {fp_results['total_fp']}")
    print(f"Showing {len(false_positives)} examples")
    print(f"{'='*80}")
    
    for i, fp in enumerate(false_positives):
        paragraph_idx = fp['index']
        activation = fp['activation']
        threshold = fp['threshold']
        
        print(f"\\n{'-'*60}")
        print(f"FALSE POSITIVE #{i+1}")
        print(f"Paragraph Index: {paragraph_idx}")
        print(f"Activation: {activation:.4f} (threshold: {threshold:.4f})")
        print(f"{'-'*60}")
        
        # Get metadata for this paragraph
        if paragraph_idx < len(metadata):
            row = metadata.iloc[paragraph_idx]
            
            # Print metadata fields
            print("METADATA:")
            for col in metadata.columns:
                if col != 'text_path':  # We'll handle text separately
                    print(f"  {col}: {row[col]}")
            
            # Load and print the actual text
            if 'text_path' in row and pd.notna(row['text_path']):
                text_path = f'/shared_data0/cgoldberg/Concept_Inversion/Data/{dataset_name}/{row["text_path"]}'
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                    
                    print(f"\\nTEXT CONTENT:")
                    print(f'"{text_content}"')
                    
                except FileNotFoundError:
                    print(f"\\nTEXT FILE NOT FOUND: {text_path}")
                except Exception as e:
                    print(f"\\nERROR READING TEXT: {e}")
            else:
                print(f"\\nNO TEXT PATH FOUND IN METADATA")
        else:
            print(f"ERROR: Paragraph index {paragraph_idx} out of range (metadata has {len(metadata)} rows)")


def analyze_concept_false_positives(dataset_name='Coco', model_name='CLIP', 
                                  concept='person', percentile=0.2,
                                  methods=['avg', 'linsep'], 
                                  sample_types=['patch', 'cls']):
    """
    Comprehensive analysis of false positives for a concept across different methods.
    """
    print(f"\\n{'='*80}")
    print(f"FALSE POSITIVE ANALYSIS FOR '{concept.upper()}'")
    print(f"Dataset: {dataset_name}, Model: {model_name}, Percentile: {percentile}")
    print(f"{'='*80}")
    
    results = {}
    
    for method in methods:
        for sample_type in sample_types:
            print(f"\\n{'-'*60}")
            print(f"Analyzing {method} {sample_type}")
            print(f"{'-'*60}")
            
            fp_results = get_false_positive_indices(
                dataset_name, model_name, sample_type, concept,
                percentile, split='test', method=method
            )
            
            if fp_results:
                results[f"{method}_{sample_type}"] = fp_results
                
                # Show top false positives
                if fp_results['false_positive_details']:
                    print(f"\\nTop 5 False Positive Images:")
                    for i, fp in enumerate(fp_results['false_positive_details'][:5]):
                        print(f"  {i+1}. Image {fp['index']}: max activation {fp['activation']:.4f} (threshold: {fp['threshold']:.4f})")
                
                # For text datasets, print the actual paragraphs
                if dataset_name in ['iSarcasm', 'Sarcasm', 'Stanford-Tree-Bank', 'IMDB', 'Jailbreak', 'GoEmotions']:
                    print_false_positive_paragraphs(fp_results, dataset_name, max_examples=10)
                
                # Visualize if it's patch method (showing which patches triggered the detection)
                elif sample_type == 'patch' and fp_results['false_positive_details']:
                    print(f"\\nVisualizing false positive images with triggering patches...")
                    visualize_false_positive_patches(fp_results, dataset_name, n_examples=6)
    
    # Compare methods
    if results:
        print(f"\\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        comparison_data = []
        for key, result in results.items():
            comparison_data.append({
                'Method': key,
                'False Positives': result['total_fp'],
                'True Positives': result['total_tp'],
                'FPR': result['total_fp'] / (result['total_fp'] + result['total_tn']),
                'FNR': result['total_fn'] / (result['total_fn'] + result['total_tp']),
                'Precision': result['total_tp'] / (result['total_tp'] + result['total_fp']) if (result['total_tp'] + result['total_fp']) > 0 else 0,
                'Recall': result['total_tp'] / (result['total_tp'] + result['total_fn']) if (result['total_tp'] + result['total_fn']) > 0 else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    return results


if __name__ == "__main__":
    # Compare patch vs CLS false positives
    results = compare_patch_vs_cls_false_positives(
        dataset_name='iSarcasm',
        model_name='Llama',
        concept='sarcastic',
        percentile=0.2,
        method='linsep',
        split='test'
    )