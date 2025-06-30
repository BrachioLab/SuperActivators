import torch
import pandas as pd
from .false_positive_extractor import get_false_positive_indices

# Get patch false positives at percentile 0.3
patch_results = get_false_positive_indices(
    dataset_name='iSarcasm',
    model_name='Llama',
    sample_type='patch',
    concept='sarcastic',
    percentile=0.3,
    split='test',
    method='linsep'
)

# Get CLS false positives at percentile 0.8
cls_results = get_false_positive_indices(
    dataset_name='iSarcasm',
    model_name='Llama', 
    sample_type='cls',
    concept='sarcastic',
    percentile=0.8,
    split='test',
    method='linsep'
)

print(f"Patch FP count: {patch_results['total_fp']}")
print(f"Patch FP indices: {sorted(patch_results['false_positive_images'])}")

print(f"\nCLS FP count: {cls_results['total_fp']}")
print(f"CLS FP indices: {sorted(cls_results['false_positive_images'])}")

# Find patch-only FPs
patch_fp_set = set(patch_results['false_positive_images'])
cls_fp_set = set(cls_results['false_positive_images'])
patch_only_fp = sorted(list(patch_fp_set - cls_fp_set))

print(f"\nPatch-only FP count: {len(patch_only_fp)}")
print(f"Patch-only FP indices: {patch_only_fp}")

# Also show the first few for verification
print("\nFirst 5 patch-only false positives:")
metadata = pd.read_csv('/shared_data0/cgoldberg/Concept_Inversion/Data/iSarcasm/metadata.csv')
for i, idx in enumerate(patch_only_fp[:5]):
    row = metadata.iloc[idx]
    print(f"\n{i+1}. Index {idx}:")
    print(f"   sarcastic label: {row['sarcastic']}")
    print(f"   split: {row['split']}")
    if pd.notna(row.get('text_path')):
        try:
            with open(f"/shared_data0/cgoldberg/Concept_Inversion/Data/iSarcasm/{row['text_path']}", 'r') as f:
                text = f.read().strip()
            if len(text) > 100:
                print(f"   text: \"{text[:100]}...\"")
            else:
                print(f"   text: \"{text}\"")
        except Exception as e:
            print(f"   text: Error reading file - {e}")