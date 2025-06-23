import torch
import pandas as pd
from optimal_f1_comparison import compare_methods_at_optimal_f1

# Run the comparison to get the results
results = compare_methods_at_optimal_f1(
    dataset_name='iSarcasm',
    model_name='Llama',
    concept='sarcastic',
    method='linsep'
)

# Get the patch-only false positives
patch_fp_set = set(results['patch_results']['false_positive_images'])
cls_fp_set = set(results['cls_results']['false_positive_images'])
patch_only_fp = sorted(list(patch_fp_set - cls_fp_set))

print(f"\n{'='*80}")
print(f"FALSE POSITIVES UNIQUE TO PATCH METHOD (at optimal F1 percentile)")
print(f"Total: {len(patch_only_fp)} paragraphs")
print(f"{'='*80}")

# Load metadata
metadata = pd.read_csv('/shared_data0/cgoldberg/Concept_Inversion/Data/iSarcasm/metadata.csv')

# Get details for each patch-only false positive
for i, paragraph_idx in enumerate(patch_only_fp):
    print(f"\n{'-'*60}")
    print(f"PATCH-ONLY FALSE POSITIVE #{i+1}")
    print(f"Paragraph Index: {paragraph_idx}")
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
            text_path = f'/shared_data0/cgoldberg/Concept_Inversion/Data/iSarcasm/{row["text_path"]}'
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                print(f"\nTEXT CONTENT:")
                print(f'"{text_content}"')
                
            except FileNotFoundError:
                print(f"\nTEXT FILE NOT FOUND: {text_path}")
            except Exception as e:
                print(f"\nERROR READING TEXT: {e}")
        else:
            print(f"\nNO TEXT PATH FOUND IN METADATA")
    else:
        print(f"ERROR: Paragraph index {paragraph_idx} out of range")

print(f"\n{'='*80}")
print(f"SUMMARY: {len(patch_only_fp)} paragraphs are false positives in PATCH but not CLS")
print(f"Patch-only FP indices: {patch_only_fp}")
print(f"{'='*80}")