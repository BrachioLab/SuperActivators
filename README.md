# Concept Inversion Research - Superdetector Tokens

This repository implements the research on **superdetector tokens** - a novel interpretability technique for transformer models that discovers sparse, highly-activated tokens reliably signaling concept presence. This enables state-of-the-art concept detection and more faithful attributions compared to traditional global aggregation methods.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Main Pipeline](#main-pipeline)
- [Alternative Pipelines](#alternative-pipelines)
- [Visualization & Analysis](#visualization--analysis)
- [Pipeline Status Monitoring](#pipeline-status-monitoring)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)

## Overview

This research addresses the "black box" problem in transformer models by:
- Discovering **superdetector tokens**: sparse subsets of highly activated tokens that reliably signal concept presence
- Providing **local, context-aware concept detection** instead of global aggregation
- Supporting both **supervised** (with ground truth labels) and **unsupervised** (k-means clustering) concept discovery
- Working across **vision** (CLIP) and **language** (Llama) transformer models

## Prerequisites

- Python >= 3.11
- CUDA-capable GPU (recommended) with at least 40GB VRAM
- Access to Hugging Face models
- ~500GB storage for embeddings and intermediate results

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Experiments
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or using the pyproject.toml
pip install -e .
```

3. Set up environment variables if needed:
```bash
export HF_HOME=/path/to/huggingface/cache
export CUDA_VISIBLE_DEVICES=0  # Select GPU
```

## Main Pipeline

The main pipeline analyzes concept detection using embeddings from transformer models. Run these scripts sequentially from the `Experiments` directory:

### For Image Datasets (CLEVR, COCO, Broden):

```bash
# 1. Extract embeddings
python scripts/embed_image_datasets.py

# 2. Compute ground truth samples (images only)
python scripts/compute_image_gt_samples.py

# 3. Learn concepts (k-means or linear separators)
python scripts/compute_all_concepts.py

# 4. Compute activations (cosine similarities)
python scripts/compute_activations.py

# 5. Find optimal thresholds
python scripts/validation_thresholds.py

# 6. Compute detection statistics
python scripts/all_detection_stats.py

# 7. Compute inversion statistics (images only)
python scripts/all_inversion_stats.py
```

### For Text Datasets (Sarcasm, iSarcasm, GoEmotions):

```bash
# 1. Extract embeddings
python scripts/embed_text_datasets.py

# 2-6. Same as steps 3-6 above
python scripts/compute_all_concepts.py
python scripts/compute_activations.py
python scripts/validation_thresholds.py
python scripts/all_detection_stats.py
# (No inversion step for text)
```

### Extended Analysis (Optional):

After the main pipeline, run these for additional insights:

```bash
# Compare with baseline aggregation methods
python scripts/baseline_detections.py

# Find optimal percentthrumodel for each concept
python scripts/per_concept_ptm_optimization.py

# Compute bootstrap confidence intervals
python scripts/detection_errors.py
```

### Configuring Pipeline Runs

Each script has configuration variables at the top. Modify these before running:

```python
# Example from scripts/embed_image_datasets.py
MODELS = ["CLIP-ViT-L-14", "Llama-3.2-11B-Vision-Base"]
DATASETS = ["CLEVR", "Coco"]
SAMPLE_TYPES = ["patch", "cls"]
PERCENTTHRUMODELS = [0, 12, 24, 36, 48, 60, 72, 84, 96]
```

## Alternative Pipelines

### 1. Prompt Concepts Pipeline

Extract concepts using vision-language models through prompting:

```bash
# Extract concepts
python scripts/extract_prompt_concepts.py --dataset CLEVR --model llama3.2-11

# Evaluate performance
python scripts/extract_prompt_concepts.py --dataset CLEVR --model llama3.2-11 --eval

# Run inversion for localization (optional)
python scripts/extract_prompt_concepts.py --dataset CLEVR --model llama3.2-11 --inversion
```

Supported models:
- `llama3.2-11` (Llama-3.2-11B-Vision-Instruct)
- `qwen2.5-vl-3` (Qwen2.5-VL-3B-Instruct)

Results are saved in `prompt_results/{dataset}/`.

### 2. SAE (Sparse Autoencoder) Pipeline

Analyze pretrained sparse autoencoders:

#### For Images:
```bash
cd scripts/pretrained_saes/
python embed_image_datasets_sae.py
python compute_activations_sae_sparse.py
python postprocess_sae_activations.py
python sae_validation_thresholds_dense.py
python sae_detection_stats_dense.py
python sae_inversion_stats_dense.py
```

#### For Text:
```bash
cd scripts/pretrained_saes/
python embed_text_datasets_sae.py
# Continue with same steps as images
```

## Visualization & Analysis

### Jupyter Notebooks

Launch Jupyter and explore the notebooks in `notebooks/`:

```bash
jupyter lab
```

Key notebooks:
- **`Visualize-Dataset.ipynb`**: Visualize dataset samples and annotations
- **`Activation-Distributions.ipynb`**: Analyze activation patterns
- **`Compare-Methods.ipynb`**: Compare detection methods
- **`Image-Concept-Evals.ipynb`**: Evaluate image concept detection with visualizations
- **`Text-Concepts.ipynb`**: Text concept analysis and visualization

### Quick Visualization of Results

```python
# Example: Load and visualize detection results
import torch
import matplotlib.pyplot as plt

# Load detection stats
results = torch.load('Quant_Results/CLEVR/detectfirst_CLEVR_supervised_CLIP-ViT-L-14_linsep_patch_embeddings_percentthrumodel_48.pt')

# Plot F1 scores
plt.figure(figsize=(10, 6))
plt.plot(results['f1_scores'])
plt.xlabel('Concept Index')
plt.ylabel('F1 Score')
plt.title('Concept Detection Performance')
plt.show()
```

## Pipeline Status Monitoring

Check pipeline completion status:

```bash
# Comprehensive status check (recommended)
python comprehensive_pipeline_status.py --show-commands

# Check specific datasets
python comprehensive_pipeline_status.py --datasets CLEVR Coco --show-commands

# Summary only
python comprehensive_pipeline_status.py --summary-only
```

This shows:
- Which pipeline stages are complete/missing
- Commands to run missing components
- Overall completion percentage

## Directory Structure

```
Experiments/
├── scripts/              # Main pipeline scripts
│   ├── embed_*.py       # Embedding extraction
│   ├── compute_*.py     # Concept learning & activation
│   ├── validation_*.py  # Threshold optimization
│   └── pretrained_saes/ # SAE pipeline scripts
├── notebooks/           # Jupyter notebooks for visualization
├── utils/               # Utility functions
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
└── comprehensive_pipeline_status.py  # Pipeline monitoring tool
```

Generated directories (created by pipeline):
- `Concepts/` - Learned concepts
- `Thresholds/` - Optimal thresholds  
- `Quant_Results/` - Detection statistics
- `Quant_Results_with_CI/` - Results with confidence intervals
- `Figs/` - Generated figures
- `prompt_results/` - Prompt concept outputs

External data directories (typically on scratch):
- `/scratch/cgoldberg/Embeddings/` - Model embeddings
- `/scratch/cgoldberg/Cosine_Similarities/` - Activation measures
- `/scratch/cgoldberg/Distances/` - Signed distances

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in scripts
   - Use CPU fallback: `CUDA_VISIBLE_DEVICES="" python script.py`

2. **Missing Files**:
   - Run `comprehensive_pipeline_status.py` to check what's missing
   - Ensure previous pipeline steps completed successfully

3. **Index Errors in Token Analysis**:
   - Check that embeddings match the model's token dimensions
   - Verify dataset preprocessing completed correctly

4. **Slow Embedding Extraction**:
   - Normal for large datasets (COCO can take 8+ hours)
   - Consider running overnight or on multiple GPUs

### Getting Help

- Check existing issues in the repository
- Look at notebook examples for usage patterns
- Review script documentation and comments

## Key Research Insights

This pipeline enables discovery of:
- **Superdetector tokens**: Sparse tokens with high concept activation
- **Local vs global detection**: Token-level outperforms CLS-level aggregation
- **Faithful attributions**: More accurate than traditional concept vector methods
- **Cross-modal applicability**: Works for both vision and language tasks

For detailed methodology and results, see the associated research paper.