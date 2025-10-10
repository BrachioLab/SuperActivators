# Concept Inversion Research - SuperActivator Tokens

This repository implements the research on **superactivator tokens** - a novel interpretability technique for transformer models that discovers sparse, highly-activated tokens reliably signaling concept presence. This enables state-of-the-art concept detection and more faithful attributions compared to traditional global aggregation methods.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Main Concept Detection Analysis](#main-concept-detection-analysis)
- [Alternative Analysis Methods](#alternative-analysis-methods)
- [Visualization & Analysis](#visualization--analysis)
- [Directory Structure](#directory-structure)

## Overview

This repository contains the implementation for studying concept detection in transformer models. The codebase focuses on understanding how transformers encode semantic concepts and developing improved methods for detecting and localizing these concepts.

The main contribution is the discovery and analysis of the **SuperActivator Mechanism** - a phenomenon where a small subset of highly-activated tokens in the extreme tail of activation distributions can reliably signal concept presence. This approach addresses limitations in standard concept detection methods that suffer from noisy activations and poor localization.

### Supported Datasets & Models

**Datasets:**
- **Vision**: CLEVR, COCO, Broden-Pascal, Broden-OpenSurfaces
- **Text**: Sarcasm, Augmented iSarcasm, Augmented GoEmotions

**Models:**
- **Vision**: CLIP ViT-L/14, Llama-3.2-11B-Vision-Instruct
- **Text**: Llama-3.2-11B-Vision-Instruct, Gemma-2-9B, Qwen3-Embedding-4B

The codebase supports:
- Both supervised and unsupervised concept learning
- Token-level (patches for images, tokens for text) and global-level analysis
- Comprehensive evaluation across multiple datasets and modalities

## Datasets

Download the prepared datasets from: https://drive.google.com/drive/folders/1rwrZjWGRF2OpWv6ESMHn87OVl55KsL65?usp=sharing

Each dataset folder in `Data/` contains:
- `metadata.csv` - Sample identifiers, concept/label information, and file paths
- Padding masks for vision models (vision datasets only)

To use these datasets:
1. Download from the Google Drive link above
2. Update file paths in `metadata.csv` if needed
3. Run the analysis scripts with appropriate dataset arguments


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

## Main Concept Detection Analysis

The concept detection analysis extracts embeddings from transformer models and evaluates concept detection performance. Run these scripts sequentially from the `scripts` directory:

### Core Analysis Steps:
```bash
# 1. Extract embeddings
# For images:
python scripts/embed_image_datasets.py
python scripts/compute_image_gt_samples.py

# For text:
python scripts/embed_text_datasets.py (does gt computation as well)

# 2. Learn concepts
python scripts/compute_all_concepts.py

# 3. Compute activations
python scripts/compute_activations.py

# 4. Find thresholds that contain top N% of gt positive calibration samples per-concept
python scripts/validation_thresholds.py

# 5. Compute detection statistics
python scripts/all_detection_stats.py

# 6. Compute direct alignment inversion statistics
python scripts/all_inversion_stats.py
```

After completing the analysis, all quantitative results (detection metrics, F1 scores, precision/recall curves, etc.) will be saved in the `Quant_Results/` folder.

### Extended Analysis (Optional):

After the main analysis, run these for additional insights:

```bash
# Compare with baseline aggregation methods (max token, mean token, last token, random token)
python scripts/baseline_detections.py

# Find optimal percentthrumodel for each concept
python scripts/per_concept_ptm_optimization.py
```

### Command Line Arguments

All analysis scripts support command line arguments. Examples:

```bash
# Process specific datasets and models
python scripts/embed_image_datasets.py --models CLIP Llama --datasets CLEVR Coco

# Use specific percentthrumodel values
python scripts/compute_all_concepts.py --percentthrumodels 0 25 50 75 100

# Process single dataset with specific model
python scripts/compute_activations.py --model CLIP --dataset CLEVR
```

Most scripts support:
- `--model` or `--models`: Specify which model(s) to use
- `--dataset` or `--datasets`: Specify which dataset(s) to process
- `--percentthrumodels`: List of layer percentages to analyze
- `--sample_type`: Choose between 'patch' (same as token in this context) or 'cls' analysis

## Alternative Analysis Methods

### 1. Prompt Concepts Pipeline

Extract concepts using vision-language models through prompting:

```bash
# Extract concepts
python scripts/extract_prompt_concepts.py --dataset CLEVR --model llama3.2-11

# Evaluate performance
python scripts/extract_prompt_concepts.py --dataset CLEVR --model llama3.2-11 --eval

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

The repository includes four analysis notebooks in the `notebooks/` directory:

```bash
jupyter lab notebooks/
```

- **`Activation-Distributions.ipynb`** - Visualizes in-concept and out-of-concept activation distributions, demonstrating the separation in the extreme tails that enables the superactivator mechanism

- **`Compare-Methods.ipynb`** - Shows quantitative results comparing concept detection performance and direct alignment inversion accuracy across different methods

- **`Image-Concept-Evals.ipynb`** - Provides qualitative examples of superactivator tokens on image datasets, visualizing which patches activate most strongly for different concepts

- **`Text-Concepts.ipynb`** - Shows qualitative examples of superactivator tokens in text datasets, highlighting which words activate most strongly for different concepts


## Directory Structure

```
Experiments/
├── scripts/              # Main analysis scripts
│   ├── embed_*.py       # Embedding extraction
│   ├── compute_*.py     # Concept learning & activation
│   ├── validation_*.py  # Threshold optimization
│   └── pretrained_saes/ # SAE analysis scripts
├── notebooks/           # Jupyter notebooks for visualization
├── utils/               # Utility functions
├── Data/                # Dataset metadata and padding masks
├── requirements.txt     # Python dependencies
└── pyproject.toml       # Project configuration
```

Pipeline Output Directories (created during analysis):
```
Embeddings/
├── {dataset}/                    # Model embeddings
│   └── *.pt                     # Chunked embedding files
Concepts/
├── {dataset}/
│   └── *.pt                     # Learned concept vectors (avg, linsep, kmeans)
Cosine_Similarities/
├── {dataset}/
│   └── *.pt                     # Cosine similarity activations
Distances/
├── {dataset}/
│   └── *.pt                     # Signed distances for linear separators
GT_Samples/
├── {dataset}/
│   └── *.pt                     # Ground truth sample indices
Thresholds/
├── {dataset}/
│   └── *.pt                     # Optimal thresholds per concept
Quant_Results/
├── {dataset}/
│   └── *.pt                     # **Final detection metrics, F1 scores, precision/recall**
activation_distributions/
├── {dataset}/
│   └── *.pt                     # Activation distributions for visualization
prompt_results/
├── {dataset}/
│   └── *.txt, *.csv             # Prompt-based concept extraction results
Best_Inversion_Percentiles_Cal/
├── {dataset}/
│   └── *.pt                     # Optimal percentiles for inversion
Best_Detection_Percentiles_Cal/
├── {dataset}/
│   └── *.pt                     # Optimal percentiles for detection
```

Where `{dataset}` is one of: CLEVR, Coco, Broden-Pascal, Broden-OpenSurfaces, Sarcasm, iSarcasm, GoEmotions



