# Concept Inversion Research - Superdetector Tokens

This repository implements the research on **superdetector tokens** - a novel interpretability technique for transformer models that discovers sparse, highly-activated tokens reliably signaling concept presence. This enables state-of-the-art concept detection and more faithful attributions compared to traditional global aggregation methods.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Main Pipeline](#main-pipeline)
- [Alternative Pipelines](#alternative-pipelines)
- [Visualization & Analysis](#visualization--analysis)
- [Directory Structure](#directory-structure)

## Overview

This repository contains the implementation and analysis pipeline for studying concept detection in transformer models. The codebase focuses on understanding how transformers encode semantic concepts and developing improved methods for detecting and localizing these concepts.

The main contribution is the discovery and analysis of the **Superdetector Mechanism** - a phenomenon where a small subset of highly-activated tokens in the extreme tail of activation distributions can reliably signal concept presence. This approach addresses limitations in standard concept detection methods that suffer from noisy activations and poor localization.

### Supported Datasets & Models

**Datasets:**
- **Vision**: CLEVR, COCO, Broden-Pascal, Broden-OpenSurfaces
- **Text**: Sarcasm, Augmented iSarcasm, Augmented GoEmotions

**Models:**
- **Vision**: CLIP ViT-L/14, Llama-3.2-11B-Vision-Instruct
- **Text**: Llama-3.2-11B-Vision-Instruct, Gemma-2-9B, Qwen3-Embedding-4B

The pipeline supports:
- Both supervised and unsupervised concept learning
- Token-level (patches for images, tokens for text) and global-level analysis
- Comprehensive evaluation across multiple datasets and modalities

## Datasets

This codebase is designed to work with the following datasets:

### Vision Datasets

- **CLEVR** - Synthetic scenes with objects of different colors (Blue, Green, Red) and shapes (Cube, Cylinder, Sphere). Generated using the CLEVR generator with single-object scenes. Full dataset with images and segmentation masks available at **[Google Drive link]**.

- **COCO** - Subset of MS COCO dataset with 80 common object categories. We reference image indices and annotations only; original images must be obtained from the [official COCO dataset](https://cocodataset.org/).

- **Broden-Pascal & Broden-OpenSurfaces** - Concept annotations from the Broden dataset for network dissection. We include metadata referencing concept labels from the original [Broden dataset](http://netdissect.csail.mit.edu/).

### Text Datasets

- **Sarcasm** - Synthetic sarcasm dataset created for this work. Contains paragraph-level and word-level sarcasm annotations. Full dataset available at **[Google Drive link]**.

- **Augmented iSarcasm** - Extended version of the iSarcasm dataset with additional context. Due to licensing restrictions, base iSarcasm text must be obtained from the [original source](https://github.com/iabufarha/iSarcasmEval).

- **Augmented GoEmotions** - Enhanced version of Google's GoEmotions dataset with additional filler text. Based on [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (CC BY 4.0). Augmented version available at **[Google Drive link]**.

### Using the Datasets

Each dataset folder in `Data/` contains:
- `metadata.csv` - Sample identifiers, concept/label information, and file paths
- `patches_w_image_mask_inputsize_(224, 224).pt` - Padding masks for CLIP (vision datasets only)
- `patches_w_image_mask_inputsize_(560, 560).pt` - Padding masks for Llama Vision (vision datasets only)

The padding masks indicate which patches contain actual image content vs padding, essential for accurate patch-level analysis.

To use these datasets:
1. Obtain datasets from their official sources or provided Drive links
2. Update the `image_path` or `text_path` columns in `metadata.csv` to reflect your local paths
3. Run the pipeline scripts with appropriate dataset arguments


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

### Core Pipeline Steps:
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

# 4. Find thresholds that contain N% of gt positive calibration samples per-concept
python scripts/validation_thresholds.py

# 5. Compute detection statistics
python scripts/all_detection_stats.py

# 6. Compute direct alignment inversion statistics
python scripts/all_inversion_stats.py
```

### Extended Analysis (Optional):

After the main pipeline, run these for additional insights:

```bash
# Compare with baseline aggregation methods (max token, mean token, last token, random token)
python scripts/baseline_detections.py

# Find optimal percentthrumodel for each concept
python scripts/per_concept_ptm_optimization.py
```

### Command Line Arguments

All pipeline scripts support command line arguments. Examples:

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
- `--sample_type`: Choose between 'patch', 'cls', or 'token' analysis

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

The repository includes four analysis notebooks in the `notebooks/` directory:

```bash
jupyter lab notebooks/
```

- **`Activation-Distributions.ipynb`** - Visualizes in-concept and out-of-concept activation distributions, demonstrating the separation in the extreme tails that enables the superdetector mechanism

- **`Compare-Methods.ipynb`** - Shows quantitative results comparing concept detection performance and direct alignment inversion accuracy across different methods

- **`Image-Concept-Evals.ipynb`** - Provides qualitative examples of superdetector activations on image datasets, visualizing which patches activate most strongly for different concepts

- **`Text-Concepts.ipynb`** - Shows qualitative examples of superdetector tokens in text datasets, highlighting which words activate most strongly for different concepts


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
└── pyproject.toml       # Project configuration
```

Generated directories (created by pipeline):
- `Embeddings/` - Model embeddings for each dataset
- `Concepts/` - Learned concepts
- `Cosine_Similarities/` - Activation measures
- `Distances/` - Signed distances for linear separators
- `Thresholds/` - Optimal thresholds  
- `Quant_Results/` - Detection statistics
- `Figs/` - Generated figures
- `prompt_results/` - Prompt concept outputs



