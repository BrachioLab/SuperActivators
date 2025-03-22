# Concept Inversion

## Overview

## Datasets
- **CLEVR**: Color(Blue, Green, Red), Shape(Cube, Cyliner, Sphere)
- **COCO**: 80 common object categories

## Models
- **CLIP (Contrastive Language-Image Pretraining)**
- **LLAMA3 Vision**

## Python Environment Setup
Run the following the setup and activate a Python development environment to run
the code:
```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision torchaudio
python -m pip install -e ".[dev]"
```

## Development Setup
To setup automatic code linting on every commit, run the following:
```sh
pre-commit install
```