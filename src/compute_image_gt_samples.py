import torch
import sys
import os
from itertools import product
sys.path.append(os.path.abspath("utils"))

import gt_concept_segmentation_utils

from gt_concept_segmentation_utils import map_concepts_to_image_indices, map_concepts_to_patch_indices, sort_mapping_by_split


PERCENT_THRU_MODEL = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_TYPES = ['cls', 'patch']
# DATASETS = ['CLEVR', 'Coco', 'Broden-Pascal', 'Broden-OpenSurfaces']
DATASETS = DATASETS = ['Coco']
MODEL_INPUT_SIZES = [(224, 224), (560, 560)]


if __name__ == "__main__":
    experiment_configs = product(DATASETS, MODEL_INPUT_SIZES, SAMPLE_TYPES)
    for dataset_name, model_input_size, sample_type in experiment_configs:
        print(f"Computing gt for dataset {dataset_name} input size {model_input_size} sample type {sample_type}")
        if sample_type == 'cls':
            gt_samples_per_concept = map_concepts_to_image_indices(dataset_name=dataset_name,
                                                                  model_input_size=model_input_size)
        else:
            gt_samples_per_concept = map_concepts_to_patch_indices(dataset_name=dataset_name,
                                                                  model_input_size=model_input_size)
        
        sort_mapping_by_split(gt_samples_per_concept, dataset_name,
                              sample_type=sample_type, model_input_size=model_input_size)
            