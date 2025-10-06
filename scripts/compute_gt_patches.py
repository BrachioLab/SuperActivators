import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.gt_concept_segmentation_utils

from utils.gt_concept_segmentation_utils import map_concepts_to_patch_indices, sort_mapping_by_split


PERCENT_THRU_MODEL = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    for DATASET_NAME in ['CLEVR', 'Coco']:
        for INPUT_IMAGE_SIZE in [(224, 224), (560, 560)]:
            gt_patches_per_concept = map_concepts_to_patch_indices(dataset_name=DATASET_NAME, model_input_size=INPUT_IMAGE_SIZE)
            gt_patches_per_concept_train, gt_patches_per_concept_test = sort_mapping_by_split(gt_patches_per_concept, DATASET_NAME,
                                                                        sample_type='patch', model_input_size=INPUT_IMAGE_SIZE)