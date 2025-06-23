import torch
import os
import pandas as pd
import importlib

# Import custom modules
import CCE_src.concept_learning
import utils
import compute_concepts_utils
import patch_alignment_utils

# Reload modules (if running interactively)
importlib.reload(CCE_src.concept_learning)

from CCE_src.concept_learning import ConceptLearner
from utils import load_images, get_split_df
from compute_concepts_utils import compute_cosine_sims

# Define dataset and parameters
DATASET_NAME = 'CLEVR'
PERCENT_THRU_MODEL = 100
NUM_CONCEPT_GROUPS = 4

# File paths
EMBEDDINGS_FILE = f'CLIP_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}.pt'
CONCEPTS_FILE = f'CCE_concepts_CLIP_cls_embeddings_percentthrumodel_{PERCENT_THRU_MODEL}_nattributes_{NUM_CONCEPT_GROUPS}.pt'
COSSIM_FILE = f'cosine_similarities_{CONCEPTS_FILE[:-3]}.csv'

if __name__ == "__main__":
    # Load embeddings
    embeds_path = f'Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}'
    if not os.path.exists(embeds_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeds_path}")

    embeds = torch.load(f"Embeddings/{DATASET_NAME}/{EMBEDDINGS_FILE}")
    
    split_df = get_split_df(DATASET_NAME)
    train_embeds = embeds[torch.tensor((split_df.values == 'train').nonzero()[0], dtype=torch.long)]

    print("Loaded embeddings:", train_embeds.shape, flush=True)

    # Initialize concept learner
    cl = ConceptLearner(samples=[None], input_to_latent=None, input_processor=None, device="cuda")

    # Learn concepts
    concepts = cl.learn_attribute_concepts(n_attributes=NUM_CONCEPT_GROUPS, patch_activations=train_embeds, 
                                           subspace_dim=100, split_method="ours-subspace")

    # Save learned concepts
    concept_dic = {i: concept for i, concept in enumerate(concepts)}
    concepts_path = f'Concepts/{DATASET_NAME}/{CONCEPTS_FILE}'

    os.makedirs(os.path.dirname(concepts_path), exist_ok=True)
    torch.save(concept_dic, concepts_path)

    # Load and print extracted concepts
    concepts = torch.load(concepts_path)
    print(f"Extracted {len(concepts.keys())} concepts")
