import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM

import sys
sys.path.append('/shared_data0/cgoldberg/Concept_Inversion/Experiments')
from src.datasets import ImageDataset
from src.inversion_methods import prompt_inversion
from src.prompt_concepts import LLMNet, RawInput
from utils.quant_concept_evals_utils import compute_concept_thresholds


class FixedImageDataset:
    """Fixed version of ImageDataset that works with absolute paths"""
    def __init__(self, root, dataset_name, split="test", transform=None, max_samples=None):
        self.root = root
        self.dataset_name = dataset_name
        self.transform = transform
        self.split = split

        # Load metadata to get concept information
        self.metadata = pd.read_csv(f"{root}/Data/{dataset_name}/metadata.csv")

        # Select images based on the split
        if split == "train":
            self.metadata = self.metadata[self.metadata["split"] == "train"].reset_index(drop=True)
        elif split == "test":
            self.metadata = self.metadata[self.metadata["split"] == "test"].reset_index(drop=True)

        # Limit samples if specified
        if max_samples is not None:
            self.metadata = self.metadata.head(max_samples).reset_index(drop=True)

        # Get concept columns (excluding non-concept columns)
        self.concept_columns = [
            col for col in self.metadata.columns if col not in ["image_path", "split", "class"]
        ]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get an image and its metadata by index.
        """
        from PIL import Image
        
        # Load image
        image_path = f"{self.root}/Data/{self.dataset_name}/{self.metadata.iloc[idx]['image_path']}"
        image = Image.open(image_path).convert("RGB")

        # Apply transforms if available
        if self.transform:
            image = self.transform(image)

        # Get concepts as a tensor
        concepts = (self.metadata.loc[idx, self.concept_columns].values,)
        
        return image, concepts

    def get_concept_names(self):
        """Return list of concept names"""
        return self.concept_columns


def main(args):
    # load model
    model = LLM(
        model=args.model,
        max_model_len=12288,
        limit_mm_per_prompt={"image": 10},
        max_num_seqs=1,
        enforce_eager=True if "llama" in args.model.lower() else False,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
    )

    # load dataset with limited samples
    data = FixedImageDataset(
        root="/shared_data0/cgoldberg/Concept_Inversion/", 
        dataset_name=args.dataset, 
        split="test",
        max_samples=args.max_samples
    )
    concept_names = data.get_concept_names()

    if "class" in concept_names:
        concept_names.remove("class")

    concept_extractors = []
    print("Concepts:", concept_names)
    for concept in concept_names:
        extractor = LLMNet(
            model,
            input_desc=f"an image which may contain concepts from the list {concept_names}",
            output_desc=f"the word 'Yes' if the image contains {concept}, otherwise 'No'",
            image_before_prompt=True,
        )
        concept_extractors.append(extractor)

    extracted_concepts = []
    inversion_results = []
    for i in tqdm(range(len(data))):
        image, _ = data[i]

        # extract concepts
        concept_outputs = []
        concept_inversion = {}
        for j, extractor in enumerate(concept_extractors):
            output = extractor.forward(RawInput(image_input=image, text_input=None))
            if "Yes" in output:
                output = 1
            else:
                output = 0
            concept_outputs.append(output)

            # get inversion if concept present
            if output == 1:
                inversion = prompt_inversion(model, concept_names[j], image)
                print("Inversion:", inversion)
                concept_inversion[concept_names[j]] = inversion
        inversion_results.append(concept_inversion)

        print("Extracted:", concept_outputs)
        extracted_concepts.append(concept_outputs)

    # Save extracted concepts to a file
    output_file = f"{args.output_dir}/{args.dataset}_{args.model.split('/')[1]}_concepts_small.txt"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index"] + concept_names)
        for idx, concepts in enumerate(extracted_concepts):
            writer.writerow([idx] + concepts)

    # Save inversion results to a file
    output_file = f"{args.output_dir}/{args.dataset}_{args.model.split('/')[1]}_inversion_small.txt"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index"] + concept_names)
        for idx, concepts in enumerate(inversion_results):
            row = [idx]
            for concept in concept_names:
                if concept in concepts:
                    row.append(concepts[concept])
                else:
                    row.append("No inversion")
            writer.writerow(row)

    print(f"Processing complete! Processed {len(data)} images.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use prompting to extract concepts from a dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to extract concepts from. Options: [CLEVR, Coco].",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2-11",
        help="The model to use for extraction. Options: [llama3.2-11].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/shared_data0/cgoldberg/Concept_Inversion/Experiments/",
        help="The output directory for extracted concepts.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to process for testing.",
    )
    args = parser.parse_args()

    if args.model == "llama3.2-11":
        args.model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    elif args.model == "qwen2.5-vl-3":
        args.model = "Qwen/Qwen2.5-VL-3B-Instruct"

    main(args)