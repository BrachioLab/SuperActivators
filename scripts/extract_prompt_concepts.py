import numpy as np
import argparse
from src.prompt_concepts import OurLLM, LLMNet, RawInput
from src.datasets import ImageDataset
from src.utils.quant_concept_evals_utils import compute_concept_thresholds
from src.inversion_methods import prompt_inversion
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch
from vllm import LLM


def concept_inversion(args):
    # load model
    # model = OurLLM(model_name=args.model)
    model = LLM(model=args.model,
                max_model_len=12288,
                limit_mm_per_prompt={"image": 10},
                max_num_seqs=1,
                enforce_eager=True if "llama" in args.model.lower() else False,
                trust_remote_code=True,
                gpu_memory_utilization=0.5,
    )

    # load dataset
    data = ImageDataset(root="/shared_data0/cgoldberg/Concept_Inversion/", dataset_name=args.dataset, split="test")
    concept_names = data.get_concept_names()

    if "class" in concept_names:
        concept_names.remove("class")

    # load detected concepts
    concepts_file = f"{args.output_dir}/{args.dataset}_{args.model.split('/')[1]}_concepts.txt"
    with open(concepts_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        detected_concepts = [list(map(lambda x: 1 if "yes" in x.lower() else 0, row[1:])) for row in reader]
    detected_concepts = np.array(detected_concepts)

    inversion_results = []
    for i in tqdm(range(len(data))):
        image, _ = data[i]

        # get concept inversion for only present concepts
        concept_inversion = {}
        for j, concept in enumerate(concept_names):
            if detected_concepts[i][j] == 1:
                inversion = prompt_inversion(model, concept, image)
                print("Inversion:", inversion)
                concept_inversion[concept] = inversion
        inversion_results.append(concept_inversion)
    
    # Save inversion results to a file
    output_file = f"{args.output_dir}/{args.dataset}_{args.model.split('/')[1]}_inversion.txt"
    with open(output_file, mode='w', newline='') as file:
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

def main(args):
    # load model
    # model = OurLLM(model_name=args.model)
    model = LLM(model=args.model,
                max_model_len=12288,
                limit_mm_per_prompt={"image": 10},
                max_num_seqs=1,
                enforce_eager=True if "llama" in args.model.lower() else False,
                trust_remote_code=True,
                gpu_memory_utilization=0.5,
    )

    # load dataset
    data = ImageDataset(root="/shared_data0/cgoldberg/Concept_Inversion/", dataset_name=args.dataset, split="test")
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
            image_before_prompt=True
        )
        concept_extractors.append(extractor)
    # for concept in concept_names:
    #     extractor = LLMNet(
    #         model,
    #         input_desc="an image",
    #         output_desc=f"a list (in the format [concept1, concept2, ...]) of the concepts that are present in the image out of the following: {concept_names}",
    #         image_before_prompt=True
    #     )
    #     concept_extractors.append(extractor)

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
    output_file = f"{args.output_dir}/{args.dataset}_{args.model.split('/')[1]}_concepts.txt"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index"] + concept_names)
        for idx, concepts in enumerate(extracted_concepts):
            writer.writerow([idx] + concepts)

    # Save inversion results to a file
    output_file = f"{args.output_dir}/{args.dataset}_{args.model.split('/')[1]}_inversion.txt"
    with open(output_file, mode='w', newline='') as file:
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


def eval(args):
    # load dataset
    data = ImageDataset(root="/shared_data0/cgoldberg/Concept_Inversion/", dataset_name=args.dataset, split="test")
    concept_names = data.get_concept_names()

    # get gt
    gt_labels = []
    for i in range(len(data)):
        _, label = data[i]
        label = label[0]
        gt_labels.append([int(label[j]) for j in range(len(label)) if concept_names[j] != "class"])
    gt_labels_array = np.array(gt_labels)

    print("GT labels size:", gt_labels_array.shape)

    if "class" in concept_names:
        concept_names.remove("class")

    # Load extracted concepts from the file
    output_file = f"{args.output_dir}/{args.dataset}_{args.model.split('/')[1]}_concepts.txt"
    with open(output_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        extracted_concepts = [list(map(int, row[1:])) for row in reader]
    
    extracted_concepts_array = np.array(extracted_concepts)

    # Evaluate the extracted concepts
    for idx, concepts in enumerate(concept_names):
        gt = gt_labels_array[:, idx]
        pred = extracted_concepts_array[:, idx]
        # Calculate F1 score
        tp = np.sum((gt == 1) & (pred == 1))
        fp = np.sum((gt == 0) & (pred == 1))
        fn = np.sum((gt == 1) & (pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"F1 for {concepts}: {f1_score:.4f}")

    # create a bar plot of the F1 scores sorted by F1 score
    f1_scores = []
    for idx, concepts in enumerate(concept_names):
        gt = gt_labels_array[:, idx]
        pred = extracted_concepts_array[:, idx]
        # Calculate F1 score
        tp = np.sum((gt == 1) & (pred == 1))
        fp = np.sum((gt == 0) & (pred == 1))
        fn = np.sum((gt == 1) & (pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)
    f1_scores = np.array(f1_scores)
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_f1_scores = f1_scores[sorted_indices]
    sorted_concept_names = np.array(concept_names)[sorted_indices]

    plt.figure(figsize=(10, 15))
    plt.barh(sorted_concept_names, sorted_f1_scores)
    plt.xlabel("F1 Score")
    plt.title("F1 Scores for Extracted Concepts")

    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # add a line at 0.5
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold')

    # save the plot
    plt.savefig(f"{args.dataset}_{args.model.split('/')[1]}_f1_scores.png")

    # load linsep concepts
    concepts_file = "linsep_concepts_CLIP_cls_embeddings_percentthrumodel_70.csv"
    linsep_dists = pd.read_csv(f"/shared_data0/cgoldberg/Concept_Inversion/Experiments/Distances/{args.dataset}/dists_{concepts_file}")
    gt_images_per_concept_test = torch.load(f'/shared_data0/cgoldberg/Concept_Inversion/Experiments/GT_Samples/{args.dataset}/gt_images_per_concept_test_image.pt')
    thresholds = compute_concept_thresholds(gt_images_per_concept_test, linsep_dists, 0.95)
    print(thresholds)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use prompting to extract concepts from a dataset.")
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
        default="/shared_data0/steinad/Concept_Inversion/",
        help="The output directory for extracted concepts."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to evaluate the concepts."
    )
    parser.add_argument(
        "--inversion",
        action="store_true",
        help="Whether to perform inversion."
    )
    args = parser.parse_args()

    if args.model == "llama3.2-11":
        args.model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    elif args.model == "qwen2.5-vl-3":
        args.model = "Qwen/Qwen2.5-VL-3B-Instruct"

    if args.eval:
        eval(args)
    elif args.inversion:
        concept_inversion(args)
    else:
        main(args)
