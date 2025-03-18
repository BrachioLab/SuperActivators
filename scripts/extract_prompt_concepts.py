import numpy as np
import argparse
from src.prompt_concepts import OurLLM, LLMNet, RawInput
from src.datasets import ImageDataset
from tqdm import tqdm
import csv


def main(args):
    # load model
    model = OurLLM(model_name=args.model)

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
            input_desc="an image",
            output_desc=f"the word 'Yes' if the image contains {concept}, otherwise 'No'",
            image_before_prompt=True
        )
        concept_extractors.append(extractor)

    extracted_concepts = []
    for i in tqdm(range(len(data))):
        image, _ = data[i]

        # extract concepts
        concept_outputs = []
        for extractor in concept_extractors:
            output = extractor.forward(RawInput(image_input=image, text_input=None))
            if "Yes" in output:
                output = 1
            else:
                output = 0
            concept_outputs.append(output)

        print("Extracted:", concept_outputs)
        extracted_concepts.append(concept_outputs)

    # Save extracted concepts to a file
    output_file = f"{args.output_dir}/{args.dataset}_{args.model.split('/')[1]}_concepts.txt"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index"] + concept_names)
        for idx, concepts in enumerate(extracted_concepts):
            writer.writerow([idx] + concepts)


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
    args = parser.parse_args()

    if args.model == "llama3.2-11":
        args.model = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    if args.eval:
        eval(args)
    else:
        main(args)
