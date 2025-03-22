import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .general_utils import get_split_df
from .patch_alignment_utils import get_patch_split_df


############# Find Thresholds for Concepts #############
def compute_avg_rand_threshold(embeddings, patch_indices, percentile, n_vectors=5, device="cuda"):
    """
    Computes the average random cosine similarity threshold over n_vectors random vectors
    in a fully vectorized manner using PyTorch.

    Args:
        embeddings (torch.Tensor): Tensor of shape (N, embedding_dim).
        patch_indices (list or 1D Tensor): Indices of patches to consider.
        percentile (float): Desired percentile (e.g., 0.95).
        n_vectors (int): Number of random vectors to sample.
        device (str): Compute device (e.g., "cuda").

    Returns:
        float: The average threshold computed over the n_vectors random vectors.
    """
    # Ensure embeddings are on the target device.
    embeddings = embeddings.to(device)
    N, embedding_dim = embeddings.shape

    # Normalize embeddings (to compute cosine similarity via dot product)
    norm_embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)

    # Generate n_vectors random vectors and normalize them.
    random_vectors = torch.randn(n_vectors, embedding_dim, device=device)
    random_vectors = random_vectors / (random_vectors.norm(dim=1, keepdim=True) + 1e-8)

    # Compute cosine similarities between each random vector and all embeddings.
    # Resulting shape: (n_vectors, N)
    cos_sims = torch.matmul(random_vectors, norm_embeddings.t())

    # Ensure patch_indices is a tensor on the correct device.
    if not torch.is_tensor(patch_indices):
        patch_indices = torch.tensor(patch_indices, device=device)

    # Select only the similarities for the specified patch indices.
    # Shape: (n_vectors, num_selected)
    relevant_cos_sims = cos_sims[:, patch_indices]

    # Sort each row in descending order.
    sorted_cos_sims, _ = torch.sort(relevant_cos_sims, dim=1, descending=True)

    # Determine the index corresponding to the desired percentile.
    n_selected = sorted_cos_sims.size(1)
    percentile_index = int(percentile * n_selected)
    percentile_index = min(percentile_index, n_selected - 1)  # safeguard against OOB

    # Gather the threshold from each random vector and average.
    thresholds = sorted_cos_sims[:, percentile_index]  # shape: (n_vectors,)
    avg_threshold = thresholds.mean().item()

    return avg_threshold


def compute_concept_thresholds(
    gt_samples_per_concept, cos_sims, percentile, device="cuda", n_concepts_to_print=0
):
    """
    GPU-accelerated and vectorized computation of cosine similarity thresholds for each concept.
    For each concept, the threshold is defined as the (1 - percentile) quantile of its cosine
    similarity scores (with NaN-padded sequences handled via torch.nanquantile). Additionally,
    an average random threshold is computed for each concept.

    Args:
        cos_sims (pd.DataFrame): Cosine similarity matrix (rows: patches, columns: concept names).
        gt_samples_per_concept (dict): Mapping of concept to list of patch indices.
            (The concept keys must correspond to the column names in cos_sims.)
        percentile (float): The desired percentile (e.g., 0.95).
        embeddings (torch.Tensor): Embeddings used for computing random thresholds.
        n_vectors (int): Number of random vectors for computing the random threshold.
        device (str): Compute device (e.g., "cuda").
        print_result (bool): If True, prints the computed thresholds.

    Returns:
        dict: Mapping from concept to a tuple (threshold, random_threshold).
              The threshold is computed from the concept's cosine similarities,
              and random_threshold is the average threshold from random vectors.
    """
    # Convert the cosine similarity DataFrame to a torch tensor on the GPU.
    cos_sims_tensor = torch.tensor(cos_sims.values, device=device)

    concept_names = list(gt_samples_per_concept.keys())
    sims_list = []

    # Gather cosine similarity scores for each concept.
    for concept in concept_names:
        # Get the column index for this concept. (Convert key to string to match DataFrame columns.)
        col_idx = cos_sims.columns.get_loc(str(concept))
        sample_indices = gt_samples_per_concept[concept]
        sims = cos_sims_tensor[sample_indices, col_idx]  # shape: (num_samples_for_concept,)
        sims_list.append(sims)

    # Pad the list of tensors to form a single tensor of shape (n_concepts, max_samples),
    # using NaN for padding so that torch.nanquantile can ignore them.
    padded_sims = pad_sequence(sims_list, batch_first=True, padding_value=float("nan"))

    # Compute the (1 - percentile) quantile for each concept.
    # (For descending-sorted values, the (1 - percentile) quantile gives the threshold such that
    #  'percentile' fraction of values are above it.)
    thresholds_tensor = torch.nanquantile(padded_sims, 1 - percentile, dim=1)

    # For each concept, compute the average random threshold.
    concept_thresholds = {}
    for i, concept in enumerate(concept_names):
        sample_indices = gt_samples_per_concept[concept]
        threshold_val = thresholds_tensor[i].item()
        concept_thresholds[concept] = threshold_val

    if n_concepts_to_print > 0:
        print(f"Concept thresholds using {percentile*100:.1f}%:")
        for i, (concept, threshold) in enumerate(concept_thresholds.items()):
            if i > n_concepts_to_print:
                break
            print(f"Concept {concept}: {threshold:.4f}")

    return concept_thresholds


def evaluate_thresholds_across_dataset(
    concept_thresholds,
    gt_samples_per_concept,
    cos_sims,
    dataset_name,
    sample_type,
    all_object_patches=None,
):
    """
    Evaluate threshold-based classification performance across a dataset.
    Computes True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN).

    Args:
        concept_thresholds (dict): Mapping from concept to (threshold, random_threshold).
        gt_samples_per_concept (dict): Mapping from concept to list of ground truth patch indices.
        cos_sims (pd.DataFrame): Cosine similarity matrix (rows: patches, columns: concept names).
        dataset_name (str): Name of the dataset.
        sample_type (str): Type of sample ('patch' or 'image').
        all_object_patches (set, optional): If provided, only consider these patch indices.

    Returns:
        tuple: (fp_count, fn_count, tp_count, tn_count) for each concept.
    """
    fp_count = defaultdict(int)
    fn_count = defaultdict(int)
    tp_count = defaultdict(int)
    tn_count = defaultdict(int)

    if sample_type == "patch":
        split_df = get_patch_split_df(dataset_name, patch_size=14, model_input_size=(224, 224))
    elif sample_type == "image":
        split_df = get_split_df(dataset_name)

    test_mask = split_df != "train"
    test_indices = np.where(test_mask)[0]

    # If filtering patches, restrict test indices
    if all_object_patches is not None:
        test_indices = np.array([idx for idx in test_indices if idx in all_object_patches])

    # Loop over each concept
    for concept, concept_indices in tqdm(gt_samples_per_concept.items()):
        # Filter concept indices if all_object_patches is provided
        if all_object_patches is not None:
            concept_indices = [idx for idx in concept_indices if idx in all_object_patches]

        # Get cosine similarity values for the test samples as a NumPy array
        cos_vals = cos_sims[str(concept)].iloc[test_indices].to_numpy()
        threshold = concept_thresholds[concept][0]
        above_threshold = cos_vals >= threshold  # Boolean array: True if above threshold

        # Create a boolean mask indicating which test indices are in the ground truth for this concept.
        test_in_gt = np.isin(test_indices, list(concept_indices))

        # Calculate counts using vectorized boolean operations
        tp = np.sum(above_threshold & test_in_gt)
        fn = np.sum(~above_threshold & test_in_gt)
        fp = np.sum(above_threshold & ~test_in_gt)
        tn = np.sum(~above_threshold & ~test_in_gt)

        tp_count[concept] = int(tp)
        fn_count[concept] = int(fn)
        fp_count[concept] = int(fp)
        tn_count[concept] = int(tn)

    return fp_count, fn_count, tp_count, tn_count


def compute_concept_metrics(fp_count, fn_count, tp_count, tn_count, concepts):
    metrics = []

    for concept in concepts:
        # Retrieve counts for each concept
        tp = tp_count[concept]
        fp = fp_count[concept]
        tn = tn_count[concept]
        fn = fn_count[concept]

        # Compute precision, recall, accuracy, f1-score, fpr, tpr, tnr, fnr
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Add metrics to the list
        metrics.append(
            {
                "concept": concept,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1": f1,
                "fpr": fpr,
                "tpr": tpr,
                "tnr": tnr,
                "fnr": fnr,
            }
        )

    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(metrics)

    return metrics_df


def print_threshold_eval_results(metrics_df, print_types):
    """
    Print metrics such as counts, rates, and summaries from the DataFrame.
    """
    # Print per-concept metrics
    if "rate" in print_types:
        for _, row in metrics_df.iterrows():
            print(f"Concept: {row['concept']}")
            print(
                f"TPR: {row['tpr']:.4f}, FPR: {row['fpr']:.4f}, TNR: {row['tnr']:.4f}, FNR: {row['fnr']:.4f}\n"
            )

    if "count" in print_types:
        for _, row in metrics_df.iterrows():
            print(f"Concept: {row['concept']}")
            print(
                f"TP: {row['precision']:.4f}, FP: {row['fpr']:.4f}, TN: {row['tnr']:.4f}, FN: {row['fnr']:.4f}\n"
            )

    # Print summary statistics if enabled
    if "summary" in print_types:
        # Top and Bottom Concepts for Precision
        top_precision = metrics_df.sort_values(by="precision", ascending=False).head(5)[
            ["concept", "precision"]
        ]
        bottom_precision = metrics_df.sort_values(by="precision", ascending=True).head(5)[
            ["concept", "precision"]
        ]

        # Top and Bottom Concepts for Recall
        top_recall = metrics_df.sort_values(by="recall", ascending=False).head(5)[
            ["concept", "recall"]
        ]
        bottom_recall = metrics_df.sort_values(by="recall", ascending=True).head(5)[
            ["concept", "recall"]
        ]

        # Top and Bottom Concepts for F1
        top_f1 = metrics_df.sort_values(by="f1", ascending=False).head(5)[["concept", "f1"]]
        bottom_f1 = metrics_df.sort_values(by="f1", ascending=True).head(5)[["concept", "f1"]]

        # Top and Bottom Concepts for FPR
        top_fpr = metrics_df.sort_values(by="fpr", ascending=True).head(5)[["concept", "fpr"]]
        bottom_fpr = metrics_df.sort_values(by="fpr", ascending=False).head(5)[["concept", "fpr"]]

        # Displaying them side by side for each metric
        print(
            "\nBest and Worst 5 Concepts by Precision (how many of the predicted positives are actually correct):"
        )
        print(
            pd.concat(
                [top_precision.reset_index(drop=True), bottom_precision.reset_index(drop=True)],
                axis=1,
            )
        )

        print(
            "\nBest and Worst 5 Concepts by Recall (how many of the actual positives were correctly identified):"
        )
        print(
            pd.concat(
                [top_recall.reset_index(drop=True), bottom_recall.reset_index(drop=True)], axis=1
            )
        )

        print("\nBest and Worst 5 Concepts by F1 (harmonic mean of precision and recall):")
        print(pd.concat([top_f1.reset_index(drop=True), bottom_f1.reset_index(drop=True)], axis=1))

        print(
            "\nBest and Worst 5 Concepts by FPR (how many of the actual negatives were incorrectly predicted as positives):"
        )
        print(
            pd.concat([top_fpr.reset_index(drop=True), bottom_fpr.reset_index(drop=True)], axis=1)
        )


def plot_metric(df, metric, y_min=None, y_max=None):
    """
    Plots either a distribution of a given metric for all concepts
    or individual bars for each concept's metric value.

    Args:
        df (pd.DataFrame): DataFrame containing concept metrics.
        metric (str): The metric to plot (e.g., 'precision', 'recall', 'f1', etc.).
        plot_individual (bool): Whether to plot individual concept bars (True) or a distribution (False).
    """
    # Bar plot for individual concepts, sorted by metric
    sorted_df = df.sort_values(by=metric, ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(x="concept", y=metric, data=sorted_df, palette="viridis")
    plt.xticks(rotation=90)  # Rotate the concept names for better visibility
    plt.title(f"{metric.capitalize()} for Each Concept")
    plt.xlabel("Concept")
    plt.ylabel(f"{metric.capitalize()}")

    # Apply y-axis limit if specified
    if y_max is not None:
        if y_min is not None:
            plt.ylim(y_min, y_max)
        else:
            plt.ylim(0, y_max)

    plt.show()


def plot_metric_distribution(df, metric):
    """
    Plots either a distribution of a given metric.

    Args:
        df (pd.DataFrame): DataFrame containing concept metrics.
        metric (str): The metric to plot (e.g., 'precision', 'recall', 'f1', etc.).
        plot_individual (bool): Whether to plot individual concept bars (True) or a distribution (False).
    """
    # Distribution plot for the selected metric across all concepts
    plt.figure(figsize=(12, 8))
    sns.histplot(df[metric], bins=20, color="purple")
    plt.title(f"Distribution of {metric.capitalize()} Across Concepts")
    plt.xlabel(f"{metric.capitalize()}")
    plt.ylabel("Number of Concepts")
    plt.show()


def compute_avg_rand_mean_and_std(embeddings, patch_indices, n_vectors=5, device="cuda"):
    embeddings = embeddings.to(device)

    # Generate n_vectors random vectors and normalize them
    random_vectors = torch.randn(n_vectors, embeddings.shape[1], device=device)
    random_vectors = F.normalize(random_vectors, p=2, dim=1)  # Normalize each random vector

    # Normalize embeddings before computing cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarities directly between embeddings and random vectors
    cos_sim_matrix = torch.matmul(embeddings, random_vectors.t())  # [n_samples, n_vectors]

    # Select relevant cosine similarities for the given patch indices
    relevant_rand_cos_sims = cos_sim_matrix[patch_indices]

    # Compute mean and std for each patch across all random vectors
    rand_means = relevant_rand_cos_sims.mean(dim=1)  # Mean across random vectors
    rand_stds = relevant_rand_cos_sims.std(dim=1)  # Std across random vectors

    # Calculate the average mean and std over all patches
    avg_mean = rand_means.mean().item()  # Averaging over all patches
    avg_std = rand_stds.mean().item()  # Averaging over all patches

    return avg_mean, avg_std


def compute_concept_cosine_stats(
    gt_patches_per_concept,
    cos_sims,
    embeddings,
    results_to_print=0,
    device="cuda",
    print_random=True,
):
    """
    Computes the mean and standard deviation of cosine similarities for each concept
    based on the patches that are known to contain the concept (using object masks).
    """
    if results_to_print > 0:
        print("Mean and Std of Cossims:")

    # Step 2: Initialize dictionary to store mean and std for each concept
    concept_cosine_stats = {}

    # Step 3: Convert cos_sims DataFrame to tensor
    cos_sims_tensor = torch.tensor(
        cos_sims.values, device=device
    )  # Convert the DataFrame to a tensor
    cos_sims_tensor = (
        cos_sims_tensor.float()
    )  # Ensure it's of float type (important for cos similarity)

    # Step 4: Calculate cosine similarities between embeddings and concepts
    for i, (concept, patch_indices) in enumerate(gt_patches_per_concept.items()):
        # Use precomputed cos_sims_tensor to extract relevant cosine similarities
        relevant_cos_sims = cos_sims_tensor[patch_indices, cos_sims.columns.get_loc(str(concept))]

        # Compute mean and standard deviation for the relevant cosine similarities
        mean_sim = relevant_cos_sims.mean().item()
        std_sim = relevant_cos_sims.std().item()

        # Do the same thing for a random vector (average over n_vectors)
        rand_mean_sim, rand_std_sim = compute_avg_rand_mean_and_std(
            embeddings, patch_indices, n_vectors=5, device=device
        )

        # Store the results in the dictionary
        concept_cosine_stats[concept] = (mean_sim, std_sim, rand_mean_sim, rand_std_sim)

        if i < results_to_print:
            print(f"Concept {concept}: mean cossim={mean_sim:.4f}, std={std_sim:.4f}")
            if print_random:
                print(
                    f"          (random: mean cossim={rand_mean_sim:.4f}, std={rand_std_sim:.4f})"
                )

    return concept_cosine_stats


### Visualizations of Quantitative Results ###
def plot_heatmap(
    concept_names, cosine_similarity_matrix, heatmap_title, save_label=None, dataset_name="CLEVR"
):
    """
    Creates and displays a heatmap of cosine similarities between concept embeddings.

    Args:
        concept_names (list of str): A list of concept names to be displayed on the heatmap axes.
        cosine_similarity_matrix (ndarray): A 2D NumPy array representing the cosine similarity values
        between concept embeddings.
        heatmap_title (str): The title of the heatmap to be displayed.
        save_label (str): label to put in path of saved image.
        dataset_name (str) : Name of the dataset

    Returns:
        None: The function directly displays the heatmap using `matplotlib` and `seaborn`.
    """
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cosine_similarity_matrix,
        xticklabels=concept_names,
        yticklabels=concept_names,
        cmap="coolwarm",
        cbar=True,
        annot=True,
        fmt=".2f",
    )

    plt.title(heatmap_title)

    if save_label:
        save_path = f"../Figs/{dataset_name}/concepts_heatmap/{save_label}.jpg"
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    plt.show()


def concept_heatmap(concept_embeddings, con_label, dataset_name="CLEVR"):
    """
    Plots a heatmap of cosine similarities between concept embeddings from a dataset.

    This function loads the concept embeddings from the specified dataset, selects a subset of
    concepts (up to 10), calculates the cosine similarities between them, and displays the result
    as a heatmap.

    Args:
        concepts_file (str): File where concept dictionary is stored.
        con_label (str): label to put in path of saved image.
        dataset_name (str, optional): The name of the dataset to load concept embeddings from.

    Returns:
        None: The function generates and displays the heatmap.
    """
    concept_names = list(concept_embeddings.keys())
    concept_names.sort()

    # Get concept names and embeddings
    embeddings = torch.stack([concept_embeddings[name] for name in concept_names])

    # Normalize embeddings to unit norm
    # norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    # normalized_embeddings = embeddings / norms

    # compute cosine similarities
    cosine_similarity_matrix = torch.matmul(embeddings, embeddings.T).cpu().numpy()

    heatmap_title = "Cosine Similarity Between Concept Embeddings"
    save_label = f'{heatmap_title.replace(" ", "_")}__{con_label}'
    plot_heatmap(
        concept_names,
        cosine_similarity_matrix,
        heatmap_title,
        dataset_name=dataset_name,
        save_label=save_label,
    )


def concept_heatmap_groupedby_concept(concepts_file, con_label, dataset_name="CLEVR"):
    """
    Plots a heatmap of cosine similarities between concept embeddings grouped by a specific concept category.

    This function allows the user to choose a concept category (e.g., color, shape) and generates a
    heatmap of cosine similarities between the embeddings of concepts in that category.

    Args:
        concepts_file (str): File where concept dictionary is stored.
        con_label (str): label to put in path of saved image.
        dataset_name (str, optional): The name of the dataset to load concept embeddings from.

    Returns:
        None: The function generates and displays the heatmap based on the chosen concept category.
    """
    concept_embeddings = torch.load(f"Concepts/{dataset_name}/{concepts_file}")

    # Have user choose concept category
    potential_concept_categories = [
        key for key in concept_embeddings.keys() if key not in ["class", "image_filename", "split"]
    ]
    concept_category = get_user_category(potential_concept_categories)[0]

    # Make heatmap just based on those categories
    concept_names = [
        key for key in list(concept_embeddings.keys()) if key.startswith(concept_category)
    ]
    embeddings = torch.stack([concept_embeddings[name] for name in concept_names])

    cosine_similarity_matrix = torch.matmul(embeddings, embeddings.T).cpu().numpy()

    heatmap_title = f"Cosine Similarities Between {concept_category} Concepts"
    save_label = f'{heatmap_title.replace(" ", "_")}__{con_label}'
    plot_heatmap(
        concept_names,
        cosine_similarity_matrix,
        heatmap_title,
        dataset_name=dataset_name,
        save_label=save_label,
    )


def concept_heatmap_random_samples(
    concept_embeddings, con_label, num_samples=15, dataset_name="CLEVR"
):
    """
    Plots a heatmap of cosine similarities between a random subset of concept embeddings from a dataset.

    This function loads the concept embeddings from the specified dataset, selects a random subset
    of concepts (up to `num_samples`), calculates the cosine similarities between them, and displays the result
    as a heatmap.

    THE SELECTED EMBEDDINGS ARE NORMALIZED WRT TO EACH OTHER BEFORE COMPUTING THEIR SIMILARITIES

    Args:
        concepts_file (str): File where the concept dictionary is stored.
        con_label (str): Label to include in the path of the saved image.
        num_samples (int, optional): The number of random concepts to sample for visualization. Default is 15.
        dataset_name (str, optional): The name of the dataset to load concept embeddings from.

    Returns:
        None
    """
    # Randomly sample `num_samples` concepts for visualization
    print(f"Sampling {num_samples} random concepts for visualization purposes.")
    concept_names = random.sample(
        list(concept_embeddings.keys()), min(num_samples, len(concept_embeddings))
    )
    concept_names.sort()

    # Extract embeddings for the sampled concepts
    embeddings = torch.stack([concept_embeddings[name] for name in concept_names])

    # Normalize embeddings to unit norm
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norms

    # Now compute the cosine similarity matrix
    cosine_similarity_matrix = (
        torch.matmul(normalized_embeddings, normalized_embeddings.T).cpu().numpy()
    )

    # Heatmap title
    heatmap_title = f"Cosine Similarity Between {num_samples} Random Concept Embeddings"

    # Create and optionally save the heatmap
    save_label = f'{heatmap_title.replace(" ", "_")}__{con_label}'
    plot_heatmap(
        concept_names,
        cosine_similarity_matrix,
        heatmap_title,
        dataset_name=dataset_name,
        save_label=save_label,
    )


def compute_cossim_hist_stats(
    gt_samples_per_concept, cos_sims, dataset_name, percentile, sample_type, all_object_patches=None
):
    """
    Computes in-sample and out-of-sample cosine similarity statistics for each concept, separated by train and test splits.

    Args:
        concept_thresholds (dict): Dictionary mapping concepts to (threshold, random_threshold).
        gt_samples_per_concept (dict): Dictionary mapping concepts to sets of true concept patch indices.
        cos_sims (pd.DataFrame): DataFrame where each column is a concept and rows are patch cosine similarities.
        dataset_name (str): The name of the dataset, used to load the correct metadata file.
        percentile (float): Percentile of in-sample patches to compute the threshold.
        all_object_patches (set, optional): Set of patch indices to consider. If provided, only these patches are considered.

    Returns:
        dict: A dictionary with per-concept cosine similarity stats, separated by train and test splits.
    """
    if sample_type == "patch":
        split_df = get_patch_split_df(dataset_name, patch_size=14, model_input_size=(224, 224))
    else:
        split_df = get_split_df(dataset_name)

    train_mask = split_df == "train"
    test_mask = split_df != "train"

    stats = {"train": {}, "test": {}}

    # Loop over each concept; vectorized operations occur per concept
    for concept, concept_indices in tqdm(gt_samples_per_concept.items()):
        concept = str(concept)
        concept_indices = set(concept_indices)

        # Apply object patches filter if provided
        if all_object_patches is not None:
            concept_indices &= all_object_patches

        # Create a boolean mask for samples belonging to this concept.
        in_gt_mask = cos_sims.index.to_series().isin(concept_indices)
        out_gt_mask = (
            cos_sims.index.to_series().isin(all_object_patches - concept_indices)
            if all_object_patches is not None
            else ~in_gt_mask
        )

        # Get the cosine similarity column for this concept.
        cos_vals = cos_sims[concept]

        # Vectorized extraction of cosine similarity values for each combination.
        in_concept_sims_train = cos_vals[train_mask & in_gt_mask].tolist()
        in_concept_sims_test = cos_vals[test_mask & in_gt_mask].tolist()
        out_concept_sims_train = cos_vals[train_mask & out_gt_mask].tolist()
        out_concept_sims_test = cos_vals[test_mask & out_gt_mask].tolist()

        # Store results for train and test splits.
        stats["train"][concept] = {
            "in_concept_sims": in_concept_sims_train,
            "out_concept_sims": out_concept_sims_train,
        }
        stats["test"][concept] = {
            "in_concept_sims": in_concept_sims_test,
            "out_concept_sims": out_concept_sims_test,
        }

    return stats


def plot_cosine_similarity_histograms(
    stats,
    concept_thresholds,
    sample_type,
    plot_type="both",
    metric_type="Cosine Similarity",
    percentile=None,
    bins=50,
):
    """
    Plots histograms of cosine similarity values for each concept using precomputed statistics.

    Args:
        stats (dict): Dictionary containing in-sample and out-of-sample cosine similarity stats for both train and test splits.
                      Expected structure:
                      {
                        'train': { concept: {'in_concept_sims': [...], 'out_concept_sims': [...]}, ... },
                        'test': { concept: {'in_concept_sims': [...], 'out_concept_sims': [...]}, ... }
                      }
        concept_thresholds (dict): Dictionary mapping concepts to (threshold, random_threshold).
        sample_type (str): Label for the sample type (e.g., "patch" or "image").
        plot_type (str): Option to plot "train", "test", or "both" datasets.
        percentile (float, optional): Percentile value for threshold line.
        bins (int): Number of bins for the histogram.

    Returns:
        None: Displays the histograms.
    """
    # Extract train and test stats
    train_stats = stats["train"]
    test_stats = stats["test"]

    # Use the keys from the train split (assume same keys in test)
    concepts = list(train_stats.keys())
    num_concepts = len(concepts)

    fig, axes = plt.subplots(nrows=num_concepts, figsize=(8, 3 * num_concepts))
    if num_concepts == 1:
        axes = [axes]

    for i, concept in enumerate(concepts):
        ax = axes[i]

        # Retrieve similarity values
        in_concept_sims_train = train_stats[concept]["in_concept_sims"]
        out_concept_sims_train = train_stats[concept]["out_concept_sims"]
        in_concept_sims_test = test_stats[concept]["in_concept_sims"]
        out_concept_sims_test = test_stats[concept]["out_concept_sims"]

        # Create a twin y-axis for in-concept sims
        ax2 = ax.twinx()

        # Plot histograms based on plot_type
        if plot_type in {"both", "train"}:
            n_out_train, _, _ = ax.hist(
                out_concept_sims_train,
                bins=bins,
                alpha=0.6,
                color="lightblue",
                label="Train - Out-of-Concept",
            )
            n_in_train, _, _ = ax2.hist(
                in_concept_sims_train,
                bins=bins,
                alpha=0.6,
                color="lightcoral",
                label="Train - In-Concept",
            )

        if plot_type in {"both", "test"}:
            n_out_test, _, _ = ax.hist(
                out_concept_sims_test,
                bins=bins,
                alpha=0.6,
                color="blue",
                label="Test - Out-of-Concept",
            )
            n_in_test, _, _ = ax2.hist(
                in_concept_sims_test, bins=bins, alpha=0.6, color="red", label="Test - In-Concept"
            )

        # Set axis labels
        ax.set_xlabel(metric_type)
        ax.set_ylabel(f"Out-of-Concept {sample_type} Count", color="blue")
        ax2.set_ylabel(f"In-Concept {sample_type} Count", color="red")

        # Set title and grid
        ax.set_title(f"{concept}")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Dynamically determine y-axis limits based on plotted data
        if plot_type == "both":
            max_out = max(n_out_train.max(), n_out_test.max())
            max_in = max(n_in_train.max(), n_in_test.max())
        elif plot_type == "train":
            max_out = n_out_train.max()
            max_in = n_in_train.max()
        elif plot_type == "test":
            max_out = n_out_test.max()
            max_in = n_in_test.max()

        ax.set_ylim(0, max_out * 1.1)
        ax2.set_ylim(0, max_in * 1.1)

        # Tick colors for clarity
        ax.tick_params(axis="y", colors="blue")
        ax2.tick_params(axis="y", colors="red")

        # Plot percentile threshold if available
        if percentile is not None and concept in concept_thresholds:
            ax.axvline(
                concept_thresholds[concept][0],
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"{percentile:.2f}% Threshold",
            )

        # Build legend handles dynamically
        handles = []
        if plot_type in {"both", "train"}:
            handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, color="lightblue", alpha=0.6, label="Train - Out-of-Concept"
                )
            )
            handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, color="lightcoral", alpha=0.6, label="Train - In-Concept"
                )
            )
        if plot_type in {"both", "test"}:
            handles.append(
                plt.Rectangle((0, 0), 1, 1, color="blue", alpha=0.6, label="Test - Out-of-Concept")
            )
            handles.append(
                plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.6, label="Test - In-Concept")
            )
        if percentile is not None:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"{percentile:.2f}% Threshold",
                )
            )

        ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    plt.show()
