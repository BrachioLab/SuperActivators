import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import torch
import math
from collections import defaultdict
from sklearn.metrics import auc
import sys
sys.path.append(os.path.abspath(".."))
os.chdir('/shared_data0/cgoldberg/Concept_Inversion/Experiments')

from utils.filter_datasets_utils import filter_concept_dict

"""""""""""

General Functions

"""""""""""
def plot_concept_metrics(metric_dfs, metric_name, title, xmin=None, xmax=None):
    """
    Plots a horizontal bar chart comparing a chosen metric across multiple metrics DataFrames.

    Args:
        metric_dfs (dict): Dictionary mapping labels (str) to pd.DataFrame of concept metrics.
        metric_name (str): The metric to plot (e.g., 'accuracy', 'f1').
        title (str): Title of the plot.

    Returns:
        None (Displays the plot).
    """
    labels = list(metric_dfs.keys())  # Extract labels
    metric_dfs = list(metric_dfs.values())  # Extract corresponding DataFrames

    colors = sns.color_palette("husl", len(metric_dfs))  # Generate distinct colors
    
    # Extract all unique concepts from the first DataFrame (assume all have the same concepts)
    concepts = metric_dfs[0]["concept"].tolist()
    
    # Increase spacing by modifying the y positions
    spacing = 0.3  # Adjust spacing factor
    y = np.arange(len(concepts)) * (len(metric_dfs) * 0.2 + spacing)  # Space out concepts
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 20))
    
    # Bar width
    bar_width = 0.2

    # Plot each metric_df's values
    for i, (label, df, color) in enumerate(zip(labels, metric_dfs, colors)):
        values = df.set_index("concept")[metric_name].reindex(concepts).values
        ax.barh(y + i * bar_width, values, height=bar_width, label=label, color=color)
    
    # Formatting
    ax.set_yticks(y + (len(metric_dfs) - 1) * bar_width / 2)
    ax.set_yticklabels(concepts)
    ax.set_xlabel(metric_name.capitalize())
    ax.set_title(title)
    
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    
    # Move legend outside the plot
    ax.legend(title="Source", loc="upper left", bbox_to_anchor=(1, 1))
    
    # Show the plot
    plt.tight_layout()
    plt.show()

    

def plot_average_metrics(metric_dfs, metric_name, title=None, xmin=None, xmax=None):
    """
    Plots a horizontal bar chart comparing average metrics across multiple methods.

    Args:
        metric_dfs (dict): Mapping from label -> pd.DataFrame (with 'concept' and metric_name columns).
        metric_name (str): Metric to plot ('f1', 'accuracy', etc).
        title (str, optional): Title for the plot.
        xmin (float, optional): Minimum x-axis value.
        xmax (float, optional): Maximum x-axis value.
    """
    labels = list(metric_dfs.keys())
    avg_metrics = [df[metric_name].mean() for df in metric_dfs.values()]

    fig, ax = plt.subplots(figsize=(8, len(labels) * 0.7))

    colors = sns.color_palette("husl", len(labels))
    bars = ax.barh(labels, avg_metrics, color=colors)

    # Add text annotations at the end of each bar
    for bar, value in zip(bars, avg_metrics):
        ax.text(
            bar.get_width() + 0.01,  # slightly offset from the end of bar
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel(metric_name.capitalize())
    ax.set_ylabel("Concept Discovery Method")
    if title:
        ax.set_title(title)

    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin if xmin is not None else 0,
                    right=xmax if xmax is not None else 1)

    plt.grid(axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    

def plot_grouped_metrics(data_dict, baseline_dfs, metric, baseline_type, curr_concepts=None, title=None, xmin=0, xmax=1):
    """
    Plots a horizontal grouped bar chart for a chosen metric for each concept,
    for each model and scheme.

    Args:
        data_dict (dict): Nested dictionary of {model: {scheme: DataFrame}}.
        metric (str): The name of the metric column to plot.
        title (str, optional): The plot title.
        xmin (float, optional): Minimum x-axis value.
        xmax (float, optional): Maximum x-axis value.
    """
    models = list(data_dict.keys())
    # Get the set of concepts for each model (assume all scheme dfs have the same concepts)
    model_concepts = {}
    for model in models:
        schemes = list(data_dict[model].keys())
        if schemes:
            df = data_dict[model][schemes[0]]
            if curr_concepts is not None:
                model_concepts[model] = [concept for concept in df['concept'].unique() if concept in curr_concepts]
            else:
                model_concepts[model] = df['concept'].unique().tolist()
        else:
            model_concepts[model] = []
    
    # Gather all schemes across models and create color mapping
    all_schemes = [scheme for scheme in data_dict[models[0]].keys()]
    cmap = plt.get_cmap("tab10")
    scheme_colors = {scheme: cmap(i) for i, scheme in enumerate(all_schemes)}
    baseline_colors = {model: cmap(i+len(all_schemes)) for i, model in enumerate(models)}

    # Determine vertical positions
    y_positions = {}  # mapping (model, concept) -> y coordinate
    model_centers = {}  # mapping model -> vertical center (for model label)
    current_y = 0
    model_gap = 0.7  # gap between models

    for model in models:
        concepts = model_concepts[model]
        for concept in concepts:
            y_positions[(model, concept)] = current_y
            current_y += 1
        model_centers[model] = np.mean([y_positions[(model, c)] for c in concepts]) if concepts else current_y
        current_y += model_gap  # add gap after each model

    # Plotting setup
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot bars for each (model, concept)
    for model in models:
        baseline_model_df = baseline_dfs[model]
        for concept in model_concepts[model]:
                
            applicable_schemes = []
            values = {}

            # Collect metric values for each scheme
            for scheme in all_schemes:
                if scheme in data_dict[model]:
                    df = data_dict[model][scheme]
                    row = df[df['concept'] == concept]
                    if not row.empty:
                        values[scheme] = row.iloc[0][metric]
                        applicable_schemes.append(scheme)

            if not applicable_schemes:
                continue

            # Bar height calculation and placement
            bar_height = 0.6 / len(applicable_schemes)
            center_y = y_positions[(model, concept)]
            for i, scheme in enumerate(applicable_schemes):
                offset = (i - (len(applicable_schemes) - 1) / 2) * bar_height
                ax.barh(center_y + offset, values[scheme], height=bar_height, color=scheme_colors[scheme],
                        label=scheme if (model == models[0] and concept == model_concepts[model][0]) else None)
                
            offset = (len(applicable_schemes) - (len(applicable_schemes) - 1) / 2) * bar_height
            baseline_val = baseline_model_df[baseline_model_df['concept'] == concept].iloc[0][metric]
            ax.barh(center_y + offset, baseline_val, height=bar_height, color=baseline_colors[model], label=f"{model} {baseline_type}")

            # Align concept labels with the center of the bars
            # Get the position for the middle of the bars for this concept
            bar_positions = []
            for i, scheme in enumerate(applicable_schemes):
                bar_positions.append(center_y + (i - (len(applicable_schemes) - 1) / 2) * bar_height)

            # The position of the concept label is aligned with the center of the bars
            concept_label_x = xmin - 0.01 * (xmax - xmin)
            concept_label_y = np.mean(bar_positions)  # The average position of all the bars for this concept
            ax.text(concept_label_x, concept_label_y, concept, va='center', ha='right', fontsize=10)

    # Remove y-ticks
    ax.set_yticks([])

    ax.set_xlabel(metric, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)

    # Position model labels
    current_xlim = ax.get_xlim()
    model_label_x = xmin - 0.1 * (xmax - xmin)
    for model in models:
        ax.text(model_label_x, model_centers[model], model, va='center', ha='right', fontsize=12, fontweight='bold')

    # Create legend
     # Retrieve the handles and labels created by the plotting calls.
    handles, labels = ax.get_legend_handles_labels()
    
    # Separate sub-label entries from baseline entries.
    sub_label_handles = []
    sub_label_labels = []
    baseline_handles = []
    baseline_labels = []
    
    for h, l in zip(handles, labels):
        # We assume baseline entries contain the baseline_type string.
        if baseline_type in l:
            if l not in baseline_labels:
                baseline_handles.append(h)
                baseline_labels.append(l)
        else:
            if l not in sub_label_labels:
                sub_label_handles.append(h)
                sub_label_labels.append(l)
    
    # Reverse the order of sub-label entries and baseline entries individually.
    sub_label_handles = list(sub_label_handles)[::-1]
    sub_label_labels = list(sub_label_labels)[::-1]
    baseline_handles = list(baseline_handles)[::-1]
    baseline_labels = list(baseline_labels)[::-1]
    
    # Combine: first the reversed sub-labels, then the reversed baseline entries.
    ordered_handles = baseline_handles + sub_label_handles
    ordered_labels = baseline_labels + sub_label_labels 
    
    # Create the legend with the reordered entries.
    ax.legend(ordered_handles, ordered_labels, title="Legend",
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    

    plt.show()



def plot_average_grouped_metrics(data_dict, baseline_dfs, metric, baseline_type, title=None, xmin=None, xmax=None):
    """
    Plots a horizontal grouped bar chart for a chosen metric averaged across concepts,
    including baseline values computed from a dictionary of baseline dataframes.

    Args:
        data_dict (dict): A dictionary where keys are overall labels and values are dictionaries.
                          The inner dictionary has sub-labels as keys and dataframes as values.
                          Each dataframe must have a column corresponding to the chosen metric.
        baseline_dfs (dict): A dictionary mapping each overall label to a dataframe.
                             Each dataframe must have a column corresponding to the chosen metric.
        metric (str): The column name for which the average is computed.
        title (str, optional): The title of the plot.
        xmin (float, optional): Minimum x-axis value.
        xmax (float, optional): Maximum x-axis value.
    """
    # Get overall labels and assume that each inner dict has the same sub-labels
    # overall_labels = list(data_dict.keys())[::-1]
    overall_labels = list(data_dict.keys())
    sub_labels = list(next(iter(data_dict.values())).keys())
    
    # Compute the average of the chosen metric for each (overall, sub) pair from data_dict
    averages = {
        ov_label: {
            sub_label: df[metric].mean() 
            for sub_label, df in sub_dict.items()
        } 
        for ov_label, sub_dict in data_dict.items()
    }
    
    # Compute baseline average metric values over concepts for each overall label.
    baseline_values = {
        ov_label: baseline_dfs[ov_label][metric].mean() 
        for ov_label in overall_labels if ov_label in baseline_dfs
    }
    
    # Set up plotting parameters for grouped horizontal bars.
    n_groups = len(overall_labels)
    n_sub = len(sub_labels)
    # Allocate space for sub-label bars and one extra for baseline within each group.
    bar_height = 0.8 / (n_sub + 1)
    
    # Create a color map for the sub-labels.
    cmap = plt.get_cmap("tab10")
    colors = {sub_label: cmap(i) for i, sub_label in enumerate(sub_labels)}
    baseline_colors = {ov_label: cmap(i+len(sub_labels)) for i, ov_label in enumerate(overall_labels)}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine y positions for each overall group.
    y_group_positions = np.arange(n_groups)
    
    # Plot bars for each overall label.
    for i, ov_label in enumerate(overall_labels):
        group_center = y_group_positions[i]
        start_y = group_center - 0.2  # Center the group vertically around the integer position.
        
        # Plot bars for each sub-label
        for j, sub_label in enumerate(sub_labels):
            if sub_label in data_dict[ov_label].keys():
                y = start_y + j * bar_height
                value = averages[ov_label][sub_label]
                ax.barh(y, value, height=bar_height, color=colors[sub_label],
                        label=sub_label if i == 0 else None)
                           
        # Plot baseline bar (placed as the last bar in the group)
        baseline_y = start_y + n_sub * bar_height
        baseline_val = baseline_values.get(ov_label, 0)
        ax.barh(baseline_y, baseline_val, height=bar_height,
                alpha=0.6, color=baseline_colors[ov_label], label=f'{ov_label} {baseline_type}')
    
    # Set y-axis ticks to show overall labels (center of each group)
    ax.set_yticks(y_group_positions)
    ax.set_yticklabels(overall_labels, fontsize=12)
    
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    
    ax.set_xlabel(metric, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    
     # Retrieve the handles and labels created by the plotting calls.
    handles, labels = ax.get_legend_handles_labels()
    
    # Separate sub-label entries from baseline entries.
    sub_label_handles = set()
    sub_label_labels = set()
    baseline_handles = set()
    baseline_labels = set()
    
    for h, l in zip(handles, labels):
        # We assume baseline entries contain the baseline_type string.
        if baseline_type in l:
            baseline_handles.add(h)
            baseline_labels.add(l)
        else:
            sub_label_handles.add(h)
            sub_label_labels.add(l)
    
    # Reverse the order of sub-label entries and baseline entries individually.
    sub_label_handles = list(sub_label_handles)[::-1]
    sub_label_labels = list(sub_label_labels)[::-1]
    baseline_handles = list(baseline_handles)[::-1]
    baseline_labels = list(baseline_labels)[::-1]
    
    # Combine: first the reversed sub-labels, then the reversed baseline entries.
    ordered_handles = baseline_handles + sub_label_handles
    ordered_labels = baseline_labels + sub_label_labels 
    
    # Create the legend with the reordered entries.
    ax.legend(ordered_handles, ordered_labels, title="Legend",
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to leave space for the legend
    plt.show()



"""""""""""

F1 Detection 

"""""""""""
def get_per_concept_prompt_scores(dataset_name, metric):
    """
    Load prompt detection scores from a saved CSV file for each concept.

    Args:
        dataset_name (str): Name of the dataset.
        metric (str): 'f1', 'tpr', or 'fpr'.

    Returns:
        dict: A dictionary mapping concept to its score.
    """
    prompt_results_dir=f'/shared_data0/cgoldberg/Concept_Inversion/Experiments/prompt_results/{dataset_name}'
    csv_files = [f for f in os.listdir(prompt_results_dir) if f.endswith('_f1_scores.csv') and dataset_name in f]

    if not csv_files:
        n_concepts = {
            'Broden-Pascal': 94,
            'Broden-OpenSurfaces': 48
        }.get(dataset_name, 46)
        return {f'concept_{i}': 0.5 for i in range(n_concepts)}

    csv_file = os.path.join(prompt_results_dir, csv_files[0])
    df = pd.read_csv(csv_file)

    return filter_concept_dict(dict(zip(df['concept'], df[metric])), dataset_name)

def get_weighted_prompt_score(dataset_name, model_name, metric, split='test'):
    """
    Calculates a single weighted prompt score for a given metric.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model.
        metric (str): 'f1', 'tpr', or 'fpr'.
        split (str, optional): Dataset split. Defaults to 'test'.

    Returns:
        float: The weighted prompt score.
    """
    per_concept_scores = get_per_concept_prompt_scores(dataset_name, metric)

    # === Load ground-truth counts for weighted average
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
    else:
        raise ValueError("Unknown model_name")

    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    gt_concepts = set(gt_samples_per_concept.keys())

    weighted_sum = 0
    total_samples = 0
    for concept, score in per_concept_scores.items():
        if concept in gt_concepts:
            count = len(gt_samples_per_concept[concept])
            weighted_sum += score * count
            total_samples += count

    return weighted_sum / total_samples if total_samples > 0 else 0.0


def get_prompt_scores(dataset_name, model_name, metric,
                      weighted_avg=True, split='test'):
    """
    Load prompt detection scores from saved CSV files and optionally compute weighted average.

    Args:
        dataset_name: Name of the dataset
        metric: 'f1', 'tpr', or 'fpr'
        weighted_avg: Whether to compute weighted average based on GT sample counts
        split: Dataset split (e.g., 'test')
        model_name: Needed to determine correct input size for gt_samples_per_concept

    Returns:
        Dict of concept -> score (if f1), or scalar for tpr/fpr
    """
    if weighted_avg:
        return get_weighted_prompt_score(dataset_name, model_name, metric, split)
    else:
        scores = get_per_concept_prompt_scores(dataset_name, metric)
        if metric == 'f1':
            return scores
        else:
            return sum(scores.values()) / len(scores) if scores else 0.0


def plot_predictions_vs_percentiles(dataset_name, model_name, sample_type, concept, scheme='avg', split='test'):
    """
    For a given concept and scheme, plots:
    - Predicted Positives and Predicted Negatives over percentiles
    - Horizontal lines for Ground Truth Positives and Negatives
    """
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # === Construct con_label based on scheme
    n_clusters = 1000 if sample_type == 'patch' else 50

    if scheme == 'avg':
        con_labels = {
            f'labeled {sample_type} avg': f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
        }
    elif scheme == 'linsep':
        con_labels = {
            f'labeled {sample_type} linsep': f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100'
        }
    elif scheme == 'kmeans':
        con_labels = {
            f'unsupervised {sample_type} kmeans': f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_100'
        }
    elif scheme == 'kmeans linsep':
        con_labels = {
            f'unsupervised {sample_type} linsep kmeans': f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_100'
        }
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    # === Plotting
    plt.figure(figsize=(10, 6))

    for method_name, con_label in con_labels.items():
        pred_pos_counts = []
        pred_neg_counts = []
        gt_pos = None
        gt_neg = None
        valid_percentiles = []

        for pct in percentiles:
            try:
                df = torch.load(
                    f'Quant_Results/{dataset_name}/detectionmetrics_per_{pct}_{con_label}.pt',
                    weights_only=False
                )
            except FileNotFoundError:
                continue

            df = df[df['concept'] == concept]
            if df.empty:
                continue

            row = df.iloc[0]
            tp, tn, fp, fn = int(row['tp']), int(row['tn']), int(row['fp']), int(row['fn'])

            pred_pos_counts.append(tp + fp)
            pred_neg_counts.append(tn + fn)
            valid_percentiles.append(pct)

            if gt_pos is None:
                gt_pos = tp + fn
                gt_neg = tn + fp

        if pred_pos_counts:
            plt.plot(valid_percentiles, pred_pos_counts, marker='o', label=f"{method_name} - Pred Pos")
            plt.plot(valid_percentiles, pred_neg_counts, marker='s', label=f"{method_name} - Pred Neg")

    if gt_pos is not None:
        plt.axhline(y=gt_pos, color='green', linestyle='--', label=f"GT Positives = {gt_pos}")
    if gt_neg is not None:
        plt.axhline(y=gt_neg, color='purple', linestyle='--', label=f"GT Negatives = {gt_neg}")

    plt.title(f"Predicted Counts vs Percentile for Concept: '{concept}' ({scheme})")
    plt.xlabel("Percentile Threshold")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_detection_scores(dataset_name, split, model_name, sample_types, metric='f1', weighted_avg=True,
                          plot_type='both'):
    save_path = f'../Figs/Paper_Figs/{model_name}_{dataset_name}_detectplot.pdf'
    plt.figure(figsize=(10, 7))
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]
    plt.rcParams.update({'font.size': 8})

    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        if plot_type in ('supervised', 'both'):
            con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
            con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100'
        if plot_type in ('unsupervised', 'both'):
            con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_100'
            con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_100'

    style_map = {
        'labeled patch avg': {'color': 'orchid', 'type': 'supervised', 'label': 'patch avg'},
        'labeled patch linsep': {'color': 'indigo', 'type': 'supervised', 'label': 'patch linsep'},
        'labeled cls avg': {'color': 'goldenrod', 'type': 'supervised', 'label': 'cls avg'},
        'labeled cls linsep': {'color': 'orangered', 'type': 'supervised', 'label': 'cls linsep'},
        'unsupervised patch kmeans': {'color': 'orchid', 'type': 'unsupervised', 'label': 'patch avg'},
        'unsupervised patch linsep kmeans': {'color': 'indigo', 'type': 'unsupervised', 'label': 'patch linsep'},
        'unsupervised cls kmeans': {'color': 'goldenrod', 'type': 'unsupervised', 'label': 'cls avg'},
        'unsupervised cls linsep kmeans': {'color': 'orangered', 'type': 'unsupervised', 'label': 'cls linsep'},
    }

    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    elif model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    # Plot prompt scores
    try:
        prompt_score = get_weighted_prompt_score(dataset_name, model_name, metric, split)
        plt.axhline(prompt_score, color='#8B4513', linestyle='-.', linewidth=2, label="Prompt")
    except Exception:
        pass

    # Baseline CSVs: random, always_yes, always_no
    baseline_style_map = {
        'random':     {'color': '#888888', 'label': 'Random'},
        'always_yes': {'color': '#bbbbbb', 'label': 'Always Pos'},
        'always_no':  {'color': '#dddddd', 'label': 'Always Neg'}
    }

    for baseline_type in ['random', 'always_yes', 'always_no']:
        style = baseline_style_map[baseline_type]
        baseline_path = f'Quant_Results/{dataset_name}/{baseline_type}_{model_name}_cls_baseline.csv'
        if not os.path.exists(baseline_path):
            continue
        df = pd.read_csv(baseline_path)
        df = df[df['concept'].isin(gt_samples_per_concept)]
        if weighted_avg:
            total = sum(len(gt_samples_per_concept[c]) for c in df['concept'])
            score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in df.iterrows()) / total
        else:
            score = df[metric].mean()
        plt.axhline(score, color=style['color'], linestyle='-.', linewidth=1.5, label=style['label'])

    seen_labels = set()
    for name, con_label in con_labels.items():
        scores = []
        for percentile in percentiles:
            detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
            detection_metrics = detection_metrics[detection_metrics['concept'].isin(gt_samples_per_concept)]
            if weighted_avg:
                total = sum(len(gt_samples_per_concept[c]) for c in detection_metrics['concept'])
                score = sum(row[metric] * len(gt_samples_per_concept[row['concept']]) for _, row in detection_metrics.iterrows()) / total
            else:
                score = detection_metrics[metric].mean()
            scores.append(score)

        style = style_map[name]
        color = style['color']
        kind = style['type']
        label = style['label']
        linestyle = ':' if plot_type == 'both' and kind == 'unsupervised' else '-'
        plot_label = label if label not in seen_labels else None
        seen_labels.add(label)
        plt.plot(percentiles, scores, color=color, linestyle=linestyle, marker='o', markersize=4, label=plot_label)

    plt.xlabel("Concept Recall Percentage", fontsize=14)
    plt.ylabel(f"{metric.upper()} Score", fontsize=14)
    plt.ylim(0, 1.05)
    plt.xlim(0, 1)
    plt.xticks(np.linspace(0, 1.0, 11), [f"{int(x*100)}%" for x in np.linspace(0, 1.0, 11)])
    if dataset_name == 'Stanford-Tree-Bank':
        plt.title("Sentence-Level Detection Performance", fontweight='bold', fontsize=14)
    elif 'Sarcasm' in dataset_name:
        plt.title("Paragraph-Level Detection Performance", fontweight='bold', fontsize=14)
    else:
        plt.title("Image-Level Detection Performance", fontweight='bold', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend(title="Concept Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(save_path, dpi=500, format='pdf', bbox_inches='tight')
    plt.show()


def plot_detection_scores_per_concept(dataset_name, split, model_name, sample_types, metric='f1',
                                      concepts_to_plot=None, plot_type='both', n_cols=3):
    """
    Creates a separate subplot for each concept showing detection performance across percentiles.

    Args:
        dataset_name (str): Name of the dataset
        split (str): 'train' or 'test' split
        model_name (str): Model name (e.g., 'CLIP', 'Llama')
        sample_types (list): List of sample types (e.g., ['patch', 'cls'])
        metric (str): Metric to plot (default: 'f1')
        concepts_to_plot (list, optional): List of specific concepts to plot. If None, plots all concepts.
        plot_type (str): 'supervised', 'unsupervised', or 'both'
        n_cols (int): Number of columns in the subplot grid
    """
    plt.rcParams.update({'font.size': 8})
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95]

    # Load ground truth samples
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    elif model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)

    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    # Determine which concepts to plot
    if concepts_to_plot is None:
        concepts_to_plot = sorted(filter_concept_dict(gt_samples_per_concept, dataset_name).keys())
    else:
        concepts_to_plot = [c for c in concepts_to_plot if c in gt_samples_per_concept]


    if not concepts_to_plot:
        print("No valid concepts to plot.")
        return

    # Setup concept labels
    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        if plot_type in ('supervised', 'both'):
            con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
            con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100'
        if plot_type in ('unsupervised', 'both'):
            con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_100'
            con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_100'

    # Style mapping
    style_map = {
        'labeled patch avg': {'color': 'orchid', 'type': 'supervised', 'label': 'patch avg'},
        'labeled patch linsep': {'color': 'indigo', 'type': 'supervised', 'label': 'patch linsep'},
        'labeled cls avg': {'color': 'goldenrod', 'type': 'supervised', 'label': 'cls avg'},
        'labeled cls linsep': {'color': 'orangered', 'type': 'supervised', 'label': 'cls linsep'},
        'unsupervised patch kmeans': {'color': 'orchid', 'type': 'unsupervised', 'label': 'patch avg'},
        'unsupervised patch linsep kmeans': {'color': 'indigo', 'type': 'unsupervised', 'label': 'patch linsep'},
        'unsupervised cls kmeans': {'color': 'goldenrod', 'type': 'unsupervised', 'label': 'cls avg'},
        'unsupervised cls linsep kmeans': {'color': 'orangered', 'type': 'unsupervised', 'label': 'cls linsep'},
    }

    # Calculate subplot layout
    n_concepts = len(concepts_to_plot)
    n_rows = math.ceil(n_concepts / n_cols)

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_concepts == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    # Get prompt scores if available
    try:
        prompt_scores = get_per_concept_prompt_scores(dataset_name, metric)
    except Exception:
        prompt_scores = None

    # Plot for each concept
    for idx, concept in enumerate(concepts_to_plot):
        ax = axes[idx]

        # Plot prompt baselines if available
        if prompt_scores and metric == 'f1' and concept in prompt_scores:
            ax.axhline(prompt_scores[concept], color='#8B4513', linestyle='-.', linewidth=1.5,
                       label="Prompt" if idx == 0 else None)

        # Plot baseline CSVs
        baseline_style_map = {
            'random': {'color': '#888888', 'label': 'Random'},
            'always_yes': {'color': '#bbbbbb', 'label': 'Always Pos'},
            'always_no': {'color': '#dddddd', 'label': 'Always Neg'}
        }

        for baseline_type in ['random', 'always_yes', 'always_no']:
            baseline_path = f'Quant_Results/{dataset_name}/{baseline_type}_{model_name}_cls_baseline.csv'
            
            # Fallback to simpler naming for some datasets
            if not os.path.exists(baseline_path):
                baseline_path = f'Quant_Results/{dataset_name}/{baseline_type}.csv'
                
            if os.path.exists(baseline_path):
                df = pd.read_csv(baseline_path)
                concept_row = df[df['concept'] == concept]
                if not concept_row.empty:
                    score = concept_row.iloc[0][metric]
                    style = baseline_style_map[baseline_type]
                    ax.axhline(score, color=style['color'], linestyle='-.', linewidth=1,
                               label=style['label'] if idx == 0 else None)

        # Plot concept discovery methods
        for name, con_label in con_labels.items():
            scores = []
            valid_percentiles = []

            for percentile in percentiles:
                try:
                    detection_metrics = torch.load(
                        f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt',
                        weights_only=False
                    )
                    concept_metrics = detection_metrics[detection_metrics['concept'] == concept]
                    if not concept_metrics.empty:
                        scores.append(concept_metrics.iloc[0][metric])
                        valid_percentiles.append(percentile)
                except FileNotFoundError:
                    continue

            if scores:
                style = style_map.get(name, {})
                color = style.get('color', 'gray')
                kind = style.get('type', 'supervised')
                label = style.get('label', name)
                linestyle = ':' if plot_type == 'both' and kind == 'unsupervised' else '-'

                # Only show label in first subplot to avoid legend duplication
                plot_label = label if idx == 0 else None
                ax.plot(valid_percentiles, scores, color=color, linestyle=linestyle,
                        marker='o', markersize=3, label=plot_label)

        # Formatting for each subplot
        ax.set_xlabel("Concept Recall %", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} Score", fontsize=10)
        ax.set_title(f"{concept}", fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1.0, 6))
        ax.set_xticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1.0, 6)])
        ax.grid(True, linestyle='--', linewidth=0.3)

    # Remove empty subplots
    for idx in range(n_concepts, len(axes)):
        fig.delaxes(axes[idx])

    # Add overall title
    fig.suptitle(f"Detection Performance by Concept - {model_name} on {dataset_name}",
                 fontsize=14, fontweight='bold')

    # Add legend to the first subplot or figure
    if n_concepts > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
                       title="Method", fontsize=8)

    plt.tight_layout()
    plt.show()


def summarize_best_detection_scores(dataset_name, split, model_name, sample_types, metric='f1', weighted_avg=True):
    """
    Returns a DataFrame summarizing the best detection score (and the percentile it occurs at)
    for each concept discovery method.
    """
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # === Construct concept label mappings
    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
        con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100'
        con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_100'
        con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_100'

    # === Load ground-truth labels
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
        gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    elif model_name == 'CLIP':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt', weights_only=False)
    elif model_name == 'Llama':
        gt_samples_per_concept = torch.load(f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt', weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)
    
    results = []

    for name, con_label in con_labels.items():
        best_score = -1
        best_percentile = None
        for percentile in percentiles:
            try:
                detection_metrics = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
            except FileNotFoundError:
                continue

            detection_metrics = detection_metrics[detection_metrics['concept'].isin(gt_samples_per_concept.keys())]

            if weighted_avg:
                total_samples = sum(len(gt_samples_per_concept[c]) for c in detection_metrics['concept'])
                weighted_sum = 0

                for _, row in detection_metrics.iterrows():
                    n_samples = len(gt_samples_per_concept[row['concept']])
                    weighted_sum += row[metric] * n_samples

                score = weighted_sum / total_samples if total_samples > 0 else 0
            else:
                score = detection_metrics[metric].mean()

            if score > best_score:
                best_score = score
                best_percentile = percentile

        results.append({
            'Method': name,
            f'Best {metric.upper()}': round(best_score, 4),
            'Percentile': best_percentile
        })

    return pd.DataFrame(results)


def summarize_best_detection_scores_per_concept(dataset_name, split, model_name, sample_types, metric='f1'):
    """
    Returns a DataFrame showing, for each concept and discovery method,
    the best detection score and the corresponding percentile.

    Columns: [concept, method, best_<metric>, percentile]
    """
    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # === Build concept discovery labels
    con_labels = {}
    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == 'patch' else 50
        con_labels[f'labeled {sample_type} avg'] = f'{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100'
        con_labels[f'labeled {sample_type} linsep'] = f'{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100'
        con_labels[f'unsupervised {sample_type} kmeans'] = f'{model_name}_kmeans_{n_clusters}_{sample_type}_embeddings_kmeans_percentthrumodel_100'
        con_labels[f'unsupervised {sample_type} linsep kmeans'] = f'{model_name}_kmeans_{n_clusters}_linsep_{sample_type}_embeddings_kmeans_percentthrumodel_100'

    # === Load GT concepts
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt'
    gt_concepts = torch.load(gt_path, weights_only=False)
    gt_concepts = filter_concept_dict(gt_concepts, dataset_name)
    results = []

    for method_name, con_label in con_labels.items():
        per_concept_best = defaultdict(lambda: (-1, None))  # concept -> (best_score, best_percentile)

        for percentile in percentiles:
            try:
                df = torch.load(f'Quant_Results/{dataset_name}/detectionmetrics_per_{percentile}_{con_label}.pt', weights_only=False)
            except FileNotFoundError:
                continue

            df = df[df['concept'].isin(gt_concepts.keys())]

            for _, row in df.iterrows():
                c = row['concept']
                score = row[metric]

                if score > per_concept_best[c][0]:
                    per_concept_best[c] = (score, percentile)

        for concept, (score, percentile) in per_concept_best.items():
            results.append({
                'concept': concept,
                'method': method_name,
                f'best_{metric}': round(score, 4),
                'percentile': percentile
            })

    return pd.DataFrame(results)
    
""" Inversions """
def compare_best_schemes(metric_type, concept_schemes, dataset_name, model_name,
                         justobj=False, superdetector_inversion=False, xmin=None, xmax=None, weighted_avg=True,
                         include_baselines=True):

    dir = f'Quant_Results/{dataset_name}'
    best_metric_dfs = {}

    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(560, 560).pt'
    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)

    for concept_scheme in concept_schemes:
        # Use pre-calibrated optimal results
        if superdetector_inversion:
            path_prefix = 'superpatch_avg_inv_'
        else:
            path_prefix = ''
            
        if concept_scheme == 'avg':
            file_name = f"{path_prefix}{model_name}_avg_patch_embeddings_percentthrumodel_100_optimal_test.csv"
        elif concept_scheme == 'linsep':
            file_name = f"{path_prefix}{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100_optimal_test.csv"
        elif concept_scheme == 'unsupervised kmeans':
            file_name = f"{path_prefix}{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100_optimal_test.csv"
        elif concept_scheme == 'unsupervised kmeans linsep':
            file_name = f"{path_prefix}{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100_optimal_test.csv"
        else:
            raise ValueError(f"Unrecognized concept scheme: {concept_scheme}")

        metric_path = f'{dir}/{file_name}'
        
        try:
            df = pd.read_csv(metric_path)
            df = df[df['concept'].isin(gt_samples_per_concept)]
            
            # Load optimal percentiles for display
            optimal_path = f"Detect_Invert_Thresholds/{dataset_name}/optimal_f1_{model_name}"
            if concept_scheme == 'avg':
                optimal_path += "_avg_patch_embeddings_percentthrumodel_100.pt"
            elif concept_scheme == 'linsep':
                optimal_path += "_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100.pt"
            elif concept_scheme == 'unsupervised kmeans':
                optimal_path += "_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100.pt"
            elif concept_scheme == 'unsupervised kmeans linsep':
                optimal_path += "_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100.pt"
            
            # Don't show percentiles since they vary by concept
            label = f"{concept_scheme}"
            best_metric_dfs[label] = df
            
        except FileNotFoundError:
            print(f"Warning: Optimal calibration file not found {metric_path}, skipping.")
            continue

    if include_baselines:
        for baseline_type in ['random', 'always_positive', 'always_negative']:
            baseline_path = f'{dir}/inversion_{baseline_type}_{model_name}_patch_baseline.csv'
            try:
                baseline_df = pd.read_csv(baseline_path)
                baseline_df = baseline_df[baseline_df['concept'].isin(gt_samples_per_concept)]

                if weighted_avg:
                    total_samples = sum(len(gt_samples_per_concept[c]) for c in baseline_df['concept'])
                    weighted_sum = sum(row[metric_type] * len(gt_samples_per_concept[row['concept']])
                                       for _, row in baseline_df.iterrows())
                    avg_metric = weighted_sum / total_samples if total_samples > 0 else 0
                else:
                    avg_metric = baseline_df[metric_type].mean()

                best_metric_dfs[f"Inversion Baseline\n({baseline_type})"] = baseline_df
                print(f"Added {baseline_type} baseline with avg {metric_type}: {avg_metric:.3f}")

            except FileNotFoundError:
                print(f"Warning: Baseline file not found {baseline_path}")

    title = f"Best {metric_type.capitalize()} for {model_name} Patch Schemes on {dataset_name}"
    if superdetector_inversion:
        title += "\n(Local Superdetector Inversion)"
    if include_baselines:
        title += "\n(Including Inversion Baselines)"

    plot_average_metrics(best_metric_dfs, metric_type, title=title, xmin=xmin, xmax=xmax)


def compare_best_schemes_per_concept(metric_type, concept_schemes, dataset_name, model_name,
                                     justobj=False,
                                     superdetector_inversion=False, concepts_to_plot=None,
                                     xmin=None, xmax=None, n_cols=3, include_baselines=True):

    base_dir = f'Quant_Results/{dataset_name}'

    # Load and filter GT
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(560, 560).pt'
    gt_samples_per_concept = torch.load(gt_path, weights_only=False)
    gt_samples_per_concept = filter_concept_dict(gt_samples_per_concept, dataset_name)

    concept_to_best_metrics = {}

    for concept_scheme in concept_schemes:
        # Use pre-calibrated optimal results
        if superdetector_inversion:
            path_prefix = 'superpatch_avg_inv_'
        else:
            path_prefix = ''
            
        if concept_scheme == 'avg':
            file_name = f"{path_prefix}{model_name}_avg_patch_embeddings_percentthrumodel_100_optimal_test.csv"
            optimal_file = f"optimal_f1_{model_name}_avg_patch_embeddings_percentthrumodel_100.pt"
        elif concept_scheme == 'linsep':
            file_name = f"{path_prefix}{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100_optimal_test.csv"
            optimal_file = f"optimal_f1_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100.pt"
        elif concept_scheme == 'unsupervised kmeans':
            file_name = f"{path_prefix}{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100_optimal_test.csv"
            optimal_file = f"optimal_f1_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100.pt"
        elif concept_scheme == 'unsupervised kmeans linsep':
            file_name = f"{path_prefix}{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100_optimal_test.csv"
            optimal_file = f"optimal_f1_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100.pt"
        else:
            raise ValueError(f"Unrecognized concept scheme: {concept_scheme}")

        metric_path = f'{base_dir}/{file_name}'
        optimal_path = f'Detect_Invert_Thresholds/{dataset_name}/{optimal_file}'
        
        try:
            df = pd.read_csv(metric_path)
            df = df[df['concept'].isin(gt_samples_per_concept)]
            
            # Load optimal percentiles
            try:
                optimal_data = torch.load(optimal_path, weights_only=False)
            except:
                optimal_data = {}
            
            best_metrics_per_concept = {}
            for idx, row in df.iterrows():
                concept = row['concept']
                score = row[metric_type]
                
                # Get the optimal percentiles for this concept
                if concept in optimal_data:
                    detect_pct = optimal_data[concept].get('detect_percentile', 'N/A')
                    invert_pct = optimal_data[concept].get('invert_percentile', 'N/A')
                else:
                    detect_pct = 'N/A'
                    invert_pct = 'N/A'
                    
                best_metrics_per_concept[concept] = (score, detect_pct, invert_pct)
                
            concept_to_best_metrics[concept_scheme] = best_metrics_per_concept
            
        except FileNotFoundError:
            print(f"Warning: Optimal calibration file not found {metric_path}, skipping.")
            concept_to_best_metrics[concept_scheme] = {}

    if include_baselines:
        for baseline_type in ['random', 'always_positive', 'always_negative']:
            baseline_path = f'{base_dir}/inversion_{baseline_type}_{model_name}_patch_baseline.csv'
            baseline_metrics_per_concept = {}

            try:
                baseline_df = pd.read_csv(baseline_path)
                baseline_df = baseline_df[baseline_df['concept'].isin(gt_samples_per_concept)]
                for idx, row in baseline_df.iterrows():
                    concept = row['concept']
                    score = row[metric_type]
                    baseline_metrics_per_concept[concept] = (score, 'N/A', 'N/A')

                concept_to_best_metrics[f'Baseline ({baseline_type})'] = baseline_metrics_per_concept
                print(f"Added {baseline_type} baseline")

            except FileNotFoundError:
                print(f"Warning: Baseline file not found {baseline_path}")

    # Plotting
    all_concepts = set()
    for scheme_best in concept_to_best_metrics.values():
        all_concepts.update(scheme_best.keys())

    if concepts_to_plot is None:
        concepts_to_plot = sorted(list(all_concepts))

    n_concepts = len(concepts_to_plot)
    n_rows = math.ceil(n_concepts / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, concept in enumerate(concepts_to_plot):
        ax = axes[idx]
        scores = []
        labels = []

        for concept_scheme in concept_to_best_metrics.keys():
            if concept in concept_to_best_metrics[concept_scheme]:
                score, detect, invert = concept_to_best_metrics[concept_scheme][concept]
                scores.append(score)
                if detect == 'N/A':
                    labels.append(f"{concept_scheme}")
                else:
                    labels.append(f"{concept_scheme}\n(detect={detect}, invert={invert})")
            else:
                scores.append(0.0)
                labels.append(f"{concept_scheme}\n(not found)")

        colors = sns.color_palette("husl", len(labels))
        bars = ax.barh(labels, scores, color=colors)

        for bar, value in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}", va='center', ha='left', fontsize=9, fontweight='bold')

        ax.set_title(f"{concept}", fontsize=11)
        ax.set_xlim(left=xmin if xmin is not None else 0, right=xmax if xmax is not None else 1)
        ax.set_xlabel(metric_type.capitalize())
        ax.grid(axis='x', linestyle='--', linewidth=0.5)

    for i in range(n_concepts, len(axes)):
        fig.delaxes(axes[i])

    title = f"Best {metric_type.capitalize()} per Concept ({model_name} on {dataset_name})"
    if include_baselines:
        title += "\n(Including Inversion Baselines)"
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def summarize_best_inversion_metrics(metric_type, concept_schemes, dataset_name, model_name,
                                     justobj=False, superdetector_inversion=False, weighted_avg=True, include_baselines=True):
    base_dir = f'Quant_Results/{dataset_name}'
    summary_rows = []

    # Load and filter ground truth
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(224, 224).pt'
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_test_inputsize_(560, 560).pt'
    gt_samples_per_concept_test = torch.load(gt_path, weights_only=False)
    gt_samples_per_concept_test = filter_concept_dict(gt_samples_per_concept_test, dataset_name)

    for concept_scheme in concept_schemes:
        # Use pre-calibrated optimal results
        if superdetector_inversion:
            path_prefix = 'superpatch_avg_inv_'
        else:
            path_prefix = ''
            
        if concept_scheme == 'avg':
            file_name = f"{path_prefix}{model_name}_avg_patch_embeddings_percentthrumodel_100_optimal_test.csv"
            optimal_file = f"optimal_f1_{model_name}_avg_patch_embeddings_percentthrumodel_100.pt"
            con_label = f"{model_name}_avg_patch_embeddings_percentthrumodel_100"
        elif concept_scheme == 'linsep':
            file_name = f"{path_prefix}{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100_optimal_test.csv"
            optimal_file = f"optimal_f1_{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100.pt"
            con_label = f"{model_name}_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100"
        elif concept_scheme == 'unsupervised kmeans':
            file_name = f"{path_prefix}{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100_optimal_test.csv"
            optimal_file = f"optimal_f1_{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100.pt"
            con_label = f"{model_name}_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100"
        elif concept_scheme == 'unsupervised kmeans linsep':
            file_name = f"{path_prefix}{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100_optimal_test.csv"
            optimal_file = f"optimal_f1_{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100.pt"
            con_label = f"{model_name}_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100"
        else:
            raise ValueError(f"Unrecognized concept scheme: {concept_scheme}")

        metric_path = os.path.join(base_dir, file_name)
        optimal_path = f'Detect_Invert_Thresholds/{dataset_name}/{optimal_file}'

        if not os.path.exists(metric_path):
            print(f"Warning: Optimal file not found {metric_path}")
            continue

        df = pd.read_csv(metric_path)

        if 'kmeans' in concept_scheme:
            alignment_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
            alignment_results = torch.load(alignment_path, weights_only=False)
            cluster_to_concept = {str(info['best_cluster']): concept for concept, info in alignment_results.items()}
            df = df.copy()
            df['concept'] = df['concept'].astype(str).map(cluster_to_concept)

        df = df[df['concept'].isin(gt_samples_per_concept_test)]

        if weighted_avg:
            total_samples = sum(len(gt_samples_per_concept_test[c]) for c in df['concept'])
            weighted_sum = sum(row[metric_type] * len(gt_samples_per_concept_test[row['concept']]) for _, row in df.iterrows())
            avg_metric = weighted_sum / total_samples if total_samples > 0 else 0
        else:
            avg_metric = df[metric_type].mean()

        scheme_name = f'{concept_scheme} superdetector cossim' if superdetector_inversion else f'{concept_scheme} concept cossim'
        summary_rows.append({
            'Scheme': scheme_name,
            f'Best Avg {metric_type.upper()}': round(avg_metric, 4),
            'Detect %': 'calibrated',
            'Invert %': 'calibrated',
            'File': file_name
        })

    if include_baselines:
        for baseline_type in ['random', 'always_positive', 'always_negative']:
            baseline_path = f'{base_dir}/inversion_{baseline_type}_{model_name}_patch_baseline.csv'
            try:
                baseline_df = pd.read_csv(baseline_path)
                baseline_df = baseline_df[baseline_df['concept'].isin(gt_samples_per_concept_test)]

                if weighted_avg:
                    total_samples = sum(len(gt_samples_per_concept_test[c]) for c in baseline_df['concept'])
                    weighted_sum = sum(row[metric_type] * len(gt_samples_per_concept_test[row['concept']]) for _, row in baseline_df.iterrows())
                    avg_metric = weighted_sum / total_samples if total_samples > 0 else 0
                else:
                    avg_metric = baseline_df[metric_type].mean()

                summary_rows.append({
                    'Scheme': f'Inversion Baseline ({baseline_type})',
                    f'Best Avg {metric_type.upper()}': round(avg_metric, 4),
                    'Detect %': 'N/A',
                    'Invert %': 'N/A',
                    'File': f'inversion_{baseline_type}_{model_name}_patch_baseline.csv'
                })
            except FileNotFoundError:
                print(f"Warning: Baseline file not found {baseline_path}")

    return pd.DataFrame(summary_rows)


# Removed summarize_best_equal_percentile_inversion_metrics as it's no longer needed with calibration approach


""" Precision/Recall Curves """
from sklearn.metrics import auc


def get_style_label_color(key, pr_auc):
    style_map = {
    'avg': {'color': 'orchid', 'type': 'supervised', 'label': 'patch avg'},
    'linsep': {'color': 'indigo', 'type': 'supervised', 'label': 'patch linsep'},
    'labeled cls avg': {'color': 'goldenrod', 'type': 'supervised', 'label': 'cls avg'},
    'labeled cls linsep': {'color': 'orangered', 'type': 'supervised', 'label': 'cls linsep'},
    'unsupervised patch kmeans': {'color': 'orchid', 'type': 'unsupervised', 'label': 'patch avg'},
    'unsupervised patch linsep kmeans': {'color': 'indigo', 'type': 'unsupervised', 'label': 'patch linsep'},
    'unsupervised cls kmeans': {'color': 'goldenrod', 'type': 'unsupervised', 'label': 'cls avg'},
    'unsupervised cls linsep kmeans': {'color': 'orangered', 'type': 'unsupervised', 'label': 'cls linsep'},
    }
    style = style_map.get(key, {})
    color = style.get('color', 'gray')
    label = f"{style.get('label', key)} (AUC={pr_auc:.2f})"
    return label, color
      
def plot_pr_curves_across_methods(
    dataset_name,
    split,
    model_name,
    sample_types,
    save_path=None,
    weighted=False,
    ax=None,
    style_map=None,
):

    percentiles = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    con_labels = {}

    for sample_type in sample_types:
        n_clusters = 1000 if sample_type == "patch" else 50
        con_labels[f"labeled {sample_type} avg"] = f"{model_name}_avg_{sample_type}_embeddings_percentthrumodel_100"
        con_labels[f"labeled {sample_type} linsep"] = f"{model_name}_linsep_{sample_type}_embeddings_BD_True_BN_False_percentthrumodel_100"

    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    elif model_name == 'ClIP':
        gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(224, 224).pt"
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_(560, 560).pt' 
    gt_concepts = torch.load(gt_path, weights_only=False)

    if weighted:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        for method_name, con_label in con_labels.items():
            weighted_prec, weighted_rec = [0.0] * len(percentiles), [0.0] * len(percentiles)
            total_weight = 0.0

            for concept, gt_idxs in gt_concepts.items():
                weight = len(gt_idxs)
                prec, rec = [], []
                for p in percentiles:
                    file_name = f"Quant_Results/{dataset_name}/detectionmetrics_per_{p}_{con_label}.pt"
                    if not os.path.exists(file_name):
                        continue
                    df = torch.load(file_name, weights_only=False)
                    row = df[df["concept"] == concept]
                    if row.empty:
                        continue
                    prec.append(row["precision"].iloc[0])
                    rec.append(row["recall"].iloc[0])

                if len(prec) == len(percentiles):
                    weighted_prec = [wp + pr * weight for wp, pr in zip(weighted_prec, prec)]
                    weighted_rec = [wr + rc * weight for wr, rc in zip(weighted_rec, rec)]
                    total_weight += weight

            if total_weight == 0:
                continue

            avg_prec = [wp / total_weight for wp in weighted_prec]
            avg_rec = [wr / total_weight for wr in weighted_rec]
            pr_auc = auc(avg_rec, avg_prec)
            best_idx = max(range(len(avg_prec)), key=lambda i: (2 * avg_prec[i] * avg_rec[i]) / (avg_prec[i] + avg_rec[i] + 1e-8))
            best_f1 = (2 * avg_prec[best_idx] * avg_rec[best_idx]) / (avg_prec[best_idx] + avg_rec[best_idx] + 1e-8)
            
            label, color = get_style_label_color(method_name, pr_auc)
            plt.plot(avg_rec, avg_prec, label=label, color=color)
            plt.plot([avg_rec[best_idx]], [avg_prec[best_idx]], marker = 'o', color=color)
            plt.plot([0, avg_rec[best_idx]], [avg_prec[best_idx]] * 2, linestyle='--', color=color, alpha=0.6)
            plt.plot([avg_rec[best_idx]] * 2, [0, avg_prec[best_idx]], linestyle='--', color=color, alpha=0.6)
            plt.text(avg_rec[best_idx], avg_prec[best_idx], f"Best F1={best_f1:.2f}", fontsize=10, color=color)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Concept Type', loc="best")
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "pr_weighted_all_methods.png"), dpi=300)
        plt.show()


def plot_pr_curves_patch_level(
    dataset_name,
    split,
    model_name,
    concept_schemes,
    percentiles,
    justobj=False,
    weighted=False,
    save_path=None,
    style_map=None
):

    base_dir = f"Quant_Results/{dataset_name}"
    if dataset_name == 'Stanford-Tree-Bank' or 'Sarcasm' in dataset_name or 'Emotion' in dataset_name:
        if model_name == 'Mistral':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text2').pt"
        elif model_name == 'Llama':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text').pt"
        elif model_name == 'Qwen':
            gt_path = f"GT_Samples/{dataset_name}/gt_samples_per_concept_{split}_inputsize_('text', 'text3').pt"
    elif model_name == 'CLIP':
        gt_path = f"GT_Samples/{dataset_name}/gt_patch_per_concept_{split}_inputsize_(224, 224).pt"
    elif model_name == 'Llama':
        gt_path = f'GT_Samples/{dataset_name}/gt_patch_per_concept_{split}_inputsize_(560, 560).pt' 
        
    gt_samples = torch.load(gt_path, weights_only=False)


    scheme_to_label = {
        'avg': '_avg_patch_embeddings_percentthrumodel_100.pt',
        'linsep': '_linsep_patch_embeddings_BD_True_BN_False_percentthrumodel_100.pt',
        'unsupervised kmeans': '_kmeans_1000_patch_embeddings_kmeans_percentthrumodel_100.pt',
        'unsupervised kmeans linsep': '_kmeans_1000_linsep_patch_embeddings_kmeans_percentthrumodel_100.pt',
    }

    if weighted:
        plt.figure(figsize=(8, 6))
        for scheme in concept_schemes:
            weighted_prec, weighted_rec = [0.0]*len(percentiles), [0.0]*len(percentiles)
            total_w = 0.0

            for concept, idxs in gt_samples.items():
                w = len(idxs)
                precs, recs = [], []
                for p in percentiles:
                    fp = f"Quant_Results/{dataset_name}/detectionmetrics_per_{p}_{model_name}{scheme_to_label[scheme]}"
                    if not os.path.exists(fp):
                        print(fp, "doesn't exist")
                        break
                    df = torch.load(fp, weights_only=False)
                    if 'kmeans' in scheme:
                        con_label = f'{model_name}_kmeans_1000{"_linsep" if "linsep" in scheme else ""}_patch_embeddings_kmeans_percentthrumodel_100'
                        align_path = f'Unsupervised_Matches/{dataset_name}/bestdetects_{con_label}.pt'
                        align_results = torch.load(align_path, weights_only=False)
                        cluster_to_concept = {str(info['best_cluster']): concept for concept, info in align_results.items()}
                        df = df.copy()
                        df['concept'] = df['concept'].astype(str).map(cluster_to_concept)

                    row = df[df['concept'] == concept]
                    if row.empty:
                        break
                    precs.append(row['precision'].iloc[0])
                    recs.append(row['recall'].iloc[0])
                else:
                    total_w += w
                    for i in range(len(percentiles)):
                        weighted_prec[i] += precs[i] * w
                        weighted_rec[i] += recs[i] * w

            if total_w > 0:
                avg_prec = [wp / total_w for wp in weighted_prec]
                avg_rec = [wr / total_w for wr in weighted_rec]
                pr_auc = auc(avg_rec, avg_prec)
                best_idx = max(range(len(avg_prec)), key=lambda i: (2 * avg_prec[i] * avg_rec[i]) / (avg_prec[i] + avg_rec[i] + 1e-8))
                best_f1 = (2 * avg_prec[best_idx] * avg_rec[best_idx]) / (avg_prec[best_idx] + avg_rec[best_idx] + 1e-8)

                label, color = get_style_label_color(scheme, pr_auc)
                plt.plot(avg_rec, avg_prec, label=label, color=color)
                plt.plot([avg_rec[best_idx]], [avg_prec[best_idx]], marker = 'o', color=color)
                plt.plot([0, avg_rec[best_idx]], [avg_prec[best_idx]] * 2, linestyle='--', color=color, alpha=0.6)
                plt.plot([avg_rec[best_idx]] * 2, [0, avg_prec[best_idx]], linestyle='--', color=color, alpha=0.6)
                plt.text(avg_rec[best_idx], avg_prec[best_idx], f"Best F1={best_f1:.2f}", fontsize=10, color=color)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Concept Type', loc='best')
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, "pr_weighted_patch_level.png"), dpi=300)
        plt.show()