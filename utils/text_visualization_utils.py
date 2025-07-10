import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from itertools import combinations
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import io
import base64
import numpy as np
from utils.general_utils import get_split_df
from itertools import chain
from collections import defaultdict

#### Computations ####
# def flatten_token_list(tokens_list):
#     """
#     Flattens a nested list of tokens into a single list.

#     Args:
#         tokens_list (List[List[str]]): A list of sentences, where each sentence is a list of tokens.

#     Returns:
#         List[str]: A flat list containing all tokens in order.
#     """
#     flat_tokens_list = []
#     for token_list in tokens_list:
#         for token in token_list:
#             flat_tokens_list.append(token)
#     return flat_tokens_list
def flatten_token_list(tokens_list):
    """
    Efficiently flattens a nested list of tokens into a single list.

    Args:
        tokens_list (List[List[str]]): A list of sentences, where each sentence is a list of tokens.

    Returns:
        List[str]: A flat list containing all tokens in order.
    """
    return list(chain.from_iterable(tokens_list))



def get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list):
    # Step 1: Determine the starting index of the sentence in the flat list
    start_idx = sum(len(tokens) for tokens in tokens_list[:sentence_idx])
    end_idx = start_idx + len(tokens_list[sentence_idx])
    return start_idx, end_idx


def get_sent_idx_from_global_token_idx(token_idx, dataset_name, model_input_size=None):
    """
    Given a global token index (from a flattened token list), return the corresponding sentence index.

    Args:
        token_idx (int): Index of a token in the flattened token list.
        dataset_name (str): Name of the dataset.
        model_input_size (tuple, optional): Model input size for model-specific files.

    Returns:
        int: Sentence index containing the token.
    """
    # Try model-specific tokens file first, fall back to generic
    if model_input_size is not None:
        model_specific_path = f'GT_Samples/{dataset_name}/tokens_inputsize_{model_input_size}.pt'
        generic_path = f'GT_Samples/{dataset_name}/tokens.pt'
        
        import os
        if os.path.exists(model_specific_path):
            tokens_list = torch.load(model_specific_path, weights_only=False)
        else:
            tokens_list = torch.load(generic_path, weights_only=False)
    else:
        tokens_list = torch.load(f'GT_Samples/{dataset_name}/tokens.pt', weights_only=False)
    
    token_counter = 0
    for sent_idx, tokens in enumerate(tokens_list):
        if token_counter + len(tokens) > token_idx:
            return sent_idx
        token_counter += len(tokens)
    return token_counter

def remove_leading_token(tokens):
    """
    Cleans up tokens that use Ġ to represent leading spaces (from BPE tokenizers).

    Args:
        tokens (List[str]): List of tokens with possible Ġ characters.

    Returns:
        str: A visually cleaned-up sentence.
    """
    cleaned_tokens = []
    for tok in tokens:
        if tok.startswith("Ġ"):
            cleaned_tokens.append(tok[1:])
        else:
            cleaned_tokens.append(tok)
    return cleaned_tokens


def get_word_from_indices(indices, tokens_list):
    """
    Retrieves tokens from a flattened token list given their global indices.

    Args:
        indices (List[int]): List of indices corresponding to positions in the flattened token list.
        tokens_list (List[List[str]]): A nested list of tokens (sentences with word tokens).

    Returns:
        List[str]: The tokens corresponding to the given indices.
    """
    tokens_list_flat = flatten_token_list(tokens_list)
    words = [tokens_list_flat[idx] for idx in indices]
    return words


def get_top_token_indices_for_concept(cos_sims, tokens_list, concept, dataset_name, top_k=5, split='test'):
    """
    Returns indices of tokens with the highest cosine similarity to the given concept.

    Args:
        cos_sims (pd.DataFrame): DataFrame with rows as tokens, columns as concept names.
        concept (str): Concept column to search by.
        top_k (int): Number of top tokens to return.

    Returns:
        List of indices (or a single index if top_k=1).
    """
    if split != 'both':
        # Step 1: Load split info
        split_df = get_split_df(dataset_name)

        # Step 2: Determine sentence indices that match the split
        valid_sentence_indices = split_df[split_df == split].index.tolist()

        # Step 3: Convert sentence-level mask to token-level indices
        valid_token_indices = []
        idx = 0
        for i, tokens in enumerate(tokens_list):
            if i in valid_sentence_indices:
                valid_token_indices.extend(list(range(idx, idx + len(tokens))))
            idx += len(tokens)

        # Step 4: Filter cos_sims to valid tokens only
        filtered = cos_sims.loc[valid_token_indices]
        top_indices = filtered[concept].nlargest(top_k).index
    else:
        # No filtering — just use all tokens
        top_indices = cos_sims[concept].nlargest(top_k).index

    return top_indices.tolist() if top_k > 1 else top_indices[0]


def user_select_concept(concepts):
    print("Select a concept from the list below:")
    for i, c in enumerate(concepts):
        print(f"{i}: {c}")
    while True:
        try:
            selection = int(input("Enter the number corresponding to the concept: "))
            return concepts[selection]
        except (ValueError, IndexError):
            print("Invalid selection. Please enter a valid index number.")
            
            
def get_sentence_category(sentence_idx, dataset_name):
    if dataset_name == 'Stanford-Tree-Bank':
        metadata = pd.read_csv(f'../Data/{dataset_name}/sentence_level_sentiment.csv')
        return str(metadata['sentiment_label'].loc[sentence_idx])
    elif 'Emotions' in dataset_name:
        metadata = pd.read_csv(f'../Data/{dataset_name}/paragraph_level_emotion.csv')
        return ""
    else:
        metadata = pd.read_csv(f'../Data/{dataset_name}/paragraph_level_sarcasm.csv')
        return str(metadata['sarcasm'].loc[sentence_idx])


def get_sentences_by_metric(cos_sims, tokens_list, dataset_name, concept=None, top_k=5, top=True, aggr_method='avg', split='test'):
    """
    Finds the top-k sentences with the highest or lowest aggregate similarity to a given concept,
    filtered by the dataset split ('train' or 'test').

    Args:
        concept (str): Concept column in cos_sims.
        cos_sims (pd.DataFrame): DataFrame where rows = flattened tokens, cols = concept names.
        tokens_list (List[List[str]]): List of tokenized sentences.
        dataset_name (str): Name of dataset to load split info from.
        top_k (int): Number of top sentences to return.
        top (bool): Whether to return top-k (True) or bottom-k (False).
        aggr_method (str): Aggregation method: 'avg', 'max', or 'min'.
        split (str): 'train' or 'test'.

    Returns:
        List of (sentence_index, score)
    """
    similarities = cos_sims[concept].to_numpy()
    
    sentence_scores = []
    idx = 0
    for i, tokens in enumerate(tokens_list):
        n = len(tokens)
        sentence_sim = similarities[idx:idx + n]
        if aggr_method == 'avg':
            metric = np.mean(sentence_sim) if n > 0 else float('-inf')
        elif aggr_method == 'max':
            metric = np.max(sentence_sim) if n > 0 else float('-inf')
        elif aggr_method == 'min':
            metric = np.min(sentence_sim) if n > 0 else float('inf')
        else:
            raise ValueError(f"Invalid aggregation method: {aggr_method}")
        sentence_scores.append((i, metric))
        idx += n

    # Step 1: Load split info
    split_df = get_split_df(dataset_name)

    # Step 2: Filter sentence indices by split, if not 'both'
    if split.lower() in ['train', 'test']:
        valid_indices = split_df[split_df == split].index.tolist()
        sentence_scores = [x for x in sentence_scores if x[0] in valid_indices]

    # Step 3: Sort and return top_k
    sorted_scores = sorted(sentence_scores, key=lambda x: -x[1] if top else x[1])
    return sorted_scores[:top_k]





#### Plotting ####
def compute_avg_token_similarities(embeddings, flat_tokens_list):
    token_to_indices = defaultdict(list)

    # Step 1: map tokens to their embedding indices
    for idx, token in enumerate(flat_tokens_list):
        token_to_indices[token].append(idx)

    token_to_stats = {}

    for token, indices in token_to_indices.items():
        count = len(indices)

        if count < 2:
            token_to_stats[token] = {"avg_sim": float('nan'), "count": count}
            continue

        # Get embeddings for the token
        token_embeds = embeddings[indices]  # shape (count, hidden_dim)

        # Compute pairwise cosine similarities
        sims = []
        for i, j in combinations(range(count), 2):
            sim = cosine_similarity(token_embeds[i].unsqueeze(0), token_embeds[j].unsqueeze(0))
            sims.append(sim.item())

        avg_sim = sum(sims) / len(sims)
        token_to_stats[token] = {"avg_sim": avg_sim, "count": count}

    return token_to_stats

def plot_sentence_similarity_heatmap(sentence_idx, tokens_list, embeds, max_tokens=None):
    """
    Compute pairwise cosine similarities between all tokens in a specific sentence.

    Args:
        sentence_idx (int): The index of the sentence in tokens_list.
        tokens_list (List[List[str]]): List of tokenized sentences.
        embeds (torch.Tensor): Flattened tensor of shape (n_tokens, hidden_dim), aligned with tokens_list.
        max_tokens (int, optional): Maximum number of tokens to include in the heatmap.
    """
    # Step 1: Determine the start and end index in the flattened embedding tensor
    start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)

    sentence_tokens = remove_leading_token(tokens_list[sentence_idx])
    sentence_embeds = embeds[start_idx:end_idx]

    # Trim if max_tokens is set
    if max_tokens is not None and len(sentence_tokens) > max_tokens:
        sentence_tokens = sentence_tokens[:max_tokens]
        sentence_embeds = sentence_embeds[:max_tokens]

    # Normalize embeddings
    sentence_embeds = torch.nn.functional.normalize(sentence_embeds, p=2, dim=1)

    # Compute cosine similarity matrix
    sim_matrix = torch.matmul(sentence_embeds, sentence_embeds.T).cpu().numpy()

    # Plot
    plt.figure(figsize=(len(sentence_tokens) * 0.5 + 6, len(sentence_tokens) * 0.5 + 6))
    ax = sns.heatmap(sim_matrix, xticklabels=sentence_tokens, yticklabels=sentence_tokens,
                     cmap="coolwarm", center=0, annot=True, fmt=".2f",
                     linewidths=0.5, square=True, cbar_kws={"label": "Cosine Similarity"})
    plt.title(f'Heatmap for Sentence {sentence_idx}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    
def get_colormap_color(score, cmap, norm):
    return matplotlib.colors.rgb2hex(cmap(norm(score)))

def plot_colorbar(vmin=0.0, vmax=1.0, cmap_name="coolwarm", orientation="vertical"):
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    figsize = (1.5, 0.5) if orientation == "vertical" else (5, 0.4)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig.subplots_adjust(left=0.2 if orientation == "vertical" else 0.3)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=orientation)
    cb.set_label("Score")

    # Convert plot to base64 for embedding
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return f'<img src="data:image/png;base64,{b64_img}" style="margin-top:10px; max-width:100%;" />'


def highlight_tokens_with_legend(tokens, scores, cmap_name="coolwarm", vmin=None, vmax=None, include_colorbar=True):
    if vmin is None:
        vmin = min(scores)
    if vmax is None:
        vmax = max(scores)         
        
    # Normalize scores and get colormap
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    
    tokens = remove_leading_token(tokens)

    html = ""
    for token, score in zip(tokens, scores):
        color = get_colormap_color(score, cmap, norm)
        html += f'<span style="background-color:{color}; padding:2px 4px; margin:2px; border-radius:3px;">{token}</span> '

    if include_colorbar:
        colorbar_html = plot_colorbar(vmin=vmin, vmax=vmax, cmap_name=cmap_name)
        html_block = f"""
        <div style="display: flex; align-items: flex-start;">
            <div style="flex: 1;">{html}</div>
            <div style="padding-left: 20px;">{colorbar_html}</div>
        </div>
        """
    else:
        html_block = f"<div>{html}</div>"

    return HTML(html_block)


def plot_most_aligned_tokens(cos_sims, tokens_list, dataset_name, concept=None, top_k=5):
    # Step 1: If no concept is passed, prompt the user
    if concept is None:
        concept = user_select_concept(cos_sims.columns)

    # Step 2: Get top token indices
    top_token_indices = get_top_token_indices_for_concept(cos_sims, tokens_list, concept, dataset_name, top_k)
    
    # Step 3: Map token indices back to token strings
    top_tokens = get_word_from_indices(top_token_indices, tokens_list)
    
    # Step 4: Plot tokens with similarity scores
    display(highlight_tokens_with_legend(top_tokens, cos_sims[concept].iloc[top_token_indices], vmin=0))
    
    
def plot_most_aligned_sentences(cos_sims, all_texts, dataset_name, concept=None, top_k=5, split='test'):
    """
    Plots the top-k most aligned sentences (CLS embeddings) for a given concept.

    Args:
        cos_sims (pd.DataFrame): DataFrame with rows as sentences, columns as concepts.
        all_texts (List[str]): Original sentences, aligned row-wise with `cos_sims`.
        dataset_name (str): Dataset name for loading split info.
        concept (str): Concept to visualize.
        top_k (int): Number of top aligned sentences to return.
        split (str): One of 'train', 'test', 'cal', or 'both'
    """
    # Step 1: Choose concept if not passed
    if concept is None:
        concept = user_select_concept(cos_sims.columns)

    # Step 2: Filter by split
    if split != 'both':
        split_df = get_split_df(dataset_name)
        valid_indices = split_df[split_df == split].index
        filtered_scores = cos_sims.loc[valid_indices, concept]
    else:
        filtered_scores = cos_sims[concept]

    # Step 3: Get top-k sentence indices
    top_sentence_indices = filtered_scores.nlargest(top_k).index.tolist()

    # Step 4: Display sentences and scores
    print(f"\nTop {top_k} sentences most aligned with concept '{concept}':\n")
    for rank, idx in enumerate(top_sentence_indices):
        score = cos_sims.loc[idx, concept]
        sentence = all_texts[idx]
        print(f"[{rank+1}] Score: {score:.4f}")
        print(f"     {sentence}\n")



def plot_tokens_in_context_byconcept(
    cos_sims,
    tokens_list,
    dataset_name,
    concept=None,
    top_k=5,
    top=True,
    aggr_method='avg',
    cmap_name="coolwarm"
):
    is_sentence_level = dataset_name == "Stanford-Tree-Bank"
    unit_type = "sentence" if is_sentence_level else "paragraph"

    if concept is None:
        concept = user_select_concept(cos_sims.columns)

    samples = get_sentences_by_metric(cos_sims, tokens_list, dataset_name, concept, top_k, top, aggr_method)

    if top:
        print(f"\nPlotting {unit_type}s MOST activated by {concept} ({aggr_method} over tokens)\n")
    else:
        print(f"\nPlotting {unit_type}s LEAST activated by {concept} ({aggr_method} over tokens)\n")

    all_scores = []
    for idx, _ in samples:
        start_idx, end_idx = get_glob_tok_indices_from_sent_idx(idx, tokens_list)
        all_scores.extend(cos_sims[concept].iloc[start_idx:end_idx].tolist())
    vmin, vmax = min(all_scores), max(all_scores)

    html_blocks = []
    for i, (idx, metric) in enumerate(samples):
        category = get_sentence_category(idx, dataset_name)
        start_idx, end_idx = get_glob_tok_indices_from_sent_idx(idx, tokens_list)
        tokens = tokens_list[idx]
        sims = cos_sims[concept].iloc[start_idx:end_idx].tolist()

        # Title for each text unit with its concept score
        title = f"<h4>Rank {i+1} : {unit_type.capitalize()} {idx} -- {category.capitalize()} ({aggr_method}={metric:.2f})</h4>"

        html = highlight_tokens_with_legend(tokens, sims, cmap_name=cmap_name, vmin=vmin, vmax=vmax, include_colorbar=False)
        html_blocks.append(f"{title}{html.data}")

    # Append shared colorbar
    colorbar_html = plot_colorbar(vmin=vmin, vmax=vmax, cmap_name=cmap_name, orientation="horizontal")

    full_html = f"""
    <div>
        {''.join(html_blocks)}
        <div style="margin-top: 10px;">{colorbar_html}</div>
    </div>
    """
    display(HTML(full_html))


def plot_all_concept_activations_on_sentence(
    sentence_idx,
    cos_sims,
    tokens_list,
    dataset_name,
    cmap_name="coolwarm",
    gt_samples_per_concept=None,
    vmin=None,
    vmax=None
):
    from IPython.display import HTML, display

    def clean_token(token):
        return token.replace("Ġ", "")  # Strip GPT/RoBERTa word boundary marker

    concepts = cos_sims.columns.tolist()
    raw_tokens = tokens_list[sentence_idx]
    tokens = [clean_token(tok) for tok in raw_tokens]
    start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sentence_idx, tokens_list)

    # Get per-concept similarity scores for this sentence
    sims_per_concept = {
        concept: cos_sims[concept].iloc[start_idx:end_idx].tolist()
        for concept in concepts
    }

    # Compute vmin/vmax from all sims if not provided
    if vmin is None or vmax is None:
        all_sims = [sim for sims in sims_per_concept.values() for sim in sims]
        vmin = min(all_sims) if vmin is None else vmin
        vmax = max(all_sims) if vmax is None else vmax

    sentence_class = get_sentence_category(sentence_idx, dataset_name)
    print(f"\nSentence {sentence_idx} ({sentence_class}):\n")

    # Optional ground truth display
    if gt_samples_per_concept is not None:
        print("Ground Truth Concept Labels:\n")
        for concept in concepts:
            concept_token_mask = [False] * len(tokens)
            for idx in gt_samples_per_concept.get(concept, []):
                if start_idx <= idx < end_idx:
                    concept_token_mask[idx - start_idx] = True

            highlighted_tokens = [
                f"<span style='background-color: yellow'>{token}</span>" if is_labeled else token
                for token, is_labeled in zip(tokens, concept_token_mask)
            ]
            sentence_html = " ".join(highlighted_tokens)
            display(HTML(f"<b>Concept: {concept}</b><br>{sentence_html}<br><br>"))

    # Render each concept activation heatmap
    html_blocks = []
    for concept in concepts:
        sims = sims_per_concept[concept]
        heatmap = highlight_tokens_with_legend(
            tokens, sims, cmap_name=cmap_name, vmin=vmin, vmax=vmax, include_colorbar=False
        )
        concept_header = f"<h4 style='margin-bottom: 4px;'>Concept: {concept} (min={min(sims):.4f}, max={max(sims):.4f}, avg={np.mean(sims):.4f})</h4>"
        html_blocks.append(f"{concept_header}{heatmap.data}")

    colorbar_html = plot_colorbar(vmin=vmin, vmax=vmax, cmap_name=cmap_name, orientation="horizontal")

    full_html = f"""
    <div>
        {''.join(html_blocks)}
        <div style="margin-top: 15px;">{colorbar_html}</div>
    </div>
    """
    display(HTML(full_html))


def plot_tokens_by_activation_and_gt(
    cos_sims,
    tokens_list,
    dataset_name,
    model_input_size,
    concept=None,
    n_examples=3,
    cmap_name="coolwarm"
):
    """
    Plots tokens in context showing paragraphs with the most positive, negative, and near-zero 
    maximum token activations, split by ground truth labels. Only includes paragraphs from the test split.
    
    Args:
        cos_sims (pd.DataFrame): DataFrame with rows as flattened tokens, cols as concept names.
        tokens_list (List[List[str]]): List of tokenized sentences.
        dataset_name (str): Name of dataset.
        model_input_size (tuple): Model input size for loading GT samples.
        concept (str): Concept to visualize.
        n_examples (int): Number of examples per category (default: 3).
        cmap_name (str): Colormap name for visualization.
    """
    # Import needed functions
    from collections import defaultdict
    import os
    
    # Select concept if not provided
    if concept is None:
        concept = user_select_concept(cos_sims.columns)
    
    # Load ground truth token indices (patches for text datasets)
    gt_path = f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_{model_input_size}.pt'
    if not os.path.exists(gt_path):
        # Try alternative path for text datasets
        gt_path = f'GT_Samples/{dataset_name}/gt_patches_per_concept_inputsize_text.pt'
    
    if os.path.exists(gt_path):
        gt_patches_per_concept = torch.load(gt_path, weights_only=False)
        gt_indices = set(gt_patches_per_concept.get(concept, []))
    else:
        print(f"Warning: Could not find GT samples file at {gt_path}")
        gt_indices = set()
    
    # Get test split indices
    split_df = get_split_df(dataset_name)
    test_sentence_indices = split_df[split_df == 'test'].index.tolist()
    
    # Compute max activation per paragraph and track which paragraphs have GT tokens
    paragraph_data = []
    idx = 0
    for sent_idx, tokens in enumerate(tokens_list):
        if sent_idx in test_sentence_indices:
            start_idx = idx
            end_idx = idx + len(tokens)
            
            # Get activations for all tokens in this paragraph
            paragraph_sims = cos_sims[concept].iloc[start_idx:end_idx]
            max_activation = paragraph_sims.max()
            
            # Check if any token in this paragraph is GT
            paragraph_token_indices = list(range(start_idx, end_idx))
            has_gt = any(token_idx in gt_indices for token_idx in paragraph_token_indices)
            
            paragraph_data.append({
                'sent_idx': sent_idx,
                'max_activation': max_activation,
                'has_gt': has_gt
            })
        idx += len(tokens)
    
    # Split by ground truth
    gt_true_paragraphs = [p for p in paragraph_data if p['has_gt']]
    gt_false_paragraphs = [p for p in paragraph_data if not p['has_gt']]
    
    # Sort by max activation
    gt_true_sorted = sorted(gt_true_paragraphs, key=lambda x: x['max_activation'], reverse=True)
    gt_false_sorted = sorted(gt_false_paragraphs, key=lambda x: x['max_activation'], reverse=True)
    
    # Get examples for each category
    categories = {}
    
    # GT True categories
    if len(gt_true_sorted) >= n_examples:
        categories["GT True - Most Positive"] = gt_true_sorted[:n_examples]
        categories["GT True - Most Negative"] = gt_true_sorted[-n_examples:]
        # For near zero, sort by absolute value of max activation
        gt_true_by_abs = sorted(gt_true_paragraphs, key=lambda x: abs(x['max_activation']))
        categories["GT True - Near Zero"] = gt_true_by_abs[:n_examples]
    else:
        categories["GT True - Most Positive"] = gt_true_sorted
        categories["GT True - Most Negative"] = []
        categories["GT True - Near Zero"] = []
    
    # GT False categories
    if len(gt_false_sorted) >= n_examples:
        categories["GT False - Most Positive"] = gt_false_sorted[:n_examples]
        categories["GT False - Most Negative"] = gt_false_sorted[-n_examples:]
        # For near zero, sort by absolute value of max activation
        gt_false_by_abs = sorted(gt_false_paragraphs, key=lambda x: abs(x['max_activation']))
        categories["GT False - Near Zero"] = gt_false_by_abs[:n_examples]
    else:
        categories["GT False - Most Positive"] = gt_false_sorted
        categories["GT False - Most Negative"] = []
        categories["GT False - Near Zero"] = []
    
    # Collect all scores for consistent colormap scale
    all_scores = []
    for category_paragraphs in categories.values():
        for paragraph in category_paragraphs:
            sent_idx = paragraph['sent_idx']
            start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
            all_scores.extend(cos_sims[concept].iloc[start_idx:end_idx].tolist())
    
    if all_scores:
        vmin, vmax = min(all_scores), max(all_scores)
    else:
        vmin, vmax = -1, 1  # Default range if no scores
    
    # Create visualization
    html_blocks = []
    
    for category_name, paragraphs in categories.items():
        if paragraphs:  # Only show categories with examples
            html_blocks.append(f"<h3>{category_name}</h3>")
            
            for i, paragraph in enumerate(paragraphs):
                sent_idx = paragraph['sent_idx']
                max_activation = paragraph['max_activation']
                start_idx, end_idx = get_glob_tok_indices_from_sent_idx(sent_idx, tokens_list)
                
                tokens = tokens_list[sent_idx]
                sims = cos_sims[concept].iloc[start_idx:end_idx].tolist()
                
                # Find which token has the max activation
                max_token_idx = np.argmax(sims)
                max_token = remove_leading_token([tokens[max_token_idx]])[0]
                
                # Get sentence category
                category = get_sentence_category(sent_idx, dataset_name)
                
                # Create title with paragraph info
                title = f"<h4>Example {i+1}: Paragraph {sent_idx} - {category} (max token: '{max_token}' = {max_activation:.3f})</h4>"
                
                # Create highlighted sentence
                html = highlight_tokens_with_legend(tokens, sims, cmap_name=cmap_name, vmin=vmin, vmax=vmax, include_colorbar=False)
                
                # Add marker for the max token
                html_with_marker = html.data.replace(
                    f'>{max_token}</span>',
                    f' style="border: 3px solid black; font-weight: bold;">{max_token}</span>'
                )
                
                html_blocks.append(f"{title}{html_with_marker}")
    
    # Add shared colorbar
    colorbar_html = plot_colorbar(vmin=vmin, vmax=vmax, cmap_name=cmap_name, orientation="horizontal")
    
    # Print summary statistics
    print(f"\nConcept: {concept}")
    print(f"GT True paragraphs in test set: {len(gt_true_paragraphs)}")
    print(f"GT False paragraphs in test set: {len(gt_false_paragraphs)}")
    print(f"Activation range: [{vmin:.3f}, {vmax:.3f}]")
    
    full_html = f"""
    <div>
        {''.join(html_blocks)}
        <div style="margin-top: 20px;">{colorbar_html}</div>
    </div>
    """
    display(HTML(full_html))

