import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict
from tqdm import tqdm
import copy

import seaborn as sns
import math

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.append(os.path.abspath("utils"))

from patch_alignment_utils import get_patch_split_df, filter_patches_by_image_presence
from quant_concept_evals_utils import find_activated_images_bypatch, compute_concept_thresholds_over_percentiles
from general_utils import create_binary_labels
from compute_concepts_utils import plot_train_history, compute_signed_distances, create_dataloaders, create_linear_model, sort_data_by_split, evaluate_model, log_progress, create_dataloader

### For Detection ###
# def compute_all_f1s_by_concept_and_percentile(sim_metrics, 
#                                               gt_images_per_concept_test, gt_patches_per_concept_test,
#                                               dataset_name, model_input_size, device,
#                                               batch_size=5000, patch_size=14,
#                                              percentiles=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
#                                                           0.6, 0.7, 0.8, 0.9, 0.95, 1.0]):
#     """
#     Computes F1 scores for every concept at every percentile.
    
#     Returns:
#         dict: f1s[concept][percentile] = F1 score
#     """
#     detect_thresholds_all = compute_concept_thresholds_over_percentiles(
#         gt_patches_per_concept_test, sim_metrics, 
#         percentiles, device, dataset_name, 
#         con_label=None, n_vectors=1, n_concepts_to_print=0
#     )

#     f1s = {concept: {} for concept in gt_images_per_concept_test.keys()}

#     for per in percentiles:
#         _, activated_images_test = find_activated_images_bypatch(
#             sim_metrics, detect_thresholds_all[per], 
#             model_input_size, dataset_name, patch_size=patch_size
#         )

#         for concept in gt_images_per_concept_test.keys():
#             gt_images = set(gt_images_per_concept_test[concept])
#             if concept not in activated_images_test:
#                 continue

#             activated_images = set(activated_images_test[concept])
#             all_images = gt_images.union(activated_images)

#             preds = [(1 if img in activated_images else 0) for img in all_images]
#             labels = [(1 if img in gt_images else 0) for img in all_images]

#             if sum(labels) == 0:
#                 continue

#             f1 = f1_score(labels, preds)
#             f1s[concept][per] = f1

#     del sim_metrics, detect_thresholds_all
#     return f1s
def compute_all_f1s_by_concept_and_percentile(sim_metrics, 
                                              gt_images_per_concept_train,
                                              gt_images_per_concept_test, 
                                              gt_patches_per_concept_test,
                                              dataset_name, model_input_size, device,
                                              batch_size=5000, patch_size=14,
                                              percentiles=[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                                                           0.6, 0.7, 0.8, 0.9, 0.95, 1.0]):
    """
    Computes train and test F1 scores for every concept at every percentile.

    Returns:
        train_f1s[concept][percentile], test_f1s[concept][percentile]
    """
    detect_thresholds_all = compute_concept_thresholds_over_percentiles(
        gt_patches_per_concept_test, sim_metrics, 
        percentiles, device, dataset_name, 
        con_label=None, n_vectors=1, n_concepts_to_print=0
    )

    train_f1s = {concept: {} for concept in gt_images_per_concept_test.keys()}
    test_f1s = {concept: {} for concept in gt_images_per_concept_test.keys()}

    for per in percentiles:
        activated_images_train, activated_images_test = find_activated_images_bypatch(
            sim_metrics, detect_thresholds_all[per], 
            model_input_size, dataset_name, patch_size=patch_size
        )

        for concept in gt_images_per_concept_test.keys():
            gt_train = set(gt_images_per_concept_train.get(concept, []))
            gt_test = set(gt_images_per_concept_test.get(concept, []))

            act_train = set(activated_images_train.get(concept, []))
            act_test = set(activated_images_test.get(concept, []))

            # Train F1
            train_all = gt_train.union(act_train)
            train_preds = [1 if img in act_train else 0 for img in train_all]
            train_labels = [1 if img in gt_train else 0 for img in train_all]
            f1_train = f1_score(train_labels, train_preds) if sum(train_labels) > 0 else 0.0

            # Test F1
            test_all = gt_test.union(act_test)
            test_preds = [1 if img in act_test else 0 for img in test_all]
            test_labels = [1 if img in gt_test else 0 for img in test_all]
            f1_test = f1_score(test_labels, test_preds) if sum(test_labels) > 0 else 0.0

            train_f1s[concept][per] = f1_train
            test_f1s[concept][per] = f1_test

    del sim_metrics, detect_thresholds_all
    return train_f1s, test_f1s


def get_best_f1s_per_concept(percentiles, sim_metrics,
                              gt_images_per_concept_test, gt_patches_per_concept_test,
                              dataset_name, model_input_size, device,
                              batch_size=5000, patch_size=14):
    """
    Selects the best F1 and corresponding percentile for each concept.

    Args:
        f1s_by_concept_and_percentile (dict): Output of compute_all_f1s_by_concept_and_percentile()

    Returns:
        dict: concept_to_best = {'f1': ..., 'detect_percentile': ...}
    """
    f1s_by_concept_and_percentile = compute_all_f1s_by_concept_and_percentile(percentiles, sim_metrics,
                                              gt_images_per_concept_test, gt_patches_per_concept_test,
                                              dataset_name, model_input_size, device,
                                              batch_size=5000, patch_size=14)
    concept_to_best = {}
    for concept, per_to_f1 in f1s_by_concept_and_percentile.items():
        if not per_to_f1:
            concept_to_best[concept] = {'f1': None, 'detect_percentile': None}
            continue
        best_per = max(per_to_f1, key=per_to_f1.get)
        concept_to_best[concept] = {'f1': per_to_f1[best_per], 'detect_percentile': best_per}
    return concept_to_best


### Training ###
def filter_concept_by_patch_activations(split, embeddings, concept_activations, top_percent,
                                        concept_labels, dataset_name, model_input_size, impose_negatives,
                                       highest_negatives, add_noise=0):
    """
    Filters and labels embeddings for a single concept by selecting the top and negative activations.

    Args:
        embeddings (torch.Tensor): Subset of all patch embeddings (shape: [n_samples, hidden_dim]).
        concept_activations (pd.Series): Activation scores for one concept (shape: [n_samples]).
        top_percent (float): Percentile of top activations to select.
        impose_negatives (bool): If True, select most negative activations instead of random negatives.

    Returns:
        (torch.Tensor, torch.Tensor): Filtered embeddings and corresponding binary labels.
    """
    # Get intersection of relevant and training/test indices
    split_df = get_patch_split_df(dataset_name, model_input_size)
    nonpadding_indices = filter_patches_by_image_presence(split_df.index, dataset_name, model_input_size).tolist()

    split_indices = split_df[split_df == split].index
    relevant_indices = sorted(list(set(split_indices).intersection(nonpadding_indices)))

    # filter concept activations 
    concept_activations = concept_activations.loc[relevant_indices]


    if concept_labels is not None:
        # Get GT-positive and GT-negative index sets
        pos_gt = sorted((concept_labels == 1).nonzero(as_tuple=True)[0].tolist())
        neg_gt = sorted((concept_labels == 0).nonzero(as_tuple=True)[0].tolist())
        
        # Keep only relevant ones
        pos_gt = sorted(list(set(pos_gt).intersection(set(relevant_indices))))
        neg_gt = sorted(list(set(neg_gt).intersection(set(relevant_indices))))
        
        n_total = min(len(pos_gt), len(neg_gt))
        n_top = int(np.ceil(top_percent * n_total))
        n_noise = int(np.ceil(add_noise * n_top))
        n_core = n_top - n_noise

        # Restrict activations to only positive and negative GTs
        pos_activations = concept_activations.loc[pos_gt]
        neg_activations = concept_activations.loc[neg_gt]
        
        # POSITIVES
        pos_core = pos_activations.nlargest(n_core).index.to_numpy()
        available_pos_noise = list(set(pos_activations.index) - set(pos_core))
        pos_noise = np.random.choice(available_pos_noise, 
                                     size=min(n_noise, len(available_pos_noise)), 
                                     replace=False) if n_noise > 0 else []
        pos_indices = np.concatenate([pos_core, pos_noise])

        # NEGATIVES
        if impose_negatives:
            neg_core = neg_activations.nsmallest(n_core).index.to_numpy()
        elif highest_negatives:
            neg_core = neg_activations.nlargest(n_core).index.to_numpy()
        else:
            neg_core = np.random.choice(neg_activations.index, size=n_core, replace=False)
        available_neg_noise = list(set(neg_activations.index) - set(neg_core))
        neg_noise = np.random.choice(available_neg_noise, 
                                     size=min(n_noise, len(available_neg_noise)), 
                                     replace=False) if n_noise > 0 else []
        neg_indices = np.concatenate([neg_core, neg_noise])

        n_to_sample = min(len(pos_indices), len(neg_indices))
        pos_indices = np.random.choice(pos_indices, size=n_to_sample, replace=False)
        neg_indices = np.random.choice(neg_indices, size=n_to_sample, replace=False)
        # print(pos_indices[:5])

        # from general_utils import load_images
        # from visualize_concepts_w_samples_utils import plot_patches_w_corr_images
        # import visualize_concepts_w_samples_utils
        # importlib.reload(visualize_concepts_w_samples_utils)
        # all_images, train_images, test_images = load_images(dataset_name='CLEVR')
        # print("pos samples", pos_indices.tolist()[:7])
        # plot_patches_w_corr_images(pos_indices.tolist()[:7], concept_activations, all_images, 'Blah', (224, 224),
        #                        save_path=None, patch_size=14, metric_type='CosSim')
        # print("neg samples", neg_indices.tolist()[:7])
        # plot_patches_w_corr_images(neg_indices.tolist()[:7], concept_activations, all_images, 'Blah', (224, 224),
        #                        save_path=None, patch_size=14, metric_type='CosSim')
        
        all_indices = np.concatenate([pos_indices, neg_indices])
        labels = concept_labels[all_indices]

    else:
        n_total = len(concept_activations)
        n_top = int(np.ceil(top_percent * n_total))
        top_indices = concept_activations.nlargest(n_top).index.to_numpy()
        if impose_negatives:
            bot_indices = concept_activations.nsmallest(n_top).index.to_numpy()
        else:
            remaining_indices = concept_activations.drop(index=top_indices).index.to_numpy()
            bot_indices = np.random.choice(remaining_indices, size=n_top, replace=False)
        
        labels = np.concatenate([np.ones(n_top), np.zeros(n_top)])
        all_indices = np.concatenate([top_indices, bot_indices])
        labels = torch.tensor(labels, dtype=torch.float32)

    filtered_embeddings = embeddings[all_indices]
    return filtered_embeddings, labels


class ModelTrainer():
    def __init__(self, train_dl, test_dl, lr, weight_decay, lr_step_size, lr_gamma, patience, tolerance, device,
                 reset_optimizer=True):
        
        self.model = create_linear_model(len(train_dl.dataset[0][0]), device)
        self.superpatches_per = 'init'

        self.train_dl = train_dl
        self.test_dl = test_dl
        
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_gamma)
        
        self.patience = patience
        self.tolerance = tolerance
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.lr = lr
    
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        self.reset_optimizer = reset_optimizer
        
        self.logs = defaultdict(list)
        self.weights_over_epochs = [self.model.weight.detach().squeeze(0).cpu()]
        self.detect_f1s_over_epochs_train = []
        self.detect_f1s_over_epochs_test = []
    
        
    def instantiate_for_finetuning(self, new_train_dl, new_test_dl):
        self.train_dl = new_train_dl
        self.test_dl = new_test_dl
        
        if self.reset_optimizer:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        
    
    def save_f1(self, train_f1s_over_percentiles, test_f1s_over_percentiles):
        self.detect_f1s_over_epochs_train.append(train_f1s_over_percentiles)
        self.detect_f1s_over_epochs_test.append(test_f1s_over_percentiles)
        
    def get_training_logs(self):
        return self.logs
    
    def get_weights_over_epochs(self):
        return self.weights_over_epochs
    
    def get_weight_at_iteration_i(self, i):
        try:
            return self.weights_over_epochs[i]
        except: #return most recent weights if you stopped training early
            return self.get_most_recent_weights()
    
    def get_most_recent_weights(self):
        return self.weights_over_epochs[-1]

    def train_one_epoch(self, epoch, total_epochs):
        if self.early_stopped:
            return None

        self.model.train()
        for batch_features, batch_labels in self.train_dl:
            batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_features.float()).view(-1)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        train_loss, train_acc, train_f1 = evaluate_model(self.model, self.train_dl, self.criterion, self.device)
        test_loss, test_acc, test_f1 = evaluate_model(self.model, self.test_dl, self.criterion, self.device)

        self.logs = log_progress(self.logs, train_loss, train_acc, train_f1, test_loss, test_acc, test_f1, epoch, total_epochs)

        self.scheduler.step()

        if self.logs['train_f1'][-1] >= 0.99:
            print(f"Early stopping at epoch {epoch + 1}")
            self.early_stopped = True
        elif epoch > 0 and (self.best_loss - self.logs['train_loss'][-1]) < self.tolerance:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                self.early_stopped = True
        else:
            self.patience_counter = 0

        self.best_loss = min(self.best_loss, self.logs['train_loss'][-1])
        
        curr_weights = self.model.weight.detach().squeeze(0).cpu()
        self.weights_over_epochs.append(curr_weights)


def get_all_concept_weights(model_trainers):
    curr_weights = {c : model_trainer.get_most_recent_weights() for c, model_trainer in model_trainers.items()}
    return curr_weights

    
def strip_dataloaders_from_trainers(trainers):
    stripped = {}
    for concept, trainer in trainers.items():
        trainer_copy = copy.deepcopy(trainer)
        trainer_copy.train_dl = None
        trainer_copy.test_dl = None
        stripped[concept] = trainer_copy
    return stripped

def compute_linear_separators_finetuned_w_superpatches(fine_tuning_params, embeds, gt_patches_per_concept, 
                                                       gt_samples_per_concept_train,
                                                       gt_patches_per_concept_test, gt_samples_per_concept_test,
                                                         dataset_name, model_input_size, device='cuda', output_file=None,
                                                         lr=0.01, train_batch_size=32, dist_batch_size=1000, patience=15, 
                                                         tolerance=3, weight_decay=1e-4, lr_step_size=10, lr_gamma=0.5,
                                                          balance_data=True, balance_negatives=False, use_gt_labels=True,
                                                         impose_negatives=False, highest_negatives=False, reset_optimizer=True, 
                                                        add_noise=0, temp_file=f'/scratch/temp/temp.csv'):
    """
    Trains and fine-tunes linear classifiers for each concept using increasing percentages of 
    superpatch activations.

    The function first trains an initial set of linear classifiers using balanced data, then iteratively
    fine-tunes each classifier using the top-k% of patches most activated by the previous round's weights.

    Args:
        fine_tuning_params (list): List of (per_superpatches, n_epochs) for training.
        embeds (torch.Tensor): Patch-level embeddings of shape (n_patches, embed_dim).
        gt_samples_per_concept (dict): Mapping from concept name to indices of patches with that concept.
        dataset_name (str): Name of the dataset (used for patch filtering).
        model_input_size (tuple): Size of input images (used for patch indexing).
        device (str): Device identifier (e.g., 'cuda').
        output_file (str or None): If provided, used to save logs or weights.
        lr (float): Learning rate.
        batch_size (int): Training batch size.
        patience (int): Early stopping patience.
        tolerance (float): Minimum improvement threshold for early stopping.
        weight_decay (float): Weight decay for regularization.
        lr_step_size (int): Step size for LR scheduler.
        lr_gamma (float): Decay factor for LR scheduler.
        balance_data (bool): Whether to balance classes during initial training.
        balance_negatives (bool): Whether to balance negatives specifically during training.
        impose_negatives (bool): If True, use most negative activations as negatives during fine-tuning.

    Returns:
        final_logs (dict): Mapping from concept -> list of training logs per fine-tuning round.
        final_concept_weights (dict): Mapping from concept -> final learned weight tensor.
    """
    model_trainers= {}
    
    print("Initial Training")
    ### INIT TRAINING ###
    #compute labels
    all_concept_labels = create_binary_labels(embeds.shape[0], gt_patches_per_concept)
    
    # Separate train and test_data (filtering out patches that don't correspond to any image locations)
    train_embeds, train_all_concept_labels = sort_data_by_split(embeds, all_concept_labels, 'train', 
                                                                dataset_name, model_input_size, 'patch')
    test_embeds, test_all_concept_labels = sort_data_by_split(embeds, all_concept_labels, 'test', 
                                                              dataset_name, model_input_size, 'patch')
    for concept in gt_patches_per_concept.keys():
        train_dl, test_dl = create_dataloaders(concept, train_embeds, train_all_concept_labels, test_embeds,
                                                        test_all_concept_labels, train_batch_size, balance_data, 
                                                        balance_negatives) 
        model_trainers[concept] = ModelTrainer(train_dl, test_dl, lr, weight_decay, 
                                            lr_step_size, lr_gamma, patience, 
                                            tolerance, device, reset_optimizer)
    
    #get f1s with randomly initialized weights
    curr_weights = get_all_concept_weights(model_trainers)
    # curr_dists = compute_signed_distances(
    #         embeds, curr_weights, dataset_name, device, output_file=None, batch_size=500
    #     )
    compute_signed_distances(
        embeds, curr_weights, dataset_name, device, output_file=temp_file, batch_size=dist_batch_size
    )
    curr_dists = pd.read_csv(temp_file)
    
    train_f1s_per_concept, test_f1s_per_concept = compute_all_f1s_by_concept_and_percentile(curr_dists,
                                                  gt_samples_per_concept_train,                                                     
                                                  gt_samples_per_concept_test, gt_patches_per_concept_test,
                                                  dataset_name, model_input_size, device,
                                                  batch_size=dist_batch_size, patch_size=14)
    for concept in gt_patches_per_concept.keys():
        model_trainers[concept].save_f1(train_f1s_per_concept[concept], test_f1s_per_concept[concept])
        
    
    total_init_epochs = fine_tuning_params[0][1]
    for epoch in range(total_init_epochs):
        #train each concept 1 epoch
        for concept in gt_patches_per_concept.keys():
            concept_model_trainer = model_trainers[concept]
            concept_model_trainer.train_one_epoch(epoch, total_init_epochs)
        
        #compute and store detection f1 and store them
        curr_weights = get_all_concept_weights(model_trainers)
        # curr_dists = compute_signed_distances(
        #     embeds, curr_weights, dataset_name, device, output_file=None, batch_size=500
        # )
        compute_signed_distances(
        embeds, curr_weights, dataset_name, device, output_file=temp_file, batch_size=dist_batch_size
        )
        curr_dists = pd.read_csv(temp_file)
        train_f1s_per_concept, test_f1s_per_concept = compute_all_f1s_by_concept_and_percentile(curr_dists,
                                                      gt_samples_per_concept_train,                                                     
                                                      gt_samples_per_concept_test, gt_patches_per_concept_test,
                                                      dataset_name, model_input_size, device,
                                                      batch_size=dist_batch_size, patch_size=14)
        for concept in gt_patches_per_concept.keys():
            model_trainers[concept].save_f1(train_f1s_per_concept[concept], test_f1s_per_concept[concept])
              
        
    ### Fine-Tuning ###
    if not use_gt_labels: #consider only superpatches as 'positive' examples, else use actual labels
        all_concept_labels = {concept:None for concept in gt_patches_per_concept.keys()}
        
    curr_weights = get_all_concept_weights(model_trainers)
    compute_signed_distances(
        embeds, curr_weights, dataset_name, device, output_file=temp_file, batch_size=dist_batch_size
        )
    curr_dists = pd.read_csv(temp_file)
    
    for i, (per, finetune_epochs) in enumerate(fine_tuning_params):
        if per == 'init':
            continue
            
        print(f"Fine tuning model with top {per*100}% of superpatches")  
        #instantiate trainers for new dataset (using superpatches)
        for concept, concept_weights in curr_weights.items():
            #get the training embeds/labels for the next round of fine-tuning based on per% superpatches
            train_embeds, train_labels = filter_concept_by_patch_activations('train', embeds, curr_dists[concept], per,
                                        all_concept_labels[concept], dataset_name, model_input_size, impose_negatives,
                                                                            highest_negatives, add_noise)
            test_embeds, test_labels = filter_concept_by_patch_activations('test', embeds, curr_dists[concept], per,
                                        all_concept_labels[concept], dataset_name, model_input_size, impose_negatives,
                                                                          highest_negatives, add_noise)
            
            print(f"Fine-tuning concept {concept} {len(train_embeds)} training samples, {len(test_embeds)} test samples")
            if train_embeds.shape[0] == 0 or test_embeds.shape[0] == 0: 
                continue
                
            new_train_dl = create_dataloader(train_embeds, train_labels, train_batch_size, shuffle=True)
            new_test_dl = create_dataloader(test_embeds, test_labels, train_batch_size, shuffle=False)
            model_trainers[concept].instantiate_for_finetuning(new_train_dl, new_test_dl)
            
            
        for epoch in range(finetune_epochs):
            for concept in gt_patches_per_concept.keys():
                concept_model_trainer = model_trainers[concept]
                concept_model_trainer.train_one_epoch(epoch, finetune_epochs)
                
            
            #compute and store detection f1 and store them
            curr_weights = get_all_concept_weights(model_trainers)
            compute_signed_distances(
            embeds, curr_weights, dataset_name, device, output_file=temp_file, batch_size=dist_batch_size
            )
            curr_dists = pd.read_csv(temp_file)
            # curr_dists = compute_signed_distances(
            #     embeds, curr_weights, dataset_name, device, output_file=None, batch_size=500
            # )

            train_f1s_per_concept, test_f1s_per_concept = compute_all_f1s_by_concept_and_percentile(curr_dists,
                                                      gt_samples_per_concept_train,                                                     
                                                      gt_samples_per_concept_test, gt_patches_per_concept_test,
                                                      dataset_name, model_input_size, device,
                                                      batch_size=dist_batch_size, patch_size=14)
            for concept in gt_patches_per_concept.keys():
                model_trainers[concept].save_f1(train_f1s_per_concept[concept], test_f1s_per_concept[concept])
    
        
    #strip trainers of dataloaders and save them
    stripped_trainers = strip_dataloaders_from_trainers(model_trainers)
    out = f'finetuned_{fine_tuning_params}_{output_file}'
    if add_noise > 0:
        out = f'noise_{add_noise}_' + out
    if impose_negatives:
        out = 'impose_neg_' + out
    if highest_negatives:
        out = 'highest_neg_' + out
    if use_gt_labels:
        out = 'gtlabels_'+ out
    if not reset_optimizer:
        out = 'noreset_'+ out
    torch.save(stripped_trainers, f'Model_Trainers/{dataset_name}/{out}')
    print(f"Concept trainers saved to Model_Trainers/{dataset_name}/{out} :)")
    
    
    return model_trainers


### Evals ###
def plot_weighted_avg_best_f1_per_epoch(trainers, fine_tuning_params, gt_samples_per_concept,
                                        max_epoch=None, baseline_trainers=None):
    """
    Plots weighted average train and test F1s across concepts per epoch.
    Legend shows scheme color and black style lines for train/test. Legend is placed to the right.
    """
    total_epochs = sum(n_epochs for _, n_epochs in fine_tuning_params) + 1  # +1 for epoch 0

    def compute_weighted_curve(trainers, attr, max_len):
        f1_sums = [0.0] * max_len
        weights = [0.0] * max_len
        for concept, trainer in trainers.items():
            f1_list = getattr(trainer, attr, [])
            weight = len(gt_samples_per_concept.get(concept, []))
            for i in range(min(len(f1_list), max_len)):
                result = f1_list[i]
                if result:
                    f1 = float(max(result.values()))
                    f1_sums[i] += f1 * weight
                    weights[i] += weight
        return [s / w if w > 0 else 0.0 for s, w in zip(f1_sums, weights)]

    avg_test_f1s = compute_weighted_curve(trainers, 'detect_f1s_over_epochs_test', total_epochs)
    avg_train_f1s = compute_weighted_curve(trainers, 'detect_f1s_over_epochs_train', total_epochs)

    if max_epoch is not None:
        avg_test_f1s = avg_test_f1s[:max_epoch]
        avg_train_f1s = avg_train_f1s[:max_epoch]

    epochs = list(range(len(avg_test_f1s)))  # Start from 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(0, 1.1)

    # Plot finetuning curves in color (e.g., blue)
    finetuning_color = 'tab:blue'
    ax.plot(epochs, avg_test_f1s, linestyle='solid', color=finetuning_color)
    ax.plot(epochs, avg_train_f1s, linestyle='dotted', color=finetuning_color)

    # Add vertical lines and labels
    curr_epoch = 0
    for patch_percent, num_epochs in fine_tuning_params:
        ax.axvline(x=curr_epoch, linestyle='dotted', color='brown', alpha=0.6)
        ymin, ymax = ax.get_ylim()
        y_offset = 0.02 * (ymax - ymin)
        if patch_percent != 'init':
            patch_percent = f"{patch_percent * 100:.0f}%"
        ax.text(curr_epoch + 0.1, ymin + y_offset, patch_percent,
                rotation=0, color='brown', ha='left', va='bottom', fontsize=9)
        curr_epoch += num_epochs

    # Plot baseline if provided
    if baseline_trainers is not None:
        baseline_test = compute_weighted_curve(baseline_trainers, 'detect_f1s_over_epochs_test', len(avg_test_f1s))
        baseline_train = compute_weighted_curve(baseline_trainers, 'detect_f1s_over_epochs_train', len(avg_train_f1s))
        baseline_epochs = list(range(len(baseline_test)))

        baseline_color = 'tab:gray'
        ax.plot(baseline_epochs, baseline_test, linestyle='solid', color=baseline_color)
        ax.plot(baseline_epochs, baseline_train, linestyle='dotted', color=baseline_color)

        # Add dummy handles for legend
        ax.plot([], [], linestyle='solid', color=baseline_color, label='Normal Training')
        ax.plot([], [], linestyle='solid', color=finetuning_color, label='Finetuning Scheme')
    else:
        # Add dummy label only for finetuning if no baseline
        ax.plot([], [], linestyle='solid', color=finetuning_color, label='Finetuning Scheme')

    # Add black lines to explain styles
    ax.plot([], [], linestyle='dotted', color='black', label='train')
    ax.plot([], [], linestyle='solid', color='black', label='test')

    if max_epoch is not None:
        ax.set_xlim(0, max_epoch - 1)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted Avg Best Detection F1')
    ax.set_title('Weighted Avg Detection F1 over Epochs')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Shift plot to make room for external legend
    plt.subplots_adjust(right=0.75)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    plt.tight_layout()
    plt.show()

    
def plot_best_f1_per_epoch_per_concept(trainers, fine_tuning_params, max_epoch=None):
    """
    Plots train and test best F1s per concept across epochs.
    Epoch 0 included. Solid = test, Dotted = train. Same color per concept.
    """
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1.1)

    concepts = list(trainers.keys())
    color_palette = sns.color_palette("husl", len(concepts))
    concept_to_color = {concept: color_palette[i] for i, concept in enumerate(concepts)}

    for concept in concepts:
        trainer = trainers[concept]

        def extract_f1(attr):
            f1s = []
            for result in getattr(trainer, attr, []):
                f1s.append(float(max(result.values())) if result else 0.0)
            return f1s[:max_epoch] if max_epoch else f1s

        test_f1s = extract_f1('detect_f1s_over_epochs_test')
        train_f1s = extract_f1('detect_f1s_over_epochs_train')
        epochs = list(range(len(test_f1s)))  # Start from 0

        color = concept_to_color[concept]
        plt.plot(epochs, test_f1s, label=f'{concept} (test)', linestyle='solid', color=color)
        plt.plot(epochs, train_f1s, label=f'{concept} (train)', linestyle='dotted', color=color)

    # Vertical lines (starting from epoch 1)
    curr_epoch = 0
    for patch_percent, num_epochs in fine_tuning_params:
        plt.axvline(x=curr_epoch, linestyle='dotted', color='brown', alpha=0.6)
        ymin, ymax = plt.ylim()
        y_offset = 0.02 * (ymax - ymin)
        if patch_percent != 'init':
            patch_percent = f"{patch_percent * 100}%"
        plt.text(curr_epoch + 0.1, ymin + y_offset, patch_percent,
                 rotation=0, color='brown', ha='left', va='bottom', fontsize=9)
        curr_epoch += num_epochs

    if max_epoch is not None:
        plt.xlim(0, max_epoch - 1)

    plt.xlabel('Epoch')
    plt.ylabel('Best Detection F1 (over percentiles)')
    plt.title('Detection F1 per Concept (Train vs Test)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def compute_f1_over_iterations(percentiles, concepts_across_iterations, embeds, 
                                   gt_images_per_concept_test, gt_patches_per_concept_test, dataset_name, 
                                   model_input_size, device, 
                                   batch_size=5000, patch_size=14):
    """
    Computes best F1 score over iterations (at image level),
    where activated_images_test is already a dict per concept.
    """

    results_per_iteration = []

    for iteration_idx, concepts_iter in enumerate(concepts_across_iterations):
        print(f"\n[Iteration {iteration_idx}] Computing similarities...")
        sim_metrics = compute_signed_distances(
            embeds, concepts_iter, dataset_name, device, output_file=None, batch_size=batch_size
        )
        concept_to_best_f1 = get_best_f1s_per_concept(percentiles, sim_metrics,
                                      gt_images_per_concept_test, gt_patches_per_concept_test,
                                      dataset_name, model_input_size, device,
                                      batch_size=5000, patch_size=14)
        results_per_iteration.apend(concept_to_best_f1)

    return results_per_iteration


def plot_f1_over_iterations(results_per_iteration, fine_tuning_params, show_percentile_table=True, concepts_to_plot=None):
    """
    Plots F1 over fine-tuning epochs for each concept and shows best detect percentile table below.

    Args:
        results_per_iteration: list of dicts (per iteration) of concept -> {'f1': f1_value, 'detect_percentile': per}
        fine_tuning_params: list of (superpatch_percent, num_epochs) tuples
        show_percentile_table: whether to show the detect percentile table below
        concepts_to_plot: list of concepts to plot (default all)
    """

    # Build list of cumulative ending epochs
    cumulative_epochs = []
    total = 0
    for _, num_epochs in fine_tuning_params:
        total += num_epochs
        cumulative_epochs.append(total)

    # Organize F1s and best detect_percentiles
    concept_to_f1s = {}
    concept_to_detect_percents = {}

    for concept in results_per_iteration[0].keys():
        f1s = []
        detect_percents = []
        for result in results_per_iteration:
            f1_val = result[concept]['f1']
            detect_p = result[concept]['detect_percentile']
            f1s.append(f1_val)
            detect_percents.append(detect_p)
        concept_to_f1s[concept] = f1s
        concept_to_detect_percents[concept] = detect_percents

    # Subset concepts if specified
    if concepts_to_plot is None:
        concepts_to_plot = list(concept_to_f1s.keys())

    # Build label mapping for superpatch phases
    phase_labels = []
    for phase, _ in fine_tuning_params:
        if isinstance(phase, str):
            phase_labels.append(phase)
        else:
            phase_labels.append(f"{int(phase*100)}%")

    # === Create Figure ===
    fig, (ax_plot, ax_table) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- Top: Line Plot ---
    for concept in concepts_to_plot:
        ax_plot.plot(cumulative_epochs, concept_to_f1s[concept], label=concept)
        ax_plot.scatter(cumulative_epochs, concept_to_f1s[concept], s=60, marker='o')

    ylim = ax_plot.get_ylim()
    for idx, epoch in enumerate(cumulative_epochs):
        label = phase_labels[idx]
        ax_plot.text(epoch, ylim[1]*1.02, label, rotation=0, ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax_plot.set_xlabel("Epoch")
    ax_plot.set_ylabel("Best F1 Score")
    ax_plot.set_title("Detection F1 over Fine-Tuning Epochs")
    ax_plot.grid(True)
    ax_plot.legend()
    
    # --- Bottom: Table ---
    if show_percentile_table:
        # Create DataFrame
        detect_percentile_table = pd.DataFrame({
            concept: [f"{int(p*100)}%" for p in concept_to_detect_percents[concept]]
            for concept in concepts_to_plot
        }, index=[f"Epoch {ep}" for ep in cumulative_epochs])

        ax_table.axis('off')  # Turn off axis

        # Plot the table
        table = ax_table.table(
            cellText=detect_percentile_table.values,
            rowLabels=detect_percentile_table.index,
            colLabels=detect_percentile_table.columns,
            loc='center',
            cellLoc='center'
        )
        table.scale(1.2, 1.5)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        ax_table.set_title('Detect Percentile Producing Best F1 per Concept per Epoch', pad=10, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
    
    
    
def get_best_detect_percentiles_per_epoch_table(trainers):
    """
    Constructs a DataFrame showing the best detection percentile per concept for each epoch.

    Args:
        trainers (dict): concept -> ModelTrainer object
                         each with .detect_f1s_over_percentiles_test: list of dicts {percentile: f1}

    Returns:
        pd.DataFrame: Rows = epochs, Columns = concepts, Values = best percentile at that epoch
    """
    # First, determine the max number of epochs across concepts
    max_epochs = max(len(trainer.detect_f1s_over_epochs_test) for trainer in trainers.values())

    # Initialize rows as list of dicts
    rows = []
    for epoch_idx in range(max_epochs):
        row = {}
        for concept, trainer in trainers.items():
            if epoch_idx < len(trainer.detect_f1s_over_epochs_test):
                epoch_f1s = trainer.detect_f1s_over_epochs_test[epoch_idx]
                if epoch_f1s:
                    best_per = max(epoch_f1s, key=epoch_f1s.get)
                else:
                    best_per = None
            else:
                best_per = None
            row[concept] = best_per
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index.name = 'Epoch'
    return df


    
    
    
#### Plot Training ####
# def plot_weighted_average_metric(model_trainers, fine_tuning_params, gt_samples_per_concept,
#                                   metric, baseline_trainers=None):
#     """
#     Plots the weighted average of a metric across all concepts using sample count weights.
#     Includes optional baseline in gray, labeled "normal training".
#     """
#     train_metric = f"train_{metric}"
#     test_metric = f"test_{metric}"
#     total_epochs = sum(n_epochs for _, n_epochs in fine_tuning_params)

#     def compute_weighted_average(trainers, max_epochs):
#         weighted_train = [0.0] * max_epochs
#         weighted_test = [0.0] * max_epochs
#         total_weights = [0.0] * max_epochs

#         for concept, model_trainer in trainers.items():
#             logs = model_trainer.get_training_logs()
#             weight = len(gt_samples_per_concept.get(concept, 1))
#             train_vals = logs.get(train_metric, [])
#             test_vals = logs.get(test_metric, [])

#             for i in range(min(len(train_vals), max_epochs)):
#                 weighted_train[i] += train_vals[i] * weight
#                 weighted_test[i] += test_vals[i] * weight
#                 total_weights[i] += weight

#         avg_train = [t / w if w > 0 else 0 for t, w in zip(weighted_train, total_weights)]
#         avg_test = [t / w if w > 0 else 0 for t, w in zip(weighted_test, total_weights)]
#         return avg_train, avg_test

#     avg_train, avg_test = compute_weighted_average(model_trainers, total_epochs)

#     plt.figure(figsize=(8, 5))
#     plt.title(f"Weighted Average F1 Across All Concepts")

#     color_palette = sns.color_palette("husl", len(fine_tuning_params))
#     seen_patch_percents = set()
#     added_legend_lines = {"train": False, "test": False}

#     start_epoch = 1
#     for stage_idx, (patch_percent, num_epochs) in enumerate(fine_tuning_params):
#         end_epoch = start_epoch + num_epochs
#         color = color_palette[stage_idx]

#         stage_epochs = list(range(start_epoch, end_epoch))
#         stage_train_vals = avg_train[start_epoch - 1:end_epoch - 1]
#         stage_test_vals = avg_test[start_epoch - 1:end_epoch - 1]

#         # Only add patch_percent once in the legend
#         patch_label = f"{patch_percent}" if patch_percent not in seen_patch_percents else None
#         seen_patch_percents.add(patch_percent)

#         # Only add 'train' and 'test' labels once
#         train_label = f"train" if not added_legend_lines["train"] else None
#         test_label = f"test" if not added_legend_lines["test"] else None
#         added_legend_lines["train"] = True
#         added_legend_lines["test"] = True

#         plt.plot(stage_epochs, stage_train_vals, linestyle='dotted', color=color, label=patch_label if train_label else None)
#         plt.plot(stage_epochs, stage_test_vals, linestyle='solid', color=color, label=test_label if test_label else None)

#         start_epoch = end_epoch

#     # Baseline (if provided)
#     if baseline_trainers is not None:
#         base_train, base_test = compute_weighted_average(baseline_trainers, total_epochs)
#         epoch_range = list(range(1, total_epochs + 1))

#         plt.plot(epoch_range, base_train, linestyle='dotted', color='gray', label='normal training')
#         plt.plot(epoch_range, base_test, linestyle='solid', color='gray')

#     plt.xlabel("Epoch")
#     plt.ylabel("Best Detection F1")
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
def plot_weighted_average_metric(model_trainers, fine_tuning_params, gt_samples_per_concept,
                                 metric_type, total_epochs=None, baseline_model_trainers=None):
    """
    Plots the weighted average of a metric across all concepts using sample count weights.
    Curves are colored by patch percentile. One black dotted and one black solid line appear in the legend
    to indicate 'train' and 'test' respectively. Optionally includes a gray baseline curve.
    """
    train_metric = f"train_{metric_type}"
    test_metric = f"test_{metric_type}"

    if total_epochs is None:
        total_epochs = sum(n_epochs for _, n_epochs in fine_tuning_params) + 1

    def compute_weighted_average(trainers):
        weighted_train = [0.0] * total_epochs
        weighted_test = [0.0] * total_epochs
        total_weights = [0.0] * total_epochs

        for concept, model_trainer in trainers.items():
            logs = model_trainer.get_training_logs()
            weight = len(gt_samples_per_concept.get(concept, 1))
            train_vals = logs.get(train_metric, [])
            test_vals = logs.get(test_metric, [])

            # Pad to total_epochs with final value
            if len(train_vals) < total_epochs and len(train_vals) > 0:
                train_vals = train_vals + [train_vals[-1]] * (total_epochs - len(train_vals))
            if len(test_vals) < total_epochs and len(test_vals) > 0:
                test_vals = test_vals + [test_vals[-1]] * (total_epochs - len(test_vals))

            for i in range(total_epochs):
                weighted_train[i] += train_vals[i] * weight
                weighted_test[i] += test_vals[i] * weight
                total_weights[i] += weight

        avg_train = [t / w if w > 0 else 0 for t, w in zip(weighted_train, total_weights)]
        avg_test = [t / w if w > 0 else 0 for t, w in zip(weighted_test, total_weights)]
        return avg_train, avg_test

    avg_train, avg_test = compute_weighted_average(model_trainers)

    color_palette = sns.color_palette("husl", len(fine_tuning_params))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Weighted Average {metric_type.capitalize()} Across All Concepts")

    seen_patch_percents = set()
    epoch_cursor = 0

    for stage_idx, (patch_percent, num_epochs) in enumerate(fine_tuning_params):
        start_epoch = epoch_cursor + 1
        end_epoch = epoch_cursor + num_epochs

        stage_epochs = list(range(start_epoch, end_epoch + 1))
        stage_train_vals = avg_train[epoch_cursor:end_epoch]
        stage_test_vals = avg_test[epoch_cursor:end_epoch]

        patch_color = color_palette[stage_idx]

        ax.plot(stage_epochs, stage_train_vals, linestyle='dotted', color=patch_color, marker='o', markersize=3)
        ax.plot(stage_epochs, stage_test_vals, linestyle='solid', color=patch_color, marker='o', markersize=3)

        if patch_percent not in seen_patch_percents:
            ax.plot([], [], linestyle='solid', color=patch_color, label=f"{patch_percent}")
            seen_patch_percents.add(patch_percent)

        epoch_cursor = end_epoch

    # Optional baseline
    if baseline_model_trainers is not None:
        base_train, base_test = compute_weighted_average(baseline_model_trainers)
        epoch_range = list(range(1, total_epochs + 1))
        ax.plot(epoch_range, base_train, linestyle='dotted', color='gray', label='baseline')
        ax.plot(epoch_range, base_test, linestyle='solid', color='gray')

    # Add black lines to legend for style explanation
    ax.plot([], [], linestyle='dotted', color='black', label='train')
    ax.plot([], [], linestyle='solid', color='black', label='test')

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_type.capitalize())
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.subplots_adjust(right=0.75)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    
def plot_all_concepts_metric(model_trainers, fine_tuning_params, metric_type):
    """
    Plots train/test metric curves over fine-tuning stages for each concept.

    Args:
        logs (dict): logs[concept][metric] = list of values
        metric_type (str): Base metric type to plot ('loss', 'accuracy', 'f1', etc.)
        fine_tuning_params (list of (patch_percent, num_epochs)): Stage info for coloring
        max_cols (int): Max number of subplots per row
        figsize_per_plot (tuple): Size per individual subplot
    """
    concepts = list(model_trainers.keys())
    n_concepts = len(concepts)
    n_cols = min(n_concepts, 3)
    n_rows = math.ceil(n_concepts / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    color_palette = sns.color_palette("husl", len(fine_tuning_params))

    for idx, (concept, model_trainer) in enumerate(model_trainers.items()):
        logs = model_trainer.get_training_logs()
        
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        
        train_metric = f"train_{metric_type}"
        test_metric = f"test_{metric_type}"
        start_epoch = 1
        for stage_idx, (patch_percent, num_epochs) in enumerate(fine_tuning_params):
            end_epoch = start_epoch + num_epochs
            color = color_palette[stage_idx]
            label_prefix = f"{patch_percent} ({num_epochs}e)"

            # Inclusive range: [start_epoch, ..., start_epoch + num_epochs]
            epoch_range = list(range(start_epoch, end_epoch))

            train_vals = logs[train_metric][start_epoch - 1: end_epoch]
            test_vals = logs[test_metric][start_epoch - 1: end_epoch]

            ax.plot(epoch_range, train_vals, linestyle='dotted', color=color, label=f"{label_prefix} - train", marker='o')
            ax.plot(epoch_range, test_vals, linestyle='solid', color=color, label=f"{label_prefix} - test", marker='o')

            start_epoch = end_epoch

        ax.set_title(concept, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_type.capitalize())
        ax.grid(True, linestyle="--", alpha=0.5)

        if idx == 0:
            ax.legend(fontsize=8)

    # Hide any empty subplots
    for i in range(n_concepts, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()
    
    