import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap

from util import patches_to_image, reembed_patch_embeddings_batch

def concept_score_cossim(embedding: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    """
    embedding: [P,D] or [B,P,D]
    concept:   [K,D]
    returns:   [B,K] (or [1,K] if input was [P,D])
    """
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)  # [1,P,D]
    
    emb = embedding.unsqueeze(2)                 # [B,P,1,D]
    con = concept.unsqueeze(0).unsqueeze(0)      # [1,1,K,D]
    sim = F.cosine_similarity(emb, con, dim=-1)  # [B,P,K]
    return sim.mean(dim=1)                       # [B,K]


def concept_score_linsep(embedding: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    """
    embedding: [P,D] or [B,P,D]
    concept:   [K,D]
    returns:   [B,K]
    """
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)  # [1,P,D]
    
    proj = torch.matmul(embedding, concept.t())  # [B,P,K]
    return proj.mean(dim=1)                      # [B,K]


def get_concept_score_fn(mode: str):
    mode = mode.lower()
    if mode == "cossim":
        return concept_score_cossim
    if mode == "linsep": 
        return concept_score_linsep
    raise ValueError("mode must be 'cossim' or 'linsep'")


def shap_concept(
    X: torch.Tensor,  
    c: int,              
    concept_vectors: torch.Tensor,  
    model,
    processor,
    model_name: str,      
    image_size: tuple,             
    score_mode: str = "cossim",  
    percent_thru_model: int = 100,
    nsamples: int = 500,
    batch_size: int = 16,
    device: str = "cuda",
) -> torch.Tensor:
    
    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError(f"X must be [1,P,Din], got {X.shape}")

    device = torch.device(device)
    concept_vectors = concept_vectors.to(device)
    concept_score_fn = get_concept_score_fn(score_mode)
    P = X.shape[1]
    Din = X.shape[2]

    def predict_fn(masks):
        masks = np.asarray(masks)
        N = masks.shape[0]
        all_scores = []
        
        for i in range(0, N, batch_size):
            batch_masks = torch.from_numpy(masks[i:i+batch_size]).to(device).float() # [B, P]
            B = batch_masks.shape[0]
            
            X_perturbed = X.expand(B, -1, -1) * batch_masks.unsqueeze(-1)
            
            with torch.no_grad():
                img_batch = patches_to_image(X_perturbed, image_size=image_size)
                emb_batch = reembed_patch_embeddings_batch(
                    images=img_batch,
                    model=model,
                    processor=processor,
                    device=str(device),
                    model_name=model_name,
                    image_size=image_size,
                    percent_thru_model=percent_thru_model,
                ).to(device) # [B, P, Dnew]
                
                scores = concept_score_fn(emb_batch, concept_vectors) # [B, K]
                target_scores = scores[:, c].cpu().numpy() # [B]
                all_scores.append(target_scores)
                
        return np.concatenate(all_scores, axis=0)

    background = np.zeros((1, P)) 
    explainer = shap.KernelExplainer(predict_fn, background)
    
    with torch.no_grad():
        shap_values = explainer.shap_values(np.ones((1, P)), nsamples=nsamples)
    
    if isinstance(shap_values, list):
        importance = torch.from_numpy(shap_values[0]).view(-1)
    else:
        importance = torch.from_numpy(shap_values).view(-1)
        
    return importance
