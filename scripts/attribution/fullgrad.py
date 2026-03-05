import torch
import torch.nn as nn
import torch.nn.functional as F

from util import patches_to_image, reembed_patch_embeddings_batch_diff


def concept_score_cossim(embedding: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    emb = embedding.unsqueeze(2)                 # [B, P, 1, D]
    con = concept.unsqueeze(0).unsqueeze(0)      # [1, 1, K, D]
    sim = F.cosine_similarity(emb, con, dim=-1)  # [B, P, K]
    return sim.mean(dim=1)                       # [B, K]


def concept_score_linsep(embedding: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    proj = torch.matmul(embedding, concept.t())  # [B, P, K]
    return proj.mean(dim=1)                      # [B, K]


def get_concept_score_fn(mode: str):
    mode = mode.lower()
    if mode == "cossim":
        return concept_score_cossim
    if mode == "linsep":
        return concept_score_linsep
    raise ValueError("mode must be 'cossim' or 'linsep'")


def fullgrad_concept(
    X: torch.Tensor,                 # [1, P, Din]
    c: int,                   
    concept_vectors: torch.Tensor,    # [K, Dnew]
    model,
    model_name: str,     
    image_size: tuple = (224, 224),
    score_mode: str = "cossim",
    percent_thru_model: int = 100,
    device: str = "cuda",
    include_bias: bool = True,
    signed: bool = False,         
    eps: float = 1e-6,
) -> torch.Tensor:
    device = torch.device(device)
    model.eval()

    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError(f"X must be [1, P, Din], got {X.shape}")

    X = X.to(device)
    concept_vectors = concept_vectors.to(device)

    concept_score_fn = get_concept_score_fn(score_mode)
    if score_mode.lower() == "cossim":
        concept_vectors = F.normalize(concept_vectors, dim=-1)

    P = X.shape[1]
    is_image = ("CLIP" in model_name) or ("ViT" in model_name)

    biases = []
    if include_bias:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                biases.append(m.bias)

    with torch.enable_grad():
        X_in = X.detach().requires_grad_(True)

        if is_image:
            img = patches_to_image(X_in, image_size=image_size)
            emb = reembed_patch_embeddings_batch_diff(
                images=img,
                model=model,
                model_name=model_name,
                image_size=image_size,
                percent_thru_model=percent_thru_model,
            )
        else:
            emb = reembed_patch_embeddings_batch_diff(
                images=None,
                tokens=X_in,
                model=model,
                model_name=model_name,
                percent_thru_model=percent_thru_model,
            )

        scores = concept_score_fn(emb, concept_vectors)  # [1, K]
        target_score = scores[0, c]                      # scalar

        grad_targets = [X_in] + biases
        grads = torch.autograd.grad(
            target_score,
            grad_targets,
            retain_graph=False,
            allow_unused=True,  
        )

        input_grad = grads[0]
        bias_grads = grads[1:]

    with torch.no_grad():
        contrib = X_in * input_grad  # [1, P, Din]
        if not signed:
            contrib = contrib.abs()
        importance = contrib.sum(dim=-1).squeeze(0)  # [P]

        if include_bias and len(biases) > 0:
            b_total = 0.0
            for b, bg in zip(biases, bias_grads):
                if bg is None:
                    continue
                bc = b * bg
                b_total = b_total + (bc if signed else bc.abs()).sum()
            importance = importance + (b_total / (P + eps))

    return importance.detach().cpu()