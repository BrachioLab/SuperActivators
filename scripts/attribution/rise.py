import torch
import torch.nn.functional as F
from tqdm import tqdm

from util import patches_to_image, reembed_patch_embeddings_batch


def concept_score_cossim(embedding: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    """
    embedding: [P, D] or [B, P, D]
    concept:   [K, D]
    returns:   [B, K]
    """
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    emb = embedding.unsqueeze(2)                 # [B, P, 1, D]
    con = concept.unsqueeze(0).unsqueeze(0)      # [1, 1, K, D]
    sim = F.cosine_similarity(emb, con, dim=-1)  # [B, P, K]
    return sim.mean(dim=1)                       # [B, K]


def concept_score_linsep(embedding: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    """
    embedding: [P, D] or [B, P, D]
    concept:   [K, D]
    returns:   [B, K]
    """
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


def rise_concept(
    X: torch.Tensor,                 # [1, P, Din]
    c: int,                          # concept index
    concept_vectors: torch.Tensor,    # [K, Dnew]
    model,
    processor,
    model_name: str,                 # "CLIP" or "Llama"
    image_size: tuple,               # (224, 224)
    score_mode: str = "cossim",       # "cossim" or "linsep"
    percent_thru_model: int = 100,
    n_masks: int = 1000,             # number of masks
    gpu_batch: int = 16,
    p1: float = 0.5,                 # keep probability
    device: str = "cuda",
    normalize_by_mask_counts: bool = True, 
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
    importance = torch.zeros(1, P, device=device)

    masks = (torch.rand(n_masks, P, device=device) < p1).float()

    for i in tqdm(range(0, n_masks, gpu_batch), desc="RISE-Concept"):
        m = masks[i : i + gpu_batch]             # [B_curr, P]
        X_masked = X * m.unsqueeze(-1)           # [B_curr, P, Din]

        with torch.no_grad():
            img_batch = patches_to_image(X_masked, image_size=image_size)
            emb_batch = reembed_patch_embeddings_batch(
                images=img_batch,
                model=model,
                processor=processor,
                device=str(device),
                model_name=model_name,
                image_size=image_size,
                percent_thru_model=percent_thru_model,
            ).to(device)                         # [B_curr, P, Dnew]

            scores = concept_score_fn(emb_batch, concept_vectors)  # [B_curr, K]
            target_scores = scores[:, c]                           # [B_curr]

        importance += (target_scores[:, None] * m).sum(dim=0, keepdim=True)

    if normalize_by_mask_counts:
        denom = masks.sum(dim=0, keepdim=True).clamp_min(eps)  # [1, P]
        importance = importance / denom
    else:
        importance = importance / (n_masks * p1 + eps)

    return importance.view(-1).detach().cpu()