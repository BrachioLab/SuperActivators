import torch
import torch.nn.functional as F
from tqdm import tqdm

from util import patches_to_image, reembed_patch_embeddings_batch_diff


def concept_score_cossim(embedding: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    """
    embedding: [P,D] or [B,P,D]
    concept:   [K,D]
    returns:   [B,K]
    """
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
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
        embedding = embedding.unsqueeze(0)
    proj = torch.matmul(embedding, concept.t())  # [B,P,K]
    return proj.mean(dim=1)                      # [B,K]


def get_concept_score_fn(mode: str):
    mode = mode.lower()
    if mode == "cossim":
        return concept_score_cossim
    if mode == "linsep":
        return concept_score_linsep
    raise ValueError("mode must be 'cossim' or 'linsep'")


def intgrad_concept(
    X: torch.Tensor,
    c: int,
    concept_vectors: torch.Tensor,
    model,
    processor,    
    model_name: str,
    image_size: tuple,
    x0: torch.Tensor = None,
    score_mode: str = "cossim",
    percent_thru_model: int = 100,
    num_steps: int = 64,
    device: str = "cuda",
    progress_bar: bool = False,
    debug: bool = False,
) -> torch.Tensor:

    device = torch.device(device)
    X = X.to(device)
    concept_vectors = concept_vectors.to(device)
    concept_score_fn = get_concept_score_fn(score_mode)

    if x0 is None:
        x0 = X.mean(dim=1, keepdim=True).expand_as(X).detach()
    else:
        x0 = x0.to(device)

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)
    total_grads = torch.zeros_like(X)
    prev_grads = None

    iterator = tqdm(alphas) if progress_bar else alphas

    with torch.enable_grad():
        for i, alpha in enumerate(iterator):
            xk = (x0 + alpha * (X - x0)).detach().requires_grad_(True)

            img_k = patches_to_image(xk, image_size=image_size)  # [B,3,H,W]
            emb_k = reembed_patch_embeddings_batch_diff(
                images=img_k,
                model=model,
                model_name=model_name,
                image_size=image_size,
                percent_thru_model=percent_thru_model,
            ) 

            score = concept_score_fn(emb_k, concept_vectors)  # [B,K]
            target = score[:, c].sum() 

            grads = torch.autograd.grad(target, xk, retain_graph=False, create_graph=False)[0]

            if prev_grads is not None:
                total_grads += 0.5 * (prev_grads + grads) * (1.0 / num_steps)
            prev_grads = grads

            if debug and i == 1:
                print("debug | target:", float(target.detach().cpu()))
                print("debug | grad_abs_mean:", float(grads.abs().mean().detach().cpu()))

    with torch.no_grad():
        importance = (X - x0) * total_grads   # [B,P,Din]
        importance = importance.sum(dim=-1)   # [B,P]
        return importance.detach().cpu()