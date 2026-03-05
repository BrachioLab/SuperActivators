import torch
import torch.nn as nn
import torch.nn.functional as F

from util import patches_to_image, reembed_patch_embeddings_batch_diff
from saliency_zoo import mfaba_sharp, mfaba_smooth, mfaba_cos, mfaba_norm

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


class ConceptModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        image_size: tuple,
        percent_thru_model: int,
        concept_vectors: torch.Tensor,   # [K,Dnew] on same device
        score_mode: str = "cossim",
        normalize_concepts_for_cossim: bool = True,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.image_size = image_size
        self.percent_thru_model = percent_thru_model

        score_mode = score_mode.lower()
        self.score_mode = score_mode
        self.score_fn = get_concept_score_fn(score_mode)

        if score_mode == "cossim" and normalize_concepts_for_cossim:
            concept_vectors = F.normalize(concept_vectors, dim=-1)
        self.register_buffer("concept_vectors", concept_vectors)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        emb = reembed_patch_embeddings_batch_diff(
            images=img,
            model=self.model,
            model_name=self.model_name,
            image_size=self.image_size,
            percent_thru_model=self.percent_thru_model,
        )
        scores = self.score_fn(emb, self.concept_vectors)
        return scores


def mfaba_concept(
    X: torch.Tensor,                 # [B, P, Din]
    c: int,                          # concept index (0 <= c < K)
    concept_vectors: torch.Tensor,   # [K, Dnew]
    model: nn.Module,
    model_name: str,        
    image_size: tuple = (224, 224),
    mfaba_type: str = "sharp",       # "sharp" | "smooth" | "cos" | "norm"
    score_mode: str = "cossim",      # "cossim" | "linsep"
    percent_thru_model: int = 100,
    device: str = "cuda",
    drop_cls_if_present: bool = True, 
    return_pixel_map: bool = False, 
    **mfaba_kwargs,
) -> torch.Tensor:
    device = torch.device(device)
    model.eval()

    if X.ndim != 3:
        raise ValueError(f"X must be [B, P, Din], got {tuple(X.shape)}")
    B, P, Din = X.shape

    X = X.to(device)
    concept_vectors = concept_vectors.to(device)

    if concept_vectors.ndim != 2:
        raise ValueError(f"concept_vectors must be [K, Dnew], got {tuple(concept_vectors.shape)}")
    K, Dnew = concept_vectors.shape
    if not (0 <= c < K):
        raise ValueError(f"concept index c={c} out of range for K={K}")

    def _is_perfect_square(n: int) -> bool:
        r = int(n ** 0.5)
        return r * r == n

    P_used = P
    X_used = X

    if not _is_perfect_square(P) and drop_cls_if_present and P > 1 and _is_perfect_square(P - 1):
        X_used = X[:, 1:, :]   # drop CLS
        P_used = P - 1

    if not _is_perfect_square(P_used):
        raise ValueError(
            f"P_used={P_used} is not a perfect square. "
            f"(Original P={P}) If CLS token exists, set drop_cls_if_present=True and ensure P-1 is a square."
        )

    grid_size = int(P_used ** 0.5)

    wrapped_model = ConceptModelWrapper(
        model=model,
        model_name=model_name,
        image_size=image_size,
        percent_thru_model=percent_thru_model,
        concept_vectors=concept_vectors,
        score_mode=score_mode,
    ).to(device)

    mfaba_type = mfaba_type.lower()
    type_mapping = {
        "sharp": mfaba_sharp,
        "smooth": mfaba_smooth,
        "cos": mfaba_cos,
        "norm": mfaba_norm,
    }
    mfaba_fn = type_mapping.get(mfaba_type)
    if mfaba_fn is None:
        raise ValueError(f"Unknown mfaba_type='{mfaba_type}'. Choose from {list(type_mapping.keys())}.")

    img = patches_to_image(X_used, image_size=image_size).to(device)

    target = torch.full((B,), int(c), device=device, dtype=torch.long)

    with torch.enable_grad():
        attribution_np = mfaba_fn(wrapped_model, img, target, **mfaba_kwargs)

    attribution = torch.from_numpy(attribution_np).to(device).float()

    # attribution shape normalization -> [B,1,H,W]
    if attribution.ndim == 4:
        # [B,C,H,W] -> [B,1,H,W]
        attribution = attribution.abs().sum(dim=1, keepdim=True)
    elif attribution.ndim == 3:
        # [B,H,W] -> [B,1,H,W]
        attribution = attribution.unsqueeze(1).abs()
    else:
        raise ValueError(f"Unexpected attribution shape from mfaba_fn: {tuple(attribution.shape)}")

    H, W = image_size
    if H % grid_size != 0 or W % grid_size != 0:
        raise ValueError(
            f"image_size={image_size} not divisible by grid_size={grid_size} derived from P_used={P_used}."
        )

    patch_h = H // grid_size
    patch_w = W // grid_size

    pooled = F.avg_pool2d(attribution, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    importance = pooled.view(B, -1)

    if B == 1:
        importance_vec = importance.squeeze(0)
    else:
        importance_vec = importance.mean(dim=0)

    importance_vec = importance_vec.detach().cpu()

    if return_pixel_map:
        pixel_map = attribution.squeeze(1).detach().cpu()  # [B,H,W]
        if B == 1:
            pixel_map = pixel_map.squeeze(0)               # [H,W]
        return importance_vec, pixel_map

    return importance_vec