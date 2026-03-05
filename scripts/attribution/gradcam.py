import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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


class ConceptModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        image_size: tuple,
        percent_thru_model: int,
        concept_vectors: torch.Tensor,  # [K, Dnew]
        score_mode: str,
        is_image: bool,
        normalize_concepts_for_cossim: bool = True,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.image_size = image_size
        self.percent_thru_model = percent_thru_model
        self.is_image = is_image

        score_mode = score_mode.lower()
        self.score_mode = score_mode
        self.score_fn = get_concept_score_fn(score_mode)

        if score_mode == "cossim" and normalize_concepts_for_cossim:
            concept_vectors = F.normalize(concept_vectors, dim=-1)
        self.register_buffer("concept_vectors", concept_vectors)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.is_image:
            emb = reembed_patch_embeddings_batch_diff(
                images=input_tensor,
                model=self.model,
                model_name=self.model_name,
                image_size=self.image_size,
                percent_thru_model=self.percent_thru_model,
            )
        else:
            emb = reembed_patch_embeddings_batch_diff(
                images=None,
                tokens=input_tensor,
                model=self.model,
                model_name=self.model_name,
                percent_thru_model=self.percent_thru_model,
            )
        return self.score_fn(emb, self.concept_vectors)  # [B,K]


def _reshape_transform_vit_tokens_to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """
    tensor: [B, N, C] (ViT tokens) or already [B,C,H,W]
    - If N-1 is square => drop CLS and reshape
    - Else if N is square => reshape directly
    Output: [B, C, H, W]
    """
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"Expected token tensor [B,N,C] or [B,C,H,W], got shape {tuple(tensor.shape)}")

    B, N, C = tensor.shape

    if (N - 1) > 0 and (math.isqrt(N - 1) ** 2 == (N - 1)):
        t = tensor[:, 1:, :]
        s = math.isqrt(N - 1)
        out = t.reshape(B, s, s, C).permute(0, 3, 1, 2)
        return out

    s = math.isqrt(N)
    if s * s != N:
        raise ValueError(f"Token length N={N} is not a perfect square (nor N-1). Provide a custom reshape_transform.")
    out = tensor.reshape(B, s, s, C).permute(0, 3, 1, 2)
    return out


def _auto_target_layers_for_clip(model: nn.Module):
    candidates = []
    try:
        candidates.append(model.vision_model.encoder.layers[-1].layer_norm2)
    except Exception:
        pass
    try:
        candidates.append(model.vision_model.encoder.layers[-1])
    except Exception:
        pass
    try:
        candidates.append(model.vision_model.encoder.layers[-1].mlp)
    except Exception:
        pass

    if not candidates:
        raise ValueError(
            "Could not auto-detect CLIP target layer. "
            "Provide target_layers explicitly (e.g., model.vision_model.encoder.layers[-1].layer_norm2)."
        )
    return [candidates[0]]


def _auto_target_layers_for_llama_vision(model: nn.Module):
    candidates = []
    try:
        candidates.append(model.vision_tower.vision_model.encoder.layers[-1].layer_norm2)
    except Exception:
        pass
    try:
        candidates.append(model.vision_tower.vision_model.encoder.layers[-1])
    except Exception:
        pass
    try:
        candidates.append(model.model.vision_tower.vision_model.encoder.layers[-1].layer_norm2)
    except Exception:
        pass
    try:
        candidates.append(model.model.vision_tower.vision_model.encoder.layers[-1])
    except Exception:
        pass
    if not candidates:
        raise ValueError(
            "Could not auto-detect Llama-Vision target layer. "
            "Provide target_layers explicitly (e.g., model.vision_tower.vision_model.encoder.layers[-1].layer_norm2)."
        )
    return [candidates[0]]


def _is_vit_image_model(model_name: str) -> bool:
    s = model_name.lower()
    return ("clip" in s) or ("vit" in s) or ("vision" in s)


def _is_perfect_square(n: int) -> bool: 
    r = int(n ** 0.5)
    return r * r == n

def _split_cls_if_present(X: torch.Tensor, drop_cls: bool = True):
    """
    Returns: (X_used, P_used, had_cls)
    Rule: if P not square but P-1 is square -> treat first token as CLS.
    """
    P = X.shape[1]
    if drop_cls and (not _is_perfect_square(P)) and P > 1 and _is_perfect_square(P - 1):
        return X[:, 1:, :], P - 1, True
    return X, P, False


def gradcam_concept(
    X: torch.Tensor,
    c: int,
    concept_vectors: torch.Tensor,
    model: nn.Module,
    model_name: str,
    image_size: tuple = (224, 224),
    score_mode: str = "cossim",
    percent_thru_model: int = 100,
    device: str = "cuda",
    target_layers=None,
    reshape_transform=None,
    drop_cls: bool = True,        
    pad_cls_with_zero: bool = True,   
) -> torch.Tensor:
    device = torch.device(device)
    model.eval()

    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError(f"X must be [1, P, Din], got {tuple(X.shape)}")

    X = X.to(device)
    concept_vectors = concept_vectors.to(device)

    is_image = _is_vit_image_model(model_name)

    wrapped_model = ConceptModelWrapper(
        model=model,
        model_name=model_name,
        image_size=image_size,
        percent_thru_model=percent_thru_model,
        concept_vectors=concept_vectors,
        score_mode=score_mode,
        is_image=is_image,
    ).to(device)

    if target_layers is None:
        s = model_name.lower()
        if "openai/clip-vit-large-patch14" in s or ("clip" in s):
            target_layers = _auto_target_layers_for_clip(model)
        elif "meta-llama/llama-3.2-11b-vision-instruct" in s or ("llama" in s and "vision" in s):
            target_layers = _auto_target_layers_for_llama_vision(model)
        else:
            try:
                target_layers = _auto_target_layers_for_clip(model)
            except Exception:
                target_layers = _auto_target_layers_for_llama_vision(model)

    if reshape_transform is None and is_image:
        reshape_transform = _reshape_transform_vit_tokens_to_bchw

    targets = [ClassifierOutputTarget(int(c))]

    if not is_image:
        raise ValueError(
            "This version is configured for ViT image models (CLIP / Llama-Vision). "
            "If you want token/text CAM, use a text-CAM implementation and token-layer targets."
        )

    X_used, P_used, had_cls = _split_cls_if_present(X, drop_cls=drop_cls) 
    cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
    img = patches_to_image(X_used, image_size=image_size).to(device).requires_grad_(True)  
    
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=img, targets=targets)  # numpy: [1,H,W]
    attribution = torch.from_numpy(grayscale_cam).to(device).float().unsqueeze(1)

    if not _is_perfect_square(P_used):  
        raise ValueError(f"P_used={P_used} is not a perfect square; cannot pool into a grid.")

    grid = int(P_used ** 0.5)  
    H, W = image_size
    if H % grid != 0 or W % grid != 0:
        raise ValueError(f"image_size={image_size} not divisible by grid={grid} derived from P_used={P_used}")

    ph, pw = H // grid, W // grid
    pooled = F.avg_pool2d(attribution, kernel_size=(ph, pw), stride=(ph, pw))
    importance = pooled.view(-1)  # [P_used]

    if had_cls and pad_cls_with_zero: 
        importance = torch.cat([torch.zeros(1, device=device), importance], dim=0)

    return importance.detach().cpu()