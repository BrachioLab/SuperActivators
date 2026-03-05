import torch
import math
from typing import Tuple, Union, List
import torch.nn.functional as F

from utils.embedding_utils import get_clip_both_embeddings, get_llama_both_embeddings

def infer_patch_size_from_X(X: torch.Tensor):
    if X.ndim != 3:
        raise ValueError(f"Expected X [B,P,d], got {X.shape}")

    d = X.shape[-1]
    if d % 3 != 0:
        raise ValueError(f"Last dim {d} not divisible by 3")

    patch_area = d // 3
    patch_size = int(math.sqrt(patch_area))

    if patch_size * patch_size != patch_area:
        raise ValueError(f"Cannot infer square patch_size from d={d}")

    return patch_size


def patches_to_image(
    patches: torch.Tensor,
    image_size: Tuple[int, int],
):
    B, P, d = patches.shape
    H, W = image_size

    patch_size = infer_patch_size_from_X(patches)

    C = 3
    gh, gw = H // patch_size, W // patch_size

    if P != gh * gw:
        raise ValueError(
            f"P={P} does not match grid {gh}x{gw} for image_size={image_size}"
        )

    x = patches.view(B, gh, gw, patch_size, patch_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)

    return x


def _ensure_image_list(images: Union[torch.Tensor, List]) -> List:
    if isinstance(images, list):
        return images
    if isinstance(images, torch.Tensor):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images [B,3,H,W], got {images.shape}")
        return [img for img in images]
    raise TypeError("images must be list or torch.Tensor [B,3,H,W]")


@torch.no_grad()
def reembed_patch_embeddings_batch(
    images: Union[torch.Tensor, List],
    model,
    processor,
    device: str,
    model_name: str,
    image_size: Tuple[int, int],
    percent_thru_model: int = 100,
):
    imgs = _ensure_image_list(images)
    B = len(imgs)

    if model_name.lower() == "clip":
        _, patch_flat = get_clip_both_embeddings(model, processor, imgs, device, percent_thru_model)
    elif model_name.lower() == "llama":
        _, patch_flat = get_llama_both_embeddings(model, processor, imgs, device, percent_thru_model)
    else:
        raise ValueError("model_name must be 'CLIP' or 'Llama'")

    D = patch_flat.shape[-1]
    P = patch_flat.shape[0] // B

    return patch_flat.view(B, P, D)


def weight_batch(sample: torch.Tensor, original: torch.Tensor, sigma: float = 1.0):
    if original.shape[0] == 1 and sample.shape[0] > 1:
        original = original.expand(sample.shape[0], -1, -1)

    distances = torch.norm(sample - original, dim=2)
    weights = torch.exp(-(distances ** 2) / (2 * sigma ** 2))
    return weights


def _pick_hidden_state(hidden_states, percent_thru_model: int):
    if hidden_states is None:
        raise ValueError("hidden_states is None; call the vision model with output_hidden_states=True")
    pct = int(percent_thru_model)
    pct = 0 if pct < 0 else 100 if pct > 100 else pct
    n_layers = len(hidden_states) - 1

    if pct == 0:
        return hidden_states[0]
    if pct == 100:
        return hidden_states[-1]

    idx = max(1, min(n_layers - 1, int(round(pct / 100.0 * n_layers))))
    return hidden_states[idx]


def _clip_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std  = images.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (images - mean) / std


def reembed_patch_embeddings_batch_diff(
    images: torch.Tensor,        
    model,     
    model_name: str, 
    image_size: Tuple[int, int],
    percent_thru_model: int = 100,
) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor [B,3,H,W] for differentiable re-embed")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected images [B,3,H,W], got {tuple(images.shape)}")

    if images.shape[-2:] != image_size:
        images = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)

    name = model_name.lower()

    if name == "clip":
        vision = getattr(model, "vision_model", None)
        vision = vision if vision is not None else model

        pixel_values = _clip_normalize(images)
        out = vision(pixel_values=pixel_values, output_hidden_states=True)

        hs = _pick_hidden_state(out.hidden_states, percent_thru_model)  # [B,1+P,D]
        if hs.ndim != 3:
            raise ValueError(f"Unexpected CLIP hidden state shape: {tuple(hs.shape)}")
        return hs[:, 1:, :] if hs.shape[1] >= 2 else hs

    elif name == "llama":
        vision = getattr(model, "vision_model", None)
        if vision is None and hasattr(model, "model"):
            vision = getattr(model.model, "vision_model", None)
        if vision is None:
            vision = getattr(model, "vision_tower", None)
        if vision is None:
            raise ValueError("Could not find a vision module on the provided Llama/Mllama model.")

        pixel_values = images

        if "pixel_values" in vision.forward.__code__.co_varnames:
            out = vision(pixel_values=pixel_values, output_hidden_states=True)
        else:
            out = vision(pixel_values)
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hs = out.hidden_states[-1]
        else:
            hs = out.last_hidden_state

        if hs.ndim != 3:
            raise ValueError(f"Unexpected Llama/Mllama vision hidden state shape: {tuple(hs.shape)}")
        return hs[:, 1:, :] if hs.shape[1] >= 2 else hs

    else:
        raise ValueError("model_name must be 'CLIP' or 'Llama'")
