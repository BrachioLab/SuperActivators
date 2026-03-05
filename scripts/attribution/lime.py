import torch
import torch.nn as nn
import torch.nn.functional as F

from util import patches_to_image, reembed_patch_embeddings_batch, weight_batch

def perturb_x(X: torch.Tensor, keep_prob: float = 0.5):
    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError(f"X must be [1,P,D], got {X.shape}")

    _, P, _ = X.shape
    mask = (torch.rand(1, P, device=X.device) < keep_prob).float()
    X0 = X * mask.unsqueeze(-1)
    return X0, mask

def concept_score_cossim(embedding: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    """
    embedding: [P,D] or [B,P,D]
    concept:   [K,D]
    returns:   [B,K] (or [1,K] if input was [P,D])
    """
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)  # [1,P,D]
    if embedding.ndim != 3:
        raise ValueError(f"embedding must be [B,P,D], got {embedding.shape}")
    if concept.ndim != 2:
        raise ValueError(f"concept must be [K,D], got {concept.shape}")

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
    if embedding.ndim != 3:
        raise ValueError(f"embedding must be [B,P,D], got {embedding.shape}")
    if concept.ndim != 2:
        raise ValueError(f"concept must be [K,D], got {concept.shape}")

    proj = torch.matmul(embedding, concept.t())  # [B,P,K]
    return proj.mean(dim=1)                      # [B,K]


def get_concept_score_fn(mode: str):
    mode = mode.lower()
    if mode == "cossim":
        return concept_score_cossim
    if mode == "linsep": 
        return concept_score_linsep
    raise ValueError("mode must be 'cossim' or 'linsep'")

class RidgeRegression(nn.Module):
    def __init__(self, num_features: int, alpha: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(num_features, 1, bias=True)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,P]
        return self.linear(x)  # [N,1]

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, feature_weights: torch.Tensor) -> torch.Tensor:
        weighted_x = x * feature_weights                 # [N,P]
        preds = self.forward(weighted_x).squeeze(1)      # [N]
        mse = torch.mean((preds - y) ** 2)
        ridge = self.alpha * torch.sum(self.linear.weight ** 2)
        return mse + ridge

def lime_concept(
    X: torch.Tensor,                 # [1, P, Din]
    c: int,                          # concept index
    concept_vectors: torch.Tensor,    # [K, Dnew] 
    perturb_fn,                       # fn(X)-> (X0 [1,P,Din], mask [1,P])
    model,
    processor,
    model_name: str,                 # "CLIP" or "Llama"
    image_size: tuple,               # (224,224) or (560,560)
    score_mode: str = "cossim",       # "cossim" or "linsep"
    percent_thru_model: int = 100,
    nperturb: int = 1000,
    embed_batch_size: int = 16,
    num_epochs: int = 1000,
    alpha: float = 1.0,
    lr: float = 1e-2,
    sigma: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError(f"X must be [1,P,Din], got {X.shape}")

    device = torch.device(device)
    concept_vectors = concept_vectors.to(device)
    concept_score_fn = get_concept_score_fn(score_mode)
    P = X.shape[1]

    with torch.no_grad():
        img_orig = patches_to_image(X, image_size=image_size)  # [1,3,H,W]
        emb_orig = reembed_patch_embeddings_batch(
            images=img_orig,
            model=model,
            processor=processor,
            device=str(device),
            model_name=model_name,
            image_size=image_size,
            percent_thru_model=percent_thru_model,
        ).to(device)  # [1,P,Dnew]

    if emb_orig.ndim != 3 or emb_orig.shape[1] != P:
        raise ValueError(f"Unexpected emb_orig shape {emb_orig.shape}, expected [1,P,Dnew] with P={P}")
    Dnew = emb_orig.shape[2]
    if concept_vectors.shape[1] != Dnew:
        raise ValueError(
            f"concept_vectors dim mismatch: concept_vectors is [K,{concept_vectors.shape[1]}] "
            f"but re-embedded patches are Dnew={Dnew}."
        )

    X_rows, y_rows, W_rows = [], [], []
    buf_X0, buf_ind = [], []

    def flush():
        nonlocal buf_X0, buf_ind, X_rows, y_rows, W_rows
        if len(buf_X0) == 0:
            return

        X0_batch = torch.cat(buf_X0, dim=0)              # [B,P,Din]
        ind_batch = torch.cat(buf_ind, dim=0).to(device) # [B,P]

        img_batch = patches_to_image(X0_batch, image_size=image_size)  # [B,3,H,W]
        emb_batch = reembed_patch_embeddings_batch(
            images=img_batch,
            model=model,
            processor=processor,
            device=str(device),
            model_name=model_name,
            image_size=image_size,
            percent_thru_model=percent_thru_model,
        ).to(device)  # [B,P,Dnew]

        W_batch = weight_batch(sample=emb_batch, original=emb_orig, sigma=sigma)  # [B,P]

        out = concept_score_fn(emb_batch, concept_vectors)  # [B,K]
        if c < 0 or c >= out.shape[1]:
            raise ValueError(f"concept index c={c} out of range for K={out.shape[1]}")
        y_batch = out[:, c]  # [B]

        X_rows.append(ind_batch)  # [B,P]
        y_rows.append(y_batch)    # [B]
        W_rows.append(W_batch)    # [B,P]

        buf_X0, buf_ind = [], []

    for _ in range(nperturb):
        X0, indicator = perturb_fn(X)     # X0: [1,P,Din], indicator: [1,P]
        buf_X0.append(X0)
        buf_ind.append(indicator)
        if len(buf_X0) >= embed_batch_size:
            flush()
    flush()

    X_lime = torch.cat(X_rows, dim=0)          # [N,P]
    y_lime = torch.cat(y_rows, dim=0).view(-1) # [N]
    W = torch.cat(W_rows, dim=0)               # [N,P]

    r_model = RidgeRegression(P, alpha=alpha).to(device)
    opt = torch.optim.Adam(r_model.parameters(), lr=lr)

    for _ in range(num_epochs):
        r_model.train()
        opt.zero_grad(set_to_none=True)
        loss = r_model.compute_loss(X_lime, y_lime, W)
        loss.backward()
        opt.step()

    with torch.no_grad():
        importance = r_model.linear.weight.detach().view(-1).cpu()  # [P]
    return importance

def make_perturb_fn(keep_prob: float):
    return lambda X: perturb_x(X, keep_prob=keep_prob)
