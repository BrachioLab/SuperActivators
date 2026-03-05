import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import Callable, Dict, List, Tuple, FrozenSet, Optional

from util import patches_to_image, reembed_patch_embeddings_batch_diff

INF = 1e9
EPS = 1e-300


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


def factorial(n: int) -> float:
    return float(math.factorial(n))


def comb(n: int, k: int) -> float:
    return float(math.comb(n, k))


def kernel_m(t: int, s: int, n: int, interaction_type: str, max_order: int) -> float:
    if interaction_type == "SII":
        return factorial(n - t - s) * factorial(t) / factorial(n - s + 1)
    elif interaction_type in ("STI", "FSI"):
        if s != max_order:
            raise ValueError(f"{interaction_type} only supports max_order ({max_order}) interactions.")
        if interaction_type == "STI":
            return float(s) * factorial(n - t - 1) * factorial(t) / factorial(n)
        else:
            num = factorial(2 * s - 1) * factorial(n - t - 1) * factorial(t + s - 1)
            den = (factorial(s - 1) ** 2) * factorial(n + s - 1)
            return num / den
    else:
        raise ValueError(f"Unknown interaction_type: {interaction_type}")


def precompute_weights(n: int, order: int, interaction_type: str) -> Dict[int, np.ndarray]:
    weights: Dict[int, np.ndarray] = {}
    s_range = range(order, order + 1) if interaction_type in ("STI", "FSI") else range(1, order + 1)
    for s in s_range:
        w_s = np.zeros((n + 1, s + 1), dtype=np.float64)
        for t in range(n + 1):
            for k in range(max(0, s + t - n), min(s, t) + 1):
                sign = (-1.0) ** (s - k)
                w_s[t, k] = sign * kernel_m(t - k, s, n, interaction_type, order)
        weights[s] = w_s
    return weights


def init_sampling_weights(n: int, order: int, kernel: str) -> np.ndarray:
    q = np.zeros(n + 1, dtype=np.float64)
    for t in range(n + 1):
        if kernel == "ksh":
            if order <= t <= n - order:
                q[t] = (factorial(n - t - order) * factorial(t - order) / factorial(n - order + 1))
            else:
                q[t] = INF
        elif kernel == "faith":
            if 1 <= t <= n - 1:
                q[t] = factorial(n - t - 1) * factorial(t - 1) / factorial(n - 1)
            else:
                q[t] = INF
        elif kernel == "unif-size":
            q[t] = 1.0
        elif kernel == "unif-set":
            den = comb(n, t)
            q[t] = 1.0 / den if den > 0 else INF
        else:
            raise ValueError(f"Unknown sampling kernel: {kernel}")
    return q


def determine_complete_subsets(n: int, budget: int, q: np.ndarray) -> Tuple[List[int], List[int]]:
    complete: List[int] = []
    incomplete: List[int] = list(range(n + 1))
    sorted_sizes: List[int] = []
    left, right = 0, n
    while left <= right:
        sorted_sizes.append(left)
        if left != right:
            sorted_sizes.append(right)
        left += 1
        right -= 1
    current_budget = budget
    for t in sorted_sizes:
        cost = int(comb(n, t))
        if q[t] == INF or current_budget >= cost:
            complete.append(t)
            incomplete.remove(t)
            if q[t] != INF:
                current_budget -= cost
        else:
            break
    return complete, incomplete


class ShapIQEstimator:
    def __init__(
        self,
        n_features: int,
        order: int = 2,
        interaction_type: str = "SII",
        budget: int = 512,
        sampling_kernel: str = "ksh",
        pairing: bool = True,
        stratification: bool = False,
        seed: int = 42,
    ) -> None:
        self.n = n_features
        self.order = order
        self.interaction_type = interaction_type
        self.budget = budget
        self.sampling_kernel = sampling_kernel
        self.pairing = pairing
        self.stratification = stratification
        self.seed = seed

        self.s_range = range(self.order, self.order + 1) if self.interaction_type in ("STI", "FSI") else range(1, self.order + 1)
        self.weights = precompute_weights(self.n, self.order, self.interaction_type)
        self.q = init_sampling_weights(self.n, self.order, self.sampling_kernel)
        self.rng = np.random.default_rng(self.seed)

    def explain(self, v_batched: Callable[[List[np.ndarray]], np.ndarray], mini_batch_size: int):
        all_interactions: Dict[int, List[FrozenSet[int]]] = {}
        for s in self.s_range:
            all_interactions[s] = [frozenset(S) for S in combinations(range(self.n), s)]

        def init_interaction_dict():
            return {s: {S: 0.0 for S in all_interactions[s]} for s in self.s_range}

        results = init_interaction_dict()
        p_t = np.zeros(self.n + 1, dtype=np.float64)
        for t in range(self.n + 1):
            if self.q[t] != INF:
                p_t[t] = self.q[t] * comb(self.n, t)
        p_sum = p_t.sum()
        p_t_norm = p_t / p_sum if p_sum > 0 else np.ones(self.n + 1) / (self.n + 1)

        complete_sizes, incomplete_sizes = determine_complete_subsets(self.n, self.budget, self.q)
        empty_mask = np.zeros(self.n, dtype=bool)
        v_empty = float(v_batched([empty_mask])[0])

        def eval_masks(masks: List[np.ndarray]) -> np.ndarray:
            vals = []
            for i in range(0, len(masks), mini_batch_size):
                batch_masks = masks[i:i + mini_batch_size]
                vals.extend(v_batched(batch_masks))
            return np.array(vals, dtype=np.float64) - v_empty

        phase1_masks, phase1_coalitions = [], []
        for t in complete_sizes:
            for T_tuple in combinations(range(self.n), t):
                T = frozenset(T_tuple)
                if len(T) == 0:
                    continue
                mask = np.zeros(self.n, dtype=bool)
                mask[list(T)] = True
                phase1_masks.append(mask)
                phase1_coalitions.append(T)

        evals_used = 1 + len(phase1_masks)
        if len(phase1_masks) > 0:
            v0_phase1 = eval_masks(phase1_masks)
            for T, v0 in zip(phase1_coalitions, v0_phase1):
                t_size = len(T)
                for s in self.s_range:
                    for S in all_interactions[s]:
                        k = len(T.intersection(S))
                        if max(0, s + t_size - self.n) <= k <= min(s, t_size):
                            results[s][S] += v0 * self.weights[s][t_size, k]

        sample_mean, sample_s2 = init_interaction_dict(), init_interaction_dict()
        n_samples = 0
        remaining_budget = self.budget - evals_used

        if remaining_budget > 0 and len(incomplete_sizes) > 0:
            p_t_inc = p_t_norm.copy()
            for t_idx in complete_sizes:
                p_t_inc[t_idx] = 0.0
            p_t_inc_sum = p_t_inc.sum()
            if p_t_inc_sum > 0:
                p_t_inc /= p_t_inc_sum
            else:
                for t_idx in incomplete_sizes:
                    p_t_inc[t_idx] = 1.0 / len(incomplete_sizes)

            target_total_phase2_evals = remaining_budget
            phase2_masks, phase2_coalitions, phase2_p_Ts = [], [], []

            while len(phase2_masks) < target_total_phase2_evals:
                t_sampled = self.rng.choice(np.arange(self.n + 1), p=p_t_inc)
                t_val = int(t_sampled)
                subset = self.rng.choice(self.n, size=t_val, replace=False)
                T = frozenset(subset)

                mask = np.zeros(self.n, dtype=bool)
                mask[list(T)] = True
                phase2_masks.append(mask)
                phase2_coalitions.append(T)
                p_T = float(p_t_inc[t_val] / comb(self.n, t_val)) if comb(self.n, t_val) > 0 else 1.0
                phase2_p_Ts.append(max(p_T, EPS))

                if len(phase2_masks) >= target_total_phase2_evals:
                    break

                if self.pairing:
                    t_c = self.n - t_val
                    if p_t_inc[t_c] > 0.0:
                        T_c = frozenset(set(range(self.n)) - T)
                        mask_c = np.zeros(self.n, dtype=bool)
                        mask_c[list(T_c)] = True
                        phase2_masks.append(mask_c)
                        phase2_coalitions.append(T_c)
                        p_Tc = float(p_t_inc[t_c] / comb(self.n, t_c)) if comb(self.n, t_c) > 0 else 1.0
                        phase2_p_Ts.append(max(p_Tc, EPS))

            if len(phase2_masks) > 0:
                evals_used += len(phase2_masks)
                chunk_size = mini_batch_size * 10
                for i in range(0, len(phase2_masks), chunk_size):
                    masks_chunk = phase2_masks[i:i + chunk_size]
                    coalitions_chunk = phase2_coalitions[i:i + chunk_size]
                    p_Ts_chunk = phase2_p_Ts[i:i + chunk_size]
                    v0_chunk = eval_masks(masks_chunk)

                    for T, p_T_val, v0 in zip(coalitions_chunk, p_Ts_chunk, v0_chunk):
                        t_size = len(T)
                        n_samples += 1
                        for s in self.s_range:
                            for S in all_interactions[s]:
                                k = len(T.intersection(S))
                                if max(0, s + t_size - self.n) <= k <= min(s, t_size):
                                    update = (v0 * self.weights[s][t_size, k]) / p_T_val
                                    delta = update - sample_mean[s][S]
                                    sample_mean[s][S] += delta / n_samples
                                    delta2 = update - sample_mean[s][S]
                                    sample_s2[s][S] += delta * delta2

        variance = init_interaction_dict()
        for s in self.s_range:
            for S in all_interactions[s]:
                results[s][S] += sample_mean[s][S]
                variance[s][S] = sample_s2[s][S] / (n_samples - 1) if n_samples > 1 else 0.0
                if abs(results[s][S]) < 1e-5:
                    results[s][S] = 0.0

        return results, variance, evals_used


def shapiq_concept(
    X: torch.Tensor,                 # [1, P, Din]
    c: int,                          # concept index
    concept_vectors: torch.Tensor,   # [K, Dnew]
    model: nn.Module,
    model_name: str,
    image_size: tuple = (224, 224),
    score_mode: str = "cossim",
    percent_thru_model: int = 100,
    device: str = "cuda",
    baseline: Optional[torch.Tensor] = None,   # [1, P, Din] Defaults to 0 if None
    order: int = 1,
    interaction_type: str = "SII",
    budget: int = 512,
    sampling_kernel: str = "ksh",
    pairing: bool = True,
    seed: int = 42,
    mini_batch_size: int = 64,
    drop_cls_if_present: bool = False,         # set True if X includes CLS and you want P-1 features
) -> torch.Tensor:
    """
    - order=1 (and interaction_type not in STI/FSI): returns [P] Shapley values
    - else: returns [P, P, ...] interaction tensor (order dims)
    """
    device = torch.device(device)
    model.eval()

    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError(f"X must be [1, P, Din], got {tuple(X.shape)}")

    X = X.to(device)
    concept_vectors = concept_vectors.to(device)

    mn = model_name.lower()
    is_image = ("clip" in mn) or ("vit" in mn) or ("vision" in mn)

    if drop_cls_if_present and X.shape[1] > 1:
        P0 = X.shape[1]
        if (int((P0 - 1) ** 0.5) ** 2) == (P0 - 1) and (int(P0 ** 0.5) ** 2) != P0:
            X = X[:, 1:, :]  # drop CLS
    P = X.shape[1]

    if concept_vectors.ndim != 2:
        raise ValueError(f"concept_vectors must be [K, Dnew], got {tuple(concept_vectors.shape)}")
    K, Dnew = concept_vectors.shape
    if not (0 <= c < K):
        raise ValueError(f"concept index c={c} out of range for K={K}")

    concept_score_fn = get_concept_score_fn(score_mode)
    if score_mode.lower() == "cossim":
        concept_vectors = F.normalize(concept_vectors, dim=-1)

    if baseline is None:
        baseline = torch.zeros_like(X)
    else:
        baseline = baseline.to(device)
        if baseline.shape != X.shape:
            raise ValueError(f"baseline must have same shape as X. baseline={tuple(baseline.shape)} X={tuple(X.shape)}")

    def v_batched(masks: List[np.ndarray]) -> np.ndarray:
        batch_size = len(masks)
        if batch_size == 0:
            return np.array([], dtype=np.float64)

        masks_t = torch.tensor(np.array(masks), dtype=torch.float32, device=device)  # [B,P]
        if masks_t.shape[1] != P:
            raise ValueError(f"Mask has P={masks_t.shape[1]} but X has P={P}")

        X_masked = baseline.expand(batch_size, -1, -1) + masks_t.unsqueeze(-1) * (X - baseline)

        with torch.no_grad():
            if is_image:
                img_batch = patches_to_image(X_masked, image_size=image_size)
                emb_batch = reembed_patch_embeddings_batch_diff(
                    images=img_batch,
                    model=model,
                    model_name=model_name,
                    image_size=image_size,
                    percent_thru_model=percent_thru_model,
                )
            else:
                emb_batch = reembed_patch_embeddings_batch_diff(
                    images=None,
                    tokens=X_masked,
                    model=model,
                    model_name=model_name,
                    percent_thru_model=percent_thru_model,
                )
            if emb_batch.shape[-1] != Dnew:
                raise ValueError(
                    f"Embedding dim mismatch: emb_batch.shape[-1]={emb_batch.shape[-1]} vs concept_vectors Dnew={Dnew}. "
                    f"Make sure concept_vectors match reembed output."
                )

            scores = concept_score_fn(emb_batch, concept_vectors)  # [B,K]
            vals = scores[:, c].detach().cpu().numpy().astype(np.float64)

        return vals

    torch.manual_seed(seed)

    estimator = ShapIQEstimator(
        n_features=P,
        order=order,
        interaction_type=interaction_type,
        budget=budget,
        sampling_kernel=sampling_kernel,
        pairing=pairing,
        seed=seed,
    )

    results, variance, evals_used = estimator.explain(v_batched, mini_batch_size=mini_batch_size)

    if order == 1 and interaction_type not in ("STI", "FSI"):
        attr_vector = np.zeros(P, dtype=np.float32)
        for j in range(P):
            attr_vector[j] = float(results[1][frozenset([j])])
        return torch.from_numpy(attr_vector).cpu()
    else:
        shape = tuple([P] * order)
        attr_tensor = np.zeros(shape, dtype=np.float32)
        for S, val in results[order].items():
            if len(S) == order:
                perms = list(itertools.permutations(list(S)))
                for p in perms:
                    attr_tensor[p] = float(val) / len(perms)
        return torch.from_numpy(attr_tensor).cpu()