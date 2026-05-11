"""Canonical naming helpers for point-model surrogate losses."""

from __future__ import annotations


CANONICAL_POINT_LOSS_TYPES = (
    "mse",
    "spo_plus",
    "weighted_mse_spo_plus",
    "spo_caching",
    "dbb",
    "nid",
    "dpo",
    "pfyl",
    "imle",
    "aimle",
    "pg",
    "pointwise_ltr",
    "pairwise_ltr",
    "pairwise_diff",
    "listwise_ltr",
    "nce",
    "nce_c",
    "map",
    "map_c",
)

BENCHMARK_SURROGATE_LOSS_CHOICES = (
    "mse",
    "SPOPlus",
    "MSE+SPOPlus",
    "SPOCaching",
    "DBB",
    "NID",
    "DPO",
    "PFYL",
    "IMLE",
    "AIMLE",
    "PG",
    "pointwiseLTR",
    "pairwiseLTR",
    "pairwiseDiff",
    "listwiseLTR",
    "NCE",
    "NCE_c",
    "contrastiveMAP",
    "MAP_c",
)


def canonical_surrogate_loss_name(value) -> str:
    loss = str(value).strip().lower()
    compact = loss.replace("_", "").replace("-", "").replace(" ", "")

    if compact == "mse":
        return "mse"
    if compact in {"spoplus", "spoplusloss", "spo+"} or loss in {"spo_plus", "spo+"}:
        return "spo_plus"
    if compact in {
        "weightedmsespoplus",
        "msespoplus",
        "mixedmsespoplus",
        "hybridmsespoplus",
    } or loss in {
        "weighted_mse_spo_plus",
        "mse_spo_plus",
        "mixed_mse_spo_plus",
        "hybrid_mse_spo_plus",
        "mse+spoplus",
        "mse+spo+",
    }:
        return "weighted_mse_spo_plus"
    if compact in {"spocaching"} or loss == "spo_caching":
        return "spo_caching"
    if compact in {"blackboxopt", "dbb"}:
        return "dbb"
    if compact in {"negativeidentity", "nid"}:
        return "nid"
    if compact in {"perturbedopt", "dpo"}:
        return "dpo"
    if compact in {"perturbedfenchelyoung", "pfyl"}:
        return "pfyl"
    if compact in {"implicitmle", "imle"}:
        return "imle"
    if compact in {"adaptiveimplicitmle", "aimle"}:
        return "aimle"
    if compact in {"perturbationgradient", "pg"}:
        return "pg"
    if compact in {"pointwiseltr", "pointwise", "predoptpointwise"} or loss in {
        "pointwise_ltr",
        "predopt_pointwise",
    }:
        return "pointwise_ltr"
    if compact in {"pairwiseltr", "pairwise", "predoptpairwise"} or loss in {
        "pairwise_ltr",
        "predopt_pairwise",
    }:
        return "pairwise_ltr"
    if compact in {"pairwisediff", "pairwisecacheddiff"} or loss == "pairwise_diff":
        return "pairwise_diff"
    if compact in {"listwiseltr", "listwise", "predoptlistwise"} or loss in {
        "listwise_ltr",
        "predopt_listwise",
    }:
        return "listwise_ltr"
    if compact == "nce":
        return "nce"
    if compact == "ncec" or loss == "nce_c":
        return "nce_c"
    if compact in {"contrastivemap", "contrastivemapestimation", "cmap", "map"} or loss == "contrastive_map":
        return "map"
    if compact in {"mapc", "mapcactual"} or loss in {"map_c", "map_c_actual"}:
        return "map_c"
    return loss


def surrogate_loss_display_name(value) -> str:
    loss = canonical_surrogate_loss_name(value)
    mapping = {
        "mse": "MSE",
        "spo_plus": "SPOPlus",
        "weighted_mse_spo_plus": "MSE+SPOPlus",
        "spo_caching": "SPOCaching",
        "dbb": "DBB",
        "nid": "NID",
        "dpo": "DPO",
        "pfyl": "PFYL",
        "imle": "IMLE",
        "aimle": "AIMLE",
        "pg": "PG",
        "pointwise_ltr": "pointwiseLTR",
        "pairwise_ltr": "pairwiseLTR",
        "pairwise_diff": "pairwiseDiff",
        "listwise_ltr": "listwiseLTR",
        "nce": "NCE",
        "nce_c": "NCE_c",
        "map": "contrastiveMAP",
        "map_c": "MAP_c",
    }
    return mapping.get(loss, str(value))


def canonical_mse_mix_weight(loss_type, mse_weight) -> float:
    loss = canonical_surrogate_loss_name(loss_type)
    if loss == "spo_plus":
        return 0.0
    if loss == "mse":
        return 1.0
    if loss == "weighted_mse_spo_plus":
        if mse_weight in {"", None}:
            return 0.5
        return float(mse_weight)
    if mse_weight in {"", None}:
        return 0.0
    return float(mse_weight)


def is_degenerate_mixed_loss_weight(loss_type, mse_weight, *, atol: float = 1.0e-12) -> bool:
    if canonical_surrogate_loss_name(loss_type) != "weighted_mse_spo_plus":
        return False
    if mse_weight in {"", None}:
        return False
    try:
        return abs(float(mse_weight)) <= atol
    except (TypeError, ValueError):
        return False
