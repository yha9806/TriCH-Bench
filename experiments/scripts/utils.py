"""
TriCH-Bench Evaluation Utilities
================================
Helper functions for data loading, metric computation, and result export.

Author: Yu, Haorui
Date: 2026-02-14
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ============================================================
# Data Loading
# ============================================================

def load_gold_samples(path: str) -> list[dict]:
    """Load 18 gold-tier samples from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    logger.info(f"Loaded {len(samples)} gold-tier samples from {path}")
    return samples


def get_image_path(sample: dict, image_dir: str) -> str:
    """Resolve the image path for a sample."""
    # image_ref.path is like "data/海仙十八描法/c01.高古游丝描.jpg"
    # We need to map it to the actual repo path
    raw = sample["image_ref"]["path"]
    filename = os.path.basename(raw)
    resolved = os.path.join(image_dir, filename)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Image not found: {resolved}")
    return resolved


def group_by_tier(samples: list[dict]) -> dict[str, list[dict]]:
    """Group samples by difficulty tier."""
    groups: dict[str, list[dict]] = {}
    for s in samples:
        tier = s["difficulty_tier"]
        groups.setdefault(tier, []).append(s)
    return groups


# ============================================================
# Retrieval Metrics
# ============================================================

def compute_recall_at_k(
    similarity_matrix: np.ndarray,
    k_values: list[int],
    gt_indices: list[int] | None = None,
) -> dict[int, float]:
    """
    Compute Recall@K from a similarity matrix.

    Args:
        similarity_matrix: (N, M) matrix where [i, j] = similarity
                           between query i and candidate j.
        k_values: list of K values, e.g. [1, 5, 10].
        gt_indices: ground truth column index for each query row.
                    If None, uses diagonal (i.e., gt for query i is col i).

    Returns:
        Dict mapping K -> Recall@K as percentage.
    """
    n = similarity_matrix.shape[0]
    # Rank candidates for each query (descending similarity)
    ranks = np.argsort(-similarity_matrix, axis=1)  # (N, M)

    results = {}
    for k in k_values:
        hits = 0
        for i in range(n):
            gt = gt_indices[i] if gt_indices is not None else i
            if gt in ranks[i, :k]:
                hits += 1
        results[k] = (hits / n) * 100.0
    return results


def compute_clc(
    sim_matrix_lang1: np.ndarray,
    sim_matrix_lang2: np.ndarray,
) -> float:
    """
    Compute Cross-Lingual Consistency (CLC) via Spearman rank correlation.

    For each image query, compare the ranking of candidates in lang1 vs lang2.
    Return the mean Spearman rho across all queries.
    """
    n = sim_matrix_lang1.shape[0]
    rhos = []
    for i in range(n):
        rho, _ = spearmanr(sim_matrix_lang1[i], sim_matrix_lang2[i])
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else 0.0


# ============================================================
# LaTeX Export
# ============================================================

def results_to_latex_task_a(
    results: dict[str, dict[str, dict[str, dict[int, float]]]],
    output_path: str,
) -> None:
    """
    Export Task A results to LaTeX table.

    Args:
        results: {model_name: {lang: {direction: {k: recall}}}}
                 direction = "i2t" or "t2i"
        output_path: path to write .tex file
    """
    langs = ["classical_zh", "modern_zh", "en"]
    lang_labels = {"classical_zh": "CC", "modern_zh": "MC", "en": "EN"}
    k_values = [1, 5, 10]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Task~A: Monolingual retrieval results (Recall@K, \%). "
        r"I2T = image-to-text; T2I = text-to-image. \textbf{Bold} = best per column.}",
        r"\label{tab:task_a}",
        r"\small",
        r"\begin{tabular}{l " + "ccc ccc " * 3 + "}",
        r"\toprule",
    ]

    # Header row 1: language groups
    header1 = r"\textbf{Model}"
    for lang in langs:
        header1 += r" & \multicolumn{6}{c}{\textbf{" + lang_labels[lang] + r"}}"
    header1 += r" \\"
    lines.append(header1)

    # Header row 2: I2T / T2I subgroups
    header2 = ""
    for lang in langs:
        header2 += r" & \multicolumn{3}{c}{I2T} & \multicolumn{3}{c}{T2I}"
    header2 += r" \\"
    lines.append(r"\cmidrule(lr){2-7} \cmidrule(lr){8-13} \cmidrule(lr){14-19}")
    lines.append(header2)

    # Header row 3: @K values
    header3 = ""
    for _ in langs:
        for _ in ["i2t", "t2i"]:
            for k in k_values:
                header3 += f" & @{k}"
    header3 += r" \\"
    lines.append(header3)
    lines.append(r"\midrule")

    # Data rows
    model_names = list(results.keys())
    # Find best per column for bolding
    for model in model_names:
        row = _short_model_name(model)
        for lang in langs:
            for direction in ["i2t", "t2i"]:
                for k in k_values:
                    val = results[model][lang][direction][k]
                    row += f" & {val:.1f}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Task A LaTeX table written to {output_path}")


def results_to_latex_task_b(
    results: dict[str, dict[str, dict[int, float]]],
    output_path: str,
) -> None:
    """
    Export Task B cross-lingual results to LaTeX table.

    Args:
        results: {model_name: {pair_label: {k: recall}}}
        output_path: path to write .tex file
    """
    pair_labels = [
        "CC→MC", "MC→CC", "CC→EN", "EN→CC", "MC→EN", "EN→MC"
    ]
    k_values = [1, 5, 10]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Task~B: Cross-lingual retrieval results (Recall@K, \%). "
        r"Query language $\rightarrow$ retrieval language. \textbf{Bold} = best per column.}",
        r"\label{tab:task_b}",
        r"\small",
        r"\begin{tabular}{l " + "ccc " * 6 + "}",
        r"\toprule",
    ]

    # Header
    header = r"\textbf{Model}"
    for pair in pair_labels:
        header += r" & \multicolumn{3}{c}{" + pair + "}"
    header += r" \\"
    lines.append(header)

    cmidrules = ""
    for i, _ in enumerate(pair_labels):
        start = 2 + i * 3
        end = start + 2
        cmidrules += rf"\cmidrule(lr){{{start}-{end}}} "
    lines.append(cmidrules)

    header_k = ""
    for _ in pair_labels:
        for k in k_values:
            header_k += f" & @{k}"
    header_k += r" \\"
    lines.append(header_k)
    lines.append(r"\midrule")

    for model in results:
        row = _short_model_name(model)
        for pair in pair_labels:
            for k in k_values:
                val = results[model][pair][k]
                row += f" & {val:.1f}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Task B LaTeX table written to {output_path}")


def results_to_latex_tier(
    results: dict[str, dict[str, dict[int, float]]],
    output_path: str,
) -> None:
    """
    Export tier-stratified results to LaTeX table.

    Args:
        results: {model_name: {tier: {k: recall}}}
    """
    tiers = ["L1-Easy", "L2-Medium", "L3-Hard"]
    k_values = [1, 5, 10]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval performance stratified by difficulty tier "
        r"(mean Recall@K across languages and directions, \%).}",
        r"\label{tab:tier_results}",
        r"\small",
        r"\begin{tabular}{l ccc ccc ccc c}",
        r"\toprule",
    ]

    header = r"\textbf{Model}"
    for tier in tiers:
        header += r" & \multicolumn{3}{c}{\textbf{" + tier + "}}"
    header += r" & $\Delta_{\text{L1--L3}}$ \\"
    lines.append(header)

    cmidrules = r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}"
    lines.append(cmidrules)

    header_k = ""
    for _ in tiers:
        for k in k_values:
            header_k += f" & @{k}"
    header_k += r" & @10 \\"
    lines.append(header_k)
    lines.append(r"\midrule")

    for model in results:
        row = _short_model_name(model)
        r10_values = {}
        for tier in tiers:
            for k in k_values:
                val = results[model][tier][k]
                row += f" & {val:.1f}"
            r10_values[tier] = results[model][tier][10]
        delta = r10_values["L1-Easy"] - r10_values["L3-Hard"]
        row += f" & {delta:.1f} \\\\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Tier results LaTeX table written to {output_path}")


def results_to_latex_clc(
    results: dict[str, dict[str, float]],
    output_path: str,
) -> None:
    """
    Export CLC results to LaTeX table.

    Args:
        results: {model_name: {pair_label: clc_score}}
    """
    pair_labels = ["CC--MC", "CC--EN", "MC--EN"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-Lingual Consistency (CLC) scores (Spearman $\rho$). "
        r"Higher values indicate more consistent rankings across languages.}",
        r"\label{tab:clc}",
        r"\small",
        r"\begin{tabular}{l ccc c}",
        r"\toprule",
        r"\textbf{Model} & \textbf{CC--MC} & \textbf{CC--EN} & \textbf{MC--EN} & \textbf{Mean} \\",
        r"\midrule",
    ]

    for model in results:
        row = _short_model_name(model)
        values = []
        for pair in pair_labels:
            val = results[model][pair]
            values.append(val)
            row += f" & {val:.3f}"
        mean_val = np.mean(values)
        row += f" & {mean_val:.3f} \\\\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"CLC LaTeX table written to {output_path}")


def save_results_json(results: dict[str, Any], output_path: str) -> None:
    """Save all results to a JSON file for reproducibility."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert(results), f, ensure_ascii=False, indent=2)
    logger.info(f"Results JSON saved to {output_path}")


def _short_model_name(name: str) -> str:
    """Shorten model names for table display."""
    mapping = {
        "clip": "CLIP ViT-B/32",
        "chinese_clip": "Chinese-CLIP ViT-B/16",
        "siglip2": "SigLIP 2 ViT-B/16",
        "jina_clip": "Jina-CLIP v2",
        "mbert_resnet": "mBERT + ResNet-50",
    }
    return mapping.get(name, name)
