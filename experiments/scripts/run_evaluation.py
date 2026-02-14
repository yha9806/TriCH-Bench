#!/usr/bin/env python3
"""
TriCH-Bench: Main Evaluation Script
=====================================
Runs Task A (monolingual retrieval), Task B (cross-lingual retrieval),
CLC analysis, and tier-stratified evaluation for all three models.

Usage:
    cd TriCH-Bench/
    python experiments/scripts/run_evaluation.py

    # Or specify individual models:
    python experiments/scripts/run_evaluation.py --models clip chinese_clip

    # CPU-only mode:
    python experiments/scripts/run_evaluation.py --device cpu

Author: Yu, Haorui
Date: 2026-02-14
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import load_model
from utils import (
    compute_clc,
    compute_recall_at_k,
    get_image_path,
    group_by_tier,
    load_gold_samples,
    results_to_latex_clc,
    results_to_latex_task_a,
    results_to_latex_task_b,
    results_to_latex_tier,
    save_results_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trich_bench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TriCH-Bench Evaluation")
    parser.add_argument(
        "--config",
        default="experiments/configs/experiment_config.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clip", "chinese_clip", "mbert_resnet"],
        help="Model keys to evaluate",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ──
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir or config["output"]["results_dir"]
    latex_dir = os.path.join(output_dir, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    # ── Setup logging to file ──
    log_path = os.path.join(output_dir, "experiment_log.txt")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"TriCH-Bench Evaluation Started: {datetime.now().isoformat()}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 60)

    # ── Load data ──
    samples = load_gold_samples(config["data"]["gold_samples"])
    image_dir = config["data"]["image_dir"]
    languages = config["evaluation"]["languages"]
    lang_labels = config["evaluation"]["language_labels"]
    k_values = config["evaluation"]["recall_k"]
    tiers = config["evaluation"]["tiers"]

    # Set seed
    np.random.seed(config["evaluation"]["seed"])

    # ── Resolve image paths ──
    image_paths = []
    for s in samples:
        try:
            img_path = get_image_path(s, image_dir)
            image_paths.append(img_path)
        except FileNotFoundError as e:
            logger.error(f"Missing image for {s['sample_id']}: {e}")
            sys.exit(1)

    logger.info(f"All {len(image_paths)} images found.")

    # ── Extract texts per language ──
    texts_by_lang: dict[str, list[str]] = {}
    for lang in languages:
        texts_by_lang[lang] = [s["text"][lang] for s in samples]

    # ── Per-model evaluation ──
    all_results: dict = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config": args.config,
            "models": args.models,
            "device": args.device,
            "num_samples": len(samples),
        },
        "task_a": {},
        "task_b": {},
        "tier": {},
        "clc": {},
    }

    # Cache similarity matrices: {model: {lang: {"i2t": matrix, "t2i": matrix}}}
    sim_cache: dict = {}

    for model_key in args.models:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating model: {model_key}")
        logger.info(f"{'='*40}")

        t_start = time.time()
        model = load_model(model_key, config, device=args.device)

        # ── Encode images (shared across languages) ──
        logger.info("Encoding images...")
        image_embeddings = model.encode_images(image_paths)
        logger.info(f"Image embeddings shape: {image_embeddings.shape}")

        # ── Encode texts for each language ──
        text_embeddings: dict[str, np.ndarray] = {}
        for lang in languages:
            logger.info(f"Encoding texts [{lang_labels[lang]}]...")
            text_embeddings[lang] = model.encode_texts(texts_by_lang[lang])
            logger.info(f"Text embeddings [{lang_labels[lang]}] shape: {text_embeddings[lang].shape}")

        # ── Task A: Monolingual retrieval ──
        logger.info("\n--- Task A: Monolingual Retrieval ---")
        all_results["task_a"][model_key] = {}
        sim_cache[model_key] = {}

        for lang in languages:
            label = lang_labels[lang]
            # I2T: image query -> text candidates
            sim_i2t = model.cosine_similarity_matrix(image_embeddings, text_embeddings[lang])
            # T2I: text query -> image candidates
            sim_t2i = model.cosine_similarity_matrix(text_embeddings[lang], image_embeddings)

            recall_i2t = compute_recall_at_k(sim_i2t, k_values)
            recall_t2i = compute_recall_at_k(sim_t2i, k_values)

            all_results["task_a"][model_key][lang] = {
                "i2t": recall_i2t,
                "t2i": recall_t2i,
            }
            sim_cache[model_key][lang] = {"i2t": sim_i2t, "t2i": sim_t2i}

            logger.info(
                f"  [{label}] I2T R@1/5/10: "
                f"{recall_i2t[1]:.1f} / {recall_i2t[5]:.1f} / {recall_i2t[10]:.1f}"
            )
            logger.info(
                f"  [{label}] T2I R@1/5/10: "
                f"{recall_t2i[1]:.1f} / {recall_t2i[5]:.1f} / {recall_t2i[10]:.1f}"
            )

        # ── Task B: Cross-lingual retrieval ──
        logger.info("\n--- Task B: Cross-lingual Retrieval ---")
        all_results["task_b"][model_key] = {}

        for query_lang, cand_lang in config["evaluation"]["cross_lingual_pairs"]:
            ql = lang_labels[query_lang]
            cl = lang_labels[cand_lang]
            pair_label = f"{ql}→{cl}"

            # Text in query_lang -> Text in cand_lang (via image as bridge)
            # Method: encode query text, find closest IMAGE, then match to cand text
            # Alternative (simpler): direct text-to-text similarity via shared embedding space
            # We use direct text-text similarity as this is standard for CLIP-family models
            sim_cross = model.cosine_similarity_matrix(
                text_embeddings[query_lang], text_embeddings[cand_lang]
            )
            recall_cross = compute_recall_at_k(sim_cross, k_values)

            all_results["task_b"][model_key][pair_label] = recall_cross

            logger.info(
                f"  [{pair_label}] R@1/5/10: "
                f"{recall_cross[1]:.1f} / {recall_cross[5]:.1f} / {recall_cross[10]:.1f}"
            )

        # ── CLC: Cross-Lingual Consistency ──
        logger.info("\n--- CLC Analysis ---")
        all_results["clc"][model_key] = {}

        clc_pairs = [
            ("classical_zh", "modern_zh", "CC--MC"),
            ("classical_zh", "en", "CC--EN"),
            ("modern_zh", "en", "MC--EN"),
        ]
        for lang1, lang2, pair_label in clc_pairs:
            # CLC: compare I2T similarity rankings across two languages
            sim1 = sim_cache[model_key][lang1]["i2t"]
            sim2 = sim_cache[model_key][lang2]["i2t"]
            clc_score = compute_clc(sim1, sim2)
            all_results["clc"][model_key][pair_label] = clc_score
            logger.info(f"  CLC [{pair_label}]: {clc_score:.3f}")

        # ── Tier-stratified evaluation ──
        logger.info("\n--- Tier-Stratified Evaluation ---")
        all_results["tier"][model_key] = {}
        tier_groups = group_by_tier(samples)

        for tier in tiers:
            if tier not in tier_groups:
                logger.warning(f"  No samples for tier {tier}")
                continue

            tier_samples = tier_groups[tier]
            tier_indices = [samples.index(s) for s in tier_samples]
            n_tier = len(tier_indices)

            # Compute mean recall across all languages and directions for this tier
            # Use tier queries against FULL candidate pool (all 18 items)
            tier_recalls = {k: [] for k in k_values}
            for lang in languages:
                for direction in ["i2t", "t2i"]:
                    sim = sim_cache[model_key][lang][direction]
                    # Take tier rows, keep all columns as candidates
                    sub_sim = sim[tier_indices, :]
                    tier_recall = compute_recall_at_k(sub_sim, k_values, gt_indices=tier_indices)
                    for k in k_values:
                        tier_recalls[k].append(tier_recall[k])

            all_results["tier"][model_key][tier] = {
                k: float(np.mean(tier_recalls[k])) for k in k_values
            }
            r10 = all_results["tier"][model_key][tier][10]
            logger.info(f"  [{tier}] Mean R@10: {r10:.1f}%")

        elapsed = time.time() - t_start
        logger.info(f"\nModel {model_key} completed in {elapsed:.1f}s")

    # ── Export results ──
    logger.info("\n--- Exporting Results ---")

    # JSON (full results for reproducibility)
    save_results_json(all_results, os.path.join(output_dir, "all_results.json"))

    # LaTeX tables
    results_to_latex_task_a(all_results["task_a"], os.path.join(latex_dir, "task_a_results.tex"))
    results_to_latex_task_b(all_results["task_b"], os.path.join(latex_dir, "task_b_results.tex"))
    results_to_latex_tier(all_results["tier"], os.path.join(latex_dir, "tier_results.tex"))
    results_to_latex_clc(all_results["clc"], os.path.join(latex_dir, "clc_results.tex"))

    logger.info(f"\nAll results saved to {output_dir}/")
    logger.info(f"LaTeX tables saved to {latex_dir}/")
    logger.info(f"Full results JSON: {output_dir}/all_results.json")
    logger.info(f"Log file: {log_path}")

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    for model_key in args.models:
        print(f"\n{model_key}:")
        for lang in languages:
            label = lang_labels[lang]
            r = all_results["task_a"][model_key][lang]
            print(f"  [{label}] I2T R@10={r['i2t'][10]:.1f}%  T2I R@10={r['t2i'][10]:.1f}%")
        clc_mean = np.mean(list(all_results["clc"][model_key].values()))
        print(f"  CLC mean: {clc_mean:.3f}")
    print(f"\nOutput: {output_dir}/")
    print(f"Next step: Copy {latex_dir}/*.tex to paper/jocch_submission/tables/")


if __name__ == "__main__":
    main()
