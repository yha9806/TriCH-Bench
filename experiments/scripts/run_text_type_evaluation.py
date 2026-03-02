#!/usr/bin/env python3
"""
TriCH-Bench: Text-Type Comparison Evaluation
=============================================
Runs Task A (T2I retrieval) for all 4 text types (original, interpretation,
catalogue, educational) across 5 models, using the same 55-image retrieval pool.

Usage:
    cd TriCH-Bench/
    python experiments/scripts/run_text_type_evaluation.py --device cuda

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import load_model
from utils import compute_recall_at_k, load_gold_samples, load_pool_index, save_results_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trich_bench_texttype")


def load_bronze_samples(path: str) -> dict[str, list[dict]]:
    """Load bronze samples grouped by text_type, ordered by work_id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    by_type: dict[str, dict[str, dict]] = {}
    for s in samples:
        tt = s["text_type"]
        wid = s["work_id"]
        by_type.setdefault(tt, {})[wid] = s

    result = {}
    for tt, wid_map in by_type.items():
        ordered = []
        for i in range(1, 19):
            hx = f"HX-{i:02d}"
            if hx in wid_map:
                ordered.append(wid_map[hx])
            else:
                logger.warning(f"Missing {hx} for text_type={tt}")
        result[tt] = ordered
        logger.info(f"Loaded {len(ordered)} {tt} samples")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TriCH-Bench Text-Type Comparison")
    parser.add_argument("--config", default="experiments/configs/experiment_config.yaml")
    parser.add_argument("--models", nargs="+",
                        default=["clip", "chinese_clip", "siglip2", "jina_clip", "mbert_resnet"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default="experiments/outputs/text_type_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "text_type_experiment_log.txt")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 60)
    logger.info(f"Text-Type Comparison Evaluation: {datetime.now().isoformat()}")
    logger.info(f"Models: {args.models}, Device: {args.device}")
    logger.info("=" * 60)

    np.random.seed(config["evaluation"]["seed"])

    # Load data
    gold_samples = load_gold_samples(config["data"]["gold_samples"])
    bronze_by_type = load_bronze_samples(config["data"]["bronze_drafts"])

    languages = config["evaluation"]["languages"]
    lang_labels = config["evaluation"]["language_labels"]
    k_values = config["evaluation"]["recall_k"]

    text_sets: dict[str, dict[str, list[str]]] = {}
    text_sets["original"] = {
        lang: [s["text"][lang] for s in gold_samples] for lang in languages
    }
    for tt, samples in bronze_by_type.items():
        text_sets[tt] = {
            lang: [s["text"][lang] for s in samples] for lang in languages
        }

    text_types = ["original", "interpretation", "catalogue", "educational"]
    logger.info(f"Text types: {text_types}")
    for tt in text_types:
        for lang in languages:
            logger.info(f"  {tt}/{lang_labels[lang]}: {len(text_sets[tt][lang])} texts, "
                        f"avg len={np.mean([len(t) for t in text_sets[tt][lang]]):.0f} chars")

    # Load 55-image pool
    pool_files, technique_indices, image_technique = load_pool_index(config["data"]["pool_index"])
    pool_image_paths = [os.path.join(config["data"]["retrieval_pool_dir"], f) for f in pool_files]

    for p in pool_image_paths:
        if not os.path.exists(p):
            logger.error(f"Pool image not found: {p}")
            sys.exit(1)

    hx_ids = [f"HX-{i+1:02d}" for i in range(18)]
    t2i_gt = [technique_indices[hx] for hx in hx_ids]

    logger.info(f"Pool: {len(pool_image_paths)} images, T2I: 18 queries -> {len(pool_image_paths)} candidates")

    all_results: dict = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment": "text_type_comparison",
            "models": args.models,
            "text_types": text_types,
            "device": args.device,
            "pool_size": len(pool_image_paths),
            "seed": config["evaluation"]["seed"],
        },
        "t2i_recall": {},
    }

    for model_key in args.models:
        logger.info(f"\n{'='*40}")
        logger.info(f"Model: {model_key}")
        logger.info(f"{'='*40}")

        t_start = time.time()
        model = load_model(model_key, config, device=args.device)

        logger.info("Encoding images...")
        image_embeddings = model.encode_images(pool_image_paths)
        logger.info(f"Image embeddings: {image_embeddings.shape}")

        all_results["t2i_recall"][model_key] = {}

        for tt in text_types:
            all_results["t2i_recall"][model_key][tt] = {}
            for lang in languages:
                label = lang_labels[lang]
                texts = text_sets[tt][lang]
                text_embeddings = model.encode_texts(texts)
                sim_t2i = model.cosine_similarity_matrix(text_embeddings, image_embeddings)
                recall = compute_recall_at_k(sim_t2i, k_values, gt_indices=t2i_gt)
                all_results["t2i_recall"][model_key][tt][lang] = recall
                logger.info(f"  [{tt}/{label}] T2I R@1/5/10: "
                            f"{recall[1]:.1f} / {recall[5]:.1f} / {recall[10]:.1f}")

        elapsed = time.time() - t_start
        logger.info(f"Model {model_key} done in {elapsed:.1f}s")

        del model
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    # Save results
    save_results_json(all_results, os.path.join(args.output_dir, "text_type_results.json"))
    _export_comparison_table(all_results, args.output_dir, lang_labels)

    print("\n" + "=" * 70)
    print("TEXT-TYPE COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\n{'Model':<22} {'Type':<16} {'CC R@10':>8} {'MC R@10':>8} {'EN R@10':>8}")
    print("-" * 70)
    for model_key in args.models:
        for tt in text_types:
            r = all_results["t2i_recall"][model_key][tt]
            print(f"{model_key:<22} {tt:<16} "
                  f"{r['classical_zh'][10]:>7.1f}% {r['modern_zh'][10]:>7.1f}% {r['en'][10]:>7.1f}%")
        print()


def _export_comparison_table(results: dict, output_dir: str, lang_labels: dict) -> None:
    """Export a compact LaTeX comparison table."""
    text_types = results["metadata"]["text_types"]
    tt_short = {"original": "Orig.", "interpretation": "Interp.", "catalogue": "Cat.", "educational": "Edu."}
    models = results["metadata"]["models"]
    model_short = {
        "clip": "CLIP", "chinese_clip": "CN-CLIP", "siglip2": "SigLIP2",
        "jina_clip": "Jina-CLIP", "mbert_resnet": "mBERT+RN",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Text-to-image Recall@$10$ (\%) by text type. "
        r"Same 55-image retrieval pool, seed=42. "
        r"Orig.\ = original expert texts (18 samples); "
        r"Interp./Cat./Edu.\ = generated text variants.}",
        r"\label{tab:text_type_comparison}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l ccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Type} & \textbf{CC} & \textbf{MC} & \textbf{EN} \\",
        r"\midrule",
    ]

    for i, model_key in enumerate(models):
        if i > 0:
            lines.append(r"\addlinespace[3pt]")
        for j, tt in enumerate(text_types):
            r = results["t2i_recall"][model_key][tt]
            name = model_short.get(model_key, model_key) if j == 0 else ""
            tt_label = tt_short.get(tt, tt)
            cc = r["classical_zh"][10]
            mc = r["modern_zh"][10]
            en = r["en"][10]
            lines.append(f"{name} & {tt_label} & {cc:.1f} & {mc:.1f} & {en:.1f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_path = os.path.join(output_dir, "text_type_comparison.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
