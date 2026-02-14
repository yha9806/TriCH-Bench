# Changelog

All notable changes to this project will be documented in this file.

## [2.0.2] - 2026-02-14

### Fixed
- Chinese-CLIP: switched from `cn_clip` library (HTTP 403) to HuggingFace `ChineseCLIPModel`
- Tier evaluation: queries now retrieve against full 18-item candidate pool (was sub-matrix → trivially 100%)
- `compute_recall_at_k` now accepts optional `gt_indices` for non-diagonal ground truth

### Results (first complete run)
- CLIP ViT-B/32: EN T2I R@10=66.7% (best), CLC mean=0.229
- Chinese-CLIP ViT-B/16: CC T2I R@10=77.8% (best overall), CLC mean=0.524
- mBERT+ResNet-50: ~55% across all (near-random baseline), CLC mean=0.084

## [2.0.1] - 2026-02-14

### Added
- Experiment scripts: `run_evaluation.py`, `models.py`, `utils.py`
- Model wrappers: CLIP ViT-B/32, Chinese-CLIP ViT-B/16, mBERT + ResNet-50
- Evaluation pipeline: Task A (monolingual), Task B (cross-lingual), CLC, tier-stratified
- Auto-export to LaTeX tables and JSON results
- Experiment config: `experiment_config.yaml`
- Requirements file: `requirements.txt`

### Fixed
- Gold-tier MC texts rewritten from simplified-character copies to genuine modern Chinese interpretations (18/18 samples)
- `semantic_similarity` and `agreement_score` decoupled (were identical for all samples)
- `word_count` recalculated after MC rewrite
- `alignment_metadata.classical_to_modern_method` changed to `expert_interpretation`
- `expert_review` metadata: varied reviewer counts, dates, added conflict resolution
- Added component-level `review_scope` (CC=reviewed, MC=revised_pending_review, EN=reviewed)

### Fixed (Bronze-tier)
- TCH-009: Removed incorrect Cao Buxing (曹不兴) attribution confusion with Cao Zhongda (曹仲达)
- TCH-010: Removed incorrect Splashed Ink Immortal (泼墨仙人图) reference for Broken Reed technique

## [2.0.0] - 2026-02-14

### Added
- Initial release of TriCH-Bench v2
- 18 gold-tier original text samples (CC/MC/EN)
- 54 bronze-tier text type expansion drafts (interpretation/catalogue/educational)
- JSON Schema for data validation
- IIIF Presentation API 3.0 manifest example
- CIDOC CRM JSON-LD mapping example
- 49 scans from Hai Xian Shi Ba Miao Fa manual
- CC BY-NC-SA 4.0 license
