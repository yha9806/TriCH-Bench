# TriCH-Bench

**TriCH-Bench: A Trilingual Cultural Heritage Benchmark for Chinese Classical Painting Archives**

An expert-validated trilingual benchmark and heritage-driven evaluation protocol for cross-lingual access to Chinese classical painting archives. Developed for the ACM JOCCH Visual Heritage special issue.

## Overview

TriCH-Bench is built from 18 gold-tier samples drawn from the *Hai Xian Shi Ba Miao Fa* (海仙十八描法, The Eighteen Line-Drawing Techniques of Hai Xian), providing parallel annotations in Classical Chinese, Modern Chinese, and English across four text types and three difficulty tiers.

### Key Features

- **Trilingual**: Classical Chinese (CC), Modern Chinese (MC), English (EN)
- **Four Text Types**: Original, Interpretation, Catalogue, Educational
- **Three Difficulty Tiers**: L1-Easy (5), L2-Medium (8), L3-Hard (5)
- **39 Annotated Cultural Concepts** with translation difficulty flags
- **Heritage-Standard Interoperability**: IIIF Presentation API 3.0 + CIDOC CRM mappings
- **FAIR-Compliant**: CC BY-NC-SA 4.0 license, persistent identifiers, open formats

## Repository Structure

```
TriCH-Bench/
├── data/
│   ├── samples/
│   │   ├── hai_xian_18_migrated.json      # 18 gold-tier original text samples
│   │   └── text_type_expansion_drafts.json # 54 bronze-tier text type drafts
│   ├── schema/
│   │   └── trich_bench_schema.json         # JSON Schema for data validation
│   ├── manifests/
│   │   └── HX-01_gaoguyousimiao.json       # IIIF Presentation API 3.0 example
│   ├── mappings/
│   │   └── HX-01_cidoc_crm.jsonld          # CIDOC CRM mapping example
│   ├── templates/
│   │   └── text_type_expansion_template.json
│   └── interop_examples_README.md
├── images/
│   ├── hai_xian_18/                        # Original manual scans (49 images)
│   │   ├── c01.高古游丝描.jpg ~ c18.枣核描.jpg  # 18 technique illustrations
│   │   ├── a01.封面.jpg                    # Cover page
│   │   ├── b01.自序.jpg ~ b10.jpg          # Preface pages
│   │   └── d01.画人物论.jpg ~ e01.封底.jpg  # Supplementary texts
│   └── retrieval_pool/                     # Expanded retrieval pool (55 images)
│       ├── pool_index.json                 # Technique→image mapping
│       ├── c01~c18 (original 18)           # Technique illustrations
│       ├── *_HX-XX.jpg (26 museum)         # CC0/PD exemplar paintings
│       └── distractor_*.jpg (11)           # Unassigned Chinese paintings
├── LICENSE                                 # CC BY-NC-SA 4.0
└── README.md
```

## Data Summary

### Gold-Tier Samples (18)

| ID | Technique (CC) | Technique (EN) | Difficulty |
|----|---------------|----------------|------------|
| HX-01 | 高古游絲描 | Ancient Gossamer Line Drawing | L3-Hard |
| HX-02 | 琴絃描 | Zither String Drawing | L2-Medium |
| HX-03 | 鐵線描 | Iron Wire Drawing | L2-Medium |
| HX-04 | 行雲流水描 | Flowing Clouds and Water Drawing | L2-Medium |
| HX-05 | 螞蟥描 | Leech Drawing | L1-Easy |
| HX-06 | 釘頭鼠尾描 | Nail Head and Rat Tail Drawing | L1-Easy |
| HX-07 | 混描 | Mixed Drawing | L1-Easy |
| HX-08 | 撅頭釘描 | Stubby Brush Drawing | L3-Hard |
| HX-09 | 曹衣描 | Cao Yi Drawing | L3-Hard |
| HX-10 | 折蘆描 | Broken Reed Drawing | L2-Medium |
| HX-11 | 柳葉描 | Willow Leaf Drawing | L2-Medium |
| HX-12 | 竹葉描 | Bamboo Leaf Drawing | L1-Easy |
| HX-13 | 戰筆水紋描 | Tremulous Water Lines Drawing | L2-Medium |
| HX-14 | 減筆描 | Reduced Brush Drawing | L3-Hard |
| HX-15 | 枯柴描 | Withered Wood Drawing | L2-Medium |
| HX-16 | 蚯蚓描 | Earthworm Drawing | L2-Medium |
| HX-17 | 橄欖描 | Olive Drawing | L1-Easy |
| HX-18 | 棗核描 | Jujube Core Drawing | L3-Hard |

### Bronze-Tier Drafts (54)

18 techniques x 3 text types (interpretation, catalogue, educational). LLM-generated initial drafts pending expert review.

### Language Statistics

| Language | Mean Characters/Words | Range |
|----------|----------------------|-------|
| Classical Chinese | 15.7 chars | 12-23 |
| Modern Chinese | 43.7 chars | 37-57 |
| English | 22.5 words | 21-26 |

## Quality Tiers

- **Gold**: Reviewed by 2+ independent experts. All quality dimensions scored >= 3.0/4.0. Suitable for primary evaluation.
- **Silver**: Single expert review with spot-checks. Suitable for extended experiments.
- **Bronze**: LLM-generated with single expert quality check. Suitable for community distribution with caveats.

## Expanded Retrieval Pool

The retrieval pool contains 55 images: 44 technique-labeled images (1-5 per technique) and 11 distractor paintings. Sources include the Metropolitan Museum of Art, Cleveland Museum of Art, Smithsonian/Freer Gallery, and Wikimedia Commons (all CC0 or Public Domain).

| Images per Technique | Techniques |
|---------------------|------------|
| 5 images | HX-14 (减笔描) |
| 4 images | HX-01 (高古游丝描) |
| 3 images | HX-03, 04, 05, 09, 11, 15, 18 |
| 2 images | HX-02, 06, 07, 08, 10 |
| 1 image  | HX-12, 13, 16, 17 |

Random baseline Recall@10 drops from 55.6% (18 images) to ~18% (55 images), enabling meaningful model differentiation.

## Evaluation Protocol

The benchmark supports three evaluation tasks:

1. **Task A**: Monolingual image-text retrieval (I2T and T2I)
2. **Task B**: Cross-lingual image-text retrieval (6 language pairs)
3. **Task C**: Cultural Semantic Preservation (CSP) expert evaluation

### CSP Dimensions

| Dimension | Description |
|-----------|-------------|
| TTA | Technique Terminology Accuracy |
| HAC | Historical and Attributional Correctness |
| SMP | Symbolic Meaning Preservation |
| RA | Register Appropriateness |
| CCC | Cultural Context Completeness |

## Citation

```bibtex
@article{yu2026trichbench,
  title={TriCH-Bench: An Expert-Validated Trilingual Benchmark and Heritage-Driven Evaluation Protocol for Cross-Lingual Access to Chinese Classical Painting Archives},
  author={Yu, Haorui and others},
  journal={ACM Journal on Computing and Cultural Heritage},
  year={2026},
  note={Under review}
}
```

## License

This dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

The original *Hai Xian Shi Ba Miao Fa* manual is in the public domain. We thank the heritage domain experts who contributed to the trilingual annotation and evaluation process.
