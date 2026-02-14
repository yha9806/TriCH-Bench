# Interoperability Examples / 互操作示例

## 概述 / Overview

本目录包含两个最小可用示例（Minimal Working Examples），用于证明 TRICHBench 数据集与文化遗产机构标准的互操作兼容性。

This directory contains two minimal working examples demonstrating the interoperability of the TRICHBench dataset with cultural heritage institutional standards.

---

## 文件说明 / File Descriptions

### 1. IIIF Presentation API 3.0 Manifest

**文件路径 / Path:** `manifests/HX-01_gaoguyousimiao.json`

**用途 / Purpose:**

符合 [IIIF Presentation API 3.0](https://iiif.io/api/presentation/3.0/) 规范的 manifest 文件，以"高古游丝描"（Ancient Gossamer Line Drawing）为示例，展示如何将三语绘画技法数据发布为 IIIF 资源。

A manifest file compliant with the IIIF Presentation API 3.0 specification, using "Ancient Gossamer Line Drawing" (高古游丝描) as an example to demonstrate how tri-lingual painting technique data can be published as IIIF resources.

**博物馆/图书馆如何使用 / How Museums and Libraries Use IIIF Manifests:**

- **在线展示**: IIIF 兼容的查看器（如 Mirador、Universal Viewer）可直接加载此 manifest，在线展示技法图像并提供三语注释浏览。
- **跨机构聚合**: 多家机构可将各自的 manifest 汇聚至统一平台（如 Europeana、日本文化遗产在线），无需统一底层数据格式。
- **学术注释**: 研究者可通过 IIIF Annotation 机制为图像添加多语言注释和学术评论。

- **Online display**: IIIF-compatible viewers (e.g., Mirador, Universal Viewer) can directly load this manifest to display technique images with tri-lingual annotation browsing.
- **Cross-institutional aggregation**: Multiple institutions can aggregate their manifests into unified platforms (e.g., Europeana, Cultural Heritage Online Japan) without unifying underlying data formats.
- **Scholarly annotation**: Researchers can add multilingual annotations and scholarly commentary to images via the IIIF Annotation mechanism.

**manifest 结构要点 / Key Structural Elements:**

| 字段 / Field | 说明 / Description |
|---|---|
| `label` | 三语标签 (文言/现代汉语/英语) / Tri-lingual labels (Classical/Modern Chinese/English) |
| `metadata` | 技法名称、朝代、来源文本、文化概念 / Technique name, dynasty, source text, cultural concepts |
| `items[0].items` | AnnotationPage 包含图像（painting motivation）/ Image annotation with painting motivation |
| `items[0].annotations` | 三语文本注释（commenting motivation）/ Tri-lingual text annotations with commenting motivation |

### 2. CIDOC CRM 映射 (JSON-LD) / CIDOC CRM Mapping (JSON-LD)

**文件路径 / Path:** `mappings/HX-01_cidoc_crm.jsonld`

**用途 / Purpose:**

使用 [CIDOC CRM](https://www.cidoc-crm.org/) 本体的 JSON-LD 映射文件，将同一技法的语义结构映射为国际文化遗产数据标准。

A JSON-LD mapping file using the CIDOC CRM ontology, mapping the semantic structure of the same technique to the international cultural heritage data standard.

**CIDOC CRM 如何支持数据互操作 / How CIDOC CRM Supports Data Interoperability:**

- **语义统一**: CIDOC CRM 提供文化遗产领域的统一概念框架。技法被建模为 `E28_Conceptual_Object`，图像为 `E36_Visual_Item`，文本为 `E33_Linguistic_Object`，确保不同机构的同类数据可语义对齐。
- **关联数据 (Linked Data)**: JSON-LD 格式天然支持关联数据发布。技法可与 Getty AAT 词汇表（用于语言标识）、其他绘画数据库等外部资源建立链接。
- **跨文化桥接**: 通过 `P72_has_language` 显式标注语言，`P67_refers_to` 关联文化概念，支持跨语言、跨文化的知识检索。

- **Semantic unification**: CIDOC CRM provides a unified conceptual framework for the cultural heritage domain. Techniques are modeled as `E28_Conceptual_Object`, images as `E36_Visual_Item`, texts as `E33_Linguistic_Object`, ensuring semantic alignment of similar data across institutions.
- **Linked Data**: The JSON-LD format natively supports Linked Data publishing. Techniques can establish links with the Getty AAT vocabulary (for language identification), other painting databases, and external resources.
- **Cross-cultural bridging**: Explicit language annotation via `P72_has_language` and cultural concept linkage via `P67_refers_to` support cross-lingual, cross-cultural knowledge retrieval.

**CRM 类与属性映射 / CRM Classes and Properties Used:**

| CRM 类 / Class | 映射对象 / Mapped To |
|---|---|
| `E28_Conceptual_Object` | 描法本身（技法概念）/ The technique itself (as concept) |
| `E36_Visual_Item` | 技法图像 / Technique illustration image |
| `E33_Linguistic_Object` | 三语文本描述 / Tri-lingual textual descriptions |
| `E55_Type` | 技法类型分类 / Technique type classification |
| `E39_Actor` | 关联画家（顾恺之）/ Associated master painter (Gu Kaizhi) |

| CRM 属性 / Property | 用途 / Usage |
|---|---|
| `P2_has_type` | 技法类型分类 / Technique type classification |
| `P3_has_note` | 文本描述 / Textual description |
| `P138_represents` | 图像表现的概念 / What the image represents |
| `P72_has_language` | 语言标注 / Language annotation |
| `P67_refers_to` | 文化概念引用 / Cultural concept reference |

---

## 如何扩展到更多样本 / How to Extend to More Samples

此示例的模式具有完全可复制性。扩展步骤如下：

This example follows a fully replicable pattern. Extension steps:

1. **IIIF Manifest**: 复制 `HX-01_gaoguyousimiao.json`，替换 `id`、`label`、`summary`、`metadata`、`body.id`（图像路径）和注释文本。每个技法对应一个 manifest 文件。
   - Copy `HX-01_gaoguyousimiao.json`, replace `id`, `label`, `summary`, `metadata`, `body.id` (image path), and annotation texts. One manifest file per technique.

2. **CIDOC CRM 映射**: 复制 `HX-01_cidoc_crm.jsonld`，替换 `@id` URI、标签、文本内容和关联画家信息。
   - Copy `HX-01_cidoc_crm.jsonld`, replace `@id` URIs, labels, textual content, and associated artist information.

3. **批量生成**: 可基于 `data/海仙十八描法/hai_xian_18_cme_dataset.json` 中的结构化数据编写脚本，自动生成全部 18 个技法的 manifest 和 CRM 映射。
   - Batch generation: Scripts can be written based on the structured data in `hai_xian_18_cme_dataset.json` to automatically generate manifests and CRM mappings for all 18 techniques.

---

## 参考链接 / References

- **IIIF Presentation API 3.0**: https://iiif.io/api/presentation/3.0/
- **IIIF Image API 3.0**: https://iiif.io/api/image/3.0/
- **CIDOC CRM (v7.1)**: https://www.cidoc-crm.org/Version/version-7.1
- **CIDOC CRM JSON-LD**: https://linked.art/ (Linked Art implementation of CIDOC CRM)
- **Getty AAT (Art & Architecture Thesaurus)**: https://www.getty.edu/research/tools/vocabularies/aat/
- **Mirador IIIF Viewer**: https://projectmirador.org/
- **Universal Viewer**: https://universalviewer.io/

---

## 许可 / License

These interoperability examples are provided under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/).
