"""
TriCH-Bench Model Wrappers
===========================
Unified interface for vision-language retrieval models:
  CLIP, Chinese-CLIP, SigLIP 2, Jina-CLIP v2, mBERT+ResNet-50.

Author: Yu, Haorui
Date: 2026-02-14
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all retrieval models."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    @abstractmethod
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode a single image into a normalized embedding vector."""

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into a normalized embedding vector."""

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        """Encode multiple images. Returns (N, D) matrix."""
        embeddings = [self.encode_image(p) for p in image_paths]
        return np.stack(embeddings)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts. Returns (N, D) matrix."""
        embeddings = [self.encode_text(t) for t in texts]
        return np.stack(embeddings)

    @staticmethod
    def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of vectors."""
        # a: (N, D), b: (M, D) -> (N, M)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return a_norm @ b_norm.T


# ============================================================
# CLIP ViT-B/32
# ============================================================

class CLIPModel(BaseModel):
    """OpenAI CLIP ViT-B/32 wrapper."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        super().__init__(device)
        from transformers import CLIPModel as HFCLIPModel, CLIPProcessor

        logger.info(f"Loading CLIP model: {model_id}")
        self.model = HFCLIPModel.from_pretrained(model_id, use_safetensors=True).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        logger.info("CLIP model loaded.")

    @torch.no_grad()
    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", truncation=True, max_length=77).to(self.device)
        emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()


# ============================================================
# Chinese-CLIP ViT-B/16
# ============================================================

class ChineseCLIPModel(BaseModel):
    """Chinese-CLIP ViT-B/16 wrapper using HuggingFace transformers."""

    def __init__(self, model_id: str = "OFA-Sys/chinese-clip-vit-base-patch16", device: str = "cuda"):
        super().__init__(device)
        from transformers import ChineseCLIPModel as HFChineseCLIPModel, ChineseCLIPProcessor

        logger.info(f"Loading Chinese-CLIP model: {model_id}")
        self.model = HFChineseCLIPModel.from_pretrained(model_id, use_safetensors=True).to(self.device).eval()
        self.processor = ChineseCLIPProcessor.from_pretrained(model_id)
        logger.info("Chinese-CLIP model loaded.")

    @torch.no_grad()
    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", truncation=True, max_length=52).to(self.device)
        emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()


# ============================================================
# SigLIP 2 (Google, 2025)
# ============================================================

class SigLIP2Model(BaseModel):
    """Google SigLIP 2 ViT-B/16 multilingual wrapper."""

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str = "cuda"):
        super().__init__(device)
        from transformers import AutoModel, AutoProcessor

        logger.info(f"Loading SigLIP 2 model: {model_id}")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        logger.info("SigLIP 2 model loaded.")

    @torch.no_grad()
    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", truncation=True, max_length=64).to(self.device)
        emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()


# ============================================================
# Jina-CLIP v2 (Jina AI, 2024)
# ============================================================

class JinaCLIPModel(BaseModel):
    """Jina-CLIP v2 multilingual multimodal wrapper."""

    def __init__(self, model_id: str = "jinaai/jina-clip-v2", device: str = "cuda"):
        super().__init__(device)
        from transformers import AutoModel

        logger.info(f"Loading Jina-CLIP v2 model: {model_id}")
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        logger.info("Jina-CLIP v2 model loaded.")

    @torch.no_grad()
    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        emb = self.model.encode_image([image], truncate_dim=512)
        emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
        return emb.squeeze()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        emb = self.model.encode_text([text], truncate_dim=512)
        emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
        return emb.squeeze()


# ============================================================
# mBERT + ResNet-50 (Projection Baseline)
# ============================================================

class MBERTResNetModel(BaseModel):
    """
    mBERT + ResNet-50 baseline with learned linear projection.

    Since there is no pretrained joint model, we:
    1. Extract [CLS] from mBERT (768-d) for text
    2. Extract avgpool from ResNet-50 (2048-d) for images
    3. Project both to a shared 512-d space via random init projection
       (this serves as a random-projection baseline; fine-tuning would
        require training data which we don't use in zero-shot)

    NOTE: This baseline is intentionally weak — it tests how a non-aligned
    multilingual text encoder + generic vision backbone performs without
    contrastive pretraining. The random projection means retrieval
    performance is expected to be near-random or poor.
    """

    def __init__(
        self,
        text_model_id: str = "google-bert/bert-base-multilingual-cased",
        vision_model_id: str = "microsoft/resnet-50",
        projection_dim: int = 512,
        device: str = "cuda",
    ):
        super().__init__(device)
        from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor

        logger.info(f"Loading mBERT: {text_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        self.text_model = AutoModel.from_pretrained(text_model_id, use_safetensors=True).to(self.device).eval()

        logger.info(f"Loading ResNet-50: {vision_model_id}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(vision_model_id)
        self.vision_model = AutoModel.from_pretrained(vision_model_id, use_safetensors=True).to(self.device).eval()

        # Fixed random projection (deterministic via seed)
        torch.manual_seed(42)
        self.text_proj = torch.nn.Linear(768, projection_dim, bias=False).to(self.device)
        self.vision_proj = torch.nn.Linear(2048, projection_dim, bias=False).to(self.device)

        # Freeze projections (no training)
        for p in self.text_proj.parameters():
            p.requires_grad = False
        for p in self.vision_proj.parameters():
            p.requires_grad = False

        logger.info("mBERT + ResNet-50 baseline loaded.")

    @torch.no_grad()
    def encode_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        outputs = self.vision_model(**inputs)
        # Global average pool of last hidden state
        features = outputs.last_hidden_state.mean(dim=[2, 3])  # (1, 2048)
        emb = self.vision_proj(features)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.text_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, 768)
        emb = self.text_proj(cls_emb)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze()


# ============================================================
# Factory
# ============================================================

def load_model(model_key: str, config: dict, device: str = "cuda") -> BaseModel:
    """Load a model by key from config."""
    model_cfg = config["models"][model_key]
    model_type = model_cfg["type"]

    if model_type == "clip":
        return CLIPModel(model_id=model_cfg["model_id"], device=device)
    elif model_type == "chinese_clip":
        return ChineseCLIPModel(model_id=model_cfg["model_id"], device=device)
    elif model_type == "siglip2":
        return SigLIP2Model(model_id=model_cfg["model_id"], device=device)
    elif model_type == "jina_clip":
        return JinaCLIPModel(model_id=model_cfg["model_id"], device=device)
    elif model_type == "mbert_resnet":
        return MBERTResNetModel(
            text_model_id=model_cfg["text_model_id"],
            vision_model_id=model_cfg["vision_model_id"],
            projection_dim=model_cfg["projection_dim"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
