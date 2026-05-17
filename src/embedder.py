"""Feature embedding generation for object crops using OpenCLIP."""

from typing import List

import cv2
import numpy as np
import torch
from PIL import Image


class ObjectEmbedder:
    """Generates embedding vectors for object crop images.

    Uses OpenCLIP (open_clip) as the default backend.
    Install: pip install open-clip-torch

    The model is loaded once on initialisation and reused across all calls.
    All returned embeddings are float32 numpy arrays. When ``normalize=True``
    (default) each vector has L2 norm 1.0, which makes cosine similarity
    equivalent to a dot product.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self.device = self._resolve_device(device)

        self._model = None
        self._preprocess = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            print("ObjectEmbedder: CUDA requested but not available, falling back to CPU.")
            return "cpu"
        return device

    def _load_model(self) -> None:
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open-clip-torch is required for embedding generation.\n"
                "Install it with: pip install open-clip-torch"
            )

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self._model.eval()

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Generate an embedding for a single BGR crop image.

        Args:
            crop: BGR numpy array (H, W, 3).

        Returns:
            1D float32 numpy array (embedding_dim,).
        """
        pil_image = self._to_pil(crop)
        image_input = self._preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(image_input)

        embedding = features.cpu().numpy().astype(np.float32).squeeze(0)
        return self._maybe_normalize(embedding)

    def embed_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Generate embeddings for a list of BGR crop images.

        Args:
            crops: List of BGR numpy arrays.

        Returns:
            2D float32 numpy array (N, embedding_dim).
            Returns empty array with shape (0,) if crops is empty.
        """
        if not crops:
            return np.empty((0,), dtype=np.float32)

        inputs = [self._preprocess(self._to_pil(c)) for c in crops]
        batch = torch.stack(inputs).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(batch)

        embeddings = features.cpu().numpy().astype(np.float32)

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            embeddings = embeddings / norms

        return embeddings

    def _to_pil(self, crop: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _maybe_normalize(self, embedding: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return embedding
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
