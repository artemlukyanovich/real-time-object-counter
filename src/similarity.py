"""Embedding similarity computations."""

from typing import List, Optional, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D embedding vectors.

    Returns:
        float in [-1.0, 1.0]. Higher is more similar.
        Returns 0.0 if either vector has zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query and every row in a gallery.

    Args:
        query: 1D array (embedding_dim,).
        gallery: 2D array (N, embedding_dim).

    Returns:
        1D float32 array of similarities (N,).
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0.0 or len(gallery) == 0:
        return np.zeros(len(gallery), dtype=np.float32)

    gallery_norms = np.linalg.norm(gallery, axis=1)
    gallery_norms = np.where(gallery_norms > 0, gallery_norms, 1.0)

    similarities = gallery.dot(query) / (gallery_norms * query_norm)
    return similarities.astype(np.float32)


def find_best_match(
    query: np.ndarray,
    gallery: np.ndarray,
    object_ids: List[int],
    threshold: float,
) -> Tuple[Optional[int], float]:
    """Find the best matching object ID from a gallery of embeddings.

    Args:
        query: 1D embedding for the new detection.
        gallery: 2D array (N, embedding_dim) of known embeddings.
        object_ids: Object IDs corresponding to each row in gallery.
        threshold: Minimum similarity score to accept a match.

    Returns:
        (matched_object_id, score) if best score >= threshold,
        (None, best_score) otherwise.
    """
    if len(gallery) == 0 or len(object_ids) == 0:
        return None, 0.0

    scores = cosine_similarity_batch(query, gallery)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        return object_ids[best_idx], best_score

    return None, best_score
