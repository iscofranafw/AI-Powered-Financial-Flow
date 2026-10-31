import os
import hashlib
import numpy as np
from functools import lru_cache
from ai.text_normalizer import normalize

EMBEDDING_DIM = 384
DEFAULT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

_MODEL_INSTANCE = None


def _get_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    if os.getenv('USE_FALLBACK_EMBEDDINGS', '').lower() in ('1', 'true', 'yes'):
        return None

    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv('EMBEDDING_MODEL', DEFAULT_MODEL_NAME)
        _MODEL_INSTANCE = SentenceTransformer(model_name)
        return _MODEL_INSTANCE
    except Exception:
        return None


def get_embedding_dimension():
    return EMBEDDING_DIM


def _fallback_embedding(text):
    # Deterministic pseudo-dense vector based on character and word n-grams for air-gapped test environments.
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tokens = normalize(text).split()
    if not tokens:
        tokens = ['empty']
    for token in tokens:
        digest = hashlib.sha256(token.encode('utf-8')).digest()
        for idx in range(0, min(len(digest), 32), 2):
            slot = int.from_bytes(digest[idx:idx + 2], 'little') % EMBEDDING_DIM
            sign = 1.0 if digest[idx] % 2 == 0 else -1.0
            vec[slot] += sign
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    else:
        vec[0] = 1.0
    return vec


def embed_text(text):
    model = _get_model()
    if model is not None:
        cleaned = normalize(text) or text
        vec = model.encode(cleaned, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
    return _fallback_embedding(text)


def embed_batch(texts):
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    if model is not None:
        cleaned_list = [normalize(t) or t for t in texts]
        vecs = model.encode(cleaned_list, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    return np.vstack([_fallback_embedding(t) for t in texts])
