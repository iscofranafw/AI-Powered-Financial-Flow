import os
import hashlib
import numpy as np
from functools import lru_cache
from ai.text_normalizer import normalize

EMBEDDING_DIM = 384
DEFAULT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

_MODEL_INSTANCE = None
_EMBED_CACHE = {}


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


def is_semantic_model_available() -> bool:
    """Return True if real neural SentenceTransformer model is loaded and active."""
    return _get_model() is not None


def get_model_name() -> str:
    """Return active embedding model identifier."""
    return os.getenv('EMBEDDING_MODEL', DEFAULT_MODEL_NAME)


def get_embedding_dimension() -> int:
    model = _get_model()
    if model is not None and hasattr(model, 'get_sentence_embedding_dimension'):
        try:
            return int(model.get_sentence_embedding_dimension())
        except Exception:
            # Safe fallback: if model dimension inquiry fails, fall back to default EMBEDDING_DIM
            pass
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
    cleaned = normalize(text) or str(text).strip()
    if cleaned in _EMBED_CACHE:
        return _EMBED_CACHE[cleaned].copy()

    model = _get_model()
    if model is not None:
        vec = model.encode(cleaned, normalize_embeddings=True)
        res = np.asarray(vec, dtype=np.float32)
    else:
        res = _fallback_embedding(text)

    if len(_EMBED_CACHE) < 8192:
        _EMBED_CACHE[cleaned] = res

    return res.copy()


def embed_batch(texts):
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    if model is not None:
        cleaned_list = [normalize(t) or t for t in texts]
        vecs = model.encode(cleaned_list, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    return np.vstack([_fallback_embedding(t) for t in texts])
