import os
from contextlib import contextmanager
from contextvars import ContextVar
import json
import time
import numpy as np
from datetime import datetime, timezone
from database.client import privileged_client
from services.auth_service import verify_actor
from ai.embedding_pipeline import (
    embed_text,
    embed_batch,
    get_embedding_dimension,
    get_model_name,
    is_semantic_model_available,
    DEFAULT_MODEL_NAME,
)
from ai.text_normalizer import normalize
from ai.store_context import get_store_business_context

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

INDEX_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vector_store')

# Invalidate pre-ownership and history-first keyword indexes on next read.
KEYWORD_TEXT_VERSION = 5

_CACHE = {}
_PREVIEW_INDEXES = ContextVar("import_preview_indexes", default=None)


def _get_store_dir(toko_id):
    path = os.path.join(INDEX_ROOT_DIR, str(toko_id))
    os.makedirs(path, exist_ok=True)
    return path


def _get_file_paths(toko_id, tipe):
    store_dir = _get_store_dir(toko_id)
    faiss_path = os.path.join(store_dir, f"{tipe}.faiss")
    meta_path = os.path.join(store_dir, f"{tipe}_meta.json")
    return faiss_path, meta_path


def get_remote_version(toko_id, tipe):
    if not toko_id or str(toko_id) in ('shop-bench', 'test-store', 'shop-1', 'shop-bulk', 'tok', 'shop-cache-test', 'shop-csv-retail'):
        return 0
    try:
        res = (
            privileged_client()
            .table('rag_index_version')
            .select('version')
            .eq('toko_id', str(toko_id))
            .eq('tipe', str(tipe))
            .limit(1)
            .execute()
        )
        if res.data:
            return int(res.data[0]['version'])
    except Exception:
        # Safe fallback: if rag_index_version table is absent or unreachable, assume version 0
        pass
    return 0


def clear_memory_cache(toko_id=None, tipe=None):
    global _CACHE
    if toko_id is None:
        _CACHE.clear()
    elif tipe is None:
        for k in list(_CACHE.keys()):
            if k[0] == str(toko_id):
                _CACHE.pop(k, None)
    else:
        _CACHE.pop((str(toko_id), str(tipe)), None)


def build_corpus_entries(toko_id, tipe, *, category_rows=None, transaction_rows=None, business_type=None):
    entries = []
    client = privileged_client() if category_rows is None or transaction_rows is None else None
    business_type = business_type or get_store_business_context(toko_id).get('bidang_usaha', 'Umum')

    cat_rows = category_rows if category_rows is not None else (
        client.table('kategori')
        .select('id, nama, deskripsi, kata_kunci, sistem')
        .eq('toko_id', str(toko_id))
        .eq('tipe', str(tipe))
        .execute()
        .data or []
    )

    tx_rows = transaction_rows if transaction_rows is not None else (
        client.table('transaksi')
        .select('id, deskripsi, kategori_id, status_kategori')
        .eq('toko_id', str(toko_id))
        .eq('tipe', str(tipe))
        .eq('status_persetujuan', 'disetujui')
        .eq('status_kategori', 'dikonfirmasi')
        .order('tanggal', desc=True)
        .limit(1000)
        .execute()
        .data or []
    )

    # Reconcile keyword ownership with this shop/type's confirmed history before
    # vectorizing. Raw database rows remain unchanged; conflicting aliases must
    # not create score-1 duplicates in unrelated categories.
    from ai.category_evidence import compile_keyword_plan
    plan = compile_keyword_plan(
        [{**c, 'tipe': str(tipe)} for c in cat_rows],
        [{**r, 'tipe': str(tipe)} for r in tx_rows], business_type, prefer_existing=True,
    )
    cat_rows = plan['categories']
    from ai.category_evidence import reconcile_purpose_rows
    tx_rows = reconcile_purpose_rows([{**r, 'tipe': str(tipe)} for r in tx_rows], cat_rows, business_type)

    cat_names = {}
    for cat in cat_rows:
        cat_id = str(cat['id'])
        name = str(cat.get('nama', '')).strip()
        cat_names[cat_id] = name
        if cat.get('sistem') or name.casefold() == 'others':
            continue

        parts = [name]
        desc = str(cat.get('deskripsi') or '').strip()
        if desc:
            parts.append(desc)
        keywords = cat.get('kata_kunci') or []
        clean_kw = [str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []
        if clean_kw:
            parts.append(' '.join(clean_kw))

        text = '. '.join(parts)
        entries.append({
            'id': f"cat_{cat_id}",
            'type': 'category',
            'store_id': str(toko_id),
            'transaction_type': str(tipe),
            'category_id': cat_id,
            'category_name': name,
            'text': text,
            'normalized_desc': normalize(text),
            'business_type': business_type,
            'verification_source': 'category_definition',
        })
        for kw_idx, kw in enumerate(clean_kw):
            kw_text = kw  # Keep the user keyword intact; category identity lives in metadata.
            entries.append({
                'id': f"cat_{cat_id}_kw_{kw_idx}",
                'type': 'category',
                'store_id': str(toko_id),
                'transaction_type': str(tipe),
                'category_id': cat_id,
                'category_name': name,
                'text': kw_text,
                'normalized_desc': normalize(kw_text),
                'business_type': business_type,
                'verification_source': 'category_keyword',
            })

    seen = set()
    for tx in tx_rows:
        cid = str(tx.get('kategori_id') or '')
        cname = cat_names.get(cid)
        if not cname or cname.casefold() == 'others':
            continue
        raw_desc = str(tx.get('deskripsi') or '').strip()
        norm_desc = normalize(raw_desc)
        if not norm_desc or (cid, norm_desc) in seen:
            continue

        seen.add((cid, norm_desc))

        entries.append({
            'id': f"tx_{tx['id']}",
            'type': 'transaction',
            'store_id': str(toko_id),
            'transaction_type': str(tipe),
            'transaction_id': str(tx['id']),
            'category_id': cid,
            'category_name': cname,
            'text': raw_desc,
            'normalized_desc': norm_desc,
            'business_type': business_type,
            'verification_source': 'dikonfirmasi',
        })

    return entries


def build_and_save_index(toko_id, tipe, entries=None, version=None):
    if version is None:
        version = get_remote_version(toko_id, tipe)

    if entries is None:
        entries = build_corpus_entries(toko_id, tipe)

    store_info = get_store_business_context(toko_id)
    business_type = store_info.get('bidang_usaha', 'Umum')

    dim = get_embedding_dimension()
    index = faiss.IndexFlatIP(dim) if FAISS_AVAILABLE else None

    if entries:
        texts = [e['text'] for e in entries]
        vectors = embed_batch(texts)
        if vectors.shape[0] > 0 and index is not None:
            index.add(vectors)

    faiss_path, meta_path = _get_file_paths(toko_id, tipe)
    if index is not None and FAISS_AVAILABLE:
        try:
            faiss.write_index(index, faiss_path)
        except Exception:
            # Safe fallback: disk write failure does not prevent in-memory cache population
            pass

    meta = {
        'store_id': str(toko_id),
        'toko_id': str(toko_id),
        'tipe': str(tipe),
        'transaction_type': str(tipe),
        'business_type': business_type,
        'embedding_model_version': get_model_name(),
        'embedding_dimension': dim,
        'keyword_text_version': KEYWORD_TEXT_VERSION,
        'degraded_mode': not is_semantic_model_available(),
        'version': int(version),
        'count': len(entries),
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'entries': entries
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    _CACHE[(str(toko_id), str(tipe))] = {
        'index': index,
        'meta': meta,
        'version': int(version)
    }

    return index, meta


@contextmanager
def import_preview_indexes(toko_id, categories, history, business_type):
    """Request-local, in-memory FAISS indexes; never publish staged categories."""
    indexes = {}
    for tipe in ('pengeluaran', 'pemasukan'):
        entries = build_corpus_entries(
            toko_id, tipe, category_rows=[c for c in categories if c.get('tipe') == tipe],
            transaction_rows=[r for r in history if r.get('tipe') == tipe], business_type=business_type)
        vectors = embed_batch([e['text'] for e in entries]) if entries else None
        index = None
        if entries and FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        indexes[(str(toko_id), tipe)] = (index, {'entries': entries})
    guard = _PREVIEW_INDEXES.set(indexes)
    try:
        yield
    finally:
        _PREVIEW_INDEXES.reset(guard)


def get_or_sync_index(toko_id, tipe, force_rebuild=False):
    preview = _PREVIEW_INDEXES.get()
    if preview is not None and (str(toko_id), str(tipe)) in preview:
        return preview[(str(toko_id), str(tipe))]

    toko_id = str(toko_id)
    tipe = str(tipe)
    cached = _CACHE.get((toko_id, tipe))
    remote_version = get_remote_version(toko_id, tipe)
    current_dim = get_embedding_dimension()

    if not force_rebuild and cached is not None and cached['version'] == remote_version:
        idx = cached.get('index')
        if idx is not None and hasattr(idx, 'd') and idx.d == current_dim and cached.get('meta', {}).get('keyword_text_version') == KEYWORD_TEXT_VERSION:
            return cached['index'], cached['meta']

    faiss_path, meta_path = _get_file_paths(toko_id, tipe)
    if not force_rebuild and os.path.exists(faiss_path) and os.path.exists(meta_path) and FAISS_AVAILABLE:
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if int(meta.get('version', 0)) == remote_version and meta.get('keyword_text_version') == KEYWORD_TEXT_VERSION:
                # Validate tenant isolation
                if str(meta.get('store_id')) != toko_id and str(meta.get('toko_id')) != toko_id:
                    return build_and_save_index(toko_id, tipe, version=remote_version)

                # Validate embedding dimension compatibility
                meta_dim = meta.get('embedding_dimension', current_dim)
                if meta_dim != current_dim:
                    return build_and_save_index(toko_id, tipe, version=remote_version)

                index = faiss.read_index(faiss_path)
                if index.d != current_dim:
                    return build_and_save_index(toko_id, tipe, version=remote_version)

                _CACHE[(toko_id, tipe)] = {
                    'index': index,
                    'meta': meta,
                    'version': remote_version,
                }
                return index, meta
        except Exception:
            # Safe fallback: corrupted index file or read error falls through to build_and_save_index rebuild
            pass

    return build_and_save_index(toko_id, tipe, version=remote_version)


def retrieve_context(toko_id, tipe, query, top_k=5, min_score=0.25):
    if not toko_id or not str(toko_id).strip() or top_k <= 0 or not query or not str(query).strip():
        return []

    if tipe not in ('pemasukan', 'pengeluaran'):
        raise ValueError("Jenis transaksi harus 'pemasukan' atau 'pengeluaran'")

    index, meta = get_or_sync_index(toko_id, tipe)
    entries = meta.get('entries', [])
    if not entries:
        return []

    q_vec = embed_text(str(query)).reshape(1, -1)

    if index is not None and hasattr(index, 'search'):
        if index.ntotal == 0:
            return []
        k = min(top_k, index.ntotal)
        scores, indices = index.search(q_vec, k)
        scores_arr = scores[0]
        indices_arr = indices[0]
    else:
        # Dense vector cosine similarity calculation via NumPy when FAISS binary is absent.
        texts = [e['text'] for e in entries]
        vectors = embed_batch(texts)
        if vectors.shape[0] == 0:
            return []
        sims = np.dot(vectors, q_vec.T).flatten()
        k = min(top_k, len(entries))
        top_idx = np.argsort(sims)[::-1][:k]
        scores_arr = sims[top_idx]
        indices_arr = top_idx

    results = []
    for score, idx in zip(scores_arr, indices_arr):
        if idx < 0 or idx >= len(entries):
            continue
        sc = float(score)
        if sc < min_score:
            continue
        item = entries[idx]
        if str(item.get('store_id', toko_id)) != str(toko_id):
            continue
        results.append({
            'score': round(sc, 4),
            'type': item['type'],
            'category_id': item['category_id'],
            'category_name': item['category_name'],
            'text': item['text'],
            'entry_id': item['id'],
            'normalized_desc': item.get('normalized_desc'),
            'verification_source': item.get('verification_source', 'verified'),
            'business_type': item.get('business_type'),
            'store_id': item.get('store_id', str(toko_id)),
        })

    return results


def retrieve_context_batch(toko_id, tipe, queries, top_k=5, min_score=0.25):
    """Batched retrieval for multiple queries using single FAISS batch search."""
    if not toko_id or not str(toko_id).strip() or top_k <= 0 or not queries:
        return [[] for _ in queries]

    if tipe not in ('pemasukan', 'pengeluaran'):
        raise ValueError("Jenis transaksi harus 'pemasukan' atau 'pengeluaran'")

    index, meta = get_or_sync_index(toko_id, tipe)
    entries = meta.get('entries', [])
    if not entries:
        return [[] for _ in queries]

    valid_queries = [str(q).strip() for q in queries]
    q_vecs = embed_batch(valid_queries)
    if q_vecs.shape[0] == 0:
        return [[] for _ in queries]

    all_results = []
    if index is not None and hasattr(index, 'search') and index.ntotal > 0:
        k = min(top_k, index.ntotal)
        scores_matrix, indices_matrix = index.search(q_vecs, k)
        for i in range(len(valid_queries)):
            res_i = []
            for score, idx in zip(scores_matrix[i], indices_matrix[i]):
                if idx < 0 or idx >= len(entries):
                    continue
                sc = float(score)
                if sc < min_score:
                    continue
                item = entries[idx]
                res_i.append({
                    'score': round(sc, 4),
                    'type': item['type'],
                    'category_id': item['category_id'],
                    'category_name': item['category_name'],
                    'text': item['text'],
                    'entry_id': item['id'],
                    'normalized_desc': item.get('normalized_desc'),
                    'verification_source': item.get('verification_source', 'verified'),
                    'business_type': item.get('business_type'),
                    'store_id': item.get('store_id', str(toko_id)),
                })
            all_results.append(res_i)
    else:
        # Fallback NumPy batch cosine calculation
        texts = [e['text'] for e in entries]
        vectors = embed_batch(texts)
        if vectors.shape[0] == 0:
            return [[] for _ in queries]
        sims_matrix = np.dot(q_vecs, vectors.T)
        k = min(top_k, len(entries))
        for i in range(len(valid_queries)):
            sims = sims_matrix[i]
            top_idx = np.argsort(sims)[::-1][:k]
            res_i = []
            for idx in top_idx:
                sc = float(sims[idx])
                if sc < min_score:
                    continue
                item = entries[idx]
                res_i.append({
                    'score': round(sc, 4),
                    'type': item['type'],
                    'category_id': item['category_id'],
                    'category_name': item['category_name'],
                    'text': item['text'],
                    'entry_id': item['id'],
                    'normalized_desc': item.get('normalized_desc'),
                    'verification_source': item.get('verification_source', 'verified'),
                    'business_type': item.get('business_type'),
                    'store_id': item.get('store_id', str(toko_id)),
                })
            all_results.append(res_i)

    return all_results


def rebuild_store_index(toko_id, tipe=None):
    """Purge cached vector files and rebuild clean reference indexes from verified records."""
    types_to_rebuild = [tipe] if tipe else ['pemasukan', 'pengeluaran']
    results = {}
    for t in types_to_rebuild:
        faiss_path, meta_path = _get_file_paths(toko_id, t)
        if os.path.exists(faiss_path):
            try:
                os.remove(faiss_path)
            except OSError:
                # Safe fallback: file lock or filesystem cleanup failure will be overwritten by rebuild
                pass
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                # Safe fallback: file lock or filesystem cleanup failure will be overwritten by rebuild
                pass
        clear_memory_cache(toko_id, t)
        remote_version = get_remote_version(toko_id, t)
        index, meta = build_and_save_index(toko_id, t, version=remote_version)
        results[t] = meta.get('count', 0)
    return results


def retrieve_context_for_actor(token, query, tipe, top_k=5, min_score=0.25):
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    return retrieve_context(actor['toko_id'], tipe, query, top_k=top_k, min_score=min_score)
