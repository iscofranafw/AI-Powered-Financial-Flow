import os
import json
import numpy as np
from datetime import datetime, timezone
from database.client import privileged_client
from services.auth_service import verify_actor
from ai.embedding_pipeline import embed_text, embed_batch, get_embedding_dimension
from ai.text_normalizer import normalize

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

INDEX_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vector_store')

_CACHE = {}


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
        pass
    return 1


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


def build_corpus_entries(toko_id, tipe):
    entries = []
    client = privileged_client()

    cat_rows = (
        client.table('kategori')
        .select('id, nama, deskripsi, kata_kunci, sistem')
        .eq('toko_id', str(toko_id))
        .eq('tipe', str(tipe))
        .execute()
        .data or []
    )

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
        if isinstance(keywords, list) and keywords:
            clean_kw = [str(k).strip() for k in keywords if str(k).strip()]
            if clean_kw:
                parts.append(' '.join(clean_kw))

        text = '. '.join(parts)
        entries.append({
            'id': f"cat_{cat_id}",
            'type': 'category',
            'category_id': cat_id,
            'category_name': name,
            'text': text
        })

    tx_rows = (
        client.table('transaksi')
        .select('id, deskripsi, kategori_id')
        .eq('toko_id', str(toko_id))
        .eq('tipe', str(tipe))
        .eq('status_persetujuan', 'disetujui')
        .neq('status_kategori', 'perlu_tinjauan')
        .order('tanggal', desc=True)
        .limit(1000)
        .execute()
        .data or []
    )

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
            'category_id': cid,
            'category_name': cname,
            'text': raw_desc
        })

    return entries


def build_and_save_index(toko_id, tipe, entries=None, version=None):
    if version is None:
        version = get_remote_version(toko_id, tipe)

    if entries is None:
        entries = build_corpus_entries(toko_id, tipe)

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
            pass

    meta = {
        'toko_id': str(toko_id),
        'tipe': str(tipe),
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


def get_or_sync_index(toko_id, tipe, force_rebuild=False):
    toko_id = str(toko_id)
    tipe = str(tipe)
    cached = _CACHE.get((toko_id, tipe))
    remote_version = get_remote_version(toko_id, tipe)

    if not force_rebuild and cached is not None and cached['version'] == remote_version:
        return cached['index'], cached['meta']

    faiss_path, meta_path = _get_file_paths(toko_id, tipe)
    if not force_rebuild and os.path.exists(faiss_path) and os.path.exists(meta_path) and FAISS_AVAILABLE:
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if int(meta.get('version', 0)) == remote_version:
                index = faiss.read_index(faiss_path)
                _CACHE[(toko_id, tipe)] = {
                    'index': index,
                    'meta': meta,
                    'version': remote_version
                }
                return index, meta
        except Exception:
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
        results.append({
            'score': round(sc, 4),
            'type': item['type'],
            'category_id': item['category_id'],
            'category_name': item['category_name'],
            'text': item['text'],
            'entry_id': item['id']
        })

    return results


def retrieve_context_for_actor(token, query, tipe, top_k=5, min_score=0.25):
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    return retrieve_context(actor['toko_id'], tipe, query, top_k=top_k, min_score=min_score)
