import logging
import json
import math
from ai.embedding_pipeline import embed_text, embed_batch, is_semantic_model_available
from ai.ollama_client import ask_json, LLMUnavailable
from ai.text_normalizer import normalize, clean_text_for_storage
from ai.rag_engine import retrieve_context
from ai.store_context import get_store_business_context

logger = logging.getLogger(__name__)

CHOICE_SCHEMA = {
    'type': 'object',
    'properties': {
        'alasan': {'type': 'string'},
        'keyakinan': {'type': 'integer', 'minimum': 0, 'maximum': 100},
        'kategori_id': {'type': 'string'},
        'nama_baru': {'type': 'string'}
    },
    'required': ['kategori_id', 'nama_baru', 'alasan']
}


def detect_type(text):
    s = normalize(text)
    capital_phrases = (
        'setor modal', 'tambah modal', 'tambahan modal', 'suntik modal',
        'tanam modal', 'modal dari', 'modal masuk', 'penambahan modal'
    )
    if any(p in s for p in capital_phrases):
        return 'pemasukan'
    income = ('omzet', 'penjualan', 'terima', 'pendapatan', 'masuk', 'setoran modal', 'tambahan modal')
    expense = ('beli', 'bayar', 'belanja', 'gaji', 'listrik', 'sewa', 'stok', 'biaya', 'gas', 'air', 'upah')
    a = sum(x in s for x in income)
    b = sum(x in s for x in expense)
    if a > b:
        return 'pemasukan'
    if b > a:
        return 'pengeluaran'
    try:
        reply = ask_json(
            'Tentukan jenis transaksi UMKM. Hanya jawab "pemasukan" atau "pengeluaran" dalam format JSON.',
            json.dumps({'transaksi': text}, ensure_ascii=False),
            {'type': 'object', 'properties': {'tipe': {'type': 'string', 'enum': ['pemasukan', 'pengeluaran']}}, 'required': ['tipe']}
        )
        if reply.get('tipe') in ('pemasukan', 'pengeluaran'):
            return reply['tipe']
    except Exception:
        # Safe fallback: if LLM classification fails, fall through to explicit ValueError below
        pass
    raise ValueError('Jenis transaksi tidak dapat ditentukan otomatis. Silakan pilih jenis secara manual.')


def _cache_key(toko_id, tipe, description, categories, business_type):
    return (str(toko_id), tipe, clean_text_for_storage(description).casefold(),
            str(business_type or ''), json.dumps(categories, sort_keys=True, default=str))


def _excluded_system_ids(tipe, cats):
    """Return category IDs excluded from LLM options (system/Others categories)."""
    return {str(c['id']) for c in cats if c.get('tipe') == tipe
            and (c.get('sistem') or str(c.get('nama', '')).strip().casefold() == 'others')}


def _llm_context(tipe, description, cats, toko_id, business_type):
    business_type = business_type or (get_store_business_context(toko_id).get('bidang_usaha') if toko_id else None) or 'Umum'
    allowed = []
    for c in _similarity_categories(tipe, cats).values():
        base_desc = c.get('deskripsi') or f"{c['nama']} untuk bidang usaha {business_type}"
        clean_kw = [str(k).strip() for k in (c.get('kata_kunci') or []) if str(k).strip()]
        desc_text = f"{base_desc} (Kata kunci: {', '.join(clean_kw[:20])})" if clean_kw else base_desc
        allowed.append({'id': str(c['id']), 'nama': c['nama'], 'deskripsi': desc_text})
    return {'bidang_usaha': business_type, 'tipe_transaksi': tipe,
            'daftar_kategori_tersedia': allowed, 'deskripsi_transaksi': clean_text_for_storage(description),
            'few_shot_examples_only': []}


CLASSIFICATION_GUIDANCE = (
    'Kamu asisten akuntansi UMKM Indonesia. Pilih kategori HANYA dari ID yang ada pada daftar_kategori_tersedia toko '
    'yang telah ditetapkan untuk jenis transaksi yang sama, berdasarkan bidang usaha, nama, dan deskripsi kategori. '
    'Dilarang membuat kategori baru atau memilih kategori di luar daftar_kategori_tersedia. '
    'Dilarang memilih kategori Others/sistem sebagai jalan pintas ketika ragu atau tidak tahu. '
    'Pada retail/minimarket/kelontong, barang untuk dijual kembali adalah persediaan/stok barang, bukan bahan baku produksi. '
    'Tentukan kategori berdasarkan TUJUAN pengeluaran, bukan sekadar kata kerja beli atau bayar. '
    'Barang yang dibeli untuk dijual kembali adalah barang dagangan. Barang yang dipakai sendiri oleh usaha '
    'untuk membungkus belanja, membersihkan, atau melayani pelanggan adalah perlengkapan/peralatan. '
    'Bahan atau komponen yang digunakan untuk memproduksi makanan dan minuman adalah bahan baku. '
    # Owner policy, 12 Nov 2026 import bug report: bakery fuel is not an ingredient.
    'Pada bakery, bahan bakar untuk menjalankan oven atau kompor adalah biaya operasional/utilitas, '
    'bukan bahan yang menjadi isi produk. Jika ada pos bahan bakar khusus gunakan pos itu. '
    'Wadah yang diserahkan kepada pelanggan bersama produk adalah kemasan. '
    'Pembayaran jasa, langganan, iuran, atau pekerjaan seseorang bukan pembelian barang: '
    'utilitas, langganan, dan retribusi termasuk operasional; imbalan kerja, termasuk upah harian dan lembur, termasuk gaji. '
    'Jika dua kategori mungkin cocok, pilih kategori yang deskripsinya sesuai dengan TUJUAN yang disebut dalam transaksi, '
    'dan sebutkan frasa tujuan yang menentukan pilihan tersebut dalam alasan. '
    'Tulis alasan singkat DULU berdasarkan deskripsi kategori, baru pilih kategori_id dari daftar_kategori_tersedia. '
    'Nilai keyakinanmu 0-100. Di bawah 70 berarti kamu tidak yakin — kembalikan kategori_id kosong agar masuk antrean perlu_tinjauan. '
    'If no category in daftar_kategori_tersedia fits confidently, return an empty kategori_id so the transaction goes to perlu_tinjauan. '
    'Guessing a wrong category is worse than abstaining. Jawab JSON sesuai schema; alasan singkat.'
)


def _classification_guidance(include_similarity_hint=False):
    if not include_similarity_hint:
        return CLASSIFICATION_GUIDANCE
    marker = 'Dilarang membuat kategori baru atau memilih kategori di luar daftar_kategori_tersedia. '
    return CLASSIFICATION_GUIDANCE.replace(marker, marker +
        'Jika tersedia petunjuk_similarity, bandingkan kandidat itu dengan tujuan transaksi dan definisi kategori; '
        'petunjuk bukan keputusan final dan kamu boleh menolaknya atau abstain. ', 1)


def _decision(reply, valid_ids, blocked_ids=None):
    if not isinstance(reply, dict) or any(not isinstance(reply.get(k), str) for k in ('kategori_id','nama_baru','alasan')):
        raise ValueError('Respons klasifikasi tidak sesuai schema')
    cid = reply['kategori_id']
    if cid and blocked_ids and cid in blocked_ids:
        logger.warning('Model returned excluded Others/system category %s; coercing to abstain (perlu_tinjauan).', cid)
        cid = ''
    elif cid and cid not in valid_ids:
        raise ValueError('ID kategori di luar kategori toko/jenis transaksi')
    confidence = reply.get('keyakinan')
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and confidence < 70:
        cid = ''
    proposal = {'nama_baru': reply['nama_baru']} if reply['nama_baru'] else {}
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        proposal['llm_confidence'] = confidence
    return {'kategori_id': cid or None, 'usulan': proposal,
            'skor': None, 'status_kategori': 'dikonfirmasi' if cid else 'perlu_tinjauan',
            'source': 'llm' if cid else 'perlu_tinjauan', 'is_unresolved': not bool(cid),
            'reason_code': 'CLASSIFIED' if cid else 'MODEL_ABSTAINED', 'message': reply['alasan']}


def _enforce_purpose(decision, tipe, description, cats, business_type):
    from ai.category_evidence import bakery_fuel_owner
    applies, owner = bakery_fuel_owner(description, cats, business_type)
    if tipe == 'pengeluaran' and applies and decision.get('kategori_id') and decision['kategori_id'] != owner:
        decision.update(kategori_id=None, status_kategori='perlu_tinjauan', source='perlu_tinjauan',
                        is_unresolved=True, reason_code='PURPOSE_CONFLICT',
                        message='Bahan bakar bakery memerlukan kategori operasional/bahan bakar yang sesuai; tinjau kembali.')
    return decision


def _no_allowed_categories():
    result = _decision({'kategori_id': '', 'nama_baru': '',
                        'alasan': 'Belum ada kategori untuk jenis transaksi ini.'}, set())
    result['reason_code'] = 'NO_ALLOWED_CATEGORIES'
    return result


def _similarity_categories(tipe, cats):
    return {str(c['id']): c for c in cats if c.get('tipe') == tipe
            and not c.get('sistem') and str(c.get('nama', '')).strip().casefold() != 'others'}


def _similarity_ranking(tipe, deskripsi, cats, toko_id=None):
    """Best cosine per allowed category, from definitions and verified history."""
    allowed = _similarity_categories(tipe, cats)
    if not allowed or not str(deskripsi or '').strip():
        return []
    # Hash fallback vectors are useful for degraded retrieval, not auto-confirmation.
    if not is_semantic_model_available():
        raise RuntimeError('Semantic embedding model unavailable')
    if not toko_id:
        ids = list(allowed)
        texts = ['. '.join(filter(None, [allowed[cid]['nama'], allowed[cid].get('deskripsi')])) for cid in ids]
        scores = embed_batch(texts) @ embed_text(deskripsi)
        best = {cid: float(score) for cid, score in zip(ids, scores)}
        kw_pairs = []
        for cid in ids:
            cname = allowed[cid]['nama']
            for kw in (allowed[cid].get('kata_kunci') or []):
                clean_kw = str(kw or '').strip()
                if clean_kw:
                    kw_text = clean_kw  # User example semantics must not be diluted by arbitrary category names.
                    kw_pairs.append((cid, kw_text))
        if kw_pairs:
            kw_scores = embed_batch([t for _, t in kw_pairs]) @ embed_text(deskripsi)
            for (cid, _), score in zip(kw_pairs, kw_scores):
                best[cid] = max(best.get(cid, -1.0), float(score))
    else:
        # Twelve documents may all belong to one category. Expand until a distinct
        # runner-up is visible or the corpus is exhausted. Keep negative scores so
        # a cutoff cannot manufacture a margin over an unseen runner-up.
        k = 12
        while True:
            hits = retrieve_context(toko_id, tipe, deskripsi, top_k=k, min_score=-1.0)
            best = {}
            for hit in hits:
                cid = str(hit.get('category_id', ''))
                if (cid not in allowed or hit.get('type') not in ('category', 'transaction')
                        or str(hit.get('store_id')) != str(toko_id)):
                    continue
                score = float(hit['score'])
                best[cid] = max(best.get(cid, -1.0), score)
            if len(best) >= 2 or len(hits) < k:
                break
            if k >= 4096:
                raise RuntimeError('Similarity runner-up not established after 4096 documents')
            k *= 2
    if any(not math.isfinite(score) or not -1.0001 <= score <= 1.0001 for score in best.values()):
        raise ValueError('Invalid cosine similarity score')
    return sorted(best.items(), key=lambda item: -item[1])


def _similarity_evaluate(tipe, deskripsi, cats, toko_id=None):
    """Evaluate similarity ranking + gate (top >= 0.80, margin >= 0.04) and return (decision|None, diagnostics)."""
    diag = {'top_category_id': None, 'top_category': None, 'top_score': None,
            'second_score': None, 'margin': None, 'gate_passed': False}
    try:
        allowed = _similarity_categories(tipe, cats)
        ranking = _similarity_ranking(tipe, deskripsi, cats, toko_id)
        if not ranking:
            return None, diag
        cid, top = ranking[0]
        second = ranking[1][1] if len(ranking) > 1 else 0.0
        margin = top - second
        diag.update({
            'top_category_id': cid,
            'top_category': allowed[cid]['nama'] if cid in allowed else None,
            'top_score': round(float(top), 4) if math.isfinite(top) else None,
            'second_score': round(float(second), 4) if math.isfinite(second) else None,
            'margin': round(float(margin), 4) if math.isfinite(margin) else None,
            'gate_passed': False,
        })
        # Revalidate at the decision boundary, including a foreign top hit.
        if cid not in allowed:
            return None, diag
        if not math.isfinite(top) or not math.isfinite(second) or top < 0.80 or margin < 0.04:
            return None, diag
        diag['gate_passed'] = True
        result = _decision({'kategori_id': cid, 'nama_baru': '',
                            'alasan': f"Kemiripan {allowed[cid]['nama']}: {top:.4f} (margin {margin:.4f})."}, set(allowed))
        result.update(skor=top, source='similarity', reason_code='SIMILARITY_AUTO')
        result['usulan'] = {**result.get('usulan', {}), 'similarity_diag': dict(diag)}
        return result, diag
    except Exception as exc:
        logger.warning('Similarity gate unavailable; continuing with LLM: %s', exc)
        return None, diag


def _similarity_decision(tipe, deskripsi, cats, toko_id=None):
    """source='similarity' means the high-score/margin gate saved an LLM call."""
    result, _ = _similarity_evaluate(tipe, deskripsi, cats, toko_id)
    return result


def _valid_cached_decision(decision, tipe, cats):
    cid = decision.get('kategori_id')
    allowed = set(_similarity_categories(tipe, cats))
    return bool(cid) and cid in allowed


def classify(tipe, deskripsi, cats, toko_id=None, business_type=None):
    from ai.hybrid_classifier import _CLASSIFICATION_CACHE
    allowed_map = _similarity_categories(tipe, cats)
    if not allowed_map:
        return _no_allowed_categories()
    payload = _llm_context(tipe, deskripsi, cats, toko_id, business_type)
    key = _cache_key(toko_id, tipe, deskripsi, payload['daftar_kategori_tersedia'], payload['bidang_usaha'])
    if toko_id and key in _CLASSIFICATION_CACHE:
        if _valid_cached_decision(_CLASSIFICATION_CACHE[key], tipe, cats):
            return json.loads(json.dumps(_CLASSIFICATION_CACHE[key]))
        del _CLASSIFICATION_CACHE[key]
    result, diag = _similarity_evaluate(tipe, deskripsi, cats, toko_id)
    if result is not None:
        result = _enforce_purpose(result, tipe, deskripsi, cats, payload['bidang_usaha'])
    if result is not None:
        logger.info(
            'Classification gate [toko=%s tipe=%s] deskripsi="%s" top=%s score=%s margin=%s gate=PASS source=similarity',
            toko_id, tipe, clean_text_for_storage(deskripsi),
            diag.get('top_category'), diag.get('top_score'), diag.get('margin'),
        )
        if toko_id:
            _CLASSIFICATION_CACHE[key] = result
        return json.loads(json.dumps(result))
    # Only grounded user keywords warrant a fallback hint; a bare category
    # name/definition is too weak to steer the model below the gate threshold.
    top_candidate = allowed_map.get(diag.get('top_category_id')) or {}
    if top_candidate.get('kata_kunci'):
        payload['petunjuk_similarity'] = diag
    try:
        reply = ask_json(_classification_guidance('petunjuk_similarity' in payload), json.dumps(payload, ensure_ascii=False), CHOICE_SCHEMA)
    except Exception as exc:
        raise LLMUnavailable('Klasifikasi AI gagal. Jalankan Ollama dan model llama3:8b (ollama serve; ollama pull llama3:8b). ' + str(exc)) from exc
    try:
        blocked = _excluded_system_ids(tipe, cats)
        result = _decision(reply, {c['id'] for c in payload['daftar_kategori_tersedia']}, blocked_ids=blocked)
        result = _enforce_purpose(result, tipe, deskripsi, cats, payload['bidang_usaha'])
    except ValueError as exc:
        raise LLMUnavailable('Model mengembalikan jawaban klasifikasi tidak valid: ' + str(exc)) from exc
    if diag.get('top_score') is not None:
        result['skor'] = diag['top_score']
    if diag.get('top_score') is not None or diag.get('top_category') is not None:
        result['usulan'] = {**result.get('usulan', {}), 'similarity_diag': diag}
    logger.info(
        'Classification llm [toko=%s tipe=%s] deskripsi="%s" top=%s score=%s margin=%s gate=FAIL source=%s picked=%s',
        toko_id, tipe, clean_text_for_storage(deskripsi),
        diag.get('top_category'), diag.get('top_score'), diag.get('margin'),
        result.get('source'), allowed_map.get(result.get('kategori_id') or '', {}).get('nama'),
    )
    if toko_id and result['source'] == 'llm':
        _CLASSIFICATION_CACHE[key] = result
    return json.loads(json.dumps(result))
