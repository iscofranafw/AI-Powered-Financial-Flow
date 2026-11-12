"""Similarity-gated LLM batches and tenant memoisation (legacy module name)."""
import json
from ai.ollama_client import ask_json, LLMUnavailable
from ai.text_normalizer import clean_text_for_storage
from ai.store_context import get_store_business_context

_CLASSIFICATION_CACHE = {}


def clear_classification_cache(toko_id=None):
    for key in list(_CLASSIFICATION_CACHE):
        if toko_id is None or key[0] == str(toko_id):
            del _CLASSIFICATION_CACHE[key]


def classify_ambiguous_batch_ai(tipe, items, allowed_categories, other_cat_id=None, business_type=None,
                                toko_id=None):
    import logging
    from ai.classifier import (_classification_guidance, CHOICE_SCHEMA, _decision, _excluded_system_ids,
                               _llm_context, _no_allowed_categories, _similarity_categories,
                               _similarity_evaluate, _enforce_purpose)
    _log = logging.getLogger(__name__)
    if not items:
        return {}
    allowed_map = _similarity_categories(tipe, allowed_categories)
    if not allowed_map:
        return {ident: _no_allowed_categories() for ident, _ in items}
    blocked_ids = _excluded_system_ids(tipe, allowed_categories)
    gated, pending, pending_diags = {}, [], {}
    for ident, desc in items:
        decision, diag = _similarity_evaluate(tipe, desc, allowed_categories, toko_id)
        if decision is None:
            pending.append((ident, desc))
            pending_diags[ident] = diag
        else:
            decision = _enforce_purpose(decision, tipe, desc, allowed_categories, business_type)
            _log.info(
                'Classification batch gate [toko=%s tipe=%s] deskripsi="%s" top=%s score=%s margin=%s gate=PASS source=similarity',
                toko_id, tipe, clean_text_for_storage(desc),
                diag.get('top_category'), diag.get('top_score'), diag.get('margin'),
            )
            gated[ident] = decision
    if not pending:
        return gated
    items = pending
    schema = {'type':'object','properties':{'klasifikasi':{'type':'array',
        'minItems': len(items), 'maxItems': len(items), 'items':{
        'type':'object','properties':{'item_id':{'type':'integer'}, **CHOICE_SCHEMA['properties']},
        'required':['item_id', *CHOICE_SCHEMA['required']]}}},'required':['klasifikasi']}
    contexts = [_llm_context(tipe, desc, allowed_categories, toko_id, business_type) for _, desc in items]
    payload = {'bidang_usaha': contexts[0]['bidang_usaha'], 'tipe_transaksi': tipe,
               'daftar_kategori_tersedia': contexts[0]['daftar_kategori_tersedia'],
               'daftar_transaksi': [
                   {'item_id': ident, 'deskripsi_transaksi': context['deskripsi_transaksi'],
                    'petunjuk_similarity': (pending_diags.get(ident) or {})
                    if (allowed_map.get((pending_diags.get(ident) or {}).get('top_category_id')) or {}).get('kata_kunci') else {}}
                   for (ident, _), context in zip(items, contexts)]}
    try:
        reply = ask_json(system=_classification_guidance(any(item.get('petunjuk_similarity') for item in payload['daftar_transaksi'])) +
                         ' Kembalikan satu keputusan untuk SETIAP item_id, termasuk ketika abstain; jangan hilangkan item.'
                         ' Daftar kategori yang tersedia berlaku untuk semua item. Batasi alasan maksimal delapan kata.',
                         user=json.dumps(payload, ensure_ascii=False),
                         schema=schema, num_predict=max(512, len(items)*160))
    except Exception as exc:
        raise LLMUnavailable('Klasifikasi batch gagal. Jalankan Ollama dengan llama3:8b. ' + str(exc)) from exc
    try:
        entries = reply['klasifikasi']
        valid = set(allowed_map)
        expected = {ident for ident, _ in items}
        desc_by_id = {ident: desc for ident, desc in items}
        result = {}
        for entry in entries:
            ident = entry['item_id']
            if type(ident) is not int or ident not in expected or ident in result:
                raise ValueError('ID item batch tidak valid/duplikat')
            dec = _decision(entry, valid, blocked_ids=blocked_ids)
            dec = _enforce_purpose(dec, tipe, desc_by_id[ident], allowed_categories, business_type)
            diag = pending_diags.get(ident) or {}
            if diag.get('top_score') is not None:
                dec['skor'] = diag['top_score']
            dec['usulan'] = {**dec.get('usulan', {}), 'similarity_diag': diag}
            _log.info(
                'Classification batch llm [toko=%s tipe=%s] deskripsi="%s" top=%s score=%s margin=%s gate=FAIL source=%s picked=%s',
                toko_id, tipe, clean_text_for_storage(desc_by_id.get(ident, '')),
                diag.get('top_category'), diag.get('top_score'), diag.get('margin'),
                dec.get('source'), allowed_map.get(dec.get('kategori_id') or '', {}).get('nama'),
            )
            result[ident] = dec
        if set(result) != expected:
            raise ValueError('Respons batch tidak lengkap')
        return {**gated, **result}
    except Exception as exc:
        raise LLMUnavailable('Model mengembalikan jawaban klasifikasi batch tidak valid: ' + str(exc)) from exc


def hybrid_bulk_classify(toko_id, categories, rows, progress_callback=None, batch_size_ai=8, business_type=None):
    from ai.classifier import _cache_key, _valid_cached_decision
    business_type = business_type or get_store_business_context(toko_id).get('bidang_usaha', 'Umum')
    output = [dict(row) for row in rows]
    groups = {}
    for idx, row in enumerate(output):
        if row.get('error'):
            continue
        tp = row.get('tipe')
        if tp not in ('pemasukan','pengeluaran'):
            from ai.classifier import detect_type
            tp = detect_type(row.get('deskripsi',''))
            row['tipe'] = tp
        allowed = [c for c in categories if c.get('tipe') == tp]
        desc = clean_text_for_storage(row.get('deskripsi',''))
        key = _cache_key(toko_id, tp, desc, allowed, business_type)
        if key in _CLASSIFICATION_CACHE and not _valid_cached_decision(_CLASSIFICATION_CACHE[key], tp, allowed):
            del _CLASSIFICATION_CACHE[key]
        if key in _CLASSIFICATION_CACHE:
            decision = _CLASSIFICATION_CACHE[key]
            row.update(json.loads(json.dumps(decision)))
            row['usulan'] = {**(rows[idx].get('usulan') or {}), **decision.get('usulan',{})}
        else:
            groups.setdefault(tp, {}).setdefault(desc, []).append(idx)
    for tp, descriptions in groups.items():
        unique = list(descriptions)
        for start in range(0,len(unique), max(1,batch_size_ai)):
            items = list(enumerate(unique[start:start+max(1,batch_size_ai)],1))
            allowed = [c for c in categories if c.get('tipe') == tp]
            results = classify_ambiguous_batch_ai(tp, items, allowed, business_type=business_type, toko_id=toko_id)
            for ident, desc in items:
                decision = results[ident]
                if decision['source'] in ('llm', 'similarity'):
                    _CLASSIFICATION_CACHE[_cache_key(toko_id,tp,desc,allowed,business_type)] = decision
                for idx in descriptions[desc]:
                    output[idx].update(json.loads(json.dumps(decision)))
                    output[idx]['usulan'] = {**(rows[idx].get('usulan') or {}), **decision.get('usulan',{})}
            if progress_callback:
                progress_callback(min(1.0, (start+len(items))/max(1,len(unique))))
    if progress_callback:
        progress_callback(1.0)
    return output
