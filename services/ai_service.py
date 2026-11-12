import datetime
from database.client import privileged_client
from services.auth_service import verify_actor
from services.validation_service import get_jakarta_today
from services import finance_service as finance
from ai.classifier import classify, detect_type
from ai.category_suggester import recommend
from ai.text_normalizer import extract_money, clean_text_for_storage
from ai.ollama_client import diagnosis, LLMUnavailable
from ai.rag_engine import retrieve_context_for_actor, get_or_sync_index
from ai.hybrid_classifier import hybrid_bulk_classify, clear_classification_cache
from ai.store_context import get_store_business_context


def prepare_categories(token):
    actor = verify_actor(token, {'pemilik_toko'})
    shop = privileged_client().table('toko').select('bidang_usaha').eq('id', actor['toko_id']).single().execute().data
    return recommend(shop['bidang_usaha'])


def discover_categories_for_import(token, rows):
    """Discover relevant financial categories from uploaded transactions before classification."""
    actor = verify_actor(token, {'pemilik_toko'})
    shop_id = actor['toko_id']
    shop_info = privileged_client().table('toko').select('nama,bidang_usaha').eq('id', shop_id).single().execute().data or {}
    business_type = shop_info.get('bidang_usaha', 'Umum')
    existing_categories = finance.list_categories(token)

    from ai.category_discovery import discover_categories_from_transactions
    try:
        return discover_categories_from_transactions(
            rows=rows,
            existing_categories=existing_categories,
            business_type=business_type,
            max_recommendations_per_type=7,
        )
    except LLMUnavailable:
        queued = bulk_classify(token, rows, categories=existing_categories, persist_pending=False)
        return {'pengeluaran': [], 'pemasukan': [], 'coverage_stats': {},
                'ai_pending_count': sum(bool((r.get('usulan') or {}).get('ai_pending')) for r in queued)}


def save_initial_categories(token, selection, enforce_ai_limit=True):
    actor = verify_actor(token, {'pemilik_toko'})
    existing = finance.list_categories(token)
    if existing:
        raise ValueError('Kategori sudah ada, silakan gunakan kelola kategori')
    for tp, cap in [('pengeluaran', 7), ('pemasukan', 7)]:
        names = [str(x).strip() for x in selection.get(tp, []) if str(x).strip()]
        if enforce_ai_limit and (len(names) > cap or len(set(x.casefold() for x in names)) != len(names)):
            raise ValueError('Kategori melewati batas atau nama ganda')
        if len(set(x.casefold() for x in names)) != len(names):
            raise ValueError('Nama kategori ganda')
        if not any(x.casefold() == 'others' for x in names):
            names.append('Others')
        if tp == 'pemasukan' and not any(x.casefold() == 'tambahan modal' for x in names):
            names.append('Tambahan Modal')
        if enforce_ai_limit and len(names) > cap:
            raise ValueError('Batas kategori tercapai setelah menambah kategori wajib')
        for name in names:
            finance.create_category(token, name, tp, system_seed=True)
    return finance.list_categories(token)


def classify_preview(token, tipe, deskripsi):
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    shop_id = actor['toko_id']
    b_type = get_store_business_context(shop_id).get('bidang_usaha', 'Umum')
    try:
        return classify(tipe, deskripsi, finance.list_categories(token), toko_id=shop_id, business_type=b_type)
    except LLMUnavailable as exc:
        # This wrapper has no amount/date: the complete quick_preview below
        # persists the row immediately, before returning it to the UI.
        return _pending_decision(exc)


def retrieve_semantic_context(token, tipe, text, top_k=5):
    return retrieve_context_for_actor(token, text, tipe, top_k=top_k)


def get_classification_index_status(token):
    """Return read-only status of the store's FAISS similarity index for UI visibility."""
    import json as _json
    import os as _os
    import ai.rag_engine as _rag_engine
    from ai.embedding_pipeline import is_semantic_model_available
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    shop_id = actor['toko_id']
    store_dir = _os.path.join(_rag_engine.INDEX_ROOT_DIR, str(shop_id))
    by_type = {}
    for tp in ('pengeluaran', 'pemasukan'):
        faiss_path = _os.path.join(store_dir, f'{tp}.faiss')
        meta_path = _os.path.join(store_dir, f'{tp}_meta.json')
        ready = _os.path.exists(faiss_path) and _os.path.exists(meta_path)
        doc_count = 0
        if ready:
            try:
                with open(meta_path, 'r', encoding='utf-8') as fh:
                    meta = _json.load(fh)
                doc_count = len(meta.get('entries', [])) if isinstance(meta, dict) else 0
            except Exception:
                doc_count = 0
        by_type[tp] = {'ready': ready, 'doc_count': doc_count}
    return {
        'toko_id': shop_id,
        'semantic_model_available': is_semantic_model_available(),
        'pengeluaran': by_type['pengeluaran'],
        'pemasukan': by_type['pemasukan'],
    }


def _pending_decision(exc):
    return {'kategori_id': None, 'status_kategori': 'perlu_tinjauan', 'skor': None,
            'source': 'perlu_tinjauan', 'is_unresolved': True,
            'usulan': {'ai_pending': True, 'ai_error': str(exc)[:300]},
            'message': 'Ollama tidak tersedia - transaksi menunggu AI (llama3:8b).'}


def bulk_classify(token, rows, progress=None, categories=None, *, persist_pending=True):
    if not rows:
        return []
    actor = verify_actor(token, {'pemilik_toko','kasir'})
    cats = categories if categories is not None else finance.list_categories(token)
    context = get_store_business_context(actor['toko_id'])
    try:
        return hybrid_bulk_classify(actor['toko_id'], cats, rows, progress_callback=progress,
                                    business_type=context.get('bidang_usaha','Umum'))
    except LLMUnavailable as exc:
        output = []
        for row in rows:
            if row.get('error'):
                output.append(dict(row))
                continue
            decision = _pending_decision(exc)
            output.append({**row, **decision, 'usulan': {**(row.get('usulan') or {}), **decision['usulan']}})
        pending = [r for r in output if (r.get('usulan') or {}).get('ai_pending') and not r.get('error')]
        if pending and persist_pending:
            from services.import_service import execute_structured_import
            result = execute_structured_import(token, pending)
            if result['failed_count']:
                raise RuntimeError('Penyimpanan antrean AI gagal: ' + str(result.get('error') or result.get('failed_batches')))
            for row in pending:
                row['_saved_ai_pending'] = True
                row['message'] += ' Sudah tersimpan.'
        return output


def pending_ai_transactions(token):
    from repositories.finance_repository import transactions
    actor = verify_actor(token, {'pemilik_toko','kasir'})
    return [r for r in transactions(actor['toko_id'])
            if (r.get('usulan') or {}).get('ai_pending') is True
            and r.get('status_persetujuan') != 'disetujui'
            and (actor['role'] == 'pemilik_toko' or r.get('dicatat_oleh') == actor['id'])]


def retry_pending_ai_classifications(token, limit=20):
    actor = verify_actor(token, {'pemilik_toko','kasir'})
    rows = pending_ai_transactions(token)
    cats = finance.list_categories(token)
    done, errors = 0, []
    for row in rows[:max(1,limit)]:
        try:
            decision = classify(row['tipe'], row['deskripsi'], cats, toko_id=actor['toko_id'])
            if decision['source'] not in ('llm', 'similarity') or not decision['kategori_id']:
                continue  # Model/similarity abstained; keep the queue marker and null category.
            saved = privileged_client().rpc('retry_pending_ai_classification', {
                'p_actor': actor['id'], 'p_shop': actor['toko_id'], 'p_id': row['id'],
                'p_category': decision['kategori_id'], 'p_updated_at': row.get('updated_at'),
            }).execute().data
            done += int(saved is True)
        except LLMUnavailable as exc:
            errors.append(str(exc))
            break
        except Exception as exc:
            errors.append('Pembaruan antrean gagal: ' + str(exc))
            break
    if done:
        clear_classification_cache(actor['toko_id'])
        from ai.rag_engine import clear_memory_cache
        clear_memory_cache(actor['toko_id'])
    return {'processed': done, 'remaining': len(pending_ai_transactions(token)), 'errors': errors}


def quick_preview(token, text, tipe=None):
    amount = extract_money(text)
    tp = tipe or detect_type(text)
    choice = classify_preview(token, tp, text)
    result = {
        'tanggal': get_jakarta_today().isoformat(),
        'deskripsi': clean_text_for_storage(text),
        'nominal': str(amount),
        'tipe': tp,
        **choice
    }

    if (result.get('usulan') or {}).get('ai_pending'):
        saved = finance.add_transaction(token, result['tanggal'], result['deskripsi'], result['nominal'],
                                        result['tipe'], None, status_kategori='perlu_tinjauan', usulan=result['usulan'])
        if not saved:
            raise RuntimeError('Antrean AI tidak tersimpan')
        result['_saved_ai_pending'] = True
        result['transaction_id'] = saved[0]['id']
    return result


def commit_preview(token, row, kategori_id):
    if row.get('_saved_ai_pending'):
        return [{'id': row.get('transaction_id')}]
    kwargs = {}
    if 'status_kategori' in row:
        kwargs['status_kategori'] = row['status_kategori']
    if 'skor' in row:
        kwargs['skor'] = row['skor']
    if 'usulan' in row:
        kwargs['usulan'] = row['usulan']
    if kwargs:
        return finance.add_transaction(
            token, row['tanggal'], row['deskripsi'], row['nominal'],
            row['tipe'], kategori_id, **kwargs
        )
    return finance.add_transaction(
        token, row['tanggal'], row['deskripsi'], row['nominal'],
        row['tipe'], kategori_id
    )


def suggest_others(token, tipe):
    import json
    from ai.ollama_client import ask_json
    from repositories.finance_repository import transactions
    actor = verify_actor(token, {'pemilik_toko'})
    cats = [c for c in finance.list_categories(token) if c['tipe'] == tipe]
    other = next((c for c in cats if c['nama'].casefold() == 'others'), None)
    if other is None:
        raise ValueError('Kategori Others tidak tersedia')
    examples = [t['deskripsi'] for t in transactions(actor['toko_id'])
                if t['tipe'] == tipe and t['kategori_id'] == other['id'] and t['status_persetujuan'] == 'disetujui']
    if not examples:
        return []
    result = ask_json(
        'Kelompokkan contoh transaksi Others dalam maksimal 3 nama kategori baru yang luas, tidak duplikatif. JSON usulan berupa array string.',
        json.dumps({'tipe': tipe, 'contoh': examples[:25], 'existing': [c['nama'] for c in cats]}, ensure_ascii=False),
        {'type': 'object', 'properties': {'usulan': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['usulan']}
    )
    existing = {c['nama'].casefold() for c in cats}
    return [str(x).strip()[:80] for x in result.get('usulan', [])[:3]
            if isinstance(x, str) and x.strip() and x.strip().casefold() not in existing]


def preview_others_reclassification(token, tipe=None):
    """Classify existing Others rows for preview only without mutating database state."""
    from repositories.finance_repository import transactions
    actor = verify_actor(token, {'pemilik_toko'})
    shop = actor['toko_id']
    cats = finance.list_categories(token)
    others_ids = {
        str(c['id']) for c in cats
        if c.get('sistem') or str(c.get('nama', '')).strip().casefold() == 'others'
    }
    if not others_ids:
        return {
            'total_others': 0, 'unique_descriptions': 0,
            'reclassified_count': 0, 'unresolved_count': 0,
            'groups': [], 'updates': [],
        }
    cat_map = {str(c['id']): c for c in cats}
    txs = [
        t for t in transactions(shop)
        if t.get('kategori_id') and str(t['kategori_id']) in others_ids
        and t.get('status_persetujuan') != 'ditolak'
        and (not tipe or t.get('tipe') == tipe)
    ]
    if not txs:
        return {
            'total_others': 0, 'unique_descriptions': 0,
            'reclassified_count': 0, 'unresolved_count': 0,
            'groups': [], 'updates': [],
        }

    # Build one representative item per unique (tipe, clean_deskripsi) to minimize LLM calls.
    grouped_txs = {}
    for t in txs:
        tp = t.get('tipe', 'pengeluaran')
        clean_desc = clean_text_for_storage(t.get('deskripsi', ''))
        key = (tp, clean_desc.casefold())
        grouped_txs.setdefault(key, {'tipe': tp, 'deskripsi': clean_desc or t.get('deskripsi', ''), 'rows': []})
        grouped_txs[key]['rows'].append(t)

    rep_rows = [
        {'tipe': g['tipe'], 'deskripsi': g['deskripsi'], '_group_key': k}
        for k, g in grouped_txs.items()
    ]
    clear_classification_cache(shop)
    b_type = get_store_business_context(shop).get('bidang_usaha', 'Umum')
    classified = hybrid_bulk_classify(shop, cats, rep_rows, business_type=b_type)

    groups_out, updates = [], []
    reclassified_count, unresolved_count = 0, 0
    for item in classified:
        key = item['_group_key']
        g = grouped_txs[key]
        new_cid = str(item.get('kategori_id') or '')
        valid_target = bool(new_cid) and new_cid in cat_map and new_cid not in others_ids
        if valid_target:
            new_cname = cat_map[new_cid]['nama']
            reclassified_count += len(g['rows'])
            for row in g['rows']:
                updates.append({
                    'transaction_id': row['id'],
                    'tanggal': row.get('tanggal'),
                    'deskripsi': row.get('deskripsi'),
                    'nominal': row.get('nominal'),
                    'tipe': row.get('tipe'),
                    'old_kategori_id': row.get('kategori_id'),
                    'new_kategori_id': new_cid,
                    'new_kategori_nama': new_cname,
                    'status_persetujuan': row.get('status_persetujuan'),
                    'source': item.get('source'),
                    'skor': item.get('skor'),
                })
        else:
            new_cname = 'Perlu Tinjauan'
            unresolved_count += len(g['rows'])

        diag = (item.get('usulan') or {}).get('similarity_diag') or {}
        groups_out.append({
            'tipe': g['tipe'],
            'deskripsi': g['deskripsi'],
            'count': len(g['rows']),
            'total_nominal': sum(float(r.get('nominal') or 0) for r in g['rows']),
            'current_category': 'Others',
            'suggested_kategori_id': new_cid if valid_target else None,
            'suggested_kategori_nama': new_cname,
            'source': item.get('source'),
            'skor': item.get('skor'),
            'top_similarity_category': diag.get('top_category'),
            'gate_passed': bool(diag.get('gate_passed')),
            'message': item.get('message', ''),
            'tx_ids': [r['id'] for r in g['rows']],
        })

    groups_out.sort(key=lambda x: (-x['count'], x['deskripsi']))
    return {
        'total_others': len(txs),
        'unique_descriptions': len(grouped_txs),
        'reclassified_count': reclassified_count,
        'unresolved_count': unresolved_count,
        'groups': groups_out,
        'updates': updates,
    }


def confirm_others_reclassification(token, updates):
    """Apply owner-confirmed Others reclassifications via audited transaction mutations and rebuild FAISS."""
    from repositories.finance_repository import transaction_by_id
    actor = verify_actor(token, {'pemilik_toko'})
    shop = actor['toko_id']
    if not updates:
        return {'updated_count': 0, 'reapproved_count': 0, 'updated_ids': []}

    cats = {str(c['id']): c for c in finance.list_categories(token)}
    others_ids = {
        cid for cid, c in cats.items()
        if c.get('sistem') or str(c.get('nama', '')).strip().casefold() == 'others'
    }

    updated_ids, reapprove_ids, updated_types = [], [], set()
    for upd in updates:
        tx_id = upd.get('transaction_id')
        new_cid = str(upd.get('new_kategori_id') or '')
        if not tx_id or not new_cid or new_cid not in cats or new_cid in others_ids:
            continue
        tx = transaction_by_id(shop, tx_id)
        if not tx or tx.get('tipe') != cats[new_cid].get('tipe'):
            continue
        was_approved = tx.get('status_persetujuan') == 'disetujui'
        finance.update_transaction(
            token, tx_id, tx['tanggal'], tx['deskripsi'], tx['nominal'], tx['tipe'], new_cid
        )
        updated_ids.append(tx_id)
        updated_types.add(tx['tipe'])
        if was_approved:
            reapprove_ids.append(tx_id)

    for start in range(0, len(reapprove_ids), 200):
        batch_ids = reapprove_ids[start:start + 200]
        finance.approve_transactions(token, batch_ids, 'disetujui')

    try:
        clear_classification_cache(shop)
        from ai.rag_engine import clear_memory_cache, rebuild_store_index
        clear_memory_cache(shop)
        for tp in updated_types:
            rebuild_store_index(shop, tp)
    except Exception:
        pass

    return {
        'updated_count': len(updated_ids),
        'reapproved_count': len(reapprove_ids),
        'updated_ids': updated_ids,
    }


def chat_message(token, text):
    from services.chat_service import handle_chat_message
    return handle_chat_message(token, text)


def confirm_chat_proposal(token, proposal):
    from services.chat_service import confirm_proposal
    return confirm_proposal(token, proposal)


def cancel_chat_proposal(proposal):
    from services.chat_service import cancel_proposal
    return cancel_proposal(proposal)
