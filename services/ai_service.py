import datetime
from database.client import privileged_client
from services.auth_service import verify_actor
from services import finance_service as finance
from ai.classifier import classify, detect_type
from ai.category_suggester import recommend
from ai.text_normalizer import extract_money, clean_text_for_storage
from ai.ollama_client import diagnosis
from ai.rag_engine import retrieve_context_for_actor, get_or_sync_index


def prepare_categories(token):
    actor = verify_actor(token, {'pemilik_toko'})
    shop = privileged_client().table('toko').select('bidang_usaha').eq('id', actor['toko_id']).single().execute().data
    return recommend(shop['bidang_usaha'])


def save_initial_categories(token, selection):
    actor = verify_actor(token, {'pemilik_toko'})
    existing = finance.list_categories(token)
    if existing:
        raise ValueError('Kategori sudah ada, silakan gunakan kelola kategori')
    for tp, cap in [('pengeluaran', 10), ('pemasukan', 5)]:
        names = [str(x).strip() for x in selection.get(tp, [])]
        if len(names) > cap or len(set(x.casefold() for x in names)) != len(names):
            raise ValueError('Kategori melewati batas atau nama ganda')
        if 'Others' not in names:
            names.append('Others')
        if tp == 'pemasukan' and 'Tambahan Modal' not in names:
            names.append('Tambahan Modal')
        if len(names) > cap:
            raise ValueError('Batas kategori tercapai setelah menambah kategori wajib')
        for name in names:
            finance.create_category(token, name, tp, system_seed=True)
    return finance.list_categories(token)


def classify_preview(token, tipe, deskripsi):
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    return classify(tipe, deskripsi, finance.list_categories(token), toko_id=actor['toko_id'])


def retrieve_semantic_context(token, tipe, text, top_k=5):
    return retrieve_context_for_actor(token, text, tipe, top_k=top_k)


def bulk_classify(token, rows, progress=None):
    if not rows:
        return []
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    shop_id = actor['toko_id']
    cats = finance.list_categories(token)
    total = len(rows)

    # Pre-sync FAISS indexes once per batch to avoid repetitive database version queries.
    for tp in ('pengeluaran', 'pemasukan'):
        try:
            get_or_sync_index(shop_id, tp)
        except Exception:
            pass

    other_cats = {
        tp: next((c for c in cats if c['tipe'] == tp and c['nama'].casefold() == 'others'), None)
        for tp in ('pengeluaran', 'pemasukan')
    }

    output = []
    for idx, row in enumerate(rows):
        row_res = dict(row)

        if row.get('error'):
            row_res['status_kategori'] = 'perlu_tinjauan'
            output.append(row_res)
            if progress:
                progress((idx + 1) / total)
            continue

        raw_desc = str(row.get('deskripsi') or '')
        clean_desc = clean_text_for_storage(raw_desc)
        if not clean_desc.strip():
            row_res['error'] = 'Deskripsi transaksi kosong'
            row_res['status_kategori'] = 'perlu_tinjauan'
            other = other_cats.get(row.get('tipe', 'pengeluaran'))
            row_res['kategori_id'] = other['id'] if other else None
            output.append(row_res)
            if progress:
                progress((idx + 1) / total)
            continue

        tp = (row.get('tipe') or '').strip().lower()
        if not tp or tp not in ('pemasukan', 'pengeluaran'):
            try:
                tp = detect_type(clean_desc)
            except Exception:
                # Ambiguous transaction type defaults to expense and flags for user review.
                tp = 'pengeluaran'
                row_res['tipe'] = tp
                row_res['error'] = 'Jenis transaksi tidak dapat ditentukan otomatis'
                row_res['status_kategori'] = 'perlu_tinjauan'
                other = other_cats['pengeluaran']
                row_res['kategori_id'] = other['id'] if other else None
                row_res['message'] = 'Jenis transaksi ambigu, disarankan untuk ditinjau'
                output.append(row_res)
                if progress:
                    progress((idx + 1) / total)
                continue

        row_res['tipe'] = tp

        try:
            decision = classify(tp, clean_desc, cats, toko_id=shop_id)
            row_res.update(decision)
        except Exception as exc:
            other = other_cats.get(tp)
            row_res['kategori_id'] = other['id'] if other else None
            row_res['usulan'] = {}
            row_res['skor'] = None
            row_res['status_kategori'] = 'perlu_tinjauan'
            row_res['message'] = f'Gagal klasifikasi: {exc}'
            row_res['error'] = str(exc)

        output.append(row_res)
        if progress:
            progress((idx + 1) / total)

    return output


def quick_preview(token, text, tipe=None):
    amount = extract_money(text)
    tp = tipe or detect_type(text)
    choice = classify_preview(token, tp, text)
    return {
        'tanggal': datetime.date.today().isoformat(),
        'deskripsi': clean_text_for_storage(text),
        'nominal': str(amount),
        'tipe': tp,
        **choice
    }


def commit_preview(token, row, kategori_id):
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
