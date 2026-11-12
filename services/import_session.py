"""Draft category approval and final-save mapping for the CSV wizard."""
import copy
import hashlib
import json
import uuid

from services import finance_service as finance
from ai.store_context import get_store_business_context
from database.client import privileged_client


def approval_signature(existing, reviewed, rows, fill_keywords):
    return hashlib.sha256(json.dumps([existing, reviewed, rows, fill_keywords],
                                   sort_keys=True, default=str).encode()).hexdigest()


def invalidate_approval(state, signature):
    if state.get('csv_approval_signature') != signature:
        for key in ('csv_category_approved', 'csv_approved_categories', 'csv_proposals'):
            state.pop(key, None)


def discard_draft(state, reset_upload=True):
    # A synchronous final save cannot be cancelled or rolled back from the UI.
    # In particular, a queued second click must not erase a partial-save report.
    if state.get('csv_commit_started'):
        return False
    for key in list(state):
        if key.startswith(('csv_', 'review_', 'exp_', 'inc_', 'btn_del_')):
            state.pop(key, None)
    if reset_upload:
        state['import_upload_generation'] = state.get('import_upload_generation', 0) + 1
    return True


def confirmed_history(token):
    actor = finance._member(token)
    return (privileged_client().table('transaksi').select('id,kategori_id,deskripsi,tipe')
            .eq('toko_id', actor['toko_id']).eq('status_persetujuan', 'disetujui')
            .eq('status_kategori', 'dikonfirmasi').order('tanggal', desc=True)
            .limit(1000).execute().data or [])


def stage_categories(token, existing, reviewed, rows, fill_keywords):
    """No INSERT/UPDATE: even mandatory system categories remain draft UUIDs."""
    cats = copy.deepcopy(existing)
    keys = {(c['tipe'], c['nama'].strip().casefold()) for c in cats}
    seeds = [dict(nama='Others', tipe=tp, sistem=True) for tp in ('pengeluaran', 'pemasukan')]
    seeds.append(dict(nama='Tambahan Modal', tipe='pemasukan', bukan_operasional=True))
    for cat in [*seeds, *reviewed]:
        key = cat['tipe'], cat['nama'].strip().casefold()
        if key not in keys:
            cats.append({**copy.deepcopy(cat), 'id': str(uuid.uuid4()), '_draft': True})
            keys.add(key)
    if fill_keywords:
        shop = finance._member(token)['toko_id']
        business = get_store_business_context(shop).get('bidang_usaha', 'Umum')
        plan = finance.plan_category_keywords(cats, confirmed_history(token), business, rows)
        planned = {c['id']: c for c in plan['categories']}
        for cat in cats:
            if not cat.get('kata_kunci') and not cat.get('sistem') and not cat.get('bukan_operasional'):
                cat['kata_kunci'] = planned[cat['id']].get('kata_kunci', [])
    return cats


def materialize_categories(token, staged):
    """Called only by the final save button. Reuse IDs on an explicit retry.

    Existing auth, tenant checks and transaction RPCs remain in their services.
    On failure the caller retains commit state and displays persisted changes.
    """
    current = finance.list_categories(token)
    by_key = {(c['tipe'], c['nama'].strip().casefold()): c for c in current}
    by_id = {c['id']: c for c in current}
    # Validate the complete existing set before creating even one draft category.
    for cat in staged:
        real = by_id.get(cat['id'])
        if not cat.get('_draft') and (real is None or real['tipe'] != cat['tipe'] or real['nama'] != cat['nama']):
            raise ValueError('Kategori toko berubah. Tinjau dan setujui ulang sebelum menyimpan.')
    mapping, created = {}, 0
    for cat in staged:
        key = cat['tipe'], cat['nama'].strip().casefold()
        real = by_key.get(key) if cat.get('_draft') else by_id.get(cat['id'])
        if real is None and not cat.get('_draft'):
            raise ValueError('Kategori toko berubah. Tinjau dan setujui ulang sebelum menyimpan.')
        if real is None:
            result = finance.create_category(token, cat['nama'], cat['tipe'],
                    description=cat.get('deskripsi', ''), keywords=cat.get('kata_kunci', []),
                    system_seed=bool(cat.get('sistem') or cat.get('bukan_operasional')))
            if not isinstance(result, list) or not result or not result[0].get('id'):
                raise RuntimeError('Database tidak mengembalikan ID kategori tersimpan; periksa ringkasan sebelum mencoba lagi.')
            real = result[0]
            by_key[key] = real
            created += 1
        elif not real.get('kata_kunci') and cat.get('kata_kunci') and not cat.get('sistem') and not cat.get('bukan_operasional'):
            finance.set_category_keywords(token, real['id'], cat['kata_kunci'])
        mapping[cat['id']] = real['id']
    return mapping, created
