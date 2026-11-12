"""Live regression on NEW disposable test shops only. Never touches owner shops.

Creates test-only confirmed auth users without sending email. Secrets stay in
memory; evidence contains shop IDs/table snapshots, never passwords or tokens.
Run explicitly, not from pytest. Leaves test shops available for inspection.
"""
import ast
import copy
import json
import secrets
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from database.client import privileged_client, public_client
from services import finance_service as finance
from services.import_category_policy import missing_category_types
from services.import_session import (stage_categories, materialize_categories, discard_draft,
                                     approval_signature, invalidate_approval)
from services.import_service import enrich_rows
from services.smart_import_engine import execute_scalable_import

OUT = ROOT / 'evidence/import_session_fix'
result = {}


def checkpoint():
    (OUT / 'live_database.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def new_shop(label):
    admin = privileged_client()
    suffix = uuid.uuid4().hex
    email, password = f'import-regression-{suffix}@example.invalid', secrets.token_urlsafe(32)
    name = f'Codex Import Test {label} 2026-11-12 {suffix[:8]}'
    user = admin.auth.admin.create_user(dict(email=email, password=password, email_confirm=True,
               user_metadata=dict(nama='Import Regression Test', nama_toko=name, bidang_usaha='Bakery'))).user
    profile = admin.table('profiles').select('*').eq('id', user.id).single().execute().data
    shop = profile['toko_id']
    row = admin.table('toko').select('nama').eq('id', shop).single().execute().data
    assert row['nama'] == name and profile['role'] == 'pemilik_toko'
    # Provisioning only the freshly-created test account; auth code is unchanged.
    admin.table('profiles').update({'status': 'aktif'}).eq('id', user.id).eq('toko_id', shop).execute()
    token = public_client().auth.sign_in_with_password(dict(email=email, password=password)).session.access_token
    result[label] = dict(shop_id=shop, shop_name=name, user_id=user.id)
    checkpoint()
    return token, shop, admin


def snapshot(admin, shop):
    return {table: admin.table(table).select('*').eq('toko_id', shop).order('id').execute().data or []
            for table in ('kategori', 'transaksi')}


def original_approval(token):
    tree = ast.parse((OUT / 'import_view_before.py').read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'btn_approve and (not missing_types)')
    wrapped = ast.parse('def run():\n    pass')
    wrapped.body[0].body = [node]
    state = {}
    scope = dict(st=MagicMock(session_state=state), finance=finance, token=token,
        btn_approve=True, missing_types=[], all_reviewed_exp=[dict(nama='Operasional', tipe='pengeluaran')],
        all_reviewed_inc=[], fill_existing_kw=False, rows_to_import=[], missing_category_types=missing_category_types)
    exec(compile(ast.fix_missing_locations(wrapped), '<original-approval>', 'exec'), scope)
    scope['run']()
    state.clear()  # Original cancellation discarded state only.


def run():
    token, shop, admin = new_shop('before')
    result['before']['initial'] = snapshot(admin, shop)
    original_approval(token)
    result['before']['after_cancel'] = snapshot(admin, shop)
    assert len(result['before']['after_cancel']['kategori']) == 4
    checkpoint()
    print('Original approval then cancel: 0 -> 4 categories; 0 -> 0 transactions', flush=True)

    token, shop, admin = new_shop('after')
    rows = [dict(tanggal='2026-11-12', deskripsi=desc, tipe='pengeluaran', nominal=215000+i,
                 source_id=f'normal-{i}') for i, desc in enumerate(
                     ['isi ulang gas lpg 12 kg', 'beli tepung terigu 25 kg'] * 50)]
    reviewed = [dict(id='ops', nama='Operasional', tipe='pengeluaran', kata_kunci=[]),
                dict(id='raw', nama='Bahan Baku', tipe='pengeluaran', kata_kunci=[]),
                dict(id='removed', nama='Pengeluaran Lainnya', tipe='pengeluaran', kata_kunci=[])]
    initial = snapshot(admin, shop)
    result['after']['initial'] = initial
    signature = approval_signature([], reviewed, rows, True)
    staged = stage_categories(token, [], reviewed, rows, True)
    # Simulate a previous owner's mapping of fuel to the category being removed.
    deleted_id = next(c['id'] for c in staged if c['nama'] == 'Pengeluaran Lainnya')
    state = dict(csv_approval_signature=signature, csv_category_approved=True,
                 csv_approved_categories=staged, csv_proposals=[{**r, 'kategori_id': deleted_id} for r in rows])
    reviewed = reviewed[:2]
    invalidate_approval(state, approval_signature([], reviewed, rows, True))
    assert 'csv_proposals' not in state
    staged = stage_categories(token, [], reviewed, rows, True)
    classified = enrich_rows(token, rows, categories=staged, preview_only=True)
    names = {c['id']: c['nama'] for c in staged}
    assert all(names.get(r['kategori_id']) == ('Operasional' if 'gas' in r['deskripsi'] else 'Bahan Baku') for r in classified)
    result['after']['preview'] = [dict(description=r['deskripsi'], category=names.get(r['kategori_id']),
                                    source=r['source'], score=r['skor']) for r in classified[:2]]
    assert discard_draft(state)
    result['after']['after_cancel'] = snapshot(admin, shop)
    assert result['after']['after_cancel'] == initial
    checkpoint()
    print('Staged approval, deletion, reclassification, cancel: tables identical (0 / 0)', flush=True)

    mapping, created = materialize_categories(token, staged)
    payload = [{**r, 'kategori_id': mapping.get(r.get('kategori_id'))} for r in classified]
    saved = execute_scalable_import(token, payload)
    assert saved['total_imported'] == 100 and not saved['failed_batches']
    result['after']['normal_save'] = dict(result=saved, categories_created=created,
        category_count=len(snapshot(admin, shop)['kategori']), transaction_count=len(snapshot(admin, shop)['transaksi']))
    ops = next(mapping[c['id']] for c in staged if c['nama'] == 'Operasional')
    try:
        finance.delete_category(token, ops)
        raise AssertionError('Used category deletion unexpectedly succeeded')
    except ValueError as exc:
        result['after']['used_category_deletion'] = str(exc)
    checkpoint()
    print('Normal save: 100 transactions; used category deletion blocked', flush=True)

    new_payload = [{**r, 'nominal': r['nominal']+1000, 'source_id': f'partial-{i}', 'usulan': {}}
                   for i, r in enumerate(payload)]
    class FailingSecondBatch:
        def __init__(self): self.calls = 0
        def table(self, *args, **kwargs): return admin.table(*args, **kwargs)
        def rpc(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('Injected failure on second test-shop batch')
            return admin.rpc(*args, **kwargs)
    with patch('services.smart_import_engine.privileged_client', return_value=FailingSecondBatch()):
        partial = execute_scalable_import(token, new_payload, batch_size=50)
    assert partial['total_imported'] == 50 and partial['failed_count'] == 50
    result['after']['partial_save'] = partial
    retried = execute_scalable_import(token, new_payload, batch_size=50, import_id=partial['import_id'])
    assert retried['total_imported'] == 50 and retried['duplicate_count'] == 50
    assert len(snapshot(admin, shop)['transaksi']) == 200
    result['after']['partial_retry'] = retried
    checkpoint()
    print('Partial: 50 saved / 50 failed; retry: 50 saved / 50 skipped; total 200', flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        # Exceptions can contain server payloads; record type only, no credentials.
        result['failure'] = type(exc).__name__
        checkpoint()
        print('Live verification failed:', type(exc).__name__, flush=True)
        raise
