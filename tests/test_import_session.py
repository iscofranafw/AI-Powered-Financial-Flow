"""Import lifecycle regressions on a fresh isolated shop; no production writes."""
import copy
from unittest.mock import patch

import pytest

from services.import_session import (approval_signature, discard_draft, invalidate_approval,
                                     materialize_categories, stage_categories)
from services.import_service import enrich_rows
from ai.category_evidence import compile_keyword_plan
from ai.classifier import _decision, _enforce_purpose, _similarity_evaluate
from ai.ollama_client import LLMUnavailable


def cat(cid, name, **extra):
    return dict(id=cid, nama=name, tipe='pengeluaran', sistem=False, kata_kunci=[], **extra)


@pytest.fixture
def shop(monkeypatch):
    from services import finance_service as finance
    db = {'kategori': [], 'transaksi': []}
    monkeypatch.setattr(finance, 'list_categories', lambda token: copy.deepcopy(db['kategori']))
    def create(token, name, tipe, description='', keywords=None, system_seed=False):
        row = dict(id=str(len(db['kategori']) + 1), nama=name, tipe=tipe,
                   deskripsi=description, kata_kunci=keywords or [], sistem=name == 'Others')
        db['kategori'].append(row)
        return [copy.deepcopy(row)]
    monkeypatch.setattr(finance, 'create_category', create)
    return db


def test_cancel_after_approval_preserves_exact_shop_tables(shop):
    before = copy.deepcopy(shop)
    staged = stage_categories('test', [], [cat('temp', 'Operasional')], [], False)
    state = dict(csv_category_approved=True, csv_approved_categories=staged,
                 csv_proposals=[{'kategori_id': staged[-1]['id']}], review_expense_cats=[])
    assert len(staged) == 4
    assert discard_draft(state)
    assert shop == before
    assert not any(k.startswith(('csv_', 'review_')) for k in state)


def test_cancel_keeps_existing_categories_keywords_and_transactions(shop):
    shop['kategori'] = [cat('old', 'Operasional')]
    shop['transaksi'] = [dict(id='historical', kategori_id='old', nominal=100)]
    before = copy.deepcopy(shop)
    with patch('services.import_session.confirmed_history', return_value=[]), \
         patch('services.finance_service._member', return_value={'toko_id': 'fresh-test'}), \
         patch('services.import_session.get_store_business_context', return_value={'bidang_usaha': 'Custom'}), \
         patch('services.finance_service.plan_category_keywords', return_value={'categories': [
             {**cat('old', 'Operasional'), 'kata_kunci': ['contoh milik pengguna']}]}):
        # Use actual compiler for seed entries as well, to verify no write occurs.
        with patch('services.finance_service.plan_category_keywords', side_effect=lambda cats, *args: {
            'categories': [{**c, 'kata_kunci': ['contoh milik pengguna']} for c in cats]}):
            staged = stage_categories('test', shop['kategori'], [], [], True)
    assert staged[0]['kata_kunci'] == ['contoh milik pengguna']
    assert discard_draft({'csv_approved_categories': staged})
    assert shop == before


@pytest.mark.parametrize('removed', ['pengeluaran lainnya', 'Kemasan'])
def test_category_deletion_invalidates_approval_and_preview(removed):
    existing = [cat('raw', 'Bahan Baku'), cat('ops', 'Operasional')]
    reviewed = [cat('deleted', removed)]
    rows = [{'deskripsi': 'isi ulang gas lpg 12 kg', 'tipe': 'pengeluaran'}]
    signature = approval_signature(existing, reviewed, rows, True)
    state = dict(csv_approval_signature=signature, csv_category_approved=True,
                 csv_approved_categories=existing + reviewed, csv_proposals=[{'kategori_id': 'deleted'}])
    invalidate_approval(state, signature)
    assert 'csv_proposals' in state
    invalidate_approval(state, approval_signature(existing, [], rows, True))
    assert 'csv_proposals' not in state and 'csv_category_approved' not in state


def test_external_category_deletion_and_row_changes_invalidate_preview():
    old = [cat('a', 'Kemasan'), cat('b', 'Operasional')]
    signature = approval_signature(old, [], [{'nominal': 10}], False)
    for existing, rows in [(old[1:], [{'nominal': 10}]), (old, [{'nominal': 20}])]:
        state = dict(csv_approval_signature=signature, csv_proposals=[{}])
        invalidate_approval(state, approval_signature(existing, [], rows, False))
        assert 'csv_proposals' not in state


@pytest.mark.parametrize('description', ['beli gas LPG 12 kg', 'isi ulang gas lpg 12 kg'])
def test_wrong_bakery_fuel_evidence_is_reclassified_before_preview(description):
    cats = [cat('raw', 'Bahan Baku'), cat('ops', 'Operasional')]
    rows = [dict(deskripsi=description, kategori_id='raw', tipe='pengeluaran')]
    original = copy.deepcopy(rows)
    plan = compile_keyword_plan(cats, rows, 'Bakery')
    dec, diag = _similarity_evaluate('pengeluaran', description, plan['categories'])
    assert dec['kategori_id'] == 'ops' and diag['gate_passed']
    assert rows == original  # No rewrite of a saved transaction during preview.


def test_removed_category_without_compatible_replacement_abstains():
    cats = [cat('raw', 'Bahan Baku')]
    decision = _decision({'kategori_id': 'raw', 'nama_baru': '', 'alasan': '', 'keyakinan': 99}, {'raw'})
    result = _enforce_purpose(decision, 'pengeluaran', 'isi ulang gas lpg 12 kg', cats, 'Bakery')
    assert result['is_unresolved'] and result['kategori_id'] is None


def test_two_fuel_categories_never_force_a_tie():
    cats = [cat('a', 'Gas LPG'), cat('b', 'Bahan Bakar Oven')]
    plan = compile_keyword_plan(cats, [dict(deskripsi='isi ulang gas lpg', kategori_id='a', tipe='pengeluaran')], 'Bakery')
    assert not any('isi ulang gas lpg' in c['kata_kunci'] for c in plan['categories'])


def test_retail_fuel_for_resale_keeps_owner_stock_assignment():
    cats = [cat('stock', 'Stok Barang Dagangan'), cat('ops', 'Operasional')]
    plan = compile_keyword_plan(cats, [dict(deskripsi='beli gas LPG 12 kg', kategori_id='stock', tipe='pengeluaran')], 'Retail')
    assert 'beli gas LPG 12 kg' in plan['categories'][0]['kata_kunci']


def test_offline_import_preview_never_saves_pending_transactions(shop):
    from services.ai_service import bulk_classify
    rows = [dict(deskripsi='biaya belum dikenal', tipe='pengeluaran', nominal=100)]
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 'new-local-shop'}), \
         patch('services.ai_service.get_store_business_context', return_value={}), \
         patch('services.ai_service.hybrid_bulk_classify', side_effect=LLMUnavailable('offline')), \
         patch('services.import_service.execute_structured_import') as save:
        result = bulk_classify('test', rows, categories=[], persist_pending=False)
    save.assert_not_called()
    assert result[0]['is_unresolved'] and not result[0].get('_saved_ai_pending')
    assert shop == {'kategori': [], 'transaksi': []}


def test_import_enrichment_uses_draft_index_and_read_only_classifier():
    rows = [dict(deskripsi='baru', tipe='pengeluaran')]
    with patch('services.import_service.verify_actor', return_value={'toko_id': 'test-shop'}), \
         patch('services.import_session.confirmed_history', return_value=[]), \
         patch('ai.store_context.get_store_business_context', return_value={}), \
         patch('ai.rag_engine.import_preview_indexes') as indexes, \
         patch('services.ai_service.bulk_classify', return_value=rows) as classify:
        assert enrich_rows('test', rows, categories=[], preview_only=True) == rows
    indexes.assert_called_once_with('test-shop', [], [], 'Umum')
    classify.assert_called_once_with('test', rows, categories=[], progress=None, persist_pending=False)


def test_final_save_materializes_ids_and_retry_reuses_categories(shop):
    staged = stage_categories('test', [], [cat('draft', 'Operasional')], [], False)
    mapping, count = materialize_categories('test', staged)
    assert count == 4 and len(shop['kategori']) == 4
    row = {'kategori_id': mapping[staged[-1]['id']], 'nominal': 100}
    shop['transaksi'].append(row)
    again, created = materialize_categories('test', staged)
    assert again == mapping and created == 0 and len(shop['transaksi']) == 1


def test_cancel_is_rejected_after_final_commit_begins():
    state = dict(csv_commit_started=True, csv_import_result={'total_imported': 500, 'failed_count': 100})
    before = copy.deepcopy(state)
    assert not discard_draft(state)
    assert state == before


def test_deleted_existing_category_is_not_silently_recreated(shop):
    with pytest.raises(ValueError, match='berubah'):
        materialize_categories('test', [cat('deleted', 'Kemasan')])
    assert shop['kategori'] == []


@pytest.mark.parametrize('confidence,unresolved', [(45, True), (69, True), (70, False), (95, False)])
def test_llama_confidence_is_distinct_from_cosine(confidence, unresolved):
    result = _decision(dict(kategori_id='ops', nama_baru='', alasan='', keyakinan=confidence), {'ops'})
    result['skor'] = .45  # Batch/single fallback retain the top cosine for diagnostics.
    assert result['is_unresolved'] is unresolved
    assert result['usulan']['llm_confidence'] == confidence
    assert result['skor'] == .45


def test_empty_upload_does_not_change_uploader_identity():
    state = {'import_upload_generation': 3}
    discard_draft(state, reset_upload=False)
    assert state['import_upload_generation'] == 3


def test_discovery_offline_uses_read_only_pending_path():
    from services.ai_service import discover_categories_for_import
    from unittest.mock import MagicMock
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {'bidang_usaha': 'Bakery'}
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 'new-test-shop'}), \
         patch('services.ai_service.privileged_client', return_value=client), \
         patch('services.finance_service.list_categories', return_value=[]), \
         patch('ai.category_discovery.discover_categories_from_transactions', side_effect=LLMUnavailable('offline')), \
         patch('services.ai_service.bulk_classify', return_value=[{'usulan': {'ai_pending': True}}]) as classify:
        result = discover_categories_for_import('test', [{'deskripsi': 'baru'}])
    assert result['ai_pending_count'] == 1
    classify.assert_called_once_with('test', [{'deskripsi': 'baru'}], categories=[], persist_pending=False)


def test_preview_indexes_are_request_scoped_and_released_after_error():
    from ai.rag_engine import import_preview_indexes, get_or_sync_index, _PREVIEW_INDEXES
    cats = [cat('ops', 'Operasional')]
    with pytest.raises(RuntimeError, match='test interruption'):
        with import_preview_indexes('preview-test-shop', cats, [], 'Bakery'):
            _, metadata = get_or_sync_index('preview-test-shop', 'pengeluaran')
            assert metadata['entries']
            assert all(e['store_id'] == 'preview-test-shop' for e in metadata['entries'])
            assert ('different-shop', 'pengeluaran') not in _PREVIEW_INDEXES.get()
            raise RuntimeError('test interruption')
    assert _PREVIEW_INDEXES.get() is None


def test_successful_category_deletion_invalidates_classifier_and_index():
    from services import finance_service as finance
    from unittest.mock import MagicMock
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    with patch.object(finance, '_owner', return_value={'toko_id': 'test-shop'}), \
         patch.object(finance, 'category_by_id', return_value={**cat('ops', 'Operasional'), 'bukan_operasional': False}), \
         patch.object(finance, 'privileged_client', return_value=client), \
         patch.object(finance, '_invalidate_classification_state') as invalidate:
        finance.delete_category('test', 'ops')
    invalidate.assert_called_once_with('test-shop', 'pengeluaran', rebuild_index=True)


def test_import_pending_panel_cannot_silently_update_existing_transactions():
    from streamlit.testing.v1 import AppTest
    with patch('services.ai_service.pending_ai_transactions', return_value=[{'toko_id': 'test-shop'}]), \
         patch('ai.ollama_client.diagnosis') as diagnosis, \
         patch('services.ai_service.retry_pending_ai_classifications') as retry:
        app = AppTest.from_string("from pages.finance import render_ai_pending_queue\nrender_ai_pending_queue('test', auto_retry=False)").run()
    assert not app.exception
    diagnosis.assert_not_called()
    retry.assert_not_called()


@pytest.mark.parametrize('deleted_name,description,expected', [
    ('Pengeluaran Lainnya', 'isi ulang gas lpg 12 kg', 'ops'),
    ('Kemasan', 'beli kemasan dus kue', None),
])
def test_streamlit_approve_preview_cancel_100_rows(shop, monkeypatch, deleted_name, description, expected):
    """Exercise actual wizard buttons and reruns with a test-only upload/store."""
    from streamlit.testing.v1 import AppTest
    from types import SimpleNamespace
    from pages import import_view
    from pages import finance as finance_page
    from services import ai_service, import_service
    shop['kategori'] = [cat('raw', 'Bahan Baku'), cat('ops', 'Operasional')]
    shop['kategori'][0]['kata_kunci'] = ['beli tepung terigu 25 kg']
    shop['kategori'][1]['kata_kunci'] = ['isi ulang gas lpg 12 kg']
    before = copy.deepcopy(shop)
    content = ('tanggal,deskripsi,nominal,tipe\n' + '\n'.join(
        f'2026-11-12,{description if i % 2 else "beli tepung terigu 25 kg"},{215000+i},pengeluaran'
        for i in range(100))).encode()
    def upload(*args, **kwargs):
        return SimpleNamespace(getvalue=lambda: content) if kwargs['key'] == 'upload_csv_0' else None
    monkeypatch.setattr(import_view.st, 'file_uploader', upload)
    from unittest.mock import MagicMock
    pending_panel = MagicMock()
    monkeypatch.setattr(finance_page, 'render_ai_pending_queue', pending_panel)
    monkeypatch.setattr(import_view.finance, '_member', lambda token: {'toko_id': 'ui-test-shop'})
    monkeypatch.setattr(import_view, 'detect_duplicates', lambda shop_id, valid_rows: (valid_rows, []))
    monkeypatch.setattr(import_view, 'get_store_business_context', lambda shop: {'bidang_usaha': 'Bakery'})
    monkeypatch.setattr(import_view, 'get_classification_index_status', lambda token: {
        'pengeluaran': {'ready': False, 'doc_count': 0}, 'pemasukan': {'ready': False, 'doc_count': 0},
        'semantic_model_available': True})
    monkeypatch.setattr(import_view, 'discover_categories_for_import', lambda *args: {
        'pengeluaran': [dict(nama=deleted_name, tipe='pengeluaran', kata_kunci=[description])],
        'pemasukan': [], 'coverage_stats': {}})
    monkeypatch.setattr(import_service, 'verify_actor', lambda *args: {'toko_id': 'ui-test-shop'})
    monkeypatch.setattr(ai_service, 'verify_actor', lambda *args: {'toko_id': 'ui-test-shop'})
    monkeypatch.setattr(ai_service, 'get_store_business_context', lambda shop: {'bidang_usaha': 'Bakery'})
    monkeypatch.setattr('ai.store_context.get_store_business_context', lambda shop: {'bidang_usaha': 'Bakery'})
    monkeypatch.setattr('services.import_session.confirmed_history', lambda token: [])
    import json
    def abstain(*args, **kwargs):
        payload = json.loads(kwargs['user'])
        return {'klasifikasi': [dict(item_id=item['item_id'], kategori_id='', nama_baru='',
                alasan='Tidak ada kategori yang cocok', keyakinan=45) for item in payload['daftar_transaksi']]}
    monkeypatch.setattr('ai.hybrid_classifier.ask_json', abstain)
    app = AppTest.from_string("from pages.import_view import render\nrender('test')", default_timeout=60).run()
    assert not app.exception
    app.checkbox(key='chk_auto_fill_existing_kw').uncheck().run()
    delete = next(button for button in app.button if str(button.key).startswith('btn_del_exp_'))
    delete.click().run()
    assert not app.exception
    app.button(key='btn_approve_and_continue').click().run()
    assert not app.exception
    proposals = app.session_state['csv_proposals']
    assert len(proposals) == 100
    assert all(row['kategori_id'] == (expected if row['deskripsi'] == description else 'raw') for row in proposals), [(r['deskripsi'], r['kategori_id'], r.get('source'), r.get('message')) for r in proposals[:2]]
    assert all(row['is_unresolved'] is (expected is None) for row in proposals if row['deskripsi'] == description)
    assert shop == before
    pending_panel.assert_called_with('test', auto_retry=False)
    app.button(key='btn_cancel_import').click().run()
    assert not app.exception
    assert shop == before
