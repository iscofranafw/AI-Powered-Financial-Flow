import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal

from ai.classifier import classify, detect_type
import services.ai_service as ai_service
import services.import_service as import_service
from ai.ollama_client import LLMUnavailable


SAMPLE_CATS = [
    {'id': 'c_bahan', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran', 'deskripsi': 'Bahan makanan dan mentah'},
    {'id': 'c_gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran', 'deskripsi': 'Gaji karyawan'},
    {'id': 'c_ops', 'nama': 'Operasional', 'tipe': 'pengeluaran', 'deskripsi': 'Listrik air sewa'},
    {'id': 'c_exp_other', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True},
    {'id': 'c_jual', 'nama': 'Penjualan', 'tipe': 'pemasukan', 'deskripsi': 'Omzet penjualan'},
    {'id': 'c_modal', 'nama': 'Tambahan Modal', 'tipe': 'pemasukan', 'bukan_operasional': True},
    {'id': 'c_inc_other', 'nama': 'Others', 'tipe': 'pemasukan', 'sistem': True},
]


def test_classification_empty_or_whitespace_description():
    res = classify('pengeluaran', '   ', SAMPLE_CATS)
    assert res['kategori_id'] == 'c_exp_other'
    assert res['status_kategori'] == 'perlu_tinjauan'
    assert 'kosong' in res['message']


def test_classification_rejects_invented_category_id():
    # LLM hallucinating a non-existent category ID must never be accepted.
    with patch('ai.classifier.ask_json', return_value={'kategori_id': 'invented-id-999', 'nama_baru': '', 'alasan': 'Hallucination'}):
        res = classify('pengeluaran', 'beli sesuatu', SAMPLE_CATS)
        assert res['kategori_id'] == 'c_exp_other'
        assert res['status_kategori'] == 'perlu_tinjauan'


def test_classification_propose_existing_name_resolves_to_id():
    with patch('ai.classifier.ask_json', return_value={'kategori_id': '', 'nama_baru': 'bahan baku', 'alasan': 'Existing'}):
        res = classify('pengeluaran', 'beli tepung terigu', SAMPLE_CATS)
        assert res['kategori_id'] == 'c_bahan'
        assert res['status_kategori'] == 'dikonfirmasi'


def test_classification_exceeding_capacity_diverts_to_others():
    full_cats = [
        {'id': f'c_{i}', 'nama': f'Cat {i}', 'tipe': 'pengeluaran'}
        for i in range(10)
    ] + [{'id': 'c_exp_other', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True}]

    with patch('ai.classifier.ask_json', return_value={'kategori_id': '', 'nama_baru': 'New Unique Category', 'alasan': 'Needs new'}):
        res = classify('pengeluaran', 'beli alat aneh', full_cats)
        assert res['kategori_id'] == 'c_exp_other'
        assert res['status_kategori'] == 'perlu_tinjauan'
        assert 'Batas kategori' in res['message']


def test_classification_no_allowed_categories():
    only_others = [{'id': 'c_exp_other', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True}]
    res = classify('pengeluaran', 'beli beras', only_others)
    assert res['kategori_id'] == 'c_exp_other'
    assert res['status_kategori'] == 'perlu_tinjauan'


def test_bulk_classify_partial_failure_and_row_preservation():
    rows = [
        {'baris': 2, 'tanggal': '2026-10-01', 'deskripsi': 'beli telur 200rb', 'nominal': '200000', 'tipe': 'pengeluaran', 'error': ''},
        {'baris': 3, 'tanggal': '2026-10-02', 'deskripsi': '', 'nominal': '100000', 'tipe': 'pengeluaran', 'error': ''},
        {'baris': 4, 'tanggal': '2026-10-03', 'deskripsi': 'format salah', 'nominal': 'invalid', 'tipe': 'pengeluaran', 'error': 'Nominal tidak valid'},
        {'baris': 5, 'tanggal': '2026-10-04', 'deskripsi': 'omzet toko sore', 'nominal': '500000', 'tipe': '', 'error': ''},
    ]

    progress_calls = []

    def mock_progress(pct):
        progress_calls.append(pct)

    with patch('services.ai_service.verify_actor', return_value={'toko_id': 'shop-bulk', 'role': 'pemilik_toko'}), \
         patch('services.ai_service.finance.list_categories', return_value=SAMPLE_CATS), \
         patch('services.ai_service.get_or_sync_index'), \
         patch('services.ai_service.classify', return_value={'kategori_id': 'c_bahan', 'usulan': {}, 'skor': 0.85, 'status_kategori': 'otomatis', 'message': 'OK'}):

        results = ai_service.bulk_classify('mock-token', rows, progress=mock_progress)

        assert len(results) == 4
        assert len(progress_calls) == 4

        # Valid row 1: enriched with category
        assert results[0]['baris'] == 2
        assert results[0]['kategori_id'] == 'c_bahan'
        assert results[0]['status_kategori'] == 'otomatis'

        # Empty description row 2: caught without terminating remaining batch
        assert results[1]['baris'] == 3
        assert results[1]['error'] == 'Deskripsi transaksi kosong'
        assert results[1]['status_kategori'] == 'perlu_tinjauan'

        # Existing format error row 3: preserved with status
        assert results[2]['baris'] == 4
        assert results[2]['error'] == 'Nominal tidak valid'
        assert results[2]['status_kategori'] == 'perlu_tinjauan'

        # Row 4 without type: auto-detected to pemasukan
        assert results[3]['baris'] == 5
        assert results[3]['tipe'] == 'pemasukan'


def test_bulk_classify_handles_ambiguous_type_gracefully():
    rows = [
        {'baris': 2, 'deskripsi': 'transaksi dana kas', 'nominal': '50000', 'tipe': '', 'error': ''}
    ]

    with patch('services.ai_service.verify_actor', return_value={'toko_id': 'shop-bulk', 'role': 'kasir'}), \
         patch('services.ai_service.finance.list_categories', return_value=SAMPLE_CATS), \
         patch('services.ai_service.get_or_sync_index'), \
         patch('services.ai_service.detect_type', side_effect=ValueError('Ambigu')):

        results = ai_service.bulk_classify('mock-token', rows)
        assert len(results) == 1
        assert results[0]['tipe'] == 'pengeluaran'
        assert results[0]['status_kategori'] == 'perlu_tinjauan'
        assert results[0]['kategori_id'] == 'c_exp_other'


def test_enrich_rows_delegates_to_bulk_classify():
    rows = [{'baris': 2, 'deskripsi': 'beli sabun', 'nominal': '10000', 'tipe': 'pengeluaran', 'error': ''}]
    with patch('services.ai_service.bulk_classify', return_value=[{'baris': 2, 'kategori_id': 'c_ops'}]) as mock_bulk:
        out = import_service.enrich_rows('token-123', rows)
        mock_bulk.assert_called_once_with('token-123', rows, progress=None)
        assert out[0]['kategori_id'] == 'c_ops'


def test_commit_preview_forwards_classification_metadata():
    row = {
        'tanggal': '2026-10-31',
        'deskripsi': 'beli tepung',
        'nominal': '250000',
        'tipe': 'pengeluaran',
        'status_kategori': 'otomatis',
        'skor': 0.9123,
        'usulan': {'nama_baru': ''}
    }

    with patch('services.ai_service.finance.add_transaction') as mock_add:
        ai_service.commit_preview('tok-owner', row, 'c_bahan')
        mock_add.assert_called_once_with(
            'tok-owner',
            '2026-10-31',
            'beli tepung',
            '250000',
            'pengeluaran',
            'c_bahan',
            status_kategori='otomatis',
            skor=0.9123,
            usulan={'nama_baru': ''}
        )


def test_classification_malformed_llm_responses():
    with patch('ai.classifier.ask_json', return_value={'kategori_id': None, 'nama_baru': None, 'alasan': None}):
        res = classify('pengeluaran', 'beli perlengkapan kantor', SAMPLE_CATS)
        assert res['kategori_id'] == 'c_exp_other'
        assert res['status_kategori'] == 'perlu_tinjauan'

    with patch('ai.classifier.ask_json', side_effect=ValueError('Balasan bukan JSON valid')):
        res = classify('pengeluaran', 'beli perlengkapan kantor', SAMPLE_CATS)
        assert res['kategori_id'] == 'c_exp_other'
        assert res['status_kategori'] == 'perlu_tinjauan'


def test_classification_missing_rag_index_fallback():
    # If FAISS index is missing or throws an error, zero-shot LLM classification continues gracefully.
    with patch('ai.classifier.retrieve_context', side_effect=Exception('Index not found')), \
         patch('ai.classifier.ask_json', return_value={'kategori_id': 'c_ops', 'nama_baru': '', 'alasan': 'Operasional'}):
        res = classify('pengeluaran', 'bayar tagihan listrik ruko', SAMPLE_CATS, toko_id='shop-missing-index')
        assert res['kategori_id'] == 'c_ops'
        assert res['status_kategori'] == 'dikonfirmasi'


def test_classification_ollama_timeout_fallback():
    # RAG provides classification when Ollama encounters a network timeout.
    mock_hits = [{'category_id': 'c_gaji', 'category_name': 'Gaji', 'score': 0.89, 'type': 'category', 'text': 'Gaji'}]
    with patch('ai.classifier.retrieve_context', return_value=mock_hits), \
         patch('ai.classifier.ask_json', side_effect=LLMUnavailable('Connection timed out')):
        res = classify('pengeluaran', 'upah bulanan kasir', SAMPLE_CATS, toko_id='shop-timeout')
        assert res['kategori_id'] == 'c_gaji'
        assert res['status_kategori'] == 'otomatis'
        assert res['skor'] == 0.89


def test_bulk_classify_empty_list():
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 'shop-1', 'role': 'pemilik_toko'}), \
         patch('services.ai_service.finance.list_categories', return_value=SAMPLE_CATS):
        res = ai_service.bulk_classify('mock-tok', [])
        assert res == []


def test_classification_cross_store_isolation():
    shop_a_cats = [{'id': 'cat_a', 'nama': 'Bahan Toko A', 'tipe': 'pengeluaran'}]
    shop_b_cats = [{'id': 'cat_b', 'nama': 'Bahan Toko B', 'tipe': 'pengeluaran'}]

    # Even if LLM returns category ID of Shop B, Shop A's invocation rejects it as foreign.
    with patch('ai.classifier.ask_json', return_value={'kategori_id': 'cat_b', 'nama_baru': '', 'alasan': 'Wrong store'}):
        res = classify('pengeluaran', 'beli bahan', shop_a_cats)
        assert res['kategori_id'] is None
        assert res['status_kategori'] == 'perlu_tinjauan'
