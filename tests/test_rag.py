import os
import json
import shutil
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ai.embedding_pipeline import embed_text, embed_batch, get_embedding_dimension, _fallback_embedding
import ai.rag_engine as rag_engine
from ai.classifier import classify
from ai.ollama_client import LLMUnavailable


@pytest.fixture(autouse=True)
def clean_rag_state():
    rag_engine.clear_memory_cache()
    yield
    rag_engine.clear_memory_cache()


def test_embedding_dimensions_and_normalization():
    vec = embed_text('belanja terigu 25kg')
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-4


def test_embedding_batch():
    texts = ['omzet harian', 'beli minyak goreng', 'gaji barista']
    batch = embed_batch(texts)
    assert batch.shape == (3, 384)
    for row in batch:
        assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-4


def test_fallback_dense_embedding():
    v1 = _fallback_embedding('beli telur ayam')
    v2 = _fallback_embedding('beli telur bebek')
    v3 = _fallback_embedding('gaji bulanan kasir')
    assert v1.shape == (384,)
    sim_similar = float(np.dot(v1, v2))
    sim_diff = float(np.dot(v1, v3))
    assert sim_similar > sim_diff


def test_build_and_save_index_persistence(tmp_path):
    shop_id = 'test-shop-persist-1'
    tipe = 'pengeluaran'
    entries = [
        {'id': 'cat_1', 'type': 'category', 'category_id': 'c1', 'category_name': 'Bahan Baku', 'text': 'Bahan Baku tepung telur gula'},
        {'id': 'cat_2', 'type': 'category', 'category_id': 'c2', 'category_name': 'Gaji', 'text': 'Gaji upah karyawan bonus'}
    ]

    with patch.object(rag_engine, 'INDEX_ROOT_DIR', str(tmp_path)):
        index, meta = rag_engine.build_and_save_index(shop_id, tipe, entries=entries, version=1)
        assert index.ntotal == 2
        assert meta['count'] == 2

        faiss_file = tmp_path / shop_id / f"{tipe}.faiss"
        meta_file = tmp_path / shop_id / f"{tipe}_meta.json"
        assert faiss_file.exists()
        assert meta_file.exists()

        with open(meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data['toko_id'] == shop_id
        assert data['version'] == 1
        assert len(data['entries']) == 2


def test_sync_index_detects_version_change(tmp_path):
    shop_id = 'test-shop-sync-1'
    tipe = 'pengeluaran'
    entries_v1 = [
        {'id': 'cat_1', 'type': 'category', 'category_id': 'c1', 'category_name': 'Operasional', 'text': 'Operasional listrik air'}
    ]
    entries_v2 = [
        {'id': 'cat_1', 'type': 'category', 'category_id': 'c1', 'category_name': 'Operasional', 'text': 'Operasional listrik air'},
        {'id': 'cat_2', 'type': 'category', 'category_id': 'c2', 'category_name': 'Bahan Baku', 'text': 'Bahan Baku'}
    ]

    with patch.object(rag_engine, 'INDEX_ROOT_DIR', str(tmp_path)), \
         patch.object(rag_engine, 'get_remote_version', return_value=1), \
         patch.object(rag_engine, 'build_corpus_entries', return_value=entries_v1):
        idx1, meta1 = rag_engine.get_or_sync_index(shop_id, tipe)
        assert idx1.ntotal == 1
        assert meta1['version'] == 1

    # Bump version in remote database and verify automatic reload.
    with patch.object(rag_engine, 'INDEX_ROOT_DIR', str(tmp_path)), \
         patch.object(rag_engine, 'get_remote_version', return_value=2), \
         patch.object(rag_engine, 'build_corpus_entries', return_value=entries_v2):
        idx2, meta2 = rag_engine.get_or_sync_index(shop_id, tipe)
        assert idx2.ntotal == 2
        assert meta2['version'] == 2


def test_strict_cross_store_isolation(tmp_path):
    shop_a = '11111111-1111-1111-1111-111111111111'
    shop_b = '22222222-2222-2222-2222-222222222222'

    entries_a = [
        {'id': 'cat_a', 'type': 'category', 'category_id': 'ca', 'category_name': 'Bahan Kue Toko A', 'text': 'Toko A tepung premix'}
    ]
    entries_b = [
        {'id': 'cat_b', 'type': 'category', 'category_id': 'cb', 'category_name': 'Sparepart Toko B', 'text': 'Toko B oli rem kampas'}
    ]

    with patch.object(rag_engine, 'INDEX_ROOT_DIR', str(tmp_path)):
        rag_engine.build_and_save_index(shop_a, 'pengeluaran', entries=entries_a, version=1)
        rag_engine.build_and_save_index(shop_b, 'pengeluaran', entries=entries_b, version=1)

        with patch.object(rag_engine, 'get_remote_version', return_value=1):
            hits_a = rag_engine.retrieve_context(shop_a, 'pengeluaran', 'oli rem dan kampas', top_k=5, min_score=0.1)
            hits_b = rag_engine.retrieve_context(shop_b, 'pengeluaran', 'oli rem dan kampas', top_k=5, min_score=0.1)

            # Store A must never contain Store B's spare parts.
            assert all(h['category_id'] != 'cb' for h in hits_a)
            # Store B must retrieve its own spare parts category.
            assert any(h['category_id'] == 'cb' for h in hits_b)


def test_type_scoping_isolation(tmp_path):
    shop_id = 'test-shop-type-iso'
    income_entries = [
        {'id': 'cat_inc', 'type': 'category', 'category_id': 'c_inc', 'category_name': 'Penjualan', 'text': 'omzet penjualan kasir'}
    ]
    expense_entries = [
        {'id': 'cat_exp', 'type': 'category', 'category_id': 'c_exp', 'category_name': 'Operasional', 'text': 'biaya listrik dan internet'}
    ]

    with patch.object(rag_engine, 'INDEX_ROOT_DIR', str(tmp_path)):
        rag_engine.build_and_save_index(shop_id, 'pemasukan', entries=income_entries, version=1)
        rag_engine.build_and_save_index(shop_id, 'pengeluaran', entries=expense_entries, version=1)

        with patch.object(rag_engine, 'get_remote_version', return_value=1):
            inc_hits = rag_engine.retrieve_context(shop_id, 'pemasukan', 'biaya listrik', top_k=5, min_score=0.1)
            exp_hits = rag_engine.retrieve_context(shop_id, 'pengeluaran', 'biaya listrik', top_k=5, min_score=0.1)

            assert all(h['category_id'] != 'c_exp' for h in inc_hits)
            assert any(h['category_id'] == 'c_exp' for h in exp_hits)


def test_retrieve_context_for_actor_enforces_auth():
    with patch('ai.rag_engine.verify_actor', return_value={'toko_id': 'shop-authenticated', 'role': 'kasir'}), \
         patch('ai.rag_engine.retrieve_context') as mock_retrieve:
        mock_retrieve.return_value = [{'category_id': 'cat1', 'score': 0.85}]
        res = rag_engine.retrieve_context_for_actor('mock-token', 'beli telur', 'pengeluaran')
        mock_retrieve.assert_called_once_with('shop-authenticated', 'pengeluaran', 'beli telur', top_k=5, min_score=0.25)
        assert len(res) == 1


def test_empty_query_and_empty_corpus_handling(tmp_path):
    shop_id = 'empty-shop'
    assert rag_engine.retrieve_context(shop_id, 'pengeluaran', '') == []
    assert rag_engine.retrieve_context(shop_id, 'pengeluaran', '   ') == []
    assert rag_engine.retrieve_context(None, 'pengeluaran', 'beli bahan') == []
    assert rag_engine.retrieve_context('', 'pengeluaran', 'beli bahan') == []
    assert rag_engine.retrieve_context(shop_id, 'pengeluaran', 'beli bahan', top_k=0) == []
    assert rag_engine.retrieve_context(shop_id, 'pengeluaran', 'beli bahan', top_k=-1) == []

    with patch.object(rag_engine, 'INDEX_ROOT_DIR', str(tmp_path)):
        rag_engine.build_and_save_index(shop_id, 'pengeluaran', entries=[], version=1)
        with patch.object(rag_engine, 'get_remote_version', return_value=1):
            hits = rag_engine.retrieve_context(shop_id, 'pengeluaran', 'beli gula', top_k=5)
            assert hits == []


def test_rag_benchmark_accuracy_on_representative_data(tmp_path):
    shop_id = 'benchmark-shop'
    corpus = [
        {'id': 'cat_1', 'type': 'category', 'category_id': 'c_bahan', 'category_name': 'Bahan Baku', 'text': 'Bahan Baku. tepung terigu telur gula mentega fermipan'},
        {'id': 'cat_2', 'type': 'category', 'category_id': 'c_ops', 'category_name': 'Operasional', 'text': 'Operasional. listrik pln air pdam sewa ruko gas lpg bensin'},
        {'id': 'cat_3', 'type': 'category', 'category_id': 'c_gaji', 'category_name': 'Gaji', 'text': 'Gaji. upah staf gaji karyawan uang makan baker kasir'},
        {'id': 'cat_4', 'type': 'category', 'category_id': 'c_kemasan', 'category_name': 'Kemasan', 'text': 'Kemasan. kotak kardus roti plastik mika kantong kresek'},
        {'id': 'tx_1', 'type': 'transaction', 'category_id': 'c_bahan', 'category_name': 'Bahan Baku', 'text': 'beli coklat bubuk dan keju parut'},
        {'id': 'tx_2', 'type': 'transaction', 'category_id': 'c_ops', 'category_name': 'Operasional', 'text': 'bayar wifi indihome toko'},
        {'id': 'tx_3', 'type': 'transaction', 'category_id': 'c_gaji', 'category_name': 'Gaji', 'text': 'gaji mingguan helper toko'},
        {'id': 'tx_4', 'type': 'transaction', 'category_id': 'c_kemasan', 'category_name': 'Kemasan', 'text': 'beli paper bag dan stiker label'}
    ]

    holdout_eval = [
        ('belanja tepung protein dan gula pasir 50kg', 'c_bahan'),
        ('beli telur ayam 2 papan', 'c_bahan'),
        ('bayar tagihan listrik pln bulanan', 'c_ops'),
        ('isi ulang tabung gas melon 3kg', 'c_ops'),
        ('gaji staf kasir bulan ini', 'c_gaji'),
        ('uang saku dan lembur baker', 'c_gaji'),
        ('beli kotak mika dan plastik roll', 'c_kemasan'),
        ('pesan kantong plastik kresek roti', 'c_kemasan'),
    ]

    with patch.object(rag_engine, 'INDEX_ROOT_DIR', str(tmp_path)):
        rag_engine.build_and_save_index(shop_id, 'pengeluaran', entries=corpus, version=1)
        with patch.object(rag_engine, 'get_remote_version', return_value=1):
            correct = 0
            for query, expected_cat in holdout_eval:
                hits = rag_engine.retrieve_context(shop_id, 'pengeluaran', query, top_k=3, min_score=0.2)
                if hits and hits[0]['category_id'] == expected_cat:
                    correct += 1

            accuracy = correct / len(holdout_eval)
            assert accuracy >= 0.85, f"Measured retrieval accuracy {accuracy:.2%} is below target 85%"


def test_classify_with_rag_context_and_confidence():
    cats = [
        {'id': 'c_bahan', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran'},
        {'id': 'c_gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran'},
        {'id': 'c_other', 'nama': 'Others', 'tipe': 'pengeluaran'}
    ]
    rag_mock_hits = [
        {'score': 0.88, 'type': 'transaction', 'category_id': 'c_bahan', 'category_name': 'Bahan Baku', 'text': 'beli tepung terigu 50kg', 'entry_id': 'tx1'}
    ]

    with patch('ai.classifier.retrieve_context', return_value=rag_mock_hits), \
         patch('ai.classifier.ask_json', return_value={'kategori_id': 'c_bahan', 'nama_baru': '', 'alasan': 'Sesuai bukti historis'}):
        res = classify('pengeluaran', 'beli tepung terigu 25kg', cats, toko_id='shop-rag')
        assert res['kategori_id'] == 'c_bahan'
        assert res['skor'] == 0.88
        assert res['status_kategori'] == 'otomatis'
        assert 'Bahan Baku' in res['message']


def test_classify_offline_fallback_uses_high_confidence_rag():
    cats = [
        {'id': 'c_bahan', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran'},
        {'id': 'c_other', 'nama': 'Others', 'tipe': 'pengeluaran'}
    ]
    rag_mock_hits = [
        {'score': 0.84, 'type': 'category', 'category_id': 'c_bahan', 'category_name': 'Bahan Baku', 'text': 'Bahan Baku', 'entry_id': 'cat1'}
    ]

    # When Ollama is offline, high-confidence vector retrieval resolves category without user frustration.
    with patch('ai.classifier.retrieve_context', return_value=rag_mock_hits), \
         patch('ai.classifier.ask_json', side_effect=LLMUnavailable('daemon down')):
        res = classify('pengeluaran', 'beli terigu segak', cats, toko_id='shop-rag')
        assert res['kategori_id'] == 'c_bahan'
        assert res['skor'] == 0.84
        assert res['status_kategori'] == 'otomatis'
        assert 'Pencarian semantik merekomendasikan' in res['message']


def test_classify_offline_low_confidence_falls_back_to_others():
    cats = [
        {'id': 'c_bahan', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran'},
        {'id': 'c_other', 'nama': 'Others', 'tipe': 'pengeluaran'}
    ]
    rag_mock_hits = [
        {'score': 0.42, 'type': 'category', 'category_id': 'c_bahan', 'category_name': 'Bahan Baku', 'text': 'Bahan Baku', 'entry_id': 'cat1'}
    ]

    with patch('ai.classifier.retrieve_context', return_value=rag_mock_hits), \
         patch('ai.classifier.ask_json', side_effect=LLMUnavailable('daemon down')):
        res = classify('pengeluaran', 'sesuatu yang asing', cats, toko_id='shop-rag')
        assert res['kategori_id'] == 'c_other'
        assert res['skor'] is None
        assert res['status_kategori'] == 'perlu_tinjauan'
