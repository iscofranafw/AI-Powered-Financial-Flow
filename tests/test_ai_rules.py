import pytest
from unittest.mock import patch
from ai.text_normalizer import extract_money, normalize
from ai.classifier import detect_type, classify
from ai.category_suggester import recommend
from ai.ollama_client import LLMUnavailable

CATS = [
    {'id': 'bahan', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran'},
    {'id': 'gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran'},
    {'id': 'other', 'nama': 'Others', 'tipe': 'pengeluaran'}
]

def test_extract_rupiah():
    assert extract_money('beli telur 3 tray 250rb') == 250000

def test_extract_rp():
    assert extract_money('beli bahan Rp250000') == 250000

def test_extract_decimal_suffix():
    from decimal import Decimal
    assert extract_money('terima omzet 1.5jt') == 1500000
    assert extract_money('tambah modal 1,5 juta') == 1500000
    assert extract_money('pesanan 1.25jt') == 1250000
    assert extract_money('beli perlengkapan 500k') == 500000
    assert extract_money('ongkir 1.5rb') == 1500
    assert extract_money('biaya admin 1,5rb') == 1500
    assert extract_money('bayar listrik Rp 250.000') == 250000
    assert extract_money('pembelian barang Rp 250.000,50') == Decimal('250000.50')

def test_extract_ambiguous_rejected():
    with pytest.raises(ValueError):
        extract_money('belanja telur 3 tray')

def test_two_amounts_rejected():
    with pytest.raises(ValueError):
        extract_money('beli 25rb dan 30rb')

def test_normalize():
    assert 'listrik' in normalize('byr lstrk Rp100000')

def test_type_expense():
    assert detect_type('beli tepung 200rb') == 'pengeluaran'

def test_type_income():
    assert detect_type('omzet hari ini 200rb') == 'pemasukan'

def test_llm_classification_success():
    with patch('ai.classifier.ask_json', return_value={'kategori_id': 'bahan', 'nama_baru': '', 'alasan': 'Cocok'}):
        res = classify('pengeluaran', 'beli telur ayam 250rb', CATS)
        assert res['kategori_id'] == 'bahan'
        assert 'Bahan Baku' in res['message']

def test_llm_suggest_new_category():
    with patch('ai.classifier.ask_json', return_value={'kategori_id': '', 'nama_baru': 'Kemasan Khusus', 'alasan': 'Kategori baru'}):
        res = classify('pengeluaran', 'beli plastik segel 50rb', CATS)
        assert res['kategori_id'] is None
        assert res['usulan']['nama_baru'] == 'Kemasan Khusus'

def test_llm_offline_fallback():
    with patch('ai.classifier.ask_json', side_effect=LLMUnavailable('offline')):
        res = classify('pengeluaran', 'beli bahan', CATS)
        assert res['kategori_id'] == 'other'

def test_initial_suggestions_respect_caps():
    for business in ('grosir sembako', 'bakery', 'bengkel motor', 'warung makan'):
        with patch('ai.category_suggester.ask_json', side_effect=LLMUnavailable('offline')):
            rec = recommend(business)
        assert len(rec['pengeluaran']) <= 10 and len(rec['pemasukan']) <= 5
        assert 'Others' in rec['pengeluaran']
        assert 'Tambahan Modal' in rec['pemasukan']
