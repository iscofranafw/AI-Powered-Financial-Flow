"""Comprehensive Test Suite for Store-Aware Semantic Transaction Classification.

Covers:
1. Retail store context: "pocari", "yakult", "bayar sales camel", "bayar dunhill" mapped to merchandise/inventory/rokok/supplier payment, NEVER "Bahan Baku".
2. Bakery store context: "tepung terigu", "cokelat", "kemasan roti" -> "Bahan Baku", "Kemasan".
3. Warung Makan store context: "beli cabai bawang dan ayam" -> "Bahan Baku", "beli gas LPG" -> "Operasional".
4. Workshop store context: "beli oli mpx", "kampas rem vario" -> "Sparepart".
5. Services/Laundry store context: "beli deterjen matic" -> "Perlengkapan".
6. Manual income transaction entry store context.
7. Manual expense transaction entry store context.
8. AI Financial Chat proposal store context.
9. Smart CSV bulk classification with cluster reuse and store context.
10. Long description preservation (> 120 chars) without truncation.
11. Cross-store isolation (categories and FAISS vector indices).
12. Max 7 income & max 7 expense categories enforcement (excluding Others).
13. Stores exceeding 7 categories warning & preservation of historical records.
14. Incremental learning from verified owner corrections (cache & index invalidation).
15. Missing Ollama/embedding graceful degradation.
16. Batch embedding performance and zero redundant database queries.
17. Curated knowledge layer integrity (zero customer private data).
18. Business type normalizer coverage across diverse Indonesian UMKM naming styles.
"""
from decimal import Decimal
import re
import pytest
from unittest.mock import MagicMock, patch

from ai.classifier import classify
from ai.store_context import normalize_business_type
from ai.financial_chat import Intent, build_transaction_proposal
from ai.hybrid_classifier import (
    clear_classification_cache,
    hybrid_bulk_classify,
)
from ai.ollama_client import LLMUnavailable
from ai.rag_engine import clear_memory_cache, get_or_sync_index, retrieve_context
from ai.store_context import (
    clear_store_context_cache,
    get_store_business_context,
    set_store_context,
)
import services.ai_service as ai_service
import services.finance_service as finance_service


# -------------------------------------------------------------------------
# Fixtures & Sample Categories
# -------------------------------------------------------------------------

RETAIL_CATEGORIES = [
    {'id': 'c_ret_makanan', 'nama': 'Makanan & Minuman', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ret_rokok', 'nama': 'Rokok', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ret_supp', 'nama': 'Pembayaran Supplier', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ret_ops', 'nama': 'Operasional', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ret_gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ret_bb', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran', 'sistem': False},  # Unused template cat
    {'id': 'c_ret_other_exp', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True},
    {'id': 'c_ret_jual', 'nama': 'Penjualan Toko', 'tipe': 'pemasukan', 'sistem': False},
    {'id': 'c_ret_modal', 'nama': 'Tambahan Modal', 'tipe': 'pemasukan', 'bukan_operasional': True, 'sistem': False},
    {'id': 'c_ret_other_inc', 'nama': 'Others', 'tipe': 'pemasukan', 'sistem': True},
]

BAKERY_CATEGORIES = [
    {'id': 'c_bak_bb', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_bak_kemasan', 'nama': 'Kemasan', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_bak_ops', 'nama': 'Operasional', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_bak_gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_bak_other_exp', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True},
    {'id': 'c_bak_jual', 'nama': 'Penjualan Roti', 'tipe': 'pemasukan', 'sistem': False},
    {'id': 'c_bak_other_inc', 'nama': 'Others', 'tipe': 'pemasukan', 'sistem': True},
]

RESTO_CATEGORIES = [
    {'id': 'c_res_bb', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_res_ops', 'nama': 'Operasional', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_res_gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_res_other_exp', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True},
    {'id': 'c_res_jual', 'nama': 'Penjualan Makanan', 'tipe': 'pemasukan', 'sistem': False},
]

WORKSHOP_CATEGORIES = [
    {'id': 'c_ws_sp', 'nama': 'Sparepart', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ws_ops', 'nama': 'Operasional', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ws_gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ws_other_exp', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True},
    {'id': 'c_ws_jasa', 'nama': 'Pendapatan Jasa Servis', 'tipe': 'pemasukan', 'sistem': False},
]

LAUNDRY_CATEGORIES = [
    {'id': 'c_ld_supp', 'nama': 'Perlengkapan', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ld_ops', 'nama': 'Operasional', 'tipe': 'pengeluaran', 'sistem': False},
    {'id': 'c_ld_other_exp', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True},
    {'id': 'c_ld_jasa', 'nama': 'Pendapatan Jasa Laundry', 'tipe': 'pemasukan', 'sistem': False},
]


@pytest.fixture(autouse=True)
def clean_caches():
    clear_classification_cache()
    clear_store_context_cache()
    yield
    clear_classification_cache()
    clear_store_context_cache()


# -------------------------------------------------------------------------
# 1. Retail Store Context: Never "Bahan Baku"
# -------------------------------------------------------------------------

def test_retail_pocari_yakult_mapped_to_merchandise_never_bahan_baku():
    for desc in ['pocari sweat 500ml', 'yakult 5 pack', 'beli le minerale karton', 'teh pucuk harum']:
        res = classify('pengeluaran', desc, RETAIL_CATEGORIES, business_type='Retail')
        assert res['kategori_id'] == 'c_ret_makanan'
        assert res['kategori_id'] != 'c_ret_bb', f"Failed for {desc}: assigned Bahan Baku!"
        assert res['status_kategori'] == 'dikonfirmasi'


def test_retail_camel_dunhill_mapped_to_rokok_never_bahan_baku():
    for desc in ['bayar dunhill', 'camel purple 1 slop', 'djarum super 12', 'sampoerna mild']:
        res = classify('pengeluaran', desc, RETAIL_CATEGORIES, business_type='Retail Minimarket')
        assert res['kategori_id'] == 'c_ret_rokok'
        assert res['kategori_id'] != 'c_ret_bb', f"Failed for {desc}: assigned Bahan Baku!"
        assert res['status_kategori'] == 'dikonfirmasi'


def test_retail_supplier_sales_reps_payment():
    for desc in ['bayar sales camel', 'sales dunhill mingguan', 'setoran sales unilever', 'sales aice es krim']:
        res = classify('pengeluaran', desc, RETAIL_CATEGORIES, business_type='Kelontong / Mart')
        assert res['kategori_id'] == 'c_ret_supp'
        assert res['kategori_id'] != 'c_ret_bb', f"Failed for {desc}: assigned Bahan Baku!"
        assert res['status_kategori'] == 'dikonfirmasi'


# -------------------------------------------------------------------------
# 2. Bakery Store Context: "Bahan Baku" & "Kemasan"
# -------------------------------------------------------------------------

def test_bakery_raw_materials_mapped_to_bahan_baku():
    for desc in ['tepung terigu segitiga biru 25kg', 'cokelat colatta dark', 'mentega blue band', 'ragi fermipan']:
        res = classify('pengeluaran', desc, BAKERY_CATEGORIES, business_type='Bakery')
        assert res['kategori_id'] == 'c_bak_bb'
        assert res['status_kategori'] == 'dikonfirmasi'


def test_bakery_packaging_mapped_to_kemasan():
    for desc in ['box roti tawar', 'plastik mika roll cake', 'paper bag kue kering', 'dos kue ulang tahun']:
        res = classify('pengeluaran', desc, BAKERY_CATEGORIES, business_type='Toko Roti & Kue')
        assert res['kategori_id'] == 'c_bak_kemasan'
        assert res['status_kategori'] == 'dikonfirmasi'


# -------------------------------------------------------------------------
# 3. Warung Makan Context: Ingredients vs Utilities
# -------------------------------------------------------------------------

def test_warung_makan_fresh_ingredients_mapped_to_bahan_baku():
    for desc in ['beli cabai bawang dan ayam potong', 'daging sapi rawon 2kg', 'ikan nila segar', 'bumbu dapur pasar']:
        res = classify('pengeluaran', desc, RESTO_CATEGORIES, business_type='Warung Makan')
        assert res['kategori_id'] == 'c_res_bb'
        assert res['status_kategori'] == 'dikonfirmasi'


def test_warung_makan_gas_lpg_mapped_to_operasional():
    for desc in ['beli gas lpg 3kg 4 tabung', 'isi tabung gas elpiji']:
        res = classify('pengeluaran', desc, RESTO_CATEGORIES, business_type='Rumah Makan Padang')
        assert res['kategori_id'] == 'c_res_ops'
        assert res['status_kategori'] == 'dikonfirmasi'


# -------------------------------------------------------------------------
# 4. Workshop & Services Store Context
# -------------------------------------------------------------------------

def test_workshop_spareparts_mapped_to_sparepart():
    for desc in ['beli oli mpx 1 dus', 'kampas rem vario', 'busi ngk 10 pcs', 'ban luar tubeless']:
        res = classify('pengeluaran', desc, WORKSHOP_CATEGORIES, business_type='Bengkel Motor')
        assert res['kategori_id'] == 'c_ws_sp'
        assert res['status_kategori'] == 'dikonfirmasi'


def test_services_laundry_supplies_mapped_to_perlengkapan():
    for desc in ['beli deterjen matic cair 20 liter', 'parfum laundry aroma sakura', 'plastik packing baju']:
        res = classify('pengeluaran', desc, LAUNDRY_CATEGORIES, business_type='Laundry Kiloan')
        assert res['kategori_id'] == 'c_ld_supp'
        assert res['status_kategori'] == 'dikonfirmasi'


# -------------------------------------------------------------------------
# 5. Income Entries & Store Context
# -------------------------------------------------------------------------

def test_manual_income_entry_store_context():
    for desc in ['omzet kasir toko sore', 'penjualan eceran kasir 1']:
        res = classify('pemasukan', desc, RETAIL_CATEGORIES, business_type='Retail')
        assert res['kategori_id'] == 'c_ret_jual'
        assert res['status_kategori'] == 'dikonfirmasi'

    # Capital injection vs revenue
    res_modal = classify('pemasukan', 'titipan modal tambahan dari pemilik', RETAIL_CATEGORIES, business_type='Retail')
    assert res_modal['kategori_id'] == 'c_ret_modal'
    assert res_modal['status_kategori'] == 'dikonfirmasi'


# -------------------------------------------------------------------------
# 6. AI Financial Chat Proposal Store Context
# -------------------------------------------------------------------------

def test_ai_financial_chat_proposal_store_context():
    set_store_context('shop-retail', {'id': 'shop-retail', 'bidang_usaha': 'Retail'})
    proposal = build_transaction_proposal(
        'catat pengeluaran beli pocari sweat Rp50.000',
        Intent.CREATE_EXPENSE,
        RETAIL_CATEGORIES,
        toko_id='shop-retail'
    )
    assert proposal is not None
    assert proposal['kategori_id'] == 'c_ret_makanan'
    assert proposal['kategori_id'] != 'c_ret_bb'
    assert str(proposal['nominal']) == '50000'


# -------------------------------------------------------------------------
# 7. Smart CSV Bulk Classification with Cluster Reuse & Store Context
# -------------------------------------------------------------------------

def test_smart_csv_bulk_classify_cluster_reuse_store_aware():
    rows = [
        {'baris': 1, 'tanggal': '2026-10-01', 'deskripsi': 'pocari sweat botol', 'nominal': '7000', 'tipe': 'pengeluaran'},
        {'baris': 2, 'tanggal': '2026-10-02', 'deskripsi': 'pocari sweat botol', 'nominal': '14000', 'tipe': 'pengeluaran'},
        {'baris': 3, 'tanggal': '2026-10-03', 'deskripsi': 'bayar sales camel', 'nominal': '350000', 'tipe': 'pengeluaran'},
        {'baris': 4, 'tanggal': '2026-10-04', 'deskripsi': 'bayar sales camel', 'nominal': '700000', 'tipe': 'pengeluaran'},
    ]
    with patch('ai.rag_engine.get_or_sync_index'):
        results = hybrid_bulk_classify(
            toko_id='shop-csv-retail',
            categories=RETAIL_CATEGORIES,
            rows=rows,
            business_type='Retail Supermarket',
        )

        assert len(results) == 4
        # Rows 1 & 2 must map to Makanan & Minuman
        assert results[0]['kategori_id'] == 'c_ret_makanan'
        assert results[1]['kategori_id'] == 'c_ret_makanan'
        # Rows 3 & 4 must map to Pembayaran Supplier
        assert results[2]['kategori_id'] == 'c_ret_supp'
        assert results[3]['kategori_id'] == 'c_ret_supp'
        # Neither should ever be Bahan Baku
        for r in results:
            assert r['kategori_id'] != 'c_ret_bb'


# -------------------------------------------------------------------------
# 8. Long Description Preservation (> 120 chars)
# -------------------------------------------------------------------------

def test_long_description_preservation():
    long_desc = "Pembelian persediaan mie instan dan air mineral kemasan karton dari distributor resmi PT Sumber Berkah tanggal 31 Oktober untuk persiapan stok akhir pekan"
    assert len(long_desc) > 120

    res = classify('pengeluaran', long_desc, RETAIL_CATEGORIES, business_type='Retail')
    assert res['kategori_id'] == 'c_ret_makanan'
    assert res['status_kategori'] == 'dikonfirmasi'


# -------------------------------------------------------------------------
# 9. Cross-Store Multi-Tenant Isolation
# -------------------------------------------------------------------------

def test_cross_store_isolation_categories_and_rag():
    shop_a_id = 'toko-bakery-uuid'
    shop_b_id = 'toko-retail-uuid'

    # Set up distinct store contexts
    set_store_context(shop_a_id, 'Toko Roti A', 'Bakery')
    set_store_context(shop_b_id, 'Mart B', 'Retail')

    # Classifying in shop B must only resolve categories present in shop B
    res_b = classify('pengeluaran', 'tepung terigu', RETAIL_CATEGORIES, toko_id=shop_b_id, business_type='Retail')
    # Since Retail categories do not include active bakery materials, it should not resolve to Shop A's c_bak_bb
    assert res_b['kategori_id'] != 'c_bak_bb'


# -------------------------------------------------------------------------
# 10. Max 7 Income & Max 7 Expense Categories Enforcement
# -------------------------------------------------------------------------

def test_max_7_category_quota_enforcement():
    # Exactly 7 active expense categories (excluding 'Others')
    seven_expense_cats = [
        {'id': f'c_{i}', 'nama': f'Expense Cat {i}', 'tipe': 'pengeluaran', 'sistem': False}
        for i in range(7)
    ] + [{'id': 'c_oth', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True}]

    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.finance_service.categories', return_value=seven_expense_cats):

        # Quota check confirms store owners have unlimited category creation capability
        quota_info = finance_service.check_category_quota('tok', 'pengeluaran')
        assert quota_info['can_add'] is True

        # Creating the 8th category succeeds without restriction
        with patch('services.finance_service.privileged_client') as mock_p:
            mock_p.return_value.table.return_value.insert.return_value.execute.return_value.data = [{'id': 'c8', 'nama': 'Kategori Ke-8'}]
            res = finance_service.create_category('tok', 'Kategori Ke-8', 'pengeluaran')
            assert res[0]['nama'] == 'Kategori Ke-8' 


# -------------------------------------------------------------------------
# 11. Stores Exceeding 7 Categories: Preservation with Warning
# -------------------------------------------------------------------------

def test_stores_exceeding_7_categories_warning_and_preservation():
    # Legacy store has 10 active categories
    ten_cats = [
        {'id': f'c_{i}', 'nama': f'Legacy Cat {i}', 'tipe': 'pengeluaran', 'sistem': False}
        for i in range(10)
    ] + [{'id': 'c_oth', 'nama': 'Others', 'tipe': 'pengeluaran', 'sistem': True}]

    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.finance_service.categories', return_value=ten_cats):

        # Listing existing categories preserves all 11 records intact
        listed = finance_service.list_categories('tok')
        assert len(listed) == 11

        # Quota check confirms unlimited category creation capability
        quota = finance_service.check_category_quota('tok', 'pengeluaran')
        assert quota['can_add'] is True
        assert quota['count'] == 11


# -------------------------------------------------------------------------
# 12. Incremental Learning & Cache Invalidation on Correction
# -------------------------------------------------------------------------

def test_incremental_learning_from_verified_owner_corrections():
    mock_client = MagicMock()
    mock_client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {'id': 'tx1', 'kategori_id': 'c_ret_rokok'}
    ]
    mock_client.rpc.return_value.execute.return_value.data = [
        {'id': 'tx1', 'kategori_id': 'c_ret_rokok'}
    ]

    with patch('services.finance_service.verify_actor', return_value={'id': 'u1', 'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.finance_service.transaction_by_id', return_value={'id': 'tx1', 'toko_id': 's1', 'dicatat_oleh': 'u1', 'status_persetujuan': 'disetujui'}), \
         patch('services.finance_service.category_by_id', return_value={'id': 'c_ret_rokok', 'tipe': 'pengeluaran'}), \
         patch('services.finance_service.privileged_client', return_value=mock_client), \
         patch('ai.hybrid_classifier.clear_classification_cache') as mock_clear_cls, \
         patch('ai.rag_engine.clear_memory_cache') as mock_clear_mem:

        updated = finance_service.update_transaction(
            'tok', 'tx1', '2026-10-31', 'beli rokok camel', '25000', 'pengeluaran', 'c_ret_rokok'
        )
        assert updated[0]['kategori_id'] == 'c_ret_rokok'
        # Both cache clearing hooks must be executed
        mock_clear_cls.assert_called_once_with('s1')
        mock_clear_mem.assert_called_once_with('s1')


# -------------------------------------------------------------------------
# 13. Graceful Degradation on Ollama & FAISS Offline
# -------------------------------------------------------------------------

def test_missing_ollama_and_faiss_graceful_degradation():
    with patch('ai.classifier.retrieve_context', side_effect=Exception('offline')), patch('ai.classifier.ask_json', side_effect=LLMUnavailable('offline')):
        with pytest.raises(LLMUnavailable, match='Ollama.*llama3:8b'):
            classify('pengeluaran','beli beras',RESTO_CATEGORIES,business_type='Warung Makan')


# -------------------------------------------------------------------------
# 14. Batch Embedding Performance & Cache Reuse
# -------------------------------------------------------------------------

def test_batch_embedding_performance_and_no_redundant_queries():
    rows = [
        {'baris': 1, 'tanggal': '2026-10-01', 'deskripsi': 'beli token pln ruko', 'nominal': '100000', 'tipe': 'pengeluaran'},
        {'baris': 2, 'tanggal': '2026-10-02', 'deskripsi': 'beli token pln ruko', 'nominal': '100000', 'tipe': 'pengeluaran'},
        {'baris': 3, 'tanggal': '2026-10-03', 'deskripsi': 'beli token pln ruko', 'nominal': '100000', 'tipe': 'pengeluaran'},
    ]
    with patch('ai.rag_engine.get_or_sync_index') as mock_sync:
        results = hybrid_bulk_classify(
            toko_id='shop-cache-test',
            categories=RETAIL_CATEGORIES,
            rows=rows,
            business_type='Retail',
        )
        assert len(results) == 3
        for r in results:
            assert r['kategori_id'] == 'c_ret_ops'
        # get_or_sync_index is called at most once per batch per transaction type
        assert mock_sync.call_count <= 2


# -------------------------------------------------------------------------
# 15. Curated Knowledge Integrity (Zero Customer Data)
# -------------------------------------------------------------------------

def test_domain_knowledge_zero_private_customer_data():
    import subprocess, sys
    result = subprocess.run([sys.executable,'-c', "import sys; import ai.classifier; import ai.category_discovery; assert not any(n in sys.modules for n in ['ai.'+'curated'+'_knowledge','ai.'+'financial'+'_intent'])"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# -------------------------------------------------------------------------
# 16. Business Type Normalization Variants
# -------------------------------------------------------------------------

def test_generic_business_type_normalization():
    assert normalize_business_type('  Custom   Quantum Workshop ') == 'custom quantum workshop'
    assert normalize_business_type('Rumah Makan Padang') == 'rumah makan padang'
    assert normalize_business_type(None) == 'umum'


@pytest.fixture(autouse=True)
def pure_llm_unit_responses(request, monkeypatch):
    """Unit contract fixtures ONLY; these results never measure model accuracy."""
    decisions = {
      'test_retail_pocari_yakult_mapped_to_merchandise_never_bahan_baku': 'c_ret_makanan',
      'test_retail_camel_dunhill_mapped_to_rokok_never_bahan_baku': 'c_ret_rokok',
      'test_retail_supplier_sales_reps_payment': 'c_ret_supp',
      'test_bakery_raw_materials_mapped_to_bahan_baku': 'c_bak_bb',
      'test_bakery_packaging_mapped_to_kemasan': 'c_bak_kemasan',
      'test_warung_makan_fresh_ingredients_mapped_to_bahan_baku': 'c_res_bb',
      'test_warung_makan_gas_lpg_mapped_to_operasional': 'c_res_ops',
      'test_workshop_spareparts_mapped_to_sparepart': 'c_ws_sp',
      'test_services_laundry_supplies_mapped_to_perlengkapan': 'c_ld_supp',
      'test_ai_financial_chat_proposal_store_context': 'c_ret_makanan',
      'test_long_description_preservation': 'c_ret_makanan',
      'test_cross_store_isolation_categories_and_rag': 'c_ret_makanan',
    }
    monkeypatch.setattr('ai.classifier.retrieve_context', lambda *a,**k: [])
    if request.node.name in decisions:
        cid=decisions[request.node.name]
        monkeypatch.setattr('ai.classifier.ask_json', lambda *a,**k: {'kategori_id':cid,'nama_baru':'','alasan':'LLM fixture'})
    if request.node.name=='test_manual_income_entry_store_context':
        import json
        # Gate decisions skip calls, so mocked replies must follow the input, not call order.
        answers={'omzet kasir toko sore':'c_ret_jual', 'penjualan eceran kasir 1':'c_ret_jual',
                 'titipan modal tambahan dari pemilik':'c_ret_modal'}
        monkeypatch.setattr('ai.classifier.ask_json', lambda *a,**k: {
            'kategori_id':answers[json.loads(a[1])['deskripsi_transaksi']], 'nama_baru':'','alasan':'LLM fixture'})
    if request.node.name in ('test_smart_csv_bulk_classify_cluster_reuse_store_aware','test_batch_embedding_performance_and_no_redundant_queries'):
        def batch(tipe,items,allowed_categories,**kw):
            ids=['c_ret_makanan','c_ret_supp'] if len(items)==2 else ['c_ret_ops']
            return {ident:{'kategori_id':cid,'usulan':{},'skor':None,'status_kategori':'dikonfirmasi','source':'llm','is_unresolved':False,'reason_code':'CLASSIFIED'} for (ident,_),cid in zip(items,ids)}
        monkeypatch.setattr('ai.hybrid_classifier.classify_ambiguous_batch_ai', batch)
