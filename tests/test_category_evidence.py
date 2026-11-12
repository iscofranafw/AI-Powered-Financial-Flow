"""Keyword ownership regressions; all phrases are test inputs, not app defaults."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ai.category_discovery import extract_keywords_from_samples
from ai.category_evidence import compile_keyword_plan
from ai.classifier import _similarity_evaluate


def category(cid, name, tipe='pengeluaran', keywords=None):
    return dict(id=cid, nama=name, tipe=tipe, kata_kunci=keywords or [], sistem=False)


@pytest.mark.parametrize('description', ['beli gula pasir 10 kg', 'beli ragi instan 1 kg'])
def test_complete_user_description_preserves_ingredient_gate(description):
    cats = [category('raw', 'Bahan Baku'), category('ops', 'Operasional')]
    plan = compile_keyword_plan(cats, [dict(deskripsi=description, tipe='pengeluaran', kategori_id='raw')], 'Bakery')
    assert plan['categories'][0]['kata_kunci'][0] == description
    decision, diag = _similarity_evaluate('pengeluaran', description, plan['categories'])
    assert decision['kategori_id'] == 'raw'
    assert diag['top_score'] >= .80 and diag['margin'] >= .04


@pytest.mark.parametrize('domain', ['Bakery', 'Retail'])
def test_delivery_phrase_is_removed_from_wages_using_unique_history_owner(domain):
    phrase = 'biaya antar pesanan ojek'
    cats = [category('wage', 'Gaji dan Upah', keywords=[phrase, 'antar pesanan ojek', 'ojek']),
            category('delivery', 'Biaya Pengiriman', keywords=[phrase, 'ojek'])]
    original = copy.deepcopy(cats)
    history = [dict(deskripsi=phrase, kategori_id='delivery', tipe='pengeluaran')]
    plan = compile_keyword_plan(cats, history, domain)
    assert cats == original
    assert not any('ojek' in kw for kw in plan['categories'][0]['kata_kunci'])
    assert plan['audit']
    dec, diag = _similarity_evaluate('pengeluaran', phrase, plan['categories'])
    assert dec['kategori_id'] == 'delivery'
    assert diag['gate_passed'] and diag['margin'] >= .04


def test_ambiguous_history_is_not_arbitrarily_assigned():
    cats = [category('a', 'Pos Pilihan Satu', keywords=['pengerjaan panel akustik']),
            category('b', 'Pos Pilihan Dua', keywords=['pengerjaan panel akustik'])]
    rows = [dict(deskripsi='pengerjaan panel akustik', kategori_id=cid, tipe='pengeluaran') for cid in ('a', 'b')]
    result = compile_keyword_plan(cats, rows, 'Akustik')
    assert all('pengerjaan panel akustik' not in c['kata_kunci'] for c in result['categories'])
    assert any(a['reason'] == 'conflicting_evidence' for a in result['audit'])


def test_owner_history_removes_cross_category_generic_words():
    cats = [category('s', 'Stok'), category('o', 'Operasional', keywords=['air'])]
    history = [dict(deskripsi='beli air mineral kemasan untuk dijual', kategori_id='s', tipe='pengeluaran'),
               dict(deskripsi='bayar air PDAM', kategori_id='o', tipe='pengeluaran')]
    result = compile_keyword_plan(cats, history, 'Retail')
    assert 'air' not in result['categories'][1]['kata_kunci']
    for row in history:
        dec, _ = _similarity_evaluate('pengeluaran', row['deskripsi'], result['categories'])
        assert dec['kategori_id'] == row['kategori_id']


def test_income_subtypes_do_not_share_other_subtype_keywords():
    cats = [category('e', 'Penjualan Eceran', 'pemasukan'), category('g', 'Penjualan Grosir', 'pemasukan'),
            {**category('m', 'Tambahan Modal', 'pemasukan'), 'bukan_operasional': True}]
    history = [dict(deskripsi='penjualan eceran hari ini', tipe='pemasukan', kategori_id='e'),
               dict(deskripsi='penjualan grosir 30 dus', tipe='pemasukan', kategori_id='g'),
               dict(deskripsi='setoran tambahan modal', tipe='pemasukan', kategori_id='m')]
    result = compile_keyword_plan(cats, history, 'Retail')
    assert not any('grosir' in kw for kw in result['categories'][0]['kata_kunci'])
    assert not any('eceran' in kw for kw in result['categories'][1]['kata_kunci'])
    for row in history:
        dec, diag = _similarity_evaluate('pemasukan', row['deskripsi'], result['categories'])
        assert dec['kategori_id'] == row['kategori_id'] and diag['gate_passed']


def test_complete_patterns_are_kept_before_multiple_compact_variants():
    examples = [f'belanja komponen seri {chr(65+i)} {i+1} kg' for i in range(16)]
    result = extract_keywords_from_samples(examples)
    assert result[:16] == examples
    assert len(result) <= 20


def test_already_filled_keywords_are_not_overwritten_by_autofill():
    from services import finance_service as fs
    from unittest.mock import MagicMock
    cats = [category('a', 'Pos Pemilik', keywords=['contoh pilihan pemilik'])]
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = []
    with patch.object(fs, '_owner', return_value={'toko_id': 'test-only'}), \
         patch.object(fs, 'privileged_client', return_value=client), \
         patch.object(fs, 'categories', return_value=cats), \
         patch('ai.store_context.get_store_business_context', return_value={'bidang_usaha':'Umum'}), \
         patch.object(fs, 'set_category_keywords') as write:
        result = fs.auto_fill_category_keywords('tok', overwrite=False)
    write.assert_not_called()
    assert result['updated_count'] == 0


def test_upload_plan_uses_actual_rows_and_never_creates_category_ids():
    from services.finance_service import plan_category_keywords
    cats = [category('pack', 'Kemasan Pilihan Pemilik'), category('stock', 'Barang Dagangan')]
    rows = [{'deskripsi':'beli wadah serat tebu untuk membungkus pesanan', 'tipe':'pengeluaran'}]
    with patch('ai.category_evidence.resolve_sample_owner', return_value={'kategori_id':'pack'}) as resolver:
        result = plan_category_keywords(cats, [], 'Retail', rows * 5)
    resolver.assert_called_once()
    assert {c['id'] for c in result['categories']} == {'pack','stock'}
    assert result['categories'][0]['kata_kunci'][0] == rows[0]['deskripsi']


def test_explicit_csv_owner_category_outweighs_old_history_in_review_plan():
    from services.finance_service import plan_category_keywords
    cats = [category('old', 'Pos Lama'), category('new', 'Pos Pilihan Pemilik')]
    desc = 'pembayaran pekerjaan akustik ruang rapat'
    history = [dict(deskripsi=desc, kategori_id='old', tipe='pengeluaran')]
    # Actual parse_and_validate_dataframe output uses csv_category, not kategori.
    csv = [dict(deskripsi=desc, csv_category='Pos Pilihan Pemilik', tipe='pengeluaran')]
    result = plan_category_keywords(cats, history, 'Akustik', csv)
    assert desc not in result['categories'][0]['kata_kunci']
    assert desc in result['categories'][1]['kata_kunci']
    assert history[0]['kategori_id'] == 'old'  # Planning never rewrites the store.
    from ai.category_discovery import matching_transaction_samples
    assert matching_transaction_samples({**cats[1], 'contoh': []}, csv) == [desc]


def test_quantity_and_purchase_action_variants_reuse_confirmed_owner():
    from services.finance_service import plan_category_keywords
    cats = [category('stock', 'Persediaan Pemilik'), category('ops', 'Operasional')]
    history = [dict(deskripsi='kulakan minyak goreng', kategori_id='stock', tipe='pengeluaran')]
    csv = [dict(deskripsi='beli minyak goreng 5 liter', tipe='pengeluaran')]
    with patch('ai.category_evidence.resolve_sample_owner') as model:
        result = plan_category_keywords(cats, history, 'Retail', csv)
    model.assert_not_called()
    assert csv[0]['deskripsi'] in result['categories'][0]['kata_kunci']


def test_guide_grounding_is_read_from_source_and_requires_uploaded_term():
    from ai.category_discovery import _reference_profiles
    from ai.category_evidence import resolve_sample_owner
    group = _reference_profiles()['retail'][0]
    root = Path(__file__).resolve().parents[1]
    text = (root / group['grounding_sources'][0]['path']).read_text(encoding='utf-8').casefold()
    assert all(term.casefold() in text for term in group['grounding_terms'])
    cats = [category('stock', 'Stok Barang Dagangan'), category('ops', 'Operasional')]
    with patch('ai.category_evidence.ask_json') as model:
        result = resolve_sample_owner('pengeluaran','beli sabun mandi 1 lusin',cats,'Retail')
    model.assert_not_called()
    assert result['kategori_id'] == 'stock' and result['references']
    with patch('ai.category_evidence.ask_json', return_value={'category_id':'1','reason':'Explicit internal use'}) as model:
        result = resolve_sample_owner('pengeluaran','beli sabun mandi untuk dipakai sendiri',cats,'Retail')
    model.assert_called_once()
    assert result['kategori_id'] == 'ops'


def test_real_corpus_keeps_genuinely_conflicting_history_tied(tmp_path):
    from unittest.mock import MagicMock
    from ai import rag_engine
    phrase = 'pengerjaan panel akustik'
    cats = [category('a', 'Pos Satu', keywords=[phrase]), category('b', 'Pos Dua', keywords=[phrase])]
    txs = [dict(id=str(i), kategori_id=cid, deskripsi=phrase) for i,cid in enumerate(('a','b'))]
    client = MagicMock()
    def table(name):
        query = MagicMock()
        if name == 'kategori':
            query.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = cats
        else:
            query.select.return_value.eq.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = txs
        return query
    client.table.side_effect = table
    shop='test-round2-conflicting-history'
    with patch.object(rag_engine,'privileged_client',return_value=client), \
         patch.object(rag_engine,'get_store_business_context',return_value={'bidang_usaha':'Akustik'}), \
         patch.object(rag_engine,'get_remote_version',return_value=1), \
         patch.object(rag_engine,'INDEX_ROOT_DIR',str(tmp_path)):
        entries=rag_engine.build_corpus_entries(shop,'pengeluaran')
        assert sum(e['type']=='transaction' for e in entries) == 2
        assert not any(e['verification_source']=='category_keyword' and e['text']==phrase for e in entries)
        rag_engine.build_and_save_index(shop,'pengeluaran',entries=entries,version=1)
        decision,diag=_similarity_evaluate('pengeluaran',phrase,cats,toko_id=shop)
    assert decision is None and not diag['gate_passed'] and diag['margin']==0
    rag_engine.clear_memory_cache(shop)


def test_current_upload_coverage_is_prioritized_over_old_patterns_at_20_limit():
    from services.finance_service import plan_category_keywords
    cats = [category('s', 'Pos Barang Pemilik')]
    old = [dict(deskripsi=f'pengadaan varian lama {chr(65+i)}', kategori_id='s', tipe='pengeluaran') for i in range(26)]
    current = [dict(deskripsi=f'pengadaan seri terkini {chr(65+i)}', kategori_id='s', tipe='pengeluaran') for i in range(14)]
    result = plan_category_keywords(cats, old, 'Umum', current)
    keywords = result['categories'][0]['kata_kunci']
    assert len(keywords) == 20
    assert keywords[:14] == [r['deskripsi'] for r in current]


def test_index_rebuild_retains_stored_csv_coverage_instead_of_evicting_it():
    stored = [f'pengadaan seri terkini {chr(65+i)}' for i in range(14)]
    cats = [category('s', 'Pos Barang Pemilik', keywords=stored)]
    old = [dict(deskripsi=f'pengadaan varian lama {chr(65+i)}', kategori_id='s', tipe='pengeluaran') for i in range(26)]
    result = compile_keyword_plan(cats, old, 'Umum', prefer_existing=True)
    assert result['categories'][0]['kata_kunci'] == stored


def test_qualified_owner_category_cannot_be_overridden_by_general_reference():
    from ai.category_evidence import resolve_sample_owner
    cats=[category('s','Stok Mesin'),category('o','Operasional')]
    with patch('ai.category_evidence.ask_json',return_value={'category_id':'','reason':'No suitable owner category'}) as model:
        result=resolve_sample_owner('pengeluaran','beli teh celup 1 dus',cats,'Retail')
    model.assert_called_once()
    assert result['kategori_id'] is None


def test_retail_wrapping_policy_respects_explicit_owner_packaging_category():
    from ai.category_evidence import resolve_sample_owner
    choices=[category('o','Operasional'),category('s','Kulakan dan Stok'),category('f','Perlengkapan dan Furnitur')]
    choices[0]['deskripsi'] = None
    with patch('ai.category_evidence.ask_json',return_value={'category_id':'0','reason':'Operating consumable'}) as model:
        resolve_sample_owner('pengeluaran','beli wadah serat untuk membungkus belanja',choices,'Retail')
    assert 'No explicit packaging category' in model.call_args.args[0]
    choices.append(category('p','Pengemasan Pilihan Pemilik'))
    with patch('ai.category_evidence.ask_json',return_value={'category_id':'3','reason':'Explicit owner category'}) as model:
        resolve_sample_owner('pengeluaran','beli wadah serat untuk membungkus belanja',choices,'Retail')
    assert 'No explicit packaging category' not in model.call_args.args[0]
