"""Regression tests using shipped references, tenant history and user categories."""
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai.category_discovery import (suggest_default_keywords, _reference_profiles,
    _discover_for_type_with_llm, matching_transaction_samples)
from services.import_category_policy import missing_category_types, preview_category_label, missing_allowed_types

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('domain,category,example', [
    ('retail', 'Stok Barang Dagangan', 'beli mie instan'),
    ('warung_makan', 'Bahan Baku', 'beli ayam dan beras'),
    ('bengkel_motor', 'Suku Cadang', 'beli stok oli mesin'),
    ('bakery', 'Bahan Baku', 'beli mentega'),
])
def test_reference_domains_prioritize_csv(domain, category, example):
    filename = 'grosir' if domain == 'retail' else domain
    rows = list(csv.DictReader((ROOT / f'data/samples/{filename}.csv').open(encoding='utf-8-sig')))
    assert example in {r['deskripsi'] for r in rows}
    kws = suggest_default_keywords(category, business_type=domain, contoh=[example], llm_keywords=['frasa discovery'])
    assert kws[0] == example
    assert kws.index('frasa discovery') < len(kws)
    if domain in ('retail', 'bengkel_motor'):
        assert not any(word in ' '.join(kws).lower() for word in ('tepung', 'gula', 'minyak', 'telur'))


def test_all_reference_keywords_have_actual_cited_sources():
    for domain, groups in _reference_profiles().items():
        for group in groups:
            corpus = '\n'.join((ROOT / source).read_text(encoding='utf-8').lower() for source in group['sources'])
            assert group['keywords'], (domain, group)
            for kw in group['keywords']:
                assert kw.lower() in corpus, (domain, kw)


@pytest.mark.parametrize('domain', ['retail', 'bengkel_motor', 'percetakan baru', 'Umum'])
def test_no_generic_kitchen_injection(domain):
    assert suggest_default_keywords('Bahan Baku', business_type=domain) == []
    kws = suggest_default_keywords('Bahan Baku', business_type=domain, contoh=['beli gula aren untuk dijual'])
    assert any('gula aren' in kw for kw in kws)
    assert not any('tepung' in kw for kw in kws)


@pytest.mark.parametrize('tp', ['pengeluaran', 'pemasukan'])
def test_empty_store_guard_both_types(tp):
    rows = [{'tipe': tp, 'deskripsi': 'transaksi baru'}]
    system = [{'nama': 'Others', 'tipe': tp, 'sistem': True},
              {'nama': 'Tambahan Modal', 'tipe': 'pemasukan', 'bukan_operasional': True}]
    assert missing_category_types(rows, system, []) == [tp]
    user = {'nama': 'Kategori Pilihan Pemilik', 'tipe': tp}
    assert missing_category_types(rows, system, [user]) == []
    assert missing_category_types(rows, system + [user], []) == []
    assert missing_category_types(rows, system + [{**user, 'aktif': False}], []) == [tp]


def test_unresolved_preview_is_distinct_from_manual_others():
    row = {'kategori_id': None, 'is_unresolved': True, 'tipe': 'pengeluaran', 'reason_code': 'NO_ALLOWED_CATEGORIES'}
    original = dict(row)
    assert preview_category_label(row, {'other': 'Others'}) == 'Belum Terpetakan (Perlu Tinjauan)'
    assert row == original
    assert missing_allowed_types([row]) == ['pengeluaran']
    assert preview_category_label({'kategori_id': 'other', 'is_unresolved': False}, {'other': 'Others'}) == 'Others'


@pytest.mark.parametrize('tp', ['pengeluaran', 'pemasukan'])
def test_discovery_retries_empty_and_grounds_examples(tp):
    samples = [('baut titanium', 'baut titanium', 2), ('pegas khusus', 'pegas khusus', 1), ('seal fluoro', 'seal fluoro', 1)]
    rec = {'nama': 'Komponen Pemilik', 'deskripsi': 'Komponen pesanan', 'contoh': [],
           'kata_kunci': ['baut titanium', 'pegas khusus', 'seal fluoro']}
    with patch('ai.category_discovery.ask_json', side_effect=[{'rekomendasi': []}, {'rekomendasi': [rec]}]) as ask:
        result = _discover_for_type_with_llm(tp, samples, [], 'Manufaktur Baru')
    assert ask.call_count == 2
    assert len(result[0]['contoh']) == 3
    assert result[0]['kata_kunci'][0] == 'baut titanium'
    schema = ask.call_args.kwargs['schema']['properties']['rekomendasi']
    assert schema['minItems'] == 1
    assert schema['items']['properties']['contoh']['minItems'] == 3
    assert set(schema['items']['required']) >= {'contoh', 'kata_kunci'}


def test_csv_samples_are_type_scoped_and_ignore_hallucinated_examples():
    rec = {'nama': 'Pesanan', 'tipe': 'pemasukan', 'contoh': ['karangan'], 'kata_kunci': ['laser']}
    assert matching_transaction_samples(rec, [
        {'tipe': 'pengeluaran', 'deskripsi': 'laser'},
        {'tipe': 'pemasukan', 'deskripsi': 'penerimaan ukir laser'},
    ]) == ['penerimaan ukir laser']


@pytest.mark.parametrize('tp', ['pengeluaran', 'pemasukan'])
def test_autofill_custom_category_uses_tenant_history_and_warns(tp):
    import services.finance_service as fs
    client = MagicMock()
    query = client.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value
    query.execute.return_value.data = [
        {'kategori_id': 'custom', 'tipe': tp, 'deskripsi': 'pengerjaan panel akustik bambu'},
        {'kategori_id': 'other-cat', 'tipe': tp, 'deskripsi': 'transaksi kategori lain'},
    ]
    cats = [{'id': 'custom', 'nama': 'Proyek Akustik', 'tipe': tp},
            {'id': 'empty', 'nama': 'Pos Baru Pemilik', 'tipe': tp}]
    with patch.object(fs, '_owner', return_value={'toko_id': 'tenant-A'}), \
         patch.object(fs, 'categories', return_value=cats), \
         patch.object(fs, 'privileged_client', return_value=client), \
         patch.object(fs, 'set_category_keywords') as save, \
         patch('ai.store_context.get_store_business_context', return_value={'bidang_usaha': 'Akustik'}):
        result = fs.auto_fill_category_keywords('token')
    assert result['updated_count'] == 1
    assert result['updated'][0]['keywords'][0] == 'pengerjaan panel akustik bambu'
    assert len(result['warnings']) == 1 and 'Pos Baru Pemilik' in result['warnings'][0]
    client.table.return_value.select.return_value.eq.assert_called_once_with('toko_id', 'tenant-A')
    save.assert_called_once()


def test_retail_14_stock_descriptions_gate_and_no_raw_materials():
    from ai.classifier import _similarity_evaluate
    rows = json.loads((ROOT / 'tests/fixtures/retail_22_descriptions.json').read_text(encoding='utf-8'))
    # Test-only category assignment by user, never an application lookup table.
    stock = [r['deskripsi'] for r in rows if r['deskripsi'].startswith('beli ')
             and 'plastik' not in r['deskripsi']]
    assert len(stock) == 14
    kw = suggest_default_keywords('Stok Barang Dagangan', business_type='retail', contoh=stock)
    cats = [dict(id='user_stock', nama='Stok Barang Dagangan', tipe='pengeluaran',
                 deskripsi='Persediaan barang untuk dijual kembali, bukan bahan produksi.', kata_kunci=kw),
            dict(id='user_supp', nama='Perlengkapan', tipe='pengeluaran',
                 deskripsi='Barang dipakai sendiri oleh toko untuk membungkus belanja.', kata_kunci=['plastik dan kantong kertas']),
            dict(id='user_raw', nama='Bahan Baku', tipe='pengeluaran',
                 deskripsi='Bahan produksi, bukan barang dijual kembali.', kata_kunci=[])]
    results = [_similarity_evaluate('pengeluaran', d, cats) for d in stock]
    assert sum(diag['gate_passed'] for _, diag in results) > len(stock) // 2
    assert all(dec is None or dec['kategori_id'] == 'user_stock' for dec, _ in results)
    for word in ('tepung', 'gula', 'minyak', 'telur'):
        if word not in ' '.join(stock):
            assert word not in ' '.join(kw)


def test_new_user_categories_generalize_to_10_expenses_and_10_incomes():
    from scripts.measure_new_user_generalization import evaluate_new_user_data
    results = evaluate_new_user_data()
    assert len(results) == 20
    assert sum(r['tipe'] == 'pengeluaran' for r in results) == 10
    assert sum(r['tipe'] == 'pemasukan' for r in results) == 10
    assert all(r['correct'] for r in results), [r for r in results if not r['correct']]
    assert all(r['predicted'] is not None for r in results)


@pytest.mark.parametrize('tp', ['pengeluaran', 'pemasukan'])
def test_streamlit_actual_approval_button_locks_and_unlocks(tp):
    """Execute the production Stage 4 guard/button block in Streamlit's test runtime."""
    import textwrap
    from streamlit.testing.v1 import AppTest
    source = (ROOT / 'pages/import_view.py').read_text(encoding='utf-8')
    block = source[source.index('    missing_types = missing_category_types('):source.index('    if btn_approve and not missing_types:')]
    prefix = f'''
import streamlit as st
from services.import_category_policy import missing_category_types
rows_to_import = [{{'tipe': {tp!r}, 'deskripsi': 'transaksi baru'}}]
existing_store_cats = [{{'nama': 'Others', 'tipe': {tp!r}, 'sistem': True}}]
all_reviewed_exp = st.session_state.get('reviewed_exp', [])
all_reviewed_inc = st.session_state.get('reviewed_inc', [])
'''
    app = AppTest.from_string(prefix + textwrap.dedent(block)).run()
    assert not app.exception
    assert app.button(key='btn_approve_and_continue').disabled
    assert f'Belum ada kategori {tp}' in app.error[0].value
    key = 'reviewed_exp' if tp == 'pengeluaran' else 'reviewed_inc'
    app.session_state[key] = [{'nama': 'Kategori Pemilik', 'tipe': tp}]
    app.run()
    assert not app.exception
    assert not app.button(key='btn_approve_and_continue').disabled


def test_user_keyword_vector_is_independent_of_category_name_length():
    from ai.classifier import _similarity_evaluate
    description = 'pengadaan mikrofon piezo untuk instrumen eksperimental'
    cats = [dict(id='custom', nama='Kategori milik pengguna dengan nama panjang yang bebas ditentukan sendiri',
                 tipe='pengeluaran', kata_kunci=[description]),
            dict(id='other', nama='Langganan Ruangan', tipe='pengeluaran', kata_kunci=['tagihan sewa tempat'])]
    decision, diag = _similarity_evaluate('pengeluaran', description, cats)
    assert decision['kategori_id'] == 'custom'
    assert diag['gate_passed'] and diag['top_score'] > .99


def test_grounding_repairs_examples_without_retaining_bad_model_keywords():
    from ai.category_discovery import ground_discovered_categories
    recommendations = [dict(nama='Kategori Pemilik', tipe='pemasukan', deskripsi='Pesanan ukiran',
                            contoh=['contoh salah'], kata_kunci=['contoh salah'], discovery_keywords=['ukiran'])]
    samples = [('pembayaran ukiran bambu', 'pembayaran ukiran bambu', 1)]
    with patch('ai.category_discovery.classify', return_value={'kategori_id': 'proposal_0'}):
        result = ground_discovered_categories(recommendations, samples, 'pemasukan', 'Kerajinan')
    assert result[0]['contoh'] == ['pembayaran ukiran bambu']
    assert result[0]['kata_kunci'][0] == 'pembayaran ukiran bambu'
    assert 'contoh salah' not in result[0]['kata_kunci']
