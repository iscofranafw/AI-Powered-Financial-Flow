from decimal import Decimal
import io
import time
from unittest.mock import MagicMock, patch
import pytest

from ai.hybrid_classifier import (
    clear_classification_cache,
    hybrid_bulk_classify,
)
from services.smart_import_engine import (
    detect_csv_dialect,
    detect_duplicates,
    execute_scalable_import,
    parse_and_validate_dataframe,
    CsvSchemaEngine,
)
import pandas as pd


SAMPLE_CATEGORIES = [
    {
        'id': 'c_gaji',
        'nama': 'Gaji',
        'tipe': 'pengeluaran',
        'kata_kunci': ['gaji', 'upah', 'honor', 'kasbon'],
    },
    {
        'id': 'c_ops',
        'nama': 'Operasional',
        'tipe': 'pengeluaran',
        'kata_kunci': ['listrik', 'token', 'air', 'pdam', 'wifi', 'sewa', 'parkir'],
    },
    {
        'id': 'c_bahan',
        'nama': 'Bahan Baku',
        'tipe': 'pengeluaran',
        'kata_kunci': ['tepung', 'telur', 'gula', 'minyak', 'rokok', 'es batu', 'crystalin', 'yakult', 'pocari'],
    },
    {
        'id': 'c_exp_other',
        'nama': 'Others',
        'tipe': 'pengeluaran',
        'sistem': True,
    },
    {
        'id': 'c_jual',
        'nama': 'Penjualan',
        'tipe': 'pemasukan',
        'kata_kunci': ['omzet', 'penjualan', 'pesanan'],
    },
    {
        'id': 'c_modal',
        'nama': 'Tambahan Modal',
        'tipe': 'pemasukan',
        'kata_kunci': ['modal', 'salah hitung modal', 'titipan modal'],
    },
    {
        'id': 'c_inc_other',
        'nama': 'Others',
        'tipe': 'pemasukan',
        'sistem': True,
    },
]


# 1. Benchmark 1,220 real rows: Parsing and normalization < 3 seconds
def test_benchmark_1220_parsing_under_3_seconds():
    raw_path = r'C:\Users\frana\.gemini\antigravity\brain\3f601137-016f-4410-9eaa-43560cb7bd45\.user_uploaded\media_1793449146801_38ebe486.csv'
    with open(raw_path, 'rb') as f:
        content = f.read()

    t0 = time.perf_counter()
    enc, delim, text = detect_csv_dialect(content)
    df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)
    engine = CsvSchemaEngine(df.head(50))
    analysis = engine.analyze_schema()
    valid_rows, rejected_rows = parse_and_validate_dataframe(
        df=df,
        mappings=analysis['mappings'],
        type_inference_mode=analysis['type_inference_mode'],
        default_type='pengeluaran',
        append_notes=True,
    )
    t_parse = time.perf_counter() - t0

    print(f"\n[BENCHMARK] 1,220 rows parsing & normalization: {t_parse:.4f} s (Target: < 3.0 s)")
    assert len(valid_rows) == 1220
    assert len(rejected_rows) == 0
    assert t_parse < 3.0, f"Parsing took {t_parse} s, exceeds 3.0 s target"


# 2. Benchmark 1,220 rows: Rule-based & cached categorization < 3 seconds
def test_benchmark_1220_categorization_under_3_seconds():
    clear_classification_cache('shop-bench')

    raw_path = r'C:\Users\frana\.gemini\antigravity\brain\3f601137-016f-4410-9eaa-43560cb7bd45\.user_uploaded\media_1793449146801_38ebe486.csv'
    with open(raw_path, 'rb') as f:
        content = f.read()

    enc, delim, text = detect_csv_dialect(content)
    df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)
    engine = CsvSchemaEngine(df.head(50))
    analysis = engine.analyze_schema()
    valid_rows, _ = parse_and_validate_dataframe(
        df=df,
        mappings=analysis['mappings'],
        type_inference_mode=analysis['type_inference_mode'],
        default_type='pengeluaran',
        append_notes=True,
    )

    t0 = time.perf_counter()
    # Mock AI batch call for ambiguous items so we measure pipeline without network latency
    mock_batch_ai = {
        item_id: {
            'kategori_id': 'c_bahan',
            'usulan': {},
            'skor': 0.85,
            'status_kategori': 'dikonfirmasi',
            'message': 'AI Classified',
        }
        for item_id in range(1, 20)
    }

    with patch('ai.hybrid_classifier.classify_ambiguous_batch_ai', side_effect=lambda tipe,items,allowed_categories,**kw: {ident:{**mock_batch_ai[ident], 'kategori_id':allowed_categories[0]['id'], 'source':'llm', 'reason_code':'CLASSIFIED'} for ident,_ in items}):
        classified = hybrid_bulk_classify(
            toko_id='shop-bench',
            categories=SAMPLE_CATEGORIES,
            rows=valid_rows,
        )
    t_cat = time.perf_counter() - t0

    print(f"\n[BENCHMARK] 1,220 rows hybrid categorization: {t_cat:.4f} s (Target: < 3.0 s)")
    assert len(classified) == 1220
    assert t_cat < 3.0, f"Categorization took {t_cat} s, exceeds 3.0 s target"


# 3. Benchmark 5,000 synthetic rows
def test_benchmark_5000_synthetic_rows():
    rows_data = ['tanggal,deskripsi,nominal,tipe']
    for i in range(1, 5001):
        tp = 'pemasukan' if i % 5 == 0 else 'pengeluaran'
        desc = 'beli telur ayam' if i % 3 == 0 else ('bayar token listrik' if i % 3 == 1 else f'gaji staf #{i%10}')
        rows_data.append(f"2026-10-15,{desc},{25000 + i % 500},{tp}")
    csv_text = '\n'.join(rows_data)

    t0 = time.perf_counter()
    enc, delim, text = detect_csv_dialect(csv_text.encode('utf-8'))
    df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)
    engine = CsvSchemaEngine(df.head(50))
    analysis = engine.analyze_schema()
    valid, rejected = parse_and_validate_dataframe(
        df=df,
        mappings=analysis['mappings'],
        type_inference_mode=analysis['type_inference_mode'],
        default_type='pengeluaran',
    )
    t_parse = time.perf_counter() - t0

    print(f"\n[BENCHMARK] 5,000 rows parsing & validation: {t_parse:.4f} s")
    assert len(valid) == 5000
    assert len(rejected) == 0
    assert t_parse < 5.0


# 4. Benchmark 10,000 synthetic rows
def test_benchmark_10000_synthetic_rows():
    rows_data = ['tanggal,deskripsi,nominal,tipe']
    for i in range(1, 10001):
        tp = 'pemasukan' if i % 4 == 0 else 'pengeluaran'
        desc = 'beli minyak goreng' if i % 2 == 0 else 'omzet penjualan harian'
        rows_data.append(f"2026-10-15,{desc},{50000 + i % 1000},{tp}")
    csv_text = '\n'.join(rows_data)

    t0 = time.perf_counter()
    enc, delim, text = detect_csv_dialect(csv_text.encode('utf-8'))
    df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)
    engine = CsvSchemaEngine(df.head(50))
    analysis = engine.analyze_schema()
    valid, rejected = parse_and_validate_dataframe(
        df=df,
        mappings=analysis['mappings'],
        type_inference_mode=analysis['type_inference_mode'],
        default_type='pengeluaran',
    )
    t_parse = time.perf_counter() - t0

    print(f"\n[BENCHMARK] 10,000 rows parsing & validation: {t_parse:.4f} s")
    assert len(valid) == 10000
    assert t_parse < 10.0


# 5. Full End-to-End Pipeline simulation under 15 seconds (Upload -> Parse -> Categorize -> Supabase Batch)
def test_full_pipeline_under_15_seconds():
    raw_path = r'C:\Users\frana\.gemini\antigravity\brain\3f601137-016f-4410-9eaa-43560cb7bd45\.user_uploaded\media_1793449146801_38ebe486.csv'
    with open(raw_path, 'rb') as f:
        content = f.read()

    t_start = time.perf_counter()

    # Step 1 & 2: Dialect & Parsing
    enc, delim, text = detect_csv_dialect(content)
    df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)

    # Step 3: Schema Analysis
    engine = CsvSchemaEngine(df.head(50))
    analysis = engine.analyze_schema()

    # Step 4: Normalization
    valid_rows, rejected_rows = parse_and_validate_dataframe(
        df=df,
        mappings=analysis['mappings'],
        type_inference_mode=analysis['type_inference_mode'],
        default_type='pengeluaran',
        append_notes=True,
    )

    # Step 5: Duplicate Detection
    new_rows, dups = detect_duplicates('s-e2e', valid_rows, existing_transactions=[])

    # Step 6: Categorization
    mock_batch_ai = {
        item_id: {
            'kategori_id': 'c_bahan',
            'usulan': {},
            'skor': 0.85,
            'status_kategori': 'dikonfirmasi',
            'message': 'AI Classified',
        }
        for item_id in range(1, 50)
    }
    with patch('ai.hybrid_classifier.classify_ambiguous_batch_ai', side_effect=lambda tipe,items,allowed_categories,**kw: {ident:{**mock_batch_ai[ident], 'kategori_id':allowed_categories[0]['id'], 'source':'llm', 'reason_code':'CLASSIFIED'} for ident,_ in items}):
        categorized = hybrid_bulk_classify(
            toko_id='s-e2e',
            categories=SAMPLE_CATEGORIES,
            rows=new_rows,
        )

    # Step 7: Supabase Batch Import
    fake_client = MagicMock()
    fake_client.table.return_value.select.return_value.eq.return_value.execute.return_value.count = 100
    fake_client.rpc.return_value.execute.return_value.data = 500

    with patch('services.smart_import_engine.verify_actor', return_value={'id': 'u1', 'toko_id': 's-e2e', 'role': 'pemilik_toko'}), \
         patch('services.smart_import_engine._validate_category'), \
         patch('services.smart_import_engine.privileged_client', return_value=fake_client), \
         patch('services.forecast_service.invalidate_forecast_cache'):

        res = execute_scalable_import('token', categorized, batch_size=500)

    t_total = time.perf_counter() - t_start

    print(f"\n[BENCHMARK] Full End-to-End Pipeline (1,220 rows): {t_total:.4f} s (Target: < 15.0 s)")
    assert res['batches_completed'] == 3
    assert t_total < 15.0, f"Full pipeline took {t_total} s, exceeds 15.0 s target"


@pytest.fixture(autouse=True)
def llm_boundary_for_pipeline_benchmark(monkeypatch):
    """Mock inference latency; reported timings are pipeline-only, never live LLM."""
    def batch(tipe,items,allowed_categories,**kwargs):
        cid=next(c['id'] for c in allowed_categories if c['tipe']==tipe)
        return {ident:{'kategori_id':cid,'usulan':{},'skor':None,'status_kategori':'dikonfirmasi','source':'llm','is_unresolved':False,'reason_code':'CLASSIFIED'} for ident,_ in items}
    monkeypatch.setattr('ai.hybrid_classifier.classify_ambiguous_batch_ai',batch)
