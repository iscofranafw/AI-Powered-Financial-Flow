from datetime import date
from decimal import Decimal
import io
import os
from unittest.mock import MagicMock, patch
import pytest

from services.import_service import (
    analyze_csv_upload,
    atomic_import,
    preview_csv,
)
from services.smart_import_engine import (
    CsvSchemaEngine,
    detect_csv_dialect,
    detect_duplicates,
    execute_scalable_import,
    normalize_amount,
    normalize_date,
    normalize_description,
    parse_and_validate_dataframe,
)
import pandas as pd


# 1. Existing application CSV format
def test_existing_application_csv_format():
    csv_bytes = b'tanggal,deskripsi,nominal,tipe\n2026-10-30,Beli bahan baku,250000,pengeluaran\n2026-10-30,Penjualan harian,750000,pemasukan\n'
    rows = preview_csv(csv_bytes)
    assert len(rows) == 2
    assert rows[0]['tanggal'] == '2026-10-30'
    assert rows[0]['deskripsi'] == 'Beli bahan baku'
    assert rows[0]['nominal'] == '250000.00' or rows[0]['nominal'] == '250000'
    assert rows[0]['tipe'] == 'pengeluaran'
    assert rows[0]['error'] == ''
    assert rows[1]['tipe'] == 'pemasukan'


# 2. Real-world seven-column transaction export (no transaction, type, staff, date, time, amount, notes)
def test_real_world_seven_column_export():
    csv_bytes = (
        b'no transaction,type,staff,date,time,amount,notes\n'
        b'I-PS261031004241,salah hitung modal,dwi,2026-10-31,06:53:30,12500,\n'
        b'E-PS261030011083,hanaang,dwi,2026-10-30,16:31:12,55000,\n'
        b'E-PS260512010705,es batu,silpi,2026-05-12,09:37:38,20000,","\n'
        b'I-PS260421004201,tambahan modal,chika,2026-04-21,08:35:24,1600000,titipan ibu\n'
    )
    analysis = analyze_csv_upload(csv_bytes)
    assert analysis['schema_analysis']['mappings']['tanggal'] == 'date'
    assert analysis['schema_analysis']['mappings']['deskripsi'] == 'type'
    assert analysis['schema_analysis']['mappings']['nominal'] == 'amount'
    assert analysis['schema_analysis']['type_inference_mode'] == 'id_prefix'

    rows = preview_csv(csv_bytes)
    assert len(rows) == 4
    # Check income derivation
    assert rows[0]['tipe'] == 'pemasukan'
    assert rows[0]['deskripsi'] == 'salah hitung modal'
    # Check expense derivation
    assert rows[1]['tipe'] == 'pengeluaran'
    assert rows[1]['deskripsi'] == 'hanaang'
    # Check notes combination: 'titipan ibu' appended
    assert rows[3]['tipe'] == 'pemasukan'
    assert 'titipan ibu' in rows[3]['deskripsi']


# 3. UTF-8 and UTF-8 BOM encoding
def test_utf8_bom_and_plain_utf8():
    raw_utf8 = 'tanggal,deskripsi,nominal,tipe\n2026-10-27,Tepung terigu,15000,pengeluaran\n'
    bom_bytes = raw_utf8.encode('utf-8-sig')
    enc, delim, text = detect_csv_dialect(bom_bytes)
    assert enc == 'utf-8-sig'
    assert delim == ','
    rows = preview_csv(bom_bytes)
    assert len(rows) == 1
    assert rows[0]['deskripsi'] == 'Tepung terigu'


# 4. Header variations (English, Indonesian, Mixed)
def test_header_variations():
    csv_bytes = b'Transaction_Date,Description,Total,Transaction_Type\n2026-10-27,Gaji Karyawan,1500000,Expense\n2026-10-28,Penjualan Tunai,500000,Income\n'
    analysis = analyze_csv_upload(csv_bytes)
    assert analysis['schema_analysis']['mappings']['tanggal'] == 'Transaction_Date'
    assert analysis['schema_analysis']['mappings']['deskripsi'] == 'Description'
    assert analysis['schema_analysis']['mappings']['nominal'] == 'Total'
    assert analysis['schema_analysis']['mappings']['tipe'] == 'Transaction_Type'

    rows = preview_csv(csv_bytes)
    assert rows[0]['tipe'] == 'pengeluaran'
    assert rows[1]['tipe'] == 'pemasukan'


# 5. Delimiter detection (semicolon, tab, pipe)
@pytest.mark.parametrize('delim', [';', '\t', '|'])
def test_delimiter_detection(delim):
    lines = [
        f'tanggal{delim}deskripsi{delim}nominal{delim}tipe',
        f'2026-10-27{delim}Beli Minyak{delim}25000{delim}pengeluaran',
        f'2026-10-28{delim}Omzet Harian{delim}50000{delim}pemasukan',
    ]
    content = '\n'.join(lines).encode('utf-8')
    enc, detected_delim, text = detect_csv_dialect(content)
    assert detected_delim == delim
    rows = preview_csv(content)
    assert len(rows) == 2
    assert rows[0]['deskripsi'] == 'Beli Minyak'


# 6. Currency and Indonesian number format normalization
def test_currency_and_number_formats():
    assert normalize_amount('12500') == Decimal('12500')
    assert normalize_amount('12.500') == Decimal('12500')
    assert normalize_amount('12.500,50') == Decimal('12500.50')
    assert normalize_amount('Rp 12.500') == Decimal('12500')
    assert normalize_amount('Rp. 1.250.000,00') == Decimal('1250000.00')
    assert normalize_amount('12,500.00') == Decimal('12500.00')

    with pytest.raises(ValueError, match='positif'):
        normalize_amount('-5000')

    with pytest.raises(ValueError, match='tidak valid'):
        normalize_amount('abc')


# 7. Date format normalization
def test_date_formats():
    assert normalize_date('2026-10-31') == '2026-10-31'
    assert normalize_date('31/10/2026') == '2026-10-31'
    assert normalize_date('31-10-2026') == '2026-10-31'
    assert normalize_date('2026/10/31') == '2026-10-31'
    assert normalize_date('2026-10-31 15:30:00') == '2026-10-31'

    with pytest.raises(ValueError):
        normalize_date('31/13/2026')  # Invalid month


# 8. Incorrect type-column assumptions (column 'type' with item names)
def test_type_column_with_item_names_not_confused_with_direction():
    df = pd.DataFrame({
        'date': ['2026-10-27', '2026-10-28'],
        'type': ['kopi susu gula aren', 'roti bakar coklat'],
        'amount': ['15000', '20000'],
    })
    engine = CsvSchemaEngine(df)
    schema = engine.analyze_schema()
    # 'type' contains coffee and bread, so it must map to deskripsi, not tipe
    assert schema['mappings']['deskripsi'] == 'type'
    assert schema['mappings']['tipe'] is None


# 9. I-/E- source identifiers mapping
def test_ie_prefix_type_derivation():
    df = pd.DataFrame({
        'no_ref': ['I-101', 'I-102', 'E-201', 'E-202'],
        'uraian': ['Penjualan A', 'Penjualan B', 'Beli Kertas', 'Beli Tinta'],
        'tgl': ['2026-10-27', '2026-10-27', '2026-10-28', '2026-10-28'],
        'jumlah': ['10000', '20000', '5000', '15000'],
    })
    engine = CsvSchemaEngine(df)
    schema = engine.analyze_schema()
    assert schema['type_inference_mode'] == 'id_prefix'

    valid, rejected = parse_and_validate_dataframe(
        df=df,
        mappings=schema['mappings'],
        type_inference_mode=schema['type_inference_mode'],
    )
    assert len(valid) == 4
    assert valid[0]['tipe'] == 'pemasukan'
    assert valid[1]['tipe'] == 'pemasukan'
    assert valid[2]['tipe'] == 'pengeluaran'
    assert valid[3]['tipe'] == 'pengeluaran'


# 10. Real 1,220-row file import simulation
def test_real_1220_row_file_parsing():
    path = r'C:\Users\frana\.gemini\antigravity\brain\3f601137-016f-4410-9eaa-43560cb7bd45\.user_uploaded\media_1793449146801_38ebe486.csv'
    if not os.path.exists(path):
        pytest.skip('Uploaded sample CSV not available on path')

    with open(path, 'rb') as f:
        content = f.read()

    rows = preview_csv(content)
    assert len(rows) == 1220
    assert all(r['error'] == '' for r in rows)
    incomes = [r for r in rows if r['tipe'] == 'pemasukan']
    expenses = [r for r in rows if r['tipe'] == 'pengeluaran']
    assert len(incomes) == 185
    assert len(expenses) == 1035


# 11. 5,000-row synthetic fixture processing
def test_synthetic_5000_row_fixture():
    rows = []
    rows.append('tanggal,deskripsi,nominal,tipe')
    for i in range(1, 5001):
        tp = 'pemasukan' if i % 4 == 0 else 'pengeluaran'
        rows.append(f'2026-10-15,Transaksi #{i},{10000 + i % 1000},{tp}')
    content = '\n'.join(rows).encode('utf-8')

    preview = preview_csv(content, max_rows=50000)
    assert len(preview) == 5000
    assert all(r['error'] == '' for r in preview)


# 12. Scalable multi-batch processing with idempotent batch IDs
def test_scalable_multi_batch_processing():
    fake_client = MagicMock()
    fake_client.rpc.return_value.execute.return_value.data = 500

    rows_to_import = [
        {
            'tanggal': '2026-10-27',
            'deskripsi': f'Barang {i}',
            'nominal': '50000',
            'tipe': 'pengeluaran',
            'kategori_id': '00000000-0000-0000-0000-000000000001',
        }
        for i in range(1250)
    ]

    with patch('services.smart_import_engine.verify_actor', return_value={'id': 'u1', 'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.smart_import_engine._validate_category'), \
         patch('services.smart_import_engine.privileged_client', return_value=fake_client), \
         patch('services.forecast_service.invalidate_forecast_cache') as mock_inv:

        res = execute_scalable_import('token', rows_to_import, batch_size=500)
        assert res['batches_completed'] == 3  # 500 + 500 + 250
        assert len(res['batch_ids']) == 3
        # Ensure batch IDs are consistent across chunks for the import operation
        assert len(set(res['batch_ids'])) == 1
        assert mock_inv.called


# 13. Rejected row reporting with row numbers and exact reasons
def test_rejected_row_reporting():
    csv_bytes = (
        b'tanggal,deskripsi,nominal,tipe\n'
        b'2026-10-27,Valid Item,15000,pengeluaran\n'
        b'bad-date,Invalid Date Item,20000,pengeluaran\n'
        b'2026-10-28,,30000,pengeluaran\n'
        b'2026-10-29,Negative Amount,-5000,pengeluaran\n'
    )
    df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')), dtype=str, keep_default_na=False)
    valid, rejected = parse_and_validate_dataframe(
        df=df,
        mappings={'tanggal': 'tanggal', 'deskripsi': 'deskripsi', 'nominal': 'nominal', 'tipe': 'tipe'},
    )
    assert len(valid) == 1
    assert len(rejected) == 3

    reasons = {r['baris']: r['alasan'] for r in rejected}
    assert 'Tanggal tidak valid' in reasons[3]
    assert 'Deskripsi tidak valid' in reasons[4]
    assert 'Nominal tidak valid' in reasons[5]


# 14. Duplicate detection (intra-file and database)
def test_duplicate_detection():
    valid_rows = [
        {'fingerprint': '2026-10-27|beli beras|50000|pengeluaran', 'source_id': 'TX01', 'deskripsi': 'beli beras'},
        {'fingerprint': '2026-10-27|beli beras|50000|pengeluaran', 'source_id': 'TX01', 'deskripsi': 'beli beras'},  # Intra-file dup
        {'fingerprint': '2026-10-28|beli telur|30000|pengeluaran', 'source_id': 'TX02', 'deskripsi': 'beli telur'},  # DB dup
        {'fingerprint': '2026-10-29|beli garam|5000|pengeluaran', 'source_id': 'TX03', 'deskripsi': 'beli garam'},   # New
    ]
    mock_db_tx = [
        {
            'tanggal': '2026-10-28',
            'deskripsi': 'beli telur',
            'nominal': 30000,
            'tipe': 'pengeluaran',
            'usulan': {'source_id': 'TX02'},
        }
    ]
    new_rows, dups = detect_duplicates('shop-1', valid_rows, existing_transactions=mock_db_tx)
    assert len(new_rows) == 2  # Row 0 and Row 3
    assert len(dups) == 2      # Row 1 (file dup) and Row 2 (db dup)


# 15. Cashier vs Owner approval rules
def test_cashier_vs_owner_approval_rules():
    fake_client = MagicMock()
    fake_client.rpc.return_value.execute.return_value.data = 1

    rows = [{
        'tanggal': '2026-10-27',
        'deskripsi': 'Uji Kasir',
        'nominal': '10000',
        'tipe': 'pengeluaran',
        'kategori_id': '00000000-0000-0000-0000-000000000001',
    }]

    # Cashier import: forecast cache should NOT be invalidated because transactions are pending approval
    with patch('services.smart_import_engine.verify_actor', return_value={'id': 'c1', 'toko_id': 's1', 'role': 'kasir'}), \
         patch('services.smart_import_engine._validate_category'), \
         patch('services.smart_import_engine.privileged_client', return_value=fake_client), \
         patch('services.forecast_service.invalidate_forecast_cache') as mock_inv:

        res = execute_scalable_import('cashier_token', rows)
        assert res['total_imported'] == 1
        assert not mock_inv.called  # Cashier transactions are pending approval, so forecast cache is preserved


# 16. Cross-store isolation
def test_cross_store_isolation():
    with patch('services.smart_import_engine.verify_actor', return_value={'id': 'u1', 'toko_id': 'STORE_A', 'role': 'pemilik_toko'}), \
         patch('services.smart_import_engine._validate_category', side_effect=ValueError('Kategori bukan milik toko Anda')):

        rows = [{
            'tanggal': '2026-10-27',
            'deskripsi': 'Item',
            'nominal': '10000',
            'tipe': 'pengeluaran',
            'kategori_id': 'STORE_B_CATEGORY',
        }]
        res = execute_scalable_import('token', rows)
        assert len(res['failed_batches']) == 1
        assert 'Kategori bukan milik toko Anda' in res['failed_batches'][0]['error']


# 17. Partial batch failure and precise reporting
def test_partial_batch_failure_reporting():
    call_count = 0

    def fake_rpc(name, params):
        nonlocal call_count
        call_count += 1
        mock_exec = MagicMock()
        if call_count == 2:
            mock_exec.execute.side_effect = Exception('Database connection lost on batch 2')
        else:
            mock_exec.execute.return_value.data = len(params['p_rows'])
        return mock_exec

    fake_client = MagicMock()
    fake_client.rpc = fake_rpc

    rows = [
        {
            'tanggal': '2026-10-27',
            'deskripsi': f'Barang #{i}',
            'nominal': '20000',
            'tipe': 'pengeluaran',
            'kategori_id': '00000000-0000-0000-0000-000000000001',
        }
        for i in range(1200)
    ]

    with patch('services.smart_import_engine.verify_actor', return_value={'id': 'u1', 'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.smart_import_engine._validate_category'), \
         patch('services.smart_import_engine.privileged_client', return_value=fake_client):

        res = execute_scalable_import('token', rows, batch_size=500)
        # Batch 1 succeeded (500), Batch 2 failed
        assert res['total_imported'] == 500
        assert res['batches_completed'] == 1
        assert len(res['failed_batches']) == 1
        assert res['failed_batches'][0]['batch_index'] == 2
        assert 'Database connection lost' in res['failed_batches'][0]['error']


# 18. Idempotent import and retry handling
def test_idempotent_import_retry():
    rows = [
        {
            'baris': 2,
            'tanggal': '2026-10-27',
            'deskripsi': 'Item Pertama',
            'nominal': '25000',
            'tipe': 'pengeluaran',
            'source_id': 'TX_ID_001',
            'fingerprint': '2026-10-27|item pertama|25000|pengeluaran',
        }
    ]
    # First run: completely new
    new_rows, dups = detect_duplicates('s1', rows, existing_transactions=[])
    assert len(new_rows) == 1
    assert len(dups) == 0

    # Second run / retry: existing transaction returned from database
    existing_db = [{
        'tanggal': '2026-10-27',
        'deskripsi': 'Item Pertama',
        'nominal': 25000,
        'tipe': 'pengeluaran',
        'usulan': {'source_id': 'TX_ID_001'},
    }]
    new_rows2, dups2 = detect_duplicates('s1', rows, existing_transactions=existing_db)
    assert len(new_rows2) == 0
    assert len(dups2) == 1
    assert dups2[0]['alasan_duplikat'] == "ID transaksi 'TX_ID_001' sudah tersimpan di database toko"

