from decimal import Decimal
from services.report_service import metrics,narrative,export_csv
from unittest.mock import patch
from ai.ollama_client import LLMUnavailable


def test_capital_excluded_from_operating_cash_flow():
    cats=[{'id':'modal','nama':'Tambahan Modal','bukan_operasional':True},
          {'id':'jualan','nama':'Penjualan','bukan_operasional':False},
          {'id':'bahan','nama':'Bahan Baku','bukan_operasional':False}]
    rows=[{'tanggal':'2026-01-01','tipe':'pemasukan','nominal':'200','kategori_id':'modal'},
          {'tanggal':'2026-01-02','tipe':'pemasukan','nominal':'100','kategori_id':'jualan'},
          {'tanggal':'2026-01-03','tipe':'pengeluaran','nominal':'60','kategori_id':'bahan'}]
    m=metrics(rows,cats)
    assert m['arus_kas_operasional']==Decimal(40)
    assert m['arus_kas_total']==Decimal(240)
    assert m['omzet']==Decimal(100)

def test_llm_narrative_fallback():
    data={'pemasukan':10,'pengeluaran':3,'arus_kas_operasional':7,'arus_kas_total':7}
    with patch('services.report_service.ask_json',side_effect=LLMUnavailable('offline')):
        assert 'Pemasukan' in narrative(data)

def test_llm_narrative_rejects_made_up_amount():
    data={'pemasukan':10,'pengeluaran':3,'arus_kas_operasional':7,'arus_kas_total':7}
    with patch('services.report_service.ask_json',return_value={'teks':'semua 100 juta','angka':{'pemasukan':100000000}}):
        assert 'semua 100 juta' not in narrative(data)

def test_csv_has_header():
    assert export_csv([],[]).decode('utf-8-sig').startswith('tanggal,deskripsi,nominal,tipe')

def test_period_rows_filter():
    from services.report_service import period_rows
    txs = [
        {'tanggal': '2026-01-05', 'status_persetujuan': 'disetujui'},
        {'tanggal': '2026-02-10', 'status_persetujuan': 'disetujui'},
        {'tanggal': '2026-03-15', 'status_persetujuan': 'disetujui'},
        {'tanggal': '2026-02-01', 'status_persetujuan': 'pending'},
    ]
    with patch('services.report_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.report_service.transactions', return_value=txs):
        filtered = period_rows('tok', start='2026-01-10', end='2026-02-28')
        assert len(filtered) == 1
        assert filtered[0]['tanggal'] == '2026-02-10'

def test_export_excel():
    from services.report_service import export_excel
    cats = [{'id': 'c1', 'nama': 'Bahan Baku'}]
    rows = [{'tanggal': '2026-01-01', 'deskripsi': 'Tepung', 'nominal': '50000', 'tipe': 'pengeluaran', 'kategori_id': 'c1', 'status_persetujuan': 'disetujui'}]
    excel_bytes = export_excel(rows, cats)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0
    assert excel_bytes.startswith(b'PK\x03\x04')
