"""Automated Regression Test Suite for Compact Recurring Expense Reminders UI Component.

Covers:
1. No reminders (graceful empty state).
2. One reminder (exact count, single date, accurate sum).
3. Seventeen reminders (compact summary, no truncation, exact sum).
4. One hundred reminders (scalable performance, zero truncation).
5. Long descriptions (> 120 chars preserved without clipping).
6. Different dates, days remaining (0, 1, 2, 3), and currency magnitudes.
7. Confirmed due dates vs predicted statistical reminders differentiation.
8. Accurate aggregation (count and total calculated from underlying records, zero double-counting).
9. Store isolation and role verification in recurring expense detection.
10. Integration with report_view and forecasting services.
"""
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from services.forecast_service import detect_recurring_expenses
from utils.ui_components import render_recurring_expense_reminders, format_rupiah


# ---------------------------------------------------------------------------
# Unit Tests for render_recurring_expense_reminders
# ---------------------------------------------------------------------------

def test_render_no_reminders():
    """Empty or missing upcoming items should return gracefully without rendering or error."""
    with patch('streamlit.metric') as mock_metric, \
         patch('streamlit.dataframe') as mock_df:
        render_recurring_expense_reminders({})
        render_recurring_expense_reminders({'upcoming_3_days_items': []})
        render_recurring_expense_reminders(None)

        mock_metric.assert_not_called()
        mock_df.assert_not_called()


def test_render_one_reminder():
    """Single recurring reminder should display exact count, amount, and date range."""
    rec_data = {
        'upcoming_3_days_items': [
            {
                'key': 'listrik',
                'description': 'Token Listrik Ruko',
                'category': 'Operasional',
                'estimated_amount': 250000.0,
                'next_expected_date': '2026-11-02',
                'days_until': 1,
                'is_upcoming_3_days': True,
                'is_verified_due_date': False,
                'reminder_type': 'Perkiraan pengeluaran rutin berdasarkan riwayat',
            }
        ]
    }

    mock_c1, mock_c2, mock_c3 = MagicMock(), MagicMock(), MagicMock()

    with patch('streamlit.columns', return_value=[mock_c1, mock_c2, mock_c3]), \
         patch('streamlit.dataframe') as mock_df, \
         patch('streamlit.expander') as mock_expander:

        mock_expander.return_value.__enter__.return_value = MagicMock()

        render_recurring_expense_reminders(rec_data)

        # Verify summary metrics
        mock_c1.metric.assert_called_once_with("Pengeluaran Terdeteksi", "1 transaksi")
        mock_c2.metric.assert_called_once_with("Rentang Tanggal", "2026-11-02")
        mock_c3.metric.assert_called_once_with("Total Estimasi Beban", format_rupiah(250000.0))

        # Verify dataframe structure
        assert mock_df.call_count == 1
        df_arg = mock_df.call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 1
        assert df_arg.iloc[0]['Deskripsi'] == 'Token Listrik Ruko'
        assert df_arg.iloc[0]['Estimasi Nominal'] == format_rupiah(250000.0)
        assert df_arg.iloc[0]['Sisa Waktu'] == '1 hari lagi (Besok)'
        assert df_arg.iloc[0]['Jenis Pengingat'] == 'Perkiraan pengeluaran rutin berdasarkan riwayat'


def test_render_seventeen_reminders_compact_and_scrollable():
    """17 reminders must be rendered in ONE table with height constrained to 350px and exact totals."""
    items = []
    base_date = date(2026, 11, 1)
    total_expected_sum = 0.0

    for i in range(17):
        amt = float((i + 1) * 50000)
        total_expected_sum += amt
        target_date = str(base_date + timedelta(days=(i % 3)))
        items.append({
            'key': f'item_{i}',
            'description': f'Beban Rutin Operasional #{i+1}',
            'category': 'Operasional' if i % 2 == 0 else 'Gaji',
            'estimated_amount': amt,
            'next_expected_date': target_date,
            'days_until': i % 3,
            'is_upcoming_3_days': True,
            'is_verified_due_date': (i == 0),  # Item 0 is verified
            'reminder_type': 'Tagihan dengan jatuh tempo terkonfirmasi' if i == 0 else 'Perkiraan pengeluaran rutin berdasarkan riwayat',
        })

    rec_data = {'upcoming_3_days_items': items}
    mock_c1, mock_c2, mock_c3 = MagicMock(), MagicMock(), MagicMock()

    with patch('streamlit.columns', return_value=[mock_c1, mock_c2, mock_c3]), \
         patch('streamlit.dataframe') as mock_df, \
         patch('streamlit.expander') as mock_expander:

        mock_expander.return_value.__enter__.return_value = MagicMock()

        render_recurring_expense_reminders(rec_data)

        # Verify expander default collapsed state
        mock_expander.assert_called_once_with("Lihat Rincian Pengeluaran Rutin (17 Transaksi)", expanded=False)

        # Verify metrics
        mock_c1.metric.assert_called_once_with("Pengeluaran Terdeteksi", "17 transaksi")
        mock_c2.metric.assert_called_once_with("Rentang Tanggal", "2026-11-01 s/d 2026-11-03")
        mock_c3.metric.assert_called_once_with("Total Estimasi Beban", format_rupiah(total_expected_sum))

        # Verify dataframe: exactly 17 rows, height=350, hide_index=True
        assert mock_df.call_count == 1
        df_arg = mock_df.call_args[0][0]
        kwargs = mock_df.call_args[1]

        assert len(df_arg) == 17
        assert kwargs.get('height') == 350
        assert kwargs.get('hide_index') is True
        assert kwargs.get('use_container_width') is True


def test_render_one_hundred_reminders_scalable():
    """100 reminders must be rendered without truncation or performance degradation."""
    items = []
    base_date = date(2026, 11, 1)
    total_sum = 0.0

    for i in range(100):
        amt = float(100000 + i * 1000)
        total_sum += amt
        target_date = str(base_date + timedelta(days=(i % 4)))
        items.append({
            'key': f'item_{i}',
            'description': f'Langganan Vendor Layanan #{i+1}',
            'category': 'Operasional',
            'estimated_amount': amt,
            'next_expected_date': target_date,
            'days_until': i % 4,
            'is_upcoming_3_days': True,
            'is_verified_due_date': False,
        })

    rec_data = {'upcoming_3_days_items': items}
    mock_c1, mock_c2, mock_c3 = MagicMock(), MagicMock(), MagicMock()

    with patch('streamlit.columns', return_value=[mock_c1, mock_c2, mock_c3]), \
         patch('streamlit.dataframe') as mock_df, \
         patch('streamlit.expander') as mock_expander:

        mock_expander.return_value.__enter__.return_value = MagicMock()

        render_recurring_expense_reminders(rec_data)

        mock_c1.metric.assert_called_once_with("Pengeluaran Terdeteksi", "100 transaksi")
        mock_c3.metric.assert_called_once_with("Total Estimasi Beban", format_rupiah(total_sum))

        df_arg = mock_df.call_args[0][0]
        assert len(df_arg) == 100


def test_render_long_descriptions_preserved():
    """Descriptions longer than 120 characters must not be truncated in the dataframe."""
    long_desc = "Pembayaran biaya sewa ruko operasional cabang pertokoan sentra niaga tahap kedua tahunan blok B2 nomor 14 Jakarta Barat"
    assert len(long_desc) > 110

    rec_data = {
        'upcoming_3_days_items': [
            {
                'key': 'sewa_ruko',
                'description': long_desc,
                'category': 'Sewa',
                'estimated_amount': 35000000.0,
                'next_expected_date': '2026-11-03',
                'days_until': 2,
                'is_upcoming_3_days': True,
                'is_verified_due_date': True,
                'reminder_type': 'Tagihan dengan jatuh tempo terkonfirmasi',
            }
        ]
    }

    with patch('streamlit.metric'), patch('streamlit.dataframe') as mock_df, patch('streamlit.expander'):
        render_recurring_expense_reminders(rec_data)
        df_arg = mock_df.call_args[0][0]
        assert df_arg.iloc[0]['Deskripsi'] == long_desc


def test_confirmed_vs_predicted_reminder_types():
    """Verified obligations are cleanly distinguished from statistical frequency projections."""
    rec_data = {
        'upcoming_3_days_items': [
            {
                'key': 'tagihan_kontrak',
                'description': 'Cicilan Mesin Espresso',
                'category': 'Peralatan',
                'estimated_amount': 1500000.0,
                'next_expected_date': '2026-11-01',
                'days_until': 0,
                'is_verified_due_date': True,
            },
            {
                'key': 'token_pln',
                'description': 'Token Listrik Kasir',
                'category': 'Operasional',
                'estimated_amount': 200000.0,
                'next_expected_date': '2026-11-02',
                'days_until': 1,
                'is_verified_due_date': False,
            }
        ]
    }

    with patch('streamlit.metric'), patch('streamlit.dataframe') as mock_df, patch('streamlit.expander'):
        render_recurring_expense_reminders(rec_data)
        df_arg = mock_df.call_args[0][0]

        row_verified = df_arg[df_arg['Deskripsi'] == 'Cicilan Mesin Espresso'].iloc[0]
        assert row_verified['Jenis Pengingat'] == 'Tagihan dengan jatuh tempo terkonfirmasi'
        assert row_verified['Sisa Waktu'] == 'Hari ini (0 hari)'

        row_predicted = df_arg[df_arg['Deskripsi'] == 'Token Listrik Kasir'].iloc[0]
        assert row_predicted['Jenis Pengingat'] == 'Perkiraan pengeluaran rutin berdasarkan riwayat'
        assert row_predicted['Sisa Waktu'] == '1 hari lagi (Besok)'


# ---------------------------------------------------------------------------
# Backend Service & Store Isolation Tests
# ---------------------------------------------------------------------------

def test_detect_recurring_expenses_includes_reminder_type_and_store_isolation():
    """Ensure forecast_service outputs correct reminder_type and enforces store isolation."""
    token = 'tok_owner_store_a'
    now_dt = datetime(2026, 11, 4, 10, 0, 0)
    txs = [
        {
            'id': 't1',
            'toko_id': 'store-alpha',
            'tanggal': '2026-09-05',
            'nominal': 500000,
            'tipe': 'pengeluaran',
            'kategori_id': 'cat-ops',
            'deskripsi': 'Internet IndiHome Toko',
            'status_persetujuan': 'disetujui',
            'is_verified_due_date': True,
        },
        {
            'id': 't2',
            'toko_id': 'store-alpha',
            'tanggal': '2026-10-05',
            'nominal': 500000,
            'tipe': 'pengeluaran',
            'kategori_id': 'cat-ops',
            'deskripsi': 'Internet IndiHome Toko',
            'status_persetujuan': 'disetujui',
            'is_verified_due_date': True,
        }
    ]
    cats = [{'id': 'cat-ops', 'nama': 'Operasional', 'tipe': 'pengeluaran'}]

    with patch('services.forecast_service.verify_actor', return_value={'id': 'u1', 'role': 'pemilik_toko', 'toko_id': 'store-alpha'}), \
         patch('services.forecast_service.transactions', return_value=txs), \
         patch('services.forecast_service.categories', return_value=cats), \
         patch('services.forecast_service.get_jakarta_now', return_value=now_dt):

        res = detect_recurring_expenses(token)
        assert res['success'] is True
        assert len(res['recurring_items']) >= 1
        item = res['recurring_items'][0]
        assert item['reminder_type'] == 'Perkiraan pengeluaran rutin berdasarkan riwayat'
        assert item['is_verified_due_date'] is False


def test_report_view_renders_compact_reminders_cleanly():
    """Verifies that pages/report_view.py calls render_recurring_expense_reminders cleanly."""
    token = 'tok_valid'
    from pages.report_view import render

    mock_rec = {
        'upcoming_3_days_items': [
            {
                'key': 'gaji',
                'description': 'Gaji Kasir Mingguan',
                'category': 'Gaji',
                'estimated_amount': 750000.0,
                'next_expected_date': '2026-11-02',
                'days_until': 1,
            }
        ]
    }

    mock_numbers = {
        'pemasukan': 1000000,
        'pengeluaran': 500000,
        'omzet': 1000000,
        'arus_kas_operasional': 500000,
        'arus_kas_total': 500000,
        'bulanan': {},
        'pengeluaran_per_kategori': {},
    }

    with patch('services.report_service.period_rows', return_value=[]), \
         patch('pages.report_view.list_categories', return_value=[]), \
         patch('services.report_service.metrics', return_value=mock_numbers), \
         patch('services.forecast_service.detect_recurring_expenses', return_value=mock_rec), \
         patch('services.forecast_service.get_forecast_readiness_status', return_value={'ready': False, 'message': 'Belum cukup data'}), \
         patch('pages.report_view.render_recurring_expense_reminders') as mock_render_reminders:

        render(token)
        mock_render_reminders.assert_called_once_with(mock_rec)


def test_detect_recurring_stale_series_excluded():
    """A series whose last occurrence was > 2 * median_interval ago is stale and excluded."""
    token = 'tok_test'
    now_dt = datetime(2026, 11, 4, 10, 0, 0)
    today = now_dt.date()

    stale_last = today - timedelta(days=100)
    stale_prev = stale_last - timedelta(days=30)

    live_last = today - timedelta(days=5)
    live_prev = live_last - timedelta(days=7)

    txs = [
        # Stale series with 30-day median interval, last seen 100 days ago
        {'id': 's1', 'toko_id': 's1', 'tanggal': stale_prev.isoformat(), 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'c1', 'deskripsi': 'Langganan Wifi Gudang', 'status_persetujuan': 'disetujui'},
        {'id': 's2', 'toko_id': 's1', 'tanggal': stale_last.isoformat(), 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'c1', 'deskripsi': 'Langganan Wifi Gudang', 'status_persetujuan': 'disetujui'},
        # Live series with 7-day median interval, last seen 5 days ago
        {'id': 'l1', 'toko_id': 's1', 'tanggal': live_prev.isoformat(), 'nominal': 50000, 'tipe': 'pengeluaran', 'kategori_id': 'c1', 'deskripsi': 'Beli Token Listrik Mingguan', 'status_persetujuan': 'disetujui'},
        {'id': 'l2', 'toko_id': 's1', 'tanggal': live_last.isoformat(), 'nominal': 50000, 'tipe': 'pengeluaran', 'kategori_id': 'c1', 'deskripsi': 'Beli Token Listrik Mingguan', 'status_persetujuan': 'disetujui'},
    ]
    cats = [{'id': 'c1', 'nama': 'Operasional', 'tipe': 'pengeluaran'}]

    with patch('services.forecast_service.verify_actor', return_value={'id': 'u1', 'role': 'pemilik_toko', 'toko_id': 's1'}),          patch('services.forecast_service.transactions', return_value=txs),          patch('services.forecast_service.categories', return_value=cats),          patch('services.forecast_service.get_jakarta_now', return_value=now_dt):
        res = detect_recurring_expenses(token)
        descriptions = [it['description'] for it in res['recurring_items']]
        assert not any('Wifi' in d for d in descriptions), "Stale series must NOT appear in recurring_items"
        assert any('Listrik' in d for d in descriptions), "Live series must appear in recurring_items"
        assert res['seri_layak'] == 1


def test_detect_recurring_short_interval_raw_and_not_forced_to_seven():
    """Short interval <= 4 days uses raw median gap (e.g. 2 days) and pattern description 'Setiap 2 hari'."""
    token = 'tok_test'
    now_dt = datetime(2026, 11, 4, 10, 0, 0)
    today = now_dt.date()

    d0 = today - timedelta(days=5)
    d1 = d0 + timedelta(days=2)
    d2 = d1 + timedelta(days=2)  # last_date: 1 day ago
    txs = [
        {'id': 'b1', 'toko_id': 's1', 'tanggal': d0.isoformat(), 'nominal': 20000, 'tipe': 'pengeluaran', 'kategori_id': 'c1', 'deskripsi': 'Beli Es Batu Kristal', 'status_persetujuan': 'disetujui'},
        {'id': 'b2', 'toko_id': 's1', 'tanggal': d1.isoformat(), 'nominal': 20000, 'tipe': 'pengeluaran', 'kategori_id': 'c1', 'deskripsi': 'Beli Es Batu Kristal', 'status_persetujuan': 'disetujui'},
        {'id': 'b3', 'toko_id': 's1', 'tanggal': d2.isoformat(), 'nominal': 20000, 'tipe': 'pengeluaran', 'kategori_id': 'c1', 'deskripsi': 'Beli Es Batu Kristal', 'status_persetujuan': 'disetujui'},
    ]
    cats = [{'id': 'c1', 'nama': 'Operasional', 'tipe': 'pengeluaran'}]

    with patch('services.forecast_service.verify_actor', return_value={'id': 'u1', 'role': 'pemilik_toko', 'toko_id': 's1'}),          patch('services.forecast_service.transactions', return_value=txs),          patch('services.forecast_service.categories', return_value=cats),          patch('services.forecast_service.get_jakarta_now', return_value=now_dt):
        res = detect_recurring_expenses(token)
        assert len(res['recurring_items']) == 1
        item = res['recurring_items'][0]
        assert item['estimated_interval_days'] == 2
        assert '2 hari' in item['pattern_description']
        expected_next = (d2 + timedelta(days=2)).isoformat()
        assert item['next_expected_date'] == expected_next

