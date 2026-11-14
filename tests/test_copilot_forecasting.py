"""Automated validation tests for Prophet Forecasting integrated with Owner AI Financial Copilot.

Tests:
1. Security, role enforcement (owner-only, cashier denial), and tenant isolation.
2. Data readiness status and empty states for insufficient data.
3. Prophet time-series model fitting and projection clipping (non-negative).
4. Accuracy holdout backtesting (MAE, RMSE, MAPE, WAPE, and zero-actual edge cases).
5. PRD FR-10: Recurring-expense detection and 3-day reminder window.
6. Cash sufficiency comparison and trend analysis with mandatory disclaimers.
7. Indonesian intent routing for all 16 required forecasting questions and follow-ups.
8. Cache invalidation consistency.
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock

from services.auth_service import AccessError
from ai.copilot_schema import ToolName, PlannerDecision, ConversationContext, FilterParams, DateRangeParams
from ai.copilot_planner import plan_stage_a_deterministic, plan_intent
from ai.copilot_tools import dispatch_tool, TOOL_REGISTRY
from services import forecast_service as fs
from services.chat_service import handle_chat_message


def make_mock_verify(actor):
    def _verify(token, allowed_roles=None):
        if allowed_roles and actor.get('role') not in allowed_roles:
            raise AccessError(f"Akses ditolak untuk peran {actor.get('role')}")
        return actor
    return _verify


@pytest.fixture
def owner_actor():
    return {
        'id': 'owner-user-001',
        'nama': 'Pak Bambang',
        'email': 'bambang@tokoberkah.com',
        'role': 'pemilik_toko',
        'toko_id': 'toko-berkah-001'
    }


@pytest.fixture
def cashier_actor():
    return {
        'id': 'cashier-user-001',
        'nama': 'Siti Barista',
        'email': 'siti@tokoberkah.com',
        'role': 'kasir',
        'toko_id': 'toko-berkah-001'
    }


@pytest.fixture
def mock_categories():
    return [
        {'id': 'cat-1', 'nama': 'Listrik & Utilitas', 'tipe': 'pengeluaran', 'bukan_operasional': False},
        {'id': 'cat-2', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran', 'bukan_operasional': False},
        {'id': 'cat-3', 'nama': 'Sewa Toko', 'tipe': 'pengeluaran', 'bukan_operasional': False},
        {'id': 'cat-4', 'nama': 'Gaji Pegawai', 'tipe': 'pengeluaran', 'bukan_operasional': False},
        {'id': 'cat-5', 'nama': 'Penjualan Produk', 'tipe': 'pemasukan', 'bukan_operasional': False},
    ]


@pytest.fixture
def synthetic_history():
    """Generates 30 days of approved expense transactions with recurring electricity & daily shopping."""
    base = date(2026, 9, 1)
    txs = []
    # Daily ingredients purchases
    for day in range(30):
        d = (base + timedelta(days=day)).isoformat()
        amt = 50000 + (day % 5) * 10000
        txs.append({
            'id': f'tx-ing-{day}',
            'toko_id': 'toko-berkah-001',
            'tanggal': d,
            'nominal': amt,
            'tipe': 'pengeluaran',
            'kategori_id': 'cat-2',
            'deskripsi': f'Bahan baku harian hari ke-{day}',
            'status_persetujuan': 'disetujui'
        })
    # Monthly utility bill on Sep 5
    txs.append({
        'id': 'tx-util-1',
        'toko_id': 'toko-berkah-001',
        'tanggal': '2026-09-05',
        'nominal': 350000,
        'tipe': 'pengeluaran',
        'kategori_id': 'cat-1',
        'deskripsi': 'Tagihan listrik PLN toko',
        'status_persetujuan': 'disetujui'
    })
    # Monthly utility bill on Oct 5
    txs.append({
        'id': 'tx-util-2',
        'toko_id': 'toko-berkah-001',
        'tanggal': '2026-10-05',
        'nominal': 365000,
        'tipe': 'pengeluaran',
        'kategori_id': 'cat-1',
        'deskripsi': 'Tagihan listrik PLN toko',
        'status_persetujuan': 'disetujui'
    })
    # Income records to establish cash balance
    for day in range(0, 30, 2):
        d = (base + timedelta(days=day)).isoformat()
        txs.append({
            'id': f'tx-inc-{day}',
            'toko_id': 'toko-berkah-001',
            'tanggal': d,
            'nominal': 250000,
            'tipe': 'pemasukan',
            'kategori_id': 'cat-5',
            'deskripsi': 'Penjualan harian',
            'status_persetujuan': 'disetujui'
        })
    return txs


# ---------------------------------------------------------------------------
# 1. Security & RBAC Enforcement Tests
# ---------------------------------------------------------------------------

def test_forecasting_denied_to_cashier(cashier_actor):
    """Cashiers must strictly be denied access to all forecasting functions and tools."""
    token = 'tok_cashier'
    decision = PlannerDecision(tool_name=ToolName.GET_EXPENSE_FORECAST)
    mock_v = make_mock_verify(cashier_actor)

    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.chat_service.verify_actor', side_effect=mock_v):

        # 1. Directly in forecast_service
        with pytest.raises(AccessError):
            fs.train_and_forecast_expenses(token)

        with pytest.raises(AccessError):
            fs.get_forecast_readiness_status(token)

        with pytest.raises(AccessError):
            fs.evaluate_forecast_accuracy(token)

        with pytest.raises(AccessError):
            fs.detect_recurring_expenses(token)

        # 2. Via copilot tool dispatcher
        res = dispatch_tool(token, decision)
        assert res['success'] is False
        assert "Akses ditolak" in res['message']

        # 3. Via chat assistant handle_chat_message
        chat_res = handle_chat_message(token, "Berapa prediksi pengeluaran bulan depan?")
        assert "Pemilik Toko" in chat_res['message']
        assert "kasir" not in chat_res['message'].lower() or "hanya dapat diakses oleh" in chat_res['message'].lower()


def test_forecasting_tenant_isolation(owner_actor, synthetic_history):
    """Forecasting must strictly query transactions matching the authenticated owner's store ID."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions') as mock_tx, \
         patch('services.forecast_service.categories', return_value=[]):
        mock_tx.return_value = synthetic_history

        fs.get_forecast_readiness_status(token)
        mock_tx.assert_called_with('toko-berkah-001')


# ---------------------------------------------------------------------------
# 2. Readiness and Empty States Tests
# ---------------------------------------------------------------------------

def test_forecast_readiness_insufficient_data(owner_actor):
    """When a store has fewer than 5 transactions, readiness returns False with clear missing parts."""
    token = 'tok_owner'
    sparse_txs = [
        {'id': '1', 'tanggal': '2026-10-01', 'nominal': 50000, 'tipe': 'pengeluaran', 'status_persetujuan': 'disetujui'},
        {'id': '2', 'tanggal': '2026-10-02', 'nominal': 75000, 'tipe': 'pengeluaran', 'status_persetujuan': 'disetujui'}
    ]
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=sparse_txs), \
         patch('services.forecast_service.categories', return_value=[]):

        readiness = fs.get_forecast_readiness_status(token)
        assert readiness['ready'] is False
        assert readiness['total_expenses_count'] == 2
        assert "Belum Mencukupi" in readiness['message']
        assert "minimal 5" in readiness['message']

        fcst = fs.train_and_forecast_expenses(token)
        assert fcst['success'] is False
        assert fcst['error'] == 'INSUFFICIENT_DATA'


def test_forecast_readiness_sufficient_data(owner_actor, synthetic_history):
    """When a store has 30+ transactions across 30 days, readiness returns True."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=[]):

        readiness = fs.get_forecast_readiness_status(token)
        assert readiness['ready'] is True
        assert readiness['total_expenses_count'] >= 30
        assert readiness['distinct_expense_days'] >= 30
        assert "Memenuhi syarat" in readiness['message']


# ---------------------------------------------------------------------------
# 3. Prophet Model Projections & Evaluation Tests
# ---------------------------------------------------------------------------

def test_prophet_model_fitting_and_non_negative_bounds(owner_actor, synthetic_history):
    """Prophet model fits on daily aggregated data and never projects negative expenses."""
    token = 'tok_owner'
    fs.invalidate_forecast_cache()
    mock_v = make_mock_verify(owner_actor)

    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=[]):

        result = fs.train_and_forecast_expenses(token, days_ahead=14)
        assert result['success'] is True
        assert result['days_ahead'] == 14
        assert result['total_predicted_expense'] > 0
        assert result['average_daily_predicted'] > 0

        # Verify uncertainty bounds and non-negative clipping
        for p in result['projections']:
            assert p['yhat'] >= 0.0
            assert p['yhat_lower'] >= 0.0
            assert p['yhat_upper'] >= 0.0
            assert p['yhat_upper'] >= p['yhat_lower']


def test_accuracy_holdout_backtest_metrics(owner_actor, synthetic_history):
    """Accuracy evaluation computes MAE, RMSE, WAPE, and MAPE honestly without false claims."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=[]):

        eval_res = fs.evaluate_forecast_accuracy(token, test_days=7)
        assert eval_res['success'] is True
        assert eval_res['ready'] is True
        assert eval_res['mae'] is not None and eval_res['mae'] >= 0.0
        assert eval_res['rmse'] is not None and eval_res['rmse'] >= 0.0
        assert eval_res['mape'] is not None

        # Verify strict target condition and multi-model comparison output
        assert "PRD" not in eval_res['message']
        assert "- **Status Target (<20% MAPE):**" in eval_res['message']
        assert eval_res.get('best_model') in fs.CANDIDATE_MODEL_ORDER
        assert len(eval_res.get('model_comparison', [])) >= 4
        msg_lower = eval_res['message'].lower()
        if "belum dapat disimpulkan" in msg_lower:
            assert eval_res['target_met'] is False
        elif eval_res['mape'] < 20.0:
            assert eval_res['target_met'] is True
            assert "tercapai" in msg_lower
        else:
            assert eval_res['target_met'] is False
            assert "belum tercapai" in msg_lower


def test_accuracy_zero_division_safety(owner_actor):
    """When test period days have 0 actual expenses, MAPE handles zero division safely."""
    token = 'tok_owner'
    txs = []
    base = date(2026, 9, 1)
    for day in range(8):
        txs.append({
            'id': f'tx-{day}',
            'toko_id': 'toko-berkah-001',
            'tanggal': (base + timedelta(days=day)).isoformat(),
            'nominal': 100000,
            'tipe': 'pengeluaran',
            'status_persetujuan': 'disetujui'
        })
    # Add dummy 15th day transaction so total span is 15 days
    txs.append({
        'id': 'tx-15',
        'toko_id': 'toko-berkah-001',
        'tanggal': (base + timedelta(days=14)).isoformat(),
        'nominal': 0,
        'tipe': 'pengeluaran',
        'status_persetujuan': 'disetujui'
    })

    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=txs), \
         patch('services.forecast_service.categories', return_value=[]):

        eval_res = fs.evaluate_forecast_accuracy(token, test_days=5)
        assert eval_res['success'] is True
        if eval_res['mape'] is None:
            assert "pembagian dengan nol dicegah" in eval_res['mape_note']


def test_accuracy_target_met_requires_minimum_active_days(owner_actor):
    """When test period has < 5 active (non-zero) days, target_met is False even if MAPE < 20%."""
    token = 'tok_owner'
    base = date(2026, 9, 1)
    txs = []
    # 18 training days with constant 100,000 expense
    for day in range(18):
        txs.append({
            'id': f'tx-train-{day}',
            'toko_id': 'toko-berkah-001',
            'tanggal': (base + timedelta(days=day)).isoformat(),
            'nominal': 100000,
            'tipe': 'pengeluaran',
            'status_persetujuan': 'disetujui'
        })
    # Test period (last 7 days of 25 days): only 2 non-zero days (days 18 and 19)
    for day in range(18, 20):
        txs.append({
            'id': f'tx-test-{day}',
            'toko_id': 'toko-berkah-001',
            'tanggal': (base + timedelta(days=day)).isoformat(),
            'nominal': 100000,
            'tipe': 'pengeluaran',
            'status_persetujuan': 'disetujui'
        })
    # Add dummy 0 transaction on day 24 so date range spans full 25 days
    txs.append({
        'id': 'tx-test-24',
        'toko_id': 'toko-berkah-001',
        'tanggal': (base + timedelta(days=24)).isoformat(),
        'nominal': 0,
        'tipe': 'pengeluaran',
        'status_persetujuan': 'disetujui'
    })

    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v),          patch('services.forecast_service.transactions', return_value=txs),          patch('services.forecast_service.categories', return_value=[]):

        eval_res = fs.evaluate_forecast_accuracy(token, test_days=7)
        assert eval_res['success'] is True
        assert eval_res['ready'] is True
        assert eval_res['mape'] is not None
        assert eval_res['mape'] < 20.0
        assert eval_res['target_met'] is False
        assert "belum dapat disimpulkan" in eval_res['message'].lower()


# ---------------------------------------------------------------------------
# 4. Recurring Expenses & FR-10 Reminders Tests
# ---------------------------------------------------------------------------

def test_detect_recurring_expenses_and_3_day_reminder(owner_actor, mock_categories):
    """Recurring monthly utility is detected, with 3-day reminder triggered when expected date is within 3 days.

    Note: In v7.1 backtest gate, a 2-occurrence candidate (< 4 occurrences) is preserved as unproven
    with terbukti_uji == False, labeled 'belum teruji', and excluded from gate-passing cakupan_nominal.
    """
    token = 'tok_owner'
    now_dt = datetime(2026, 11, 4, 10, 0, 0)
    txs = [
        {
            'id': 'u1',
            'toko_id': 'toko-berkah-001',
            'tanggal': '2026-09-05',
            'nominal': 350000,
            'tipe': 'pengeluaran',
            'kategori_id': 'cat-1',
            'deskripsi': 'Listrik PLN Toko',
            'status_persetujuan': 'disetujui'
        },
        {
            'id': 'u2',
            'toko_id': 'toko-berkah-001',
            'tanggal': '2026-10-05',
            'nominal': 360000,
            'tipe': 'pengeluaran',
            'kategori_id': 'cat-1',
            'deskripsi': 'Listrik PLN Toko',
            'status_persetujuan': 'disetujui'
        }
    ]

    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=txs), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.get_jakarta_now', return_value=now_dt):

        rec = fs.detect_recurring_expenses(token)
        assert rec['success'] is True
        assert len(rec['recurring_items']) >= 1
        item = rec['recurring_items'][0]
        assert 'Listrik' in item['description']
        assert item['is_upcoming_3_days'] is True
        assert rec['upcoming_3_days_count'] >= 1
        assert "Pengingat Tagihan dalam 3 Hari" in rec['message']

        # Assert v7.1 backtest gate contract for < 4 occurrences
        assert item['terbukti_uji'] is False
        assert item['mape_uji'] is None
        assert item['kejadian'] == 0
        assert item['tepat_tanggal_uji'] == 0
        assert "belum teruji" in rec['message']
        assert rec['seri_layak'] == 1
        assert rec['seri_lolos'] == 0
        assert rec['seri_belum_teruji'] == 1
        assert rec['cakupan_nominal'] == 0.0


def test_recurring_backtest_gate_passes_regular_series(owner_actor, mock_categories):
    """(a) Synthetic series with a perfect monthly pattern (>= 4 occurrences, same day each month, stable amounts) passes gate and IS shown with terbukti_uji == True."""
    token = 'tok_owner'
    now_dt = datetime(2026, 11, 4, 10, 0, 0)
    txs = [
        {'id': 't1', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-06-05', 'nominal': 500000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-3', 'deskripsi': 'Sewa Toko Bulanan', 'status_persetujuan': 'disetujui'},
        {'id': 't2', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-07-05', 'nominal': 500000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-3', 'deskripsi': 'Sewa Toko Bulanan', 'status_persetujuan': 'disetujui'},
        {'id': 't3', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-08-05', 'nominal': 500000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-3', 'deskripsi': 'Sewa Toko Bulanan', 'status_persetujuan': 'disetujui'},
        {'id': 't4', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-09-05', 'nominal': 500000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-3', 'deskripsi': 'Sewa Toko Bulanan', 'status_persetujuan': 'disetujui'},
        {'id': 't5', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-10-05', 'nominal': 500000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-3', 'deskripsi': 'Sewa Toko Bulanan', 'status_persetujuan': 'disetujui'},
    ]

    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=txs), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.get_jakarta_now', return_value=now_dt):

        rec = fs.detect_recurring_expenses(token)
        assert rec['success'] is True
        assert rec['seri_layak'] == 1
        assert rec['seri_lolos'] == 1
        assert rec['seri_belum_teruji'] == 0
        assert rec['cakupan_nominal'] == 1.0
        assert len(rec['recurring_items']) == 1

        item = rec['recurring_items'][0]
        assert item['terbukti_uji'] is True
        assert item['kejadian'] >= 3
        assert item['tepat_tanggal_uji'] >= 3
        assert item['mape_uji'] is not None
        assert item['mape_uji'] <= 20.0
        assert "belum teruji" not in rec['message']
        assert "Cakupan prediksi: 100.0%" in rec['message']


def test_recurring_backtest_gate_fails_irregular_dates(owner_actor, mock_categories):
    """Irregular series retains failed backtest evidence and appears without invented dates."""
    token = 'tok_owner'
    now_dt = datetime(2026, 11, 4, 10, 0, 0)
    txs = [
        {'id': 'irr-1', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-08-17', 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-1', 'deskripsi': 'Langganan Software Pos', 'status_persetujuan': 'disetujui'},
        {'id': 'irr-2', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-08-21', 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-1', 'deskripsi': 'Langganan Software Pos', 'status_persetujuan': 'disetujui'},
        {'id': 'irr-3', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-09-13', 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-1', 'deskripsi': 'Langganan Software Pos', 'status_persetujuan': 'disetujui'},
        {'id': 'irr-4', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-09-18', 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-1', 'deskripsi': 'Langganan Software Pos', 'status_persetujuan': 'disetujui'},
        {'id': 'irr-5', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-11-03', 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-1', 'deskripsi': 'Langganan Software Pos', 'status_persetujuan': 'disetujui'},
    ]

    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=txs), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.get_jakarta_now', return_value=now_dt):

        rec = fs.detect_recurring_expenses(token)
        assert rec['success'] is True
        assert rec['seri_layak'] == 1
        assert rec['seri_lolos'] == 0
        assert rec['seri_belum_teruji'] == 0
        assert rec['cakupan_nominal'] == 0.0
        assert len(rec['recurring_items']) == 1
        item = rec['recurring_items'][0]
        assert item['schedule_regular'] is False
        assert item['next_expected_date'] is None
        assert item['days_until'] is None
        assert item['pattern_description'] == 'jadwal belum dapat dipastikan'
        assert item['terbukti_uji'] is False
        assert item['kejadian'] > 0
        assert 'mape_uji' in item and 'tepat_tanggal_uji' in item
        assert len(rec['upcoming_3_days_items']) == 0
        assert "Cakupan prediksi: 0.0%" in rec['message']


def test_recurring_backtest_gate_fails_wildly_varying_amounts(owner_actor, mock_categories):
    """Regular dates remain scheduled with a historical median despite a failed amount backtest."""
    token = 'tok_owner'
    now_dt = datetime(2026, 11, 4, 10, 0, 0)
    txs = [
        {'id': 'w1', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-06-05', 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-4', 'deskripsi': 'Gaji Lembur Lepas', 'status_persetujuan': 'disetujui'},
        {'id': 'w2', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-07-05', 'nominal': 100000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-4', 'deskripsi': 'Gaji Lembur Lepas', 'status_persetujuan': 'disetujui'},
        {'id': 'w3', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-08-05', 'nominal': 500000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-4', 'deskripsi': 'Gaji Lembur Lepas', 'status_persetujuan': 'disetujui'},
        {'id': 'w4', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-09-05', 'nominal': 50000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-4', 'deskripsi': 'Gaji Lembur Lepas', 'status_persetujuan': 'disetujui'},
        {'id': 'w5', 'toko_id': 'toko-berkah-001', 'tanggal': '2026-10-05', 'nominal': 800000, 'tipe': 'pengeluaran', 'kategori_id': 'cat-4', 'deskripsi': 'Gaji Lembur Lepas', 'status_persetujuan': 'disetujui'},
    ]

    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=txs), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.get_jakarta_now', return_value=now_dt):

        rec = fs.detect_recurring_expenses(token)
        assert rec['success'] is True
        assert rec['seri_layak'] == 1
        assert rec['seri_lolos'] == 0
        assert rec['seri_belum_teruji'] == 0
        assert rec['cakupan_nominal'] == 0.0
        assert len(rec['recurring_items']) == 1
        item = rec['recurring_items'][0]
        assert item['schedule_regular'] is True
        assert item['schedule_kind'] == 'monthly'
        assert item['anchor_day'] == 5
        assert item['next_expected_date'] == '2026-11-05'
        assert item['estimated_amount'] == 100000
        assert item['is_verified_due_date'] is False
        assert item['terbukti_uji'] is False
        assert item['mape_uji'] > 20
        assert item['kejadian'] > 0
        assert len(rec['upcoming_3_days_items']) == 1
        assert "Cakupan prediksi: 0.0%" in rec['message']


# ---------------------------------------------------------------------------
# 5. Cash Balance Sufficiency & Trend Tests
# ---------------------------------------------------------------------------

def test_check_forecast_against_cash_balance(owner_actor, synthetic_history, mock_categories):
    """Compares current recorded net cash balance against upcoming predicted expenses with mandatory disclaimer."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.period_rows', return_value=synthetic_history):

        res = fs.check_forecast_against_cash_balance(token, days_ahead=30)
        assert res['success'] is True
        assert 'cash_balance' in res
        assert 'predicted_expense' in res
        assert 'difference' in res
        assert 'disclaimer' in res
        assert "bukan jaminan likuiditas mutlak" in res['message']


def test_get_expense_trend_analysis(owner_actor, synthetic_history, mock_categories):
    """Evaluates expense trajectory (meningkat/menurun/stabil) and highlights top category drivers."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories):

        trend = fs.get_expense_trend(token, days_ahead=30)
        assert trend['success'] is True
        assert trend['trend'] in ('meningkat', 'menurun', 'stabil')
        assert len(trend['top_categories']) > 0
        assert "Analisis Tren Pengeluaran Toko" in trend['message']


# ---------------------------------------------------------------------------
# 6. Intent Routing & Chat Assistant End-to-End Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_tool", [
    ("Berapa prediksi pengeluaran bulan depan?", ToolName.GET_FORECAST_FOR_PERIOD),
    ("Bagaimana perkiraan pengeluaran minggu depan?", ToolName.GET_FORECAST_FOR_PERIOD),
    ("Kapan pengeluaran berikutnya?", ToolName.GET_NEXT_PREDICTED_EXPENSE),
    ("Pengeluaran rutin apa saja yang akan datang?", ToolName.GET_UPCOMING_RECURRING_EXPENSES),
    ("Apakah ada tagihan yang harus saya persiapkan dalam tiga hari ke depan?", ToolName.GET_UPCOMING_RECURRING_EXPENSES),
    ("Prediksi pengeluaran listrik bulan depan berapa?", ToolName.GET_FORECAST_FOR_PERIOD),
    ("Bandingkan pengeluaran bulan ini dengan prediksi bulan depan.", ToolName.COMPARE_ACTUAL_VS_FORECAST),
    ("Apakah pengeluaran toko saya akan meningkat?", ToolName.GET_EXPENSE_TREND),
    ("Seberapa akurat prediksi Prophet?", ToolName.GET_FORECAST_ACCURACY),
    ("Berapa nilai MAPE model prediksi toko saya?", ToolName.GET_FORECAST_ACCURACY),
    ("Kenapa prediksi pengeluaran naik?", ToolName.GET_EXPENSE_TREND),
    ("Apakah saldo kas saat ini mencukupi pengeluaran yang diprediksi?", ToolName.CHECK_FORECAST_AGAINST_CASH_BALANCE),
    ("Tampilkan prediksi tujuh hari ke depan.", ToolName.GET_FORECAST_FOR_PERIOD),
    ("Bulan mana yang diperkirakan memiliki pengeluaran tertinggi?", ToolName.GET_EXPENSE_FORECAST),
    ("Apakah data toko saya cukup untuk prediksi?", ToolName.GET_FORECAST_READINESS),
    ("Bagaimana cara kerja forecasting Prophet?", ToolName.GET_AVAILABLE_FORECAST),
])
def test_all_16_forecasting_intent_routings(query, expected_tool):
    """Every Indonesian forecasting query variation routes strictly to its expected ToolName."""
    decision = plan_stage_a_deterministic(query)
    assert decision is not None, f"Query '{query}' returned None"
    assert decision.tool_name == expected_tool, f"Query '{query}' resolved to {decision.tool_name}, expected {expected_tool}"


def test_conversational_follow_up_comparison():
    """Follow-up 'Bandingkan dengan bulan ini' routes to COMPARE_ACTUAL_VS_FORECAST when preceding turn was forecast."""
    ctx = ConversationContext(last_tool=ToolName.GET_FORECAST_FOR_PERIOD.value)
    decision = plan_stage_a_deterministic("Bandingkan dengan bulan ini.", context=ctx)
    assert decision is not None
    assert decision.tool_name == ToolName.COMPARE_ACTUAL_VS_FORECAST
    assert decision.follow_up_resolved is True


def test_chat_message_forecasting_end_to_end(owner_actor, synthetic_history, mock_categories):
    """End-to-end execution of handle_chat_message for a forecast query by authorized owner."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories):

        res = handle_chat_message(token, "Berapa nilai MAPE model prediksi toko saya?")
        assert res['tool'] == ToolName.GET_FORECAST_ACCURACY.value
        assert "Evaluasi & Perbandingan Model Peramalan" in res['message']
        assert res['data'].get('best_model') in fs.CANDIDATE_MODEL_ORDER
        assert "MAE" in res['message']
        assert "RMSE" in res['message']


# ---------------------------------------------------------------------------
# 7. Additional QA Scope: Deep Invariants & Edge Cases
# ---------------------------------------------------------------------------

def test_cross_intent_negative_invariants():
    """Verify that forecasting queries never route to financial totals, categories, or cashier tools."""
    forecast_queries = [
        "Berapa prediksi pengeluaran bulan depan?",
        "Bagaimana perkiraan pengeluaran minggu depan?",
        "Kapan pengeluaran berikutnya?",
        "Pengeluaran rutin apa saja yang akan datang?",
        "Apakah ada tagihan yang harus saya persiapkan dalam tiga hari ke depan?",
        "Apakah pengeluaran toko saya akan meningkat?",
        "Seberapa akurat prediksi Prophet?",
        "Apakah saldo kas saat ini mencukupi pengeluaran yang diprediksi?",
    ]
    forbidden_tools = {
        ToolName.GET_EXPENSE_SUMMARY,
        ToolName.GET_INCOME_SUMMARY,
        ToolName.GET_CASH_FLOW_SUMMARY,
        ToolName.GET_STORE_PROFILE,
        ToolName.COUNT_CASHIERS,
        ToolName.LIST_CASHIERS,
        ToolName.LIST_ALL_CATEGORIES,
        ToolName.RANK_EXPENSE_CATEGORIES,
    }
    for q in forecast_queries:
        dec = plan_stage_a_deterministic(q)
        assert dec is not None, f"Query '{q}' returned None"
        assert dec.tool_name not in forbidden_tools, (
            f"Query '{q}' was misrouted to {dec.tool_name}!"
        )


def test_chat_and_service_exact_amount_consistency(owner_actor, synthetic_history, mock_categories):
    """Verify forecast amounts in chat message exactly match the backend forecasting service."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories):

        # Direct service call
        direct_fcst = fs.get_forecast_for_period(token, period_label="Bulan Depan")
        assert direct_fcst['success'] is True
        expected_total = direct_fcst['total_predicted']

        # Chat message call
        chat_resp = handle_chat_message(token, "Berapa prediksi pengeluaran bulan depan?")
        assert chat_resp['tool'] == ToolName.GET_FORECAST_FOR_PERIOD.value
        from utils.ui_components import format_rupiah
        assert format_rupiah(expected_total) in chat_resp['message']
        assert chat_resp['data']['total_predicted'] == expected_total


def test_dashboard_and_chat_consistency(owner_actor, synthetic_history, mock_categories):
    """Verify dashboard metrics and chat assistant return identical 30-day forecast and recurring numbers."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories):

        from utils.ui_components import format_rupiah
        dash_fcst = fs.train_and_forecast_expenses(token, days_ahead=30)
        dash_rec = fs.detect_recurring_expenses(token)

        chat_fcst = handle_chat_message(token, "Bulan mana yang diperkirakan memiliki pengeluaran tertinggi?")
        chat_rec = handle_chat_message(token, "Pengeluaran rutin apa saja yang akan datang?")

        assert format_rupiah(dash_fcst['total_predicted_expense']) in chat_fcst['message']
        for item in dash_rec['recurring_items']:
            assert item['description'] in chat_rec['message']
            assert format_rupiah(item['estimated_amount']) in chat_rec['message']


def test_historical_actual_vs_future_predicted_labeling(owner_actor, synthetic_history, mock_categories):
    """Ensure clear linguistic distinction between historical actuals and future predictions."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories):

        comp = handle_chat_message(token, "Bandingkan pengeluaran bulan ini dengan prediksi bulan depan.")
        assert "Realisasi Pengeluaran" in comp['message']
        assert "Prediksi Pengeluaran" in comp['message']
        assert "Proyeksi pengeluaran diperkirakan" in comp['message']


def test_asia_jakarta_timezone_resolution():
    """Verify relative dates resolve using Asia/Jakarta (UTC+7)."""
    from ai.financial_chat import get_jakarta_now
    now_jkt = get_jakarta_now()
    assert now_jkt.utcoffset() == timedelta(hours=7)

    dec_period = plan_stage_a_deterministic("Berapa prediksi pengeluaran bulan depan?")
    assert dec_period.date_params.period_label == "Bulan Depan"


def test_recurring_expense_obligation_disclaimer(owner_actor, mock_categories):
    """Recurring expense detection must explicitly state that dates and amounts are estimates, not confirmed debt."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    txs = [
        {'id': '1', 'tanggal': '2026-09-01', 'nominal': 100000, 'tipe': 'pengeluaran', 'deskripsi': 'Wifi Indihome Toko', 'status_persetujuan': 'disetujui', 'kategori_id': 'cat-1'},
        {'id': '2', 'tanggal': '2026-10-01', 'nominal': 100000, 'tipe': 'pengeluaran', 'deskripsi': 'Wifi Indihome Toko', 'status_persetujuan': 'disetujui', 'kategori_id': 'cat-1'}
    ]
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=txs), \
         patch('services.forecast_service.categories', return_value=mock_categories):

        res = fs.detect_recurring_expenses(token)
        assert res['success'] is True
        assert "bukan tagihan resmi" in res['message']


def test_prophet_model_failure_graceful_handling(owner_actor, synthetic_history):
    """If one candidate fails, skip it and continue with remaining models; if all fail, return FIT_ERROR cleanly."""
    token = 'tok_owner'
    fs.invalidate_forecast_cache()
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=[]), \
         patch('services.forecast_service.Prophet.fit', side_effect=RuntimeError("Stan engine allocation failed")):

        res = fs.train_and_forecast_expenses(token)
        assert res['success'] is True
        assert res['selected_model'] != 'Prophet'
        assert any(s['model'] == 'Prophet' for s in res.get('skipped_models', []))

    fs.invalidate_forecast_cache()
    failing_fitters = {name: (lambda df, h: (_ for _ in ()).throw(RuntimeError("All failed"))) for name in fs.CANDIDATE_MODEL_ORDER}
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=[]), \
         patch.dict('services.forecast_service._MODEL_FITTERS', failing_fitters, clear=True):

        res_all_fail = fs.train_and_forecast_expenses(token)
        assert res_all_fail['success'] is False
        assert res_all_fail['error'] == 'FIT_ERROR'
        assert "kendala teknis" in res_all_fail['message']


def test_cross_store_isolation_in_prompts(owner_actor, synthetic_history):
    """Attempts to inject foreign store ID into prompt query must be completely ignored."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions') as mock_tx, \
         patch('services.forecast_service.categories', return_value=[]):
        mock_tx.return_value = synthetic_history

        dec = plan_stage_a_deterministic("Berapa prediksi pengeluaran toko-victim-999 bulan depan?")
        res = fs.train_and_forecast_expenses(token)
        mock_tx.assert_called_with('toko-berkah-001')


def test_multi_turn_conversational_forecast_flow(owner_actor, synthetic_history, mock_categories):
    """Multi-turn flow: forecast -> follow-up comparison -> cash sufficiency check."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.period_rows', return_value=synthetic_history):

        # Turn 1: Forecast
        t1 = handle_chat_message(token, "Berapa prediksi pengeluaran bulan depan?")
        assert t1['tool'] == ToolName.GET_FORECAST_FOR_PERIOD.value

        # Turn 2: Follow-up comparison
        ctx = ConversationContext(last_tool=ToolName.GET_FORECAST_FOR_PERIOD.value)
        t2_dec = plan_stage_a_deterministic("Bandingkan dengan bulan ini.", context=ctx)
        assert t2_dec.tool_name == ToolName.COMPARE_ACTUAL_VS_FORECAST
        assert t2_dec.follow_up_resolved is True

        # Turn 3: Cash balance sufficiency
        t3_dec = plan_stage_a_deterministic("Apakah saldo kas saat ini mencukupi pengeluaran yang diprediksi?")
        assert t3_dec.tool_name == ToolName.CHECK_FORECAST_AGAINST_CASH_BALANCE


def test_cache_invalidation_lifecycle(owner_actor, synthetic_history):
    """Verify cache invalidation purges cached forecast when data changes."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=[]):

        fs.invalidate_forecast_cache()
        r1 = fs.train_and_forecast_expenses(token, days_ahead=7)
        assert r1['success'] is True

        fs.invalidate_forecast_cache('toko-berkah-001')
        assert len(fs._FORECAST_CACHE) == 0


# ---------------------------------------------------------------------------
# 9. Validation of Problems A, B, C, D & Conversational Follow-up
# ---------------------------------------------------------------------------

def test_problem_a_tomorrow_one_day_forecast(owner_actor, synthetic_history, mock_categories):
    """Problem A: 'Prediksi pengeluaran untuk besok?' must return 1-day prediction for tomorrow only."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.get_store_name', return_value='Toko Berkah'):

        fs.invalidate_forecast_cache()

        # Query variant 1
        resp1 = handle_chat_message(token, "Prediksi pengeluaran untuk besok?")
        assert resp1['tool'] == ToolName.GET_FORECAST_FOR_PERIOD.value
        assert resp1['data']['output_mode'] == 'single_day'
        assert resp1['data']['days_count'] == 1
        assert resp1['data']['start_date'] == tomorrow
        assert resp1['data']['end_date'] == tomorrow
        assert "Prediksi Pengeluaran Besok — Toko Berkah" in resp1['message']
        assert tomorrow in resp1['message']
        assert "30 hari" not in resp1['message']
        assert "Puncak Pengeluaran" not in resp1['message']
        chosen_a = resp1['data']['selected_model']
        assert f"Estimasi pengeluaran Toko Berkah dihitung menggunakan model terpilih **{chosen_a}**" in resp1['message']

        # Query variant 2
        resp2 = handle_chat_message(token, "Prediksi pengeluaran besok?")
        assert resp2['tool'] == ToolName.GET_FORECAST_FOR_PERIOD.value
        assert resp2['data']['output_mode'] == 'single_day'
        assert resp2['data']['days_count'] == 1


def test_problem_b_seven_day_daily_breakdown_table(owner_actor, synthetic_history, mock_categories):
    """Problem B: 'Bisa berikan rincian prediksi pengeluaran 7 hari ke depan?' returns 7-row table."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)

    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.get_store_name', return_value='Toko Roti Makmur'):

        fs.invalidate_forecast_cache()

        resp = handle_chat_message(token, "Bisa berikan rincian prediksi pengeluaran 7 hari ke depan?")
        assert resp['tool'] == ToolName.GET_FORECAST_FOR_PERIOD.value
        assert resp['data']['output_mode'] == 'daily_breakdown'
        assert resp['data']['days_count'] == 7
        assert len(resp['data']['projections']) == 7

        msg = resp['message']
        assert "Rincian Prediksi Pengeluaran Harian" in msg
        assert "Toko Roti Makmur" in msg
        assert "| Tanggal | Prediksi Pengeluaran | Rentang Estimasi (80% CI) |" in msg
        assert "| :--- | :--- | :--- |" in msg
        # Count rows in markdown table
        table_lines = [line for line in msg.splitlines() if line.startswith("|") and not line.startswith("| :---") and not "Tanggal" in line]
        assert len(table_lines) == 7

        # Total and average summary
        assert "Total Estimasi (7 Hari):" in msg
        assert "Rata-rata Harian:" in msg
        chosen_b = resp['data']['selected_model']
        assert f"Estimasi pengeluaran Toko Roti Makmur dihitung menggunakan model terpilih **{chosen_b}**" in msg


def test_problem_c_dynamic_store_name_footer_resolution(owner_actor, synthetic_history, mock_categories):
    """Problem C: Real store name in dynamic footer, fallback to 'toko Anda', no hardcoded names."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)

    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories):

        # Subtest 1: Named store
        with patch('services.forecast_service.get_store_name', return_value='Kedai Kopi Nusantara'):
            fs.invalidate_forecast_cache()
            resp1 = handle_chat_message(token, "Berapa prediksi pengeluaran bulan depan?")
            assert "Kedai Kopi Nusantara" in resp1['message']
            chosen_1 = resp1['data']['selected_model']
            assert f"Estimasi pengeluaran Kedai Kopi Nusantara dihitung menggunakan model terpilih **{chosen_1}**" in resp1['message']
            assert "Tewiwi Bakery" not in resp1['message']

        # Subtest 2: Fallback when store name is None / unavailable
        with patch('services.forecast_service.get_store_name', return_value='toko Anda'):
            fs.invalidate_forecast_cache()
            resp2 = handle_chat_message(token, "Berapa prediksi pengeluaran bulan depan?")
            chosen_2 = resp2['data']['selected_model']
            assert f"Estimasi pengeluaran toko Anda dihitung menggunakan model terpilih **{chosen_2}**" in resp2['message']
            assert "Tewiwi Bakery" not in resp2['message']


def test_problem_d_date_and_horizon_distinction(owner_actor, synthetic_history, mock_categories):
    """Problem D: Distinguish single-day, calendar week, calendar month, and specific dates."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)

    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.get_store_name', return_value='Toko Serba Ada'):

        fs.invalidate_forecast_cache()

        # 1. Specific date
        dec_date = plan_stage_a_deterministic("Prediksi pengeluaran tanggal 5 November 2026?")
        assert dec_date.tool_name == ToolName.GET_FORECAST_FOR_PERIOD
        assert dec_date.date_params.start_date == '2026-11-05'
        assert dec_date.date_params.end_date == '2026-11-05'

        # 2. Next calendar week
        dec_week = plan_stage_a_deterministic("Prediksi pengeluaran minggu depan?")
        assert dec_week.tool_name == ToolName.GET_FORECAST_FOR_PERIOD
        assert dec_week.date_params.period_label == 'Minggu Depan'

        # 3. Next calendar month
        dec_month = plan_stage_a_deterministic("Prediksi pengeluaran bulan depan?")
        assert dec_month.tool_name == ToolName.GET_FORECAST_FOR_PERIOD
        assert dec_month.date_params.period_label == 'Bulan Depan'


def test_conversational_follow_up_preserves_period_and_breaks_down(owner_actor, synthetic_history, mock_categories):
    """Conversational Follow-up: Turn 1 forecast summary -> Turn 2 'Tampilkan rinciannya' outputs daily breakdown."""
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)

    with patch('services.chat_service.verify_actor', side_effect=mock_v), \
         patch('ai.copilot_tools.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=mock_categories), \
         patch('services.forecast_service.get_store_name', return_value='Toko Berkah'):

        fs.invalidate_forecast_cache()

        # Turn 1: 7-day summary
        t1_resp = handle_chat_message(token, "Prediksi pengeluaran 7 hari ke depan")
        assert t1_resp['tool'] == ToolName.GET_FORECAST_FOR_PERIOD.value
        assert t1_resp['data']['output_mode'] == 'summary'
        assert t1_resp['data']['days_count'] == 7
        assert "Total Estimasi (7 Hari)" not in t1_resp['message']  # Summary mode, not breakdown table

        # History with Turn 1
        history = [
            {'role': 'user', 'content': 'Prediksi pengeluaran 7 hari ke depan'},
            {'role': 'assistant', 'content': t1_resp['message'], 'metadata': t1_resp.get('metadata')}
        ]

        # Turn 2: "Tampilkan rinciannya"
        t2_resp = handle_chat_message(token, "Tampilkan rinciannya", history=history)
        assert t2_resp['tool'] == ToolName.GET_FORECAST_FOR_PERIOD.value
        assert t2_resp['data']['output_mode'] == 'daily_breakdown'
        assert t2_resp['data']['days_count'] == 7
        assert "| Tanggal | Prediksi Pengeluaran | Rentang Estimasi (80% CI) |" in t2_resp['message']
        assert "Total Estimasi (7 Hari):" in t2_resp['message']

        # Turn 2 variant: "Bisa berikan rinciannya?"
        t2_var = handle_chat_message(token, "Bisa berikan rinciannya?", history=history)
        assert t2_var['tool'] == ToolName.GET_FORECAST_FOR_PERIOD.value
        assert t2_var['data']['output_mode'] == 'daily_breakdown'
        assert t2_var['data']['days_count'] == 7
        assert "| Tanggal | Prediksi Pengeluaran | Rentang Estimasi (80% CI) |" in t2_var['message']


# =====================================================================
# 10. REVISI 3: MULTI-MODEL SELECTION & 3-90 DAY HORIZON VALIDATION
# =====================================================================

def test_revisi3_compare_forecast_models_selects_lowest_mape_and_labels_target(owner_actor):
    """Compare all 6 candidates on the same holdout; select lowest MAPE and show honest target label."""
    from datetime import date, timedelta
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    base = date(2026, 6, 1)

    # Dense 120-day series with clean weekly seasonality + slight upward trend -> best MAPE < 20%
    dense_txs = []
    for i in range(120):
        d = base + timedelta(days=i)
        amt = 100000 + (i * 250) + (20000 if d.weekday() in (4, 5) else 0)
        dense_txs.append({
            'id': f'tx-d-{i}',
            'toko_id': owner_actor['toko_id'],
            'tanggal': d.isoformat(),
            'nominal': amt,
            'tipe': 'pengeluaran',
            'status_persetujuan': 'disetujui',
        })

    fs.invalidate_forecast_cache()
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=dense_txs), \
         patch('services.forecast_service.categories', return_value=[]):
        comp = fs.compare_forecast_models(token, test_days=28)

    assert comp['success'] is True
    assert comp['test_days'] == 28
    evaluated_names = {r['model'] for r in comp['model_comparison']}
    assert {'Prophet', 'Seasonal Naive', 'MA7', 'ETS', 'ARIMA', 'LinearRegression'} <= evaluated_names
    assert comp['best_model'] == comp['model_comparison'][0]['model']
    assert comp['best_mape'] < 20.0
    assert comp['target_met'] is True
    assert 'target <20% tercapai' in comp['target_status']

    # Volatile/erratic series -> best MAPE >= 20%, still picks lowest MAPE and labels honestly
    volatile_txs = []
    for i in range(70):
        d = base + timedelta(days=i)
        amt = 500000 if i % 7 == 0 else (25000 if i % 2 == 0 else 180000)
        volatile_txs.append({
            'id': f'tx-v-{i}',
            'toko_id': owner_actor['toko_id'],
            'tanggal': d.isoformat(),
            'nominal': amt,
            'tipe': 'pengeluaran',
            'status_persetujuan': 'disetujui',
        })

    fs.invalidate_forecast_cache()
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=volatile_txs), \
         patch('services.forecast_service.categories', return_value=[]):
        comp_v = fs.compare_forecast_models(token, test_days=28)

    assert comp_v['success'] is True
    assert comp_v['best_model'] == comp_v['model_comparison'][0]['model']
    assert comp_v['best_mape'] >= 20.0
    assert comp_v['target_met'] is False
    assert f"target <20% belum tercapai (MAPE terbaik {comp_v['best_mape']:.2f}%)" in comp_v['target_status']


def test_revisi3_horizon_3_to_90_validation_across_service_schema_and_planner(owner_actor, synthetic_history):
    """Enforce 3..90 days across validate_forecast_horizon, FilterParams, plan_intent, and service."""
    from pydantic import ValidationError
    from ai.copilot_schema import FilterParams
    from ai.copilot_planner import plan_intent

    assert fs.validate_forecast_horizon(None) == 30
    assert fs.validate_forecast_horizon(3) == 3
    assert fs.validate_forecast_horizon(45) == 45
    assert fs.validate_forecast_horizon(90) == 90

    for bad_days in (0, 1, 2, 91, 180, 365):
        with pytest.raises(ValueError):
            fs.validate_forecast_horizon(bad_days)
        with pytest.raises(ValidationError):
            FilterParams(days_ahead=bad_days)

    assert FilterParams(days_ahead=3).days_ahead == 3
    assert FilterParams(days_ahead=90).days_ahead == 90

    # Planner rejects out-of-range explicit N hari requests clearly while preserving 'besok'
    d_too_low = plan_intent("Prediksi pengeluaran 2 hari ke depan")
    assert d_too_low.tool_name == ToolName.UNKNOWN
    assert "3 hingga 90 hari" in (d_too_low.clarification_question or "")

    d_too_high = plan_intent("Ramalkan pengeluaran 120 hari ke depan")
    assert d_too_high.tool_name == ToolName.UNKNOWN
    assert "3 hingga 90 hari" in (d_too_high.clarification_question or "")

    d_three = plan_intent("Prediksi pengeluaran 3 hari ke depan")
    assert d_three.tool_name in (ToolName.GET_EXPENSE_FORECAST, ToolName.GET_FORECAST_FOR_PERIOD)

    d_45 = plan_intent("Ramalkan pengeluaran 45 hari")
    assert d_45.tool_name == ToolName.GET_EXPENSE_FORECAST
    assert d_45.filter_params.days_ahead == 45

    d_besok = plan_intent("Prediksi pengeluaran untuk besok?")
    assert d_besok.tool_name == ToolName.GET_FORECAST_FOR_PERIOD

    # Service end-to-end validates 3..90 and returns exact requested projection length
    token = 'tok_owner'
    mock_v = make_mock_verify(owner_actor)
    fs.invalidate_forecast_cache()
    with patch('services.forecast_service.verify_actor', side_effect=mock_v), \
         patch('services.forecast_service.transactions', return_value=synthetic_history), \
         patch('services.forecast_service.categories', return_value=[]):
        with pytest.raises(ValueError):
            fs.train_and_forecast_expenses(token, days_ahead=2)
        with pytest.raises(ValueError):
            fs.train_and_forecast_expenses(token, days_ahead=91)

        r3 = fs.train_and_forecast_expenses(token, days_ahead=3)
        assert r3['success'] is True
        assert r3['days_ahead'] == 3
        assert len(r3['projections']) == 3

        r45 = fs.train_and_forecast_expenses(token, days_ahead=45)
        assert r45['success'] is True
        assert r45['days_ahead'] == 45
        assert len(r45['projections']) == 45
