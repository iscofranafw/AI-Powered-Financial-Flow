"""Multi-Model Expense Forecasting Service for AI-Powered Financial Recording & Cash Flow Assistant.

Provides store-specific, owner-only expense forecasting, multi-model holdout selection
(Prophet, Seasonal Naive, MA7, ETS, ARIMA, LinearRegression), recurring-expense detection,
accuracy backtesting (MAE, RMSE, MAPE, WAPE), cash-sufficiency analysis, and trend evaluation.
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

# Suppress cmdstanpy and prophet console log spam
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

try:
    from prophet import Prophet
    HAS_PROPHET = True
except (ImportError, ModuleNotFoundError):
    Prophet = None
    HAS_PROPHET = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS_ETS = True
except (ImportError, ModuleNotFoundError):
    ExponentialSmoothing = None
    HAS_STATSMODELS_ETS = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS_ARIMA = True
except (ImportError, ModuleNotFoundError):
    ARIMA = None
    HAS_STATSMODELS_ARIMA = False

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except (ImportError, ModuleNotFoundError):
    LinearRegression = None
    HAS_SKLEARN = False

from services.auth_service import verify_actor, AccessError
from repositories.finance_repository import transactions, categories
from services.report_service import period_rows, metrics
from utils.ui_components import format_rupiah
from ai.financial_chat import get_jakarta_now
from utils.recurring_calendar import infer_schedule, scheduled_dates

MIN_FORECAST_DAYS = 3
MAX_FORECAST_DAYS = 90
DEFAULT_FORECAST_DAYS = 30
TARGET_MAPE_THRESHOLD = 20.0

CANDIDATE_MODEL_ORDER = (
    'Prophet',
    'Seasonal Naive',
    'MA7',
    'ETS',
    'ARIMA',
    'LinearRegression',
)

# In-memory forecast cache keyed by (store_id, expense_count, latest_tx_id, days_ahead, category)
_FORECAST_CACHE: Dict[Tuple[str, int, str, int, Optional[str]], Dict[str, Any]] = {}


def invalidate_forecast_cache(store_id: Optional[str] = None) -> None:
    """Invalidate in-memory forecasting cache for a specific store or globally."""
    global _FORECAST_CACHE
    if store_id:
        _FORECAST_CACHE = {k: v for k, v in _FORECAST_CACHE.items() if k[0] != store_id}
    else:
        _FORECAST_CACHE.clear()


def validate_forecast_horizon(days_ahead: Optional[int]) -> int:
    """Enforce forecast horizon within 3..90 days (default 30)."""
    if days_ahead is None:
        return DEFAULT_FORECAST_DAYS
    if isinstance(days_ahead, bool):
        raise ValueError(
            f"Horizon peramalan harus berupa angka antara {MIN_FORECAST_DAYS} hingga {MAX_FORECAST_DAYS} hari."
        )
    try:
        days = int(days_ahead)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Horizon peramalan tidak valid ({days_ahead}). Gunakan angka antara {MIN_FORECAST_DAYS} hingga {MAX_FORECAST_DAYS} hari."
        ) from exc
    if days < MIN_FORECAST_DAYS or days > MAX_FORECAST_DAYS:
        raise ValueError(
            f"Horizon peramalan harus antara {MIN_FORECAST_DAYS} hingga {MAX_FORECAST_DAYS} hari "
            f"(diminta: {days} hari)."
        )
    return days


def _get_eligible_expense_rows(token: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retrieve and validate authorized store expense transactions for the authenticated owner."""
    actor = verify_actor(token, {'pemilik_toko'})
    shop_id = actor['toko_id']
    all_tx = transactions(shop_id)
    all_cats = categories(shop_id)

    approved_expenses = [
        t for t in all_tx
        if t.get('tipe') == 'pengeluaran' and t.get('status_persetujuan') == 'disetujui'
    ]
    # Sort ascending by transaction date
    approved_expenses.sort(key=lambda x: str(x.get('tanggal', '')))
    return actor, approved_expenses, all_cats


def get_store_name(shop_id: Optional[str]) -> str:
    """Resolve the real store name from the database given shop_id, safely falling back to 'toko Anda'."""
    if not shop_id:
        return "toko Anda"
    try:
        from database.client import privileged_client
        res = privileged_client().table('toko').select('nama').eq('id', shop_id).single().execute()
        if res and res.data and res.data.get('nama'):
            name = str(res.data['nama']).strip()
            if name:
                return name
    except Exception as exc:
        logging.debug(f"Could not resolve store name for {shop_id}: {exc}")
    return "toko Anda"


def get_dynamic_forecast_footer(store_name: str, model_name: Optional[str] = None) -> str:
    """Produce the dynamic disclaimer footer scoped to the authenticated store and selected model."""
    store_display = store_name if store_name and store_name != "toko Anda" else "toko Anda"
    chosen = (model_name or "Prophet / Multi-Model Terbaik").strip()
    return (
        f"Estimasi pengeluaran {store_display} dihitung menggunakan model terpilih **{chosen}** "
        f"berdasarkan perbandingan akurasi pada riwayat transaksi pengeluaran yang telah disetujui "
        f"pada toko tersebut. Hasil ini merupakan perkiraan statistik, bukan kepastian pengeluaran di masa mendatang."
    )


def get_forecast_readiness_status(token: str) -> Dict[str, Any]:
    """Check if the store has sufficient historical expense transactions for forecasting."""
    actor, approved_expenses, _ = _get_eligible_expense_rows(token)

    total_count = len(approved_expenses)
    unique_dates = sorted(list({str(t.get('tanggal')) for t in approved_expenses if t.get('tanggal')}))
    distinct_days = len(unique_dates)

    min_date = unique_dates[0] if unique_dates else None
    max_date = unique_dates[-1] if unique_dates else None

    date_span_days = 0
    if min_date and max_date:
        d_min = date.fromisoformat(min_date)
        d_max = date.fromisoformat(max_date)
        date_span_days = (d_max - d_min).days + 1

    req_tx = 5
    req_days = 5
    req_span = 7

    is_ready = (total_count >= req_tx) and (distinct_days >= req_days) and (date_span_days >= req_span)

    if is_ready:
        msg = (
            f"**Data Keuangan Siap untuk Peramalan Pengeluaran (Multi-Model)**\n\n"
            f"- **Jumlah Transaksi Pengeluaran:** {total_count} transaksi\n"
            f"- **Jumlah Hari Unik:** {distinct_days} hari\n"
            f"- **Rentang Tanggal:** {min_date} s.d. {max_date} ({date_span_days} hari)\n"
            f"- **Status Kesiapan:** Memenuhi syarat minimum peramalan deret waktu."
        )
    else:
        missing_parts = []
        if total_count < req_tx:
            missing_parts.append(f"transaksi pengeluaran ({total_count}/{req_tx})")
        if distinct_days < req_days:
            missing_parts.append(f"hari pencatatan unik ({distinct_days}/{req_days})")
        if date_span_days < req_span:
            missing_parts.append(f"rentang waktu historis ({date_span_days}/{req_span} hari)")

        msg = (
            f"**Data Belum Mencukupi untuk Peramalan Pengeluaran yang Andal**\n\n"
            f"Mesin peramalan memerlukan riwayat data yang memadai untuk mengenali tren dan pola pengeluaran toko. "
            f"Saat ini data toko Anda:\n"
            f"- **Transaksi Pengeluaran Tercatat:** {total_count} transaksi (minimal {req_tx})\n"
            f"- **Hari Aktif dengan Transaksi:** {distinct_days} hari (minimal {req_days})\n"
            f"- **Rentang Waktu Historis:** {date_span_days} hari (minimal {req_span} hari)\n\n"
            f"**Kebutuhan Data yang Belum Terpenuhi:** {', '.join(missing_parts)}.\n"
            f"Silakan lanjutkan pencatatan pengeluaran operasional toko agar peramalan dapat dilakukan secara akurat."
        )

    return {
        'ready': is_ready,
        'total_expenses_count': total_count,
        'distinct_expense_days': distinct_days,
        'min_date': min_date,
        'max_date': max_date,
        'date_span_days': date_span_days,
        'requirements': {
            'min_transactions': req_tx,
            'min_distinct_days': req_days,
            'min_date_span_days': req_span,
        },
        'message': msg
    }


def _build_daily_expense_df(approved_expenses: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, int, pd.Timestamp, pd.Timestamp]:
    """Aggregate approved expense transactions to a continuous daily Pandas DataFrame."""
    records = []
    for t in approved_expenses:
        try:
            records.append({
                'ds': pd.Timestamp(str(t['tanggal'])[:10]),
                'nominal': float(t['nominal'])
            })
        except (ValueError, TypeError):
            continue

    if not records:
        return pd.DataFrame(columns=['ds', 'y']), 0, pd.NaT, pd.NaT

    df_raw = pd.DataFrame(records)
    daily = df_raw.groupby('ds')['nominal'].sum()
    min_d = daily.index.min()
    max_d = daily.index.max()
    full_idx = pd.date_range(start=min_d, end=max_d, freq='D')
    daily_series = daily.reindex(full_idx, fill_value=0.0)
    daily_indexed = pd.DataFrame({
        'ds': pd.to_datetime(daily_series.index),
        'y': daily_series.values.astype(float)
    })
    return daily_indexed, len(daily), min_d, max_d


def _filter_expenses_by_category(
    approved_expenses: List[Dict[str, Any]],
    all_cats: List[Dict[str, Any]],
    category_name: Optional[str],
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
    if not category_name:
        return approved_expenses, None, None

    target_cat_id = None
    target_cat_label = None
    for c in all_cats:
        if category_name.lower() in c.get('nama', '').lower():
            target_cat_id = c['id']
            target_cat_label = c.get('nama')
            break

    if not target_cat_id:
        return approved_expenses, None, {
            'success': False,
            'error': 'CATEGORY_NOT_FOUND',
            'message': f"Kategori pengeluaran **'{category_name}'** tidak ditemukan di daftar kategori toko Anda."
        }

    filtered = [t for t in approved_expenses if t.get('kategori_id') == target_cat_id]
    if len(filtered) < 3:
        return filtered, target_cat_label, {
            'success': False,
            'error': 'INSUFFICIENT_CATEGORY_DATA',
            'message': (
                f"Data pengeluaran untuk kategori **'{target_cat_label}'** baru memiliki {len(filtered)} transaksi. "
                f"Dibutuhkan minimal 3 transaksi pada kategori ini untuk membuat peramalan khusus."
            )
        }
    return filtered, target_cat_label, None


def _future_dates(max_dt: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = pd.to_datetime(max_dt) + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq='D')


def _empirical_bounds(y_hist: np.ndarray, yhat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(y_hist) >= 2:
        spread = float(np.std(y_hist, ddof=1))
    else:
        spread = float(np.mean(y_hist)) * 0.2 if len(y_hist) else 0.0
    margin = 1.28 * spread * 0.5
    lower = np.clip(yhat - margin, 0.0, None)
    upper = np.clip(yhat + margin, 0.0, None)
    return lower, upper


def _fit_predict_prophet(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not HAS_PROPHET or Prophet is None:
        raise RuntimeError("Prophet belum terpasang")
    span = len(train_df)
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=bool(span >= 14),
        yearly_seasonality=bool(span >= 365),
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    model.fit(train_df[['ds', 'y']])
    future = model.make_future_dataframe(periods=horizon, freq='D')
    fcst = model.predict(future)
    max_dt = pd.to_datetime(train_df['ds'].max())
    future_part = fcst[fcst['ds'] > max_dt].head(horizon).copy()
    future_part['yhat'] = future_part['yhat'].clip(lower=0.0)
    future_part['yhat_lower'] = future_part['yhat_lower'].clip(lower=0.0)
    future_part['yhat_upper'] = future_part['yhat_upper'].clip(lower=0.0)
    return future_part[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)


def _fit_predict_seasonal_naive(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = train_df['y'].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError("Data latih kosong")
    fut_dates = _future_dates(train_df['ds'].max(), horizon)
    preds = np.zeros(horizon, dtype=float)
    if len(y) >= 7:
        pattern = y[-7:]
        for i in range(horizon):
            preds[i] = max(0.0, float(pattern[i % 7]))
    else:
        mean_val = max(0.0, float(np.mean(y)))
        preds[:] = mean_val
    low, up = _empirical_bounds(y, preds)
    return pd.DataFrame({'ds': fut_dates, 'yhat': preds, 'yhat_lower': low, 'yhat_upper': up})


def _fit_predict_ma7(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = train_df['y'].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError("Data latih kosong")
    fut_dates = _future_dates(train_df['ds'].max(), horizon)
    history = list(y.astype(float))
    preds = np.zeros(horizon, dtype=float)
    for i in range(horizon):
        window = history[-7:] if len(history) >= 7 else history
        val = max(0.0, float(np.mean(window)))
        preds[i] = val
        history.append(val)
    low, up = _empirical_bounds(y, preds)
    return pd.DataFrame({'ds': fut_dates, 'yhat': preds, 'yhat_lower': low, 'yhat_upper': up})


def _fit_predict_ets(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not HAS_STATSMODELS_ETS or ExponentialSmoothing is None:
        raise RuntimeError("statsmodels ExponentialSmoothing belum terpasang")
    y = train_df['y'].to_numpy(dtype=float)
    if len(y) < 4:
        raise ValueError("Data latih terlalu pendek untuk ETS")
    fut_dates = _future_dates(train_df['ds'].max(), horizon)
    if float(np.std(y)) < 1e-9:
        preds = np.full(horizon, max(0.0, float(y[-1])), dtype=float)
        low, up = _empirical_bounds(y, preds)
        return pd.DataFrame({'ds': fut_dates, 'yhat': preds, 'yhat_lower': low, 'yhat_upper': up})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        configs = []
        if len(y) >= 14:
            configs.append(dict(trend='add', damped_trend=True, seasonal='add', seasonal_periods=7))
        configs.append(dict(trend='add', damped_trend=True, seasonal=None))
        configs.append(dict(trend=None, seasonal=None))

        last_err = None
        for cfg in configs:
            try:
                fitted = ExponentialSmoothing(y, **cfg).fit(optimized=True)
                raw_pred = np.asarray(fitted.forecast(horizon), dtype=float)
                if np.all(np.isfinite(raw_pred)):
                    preds = np.clip(raw_pred, 0.0, None)
                    low, up = _empirical_bounds(y, preds)
                    return pd.DataFrame({'ds': fut_dates, 'yhat': preds, 'yhat_lower': low, 'yhat_upper': up})
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(f"Semua konfigurasi ETS gagal fit: {last_err}")


def _fit_predict_arima(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not HAS_STATSMODELS_ARIMA or ARIMA is None:
        raise RuntimeError("statsmodels ARIMA belum terpasang")
    y = train_df['y'].to_numpy(dtype=float)
    if len(y) < 6:
        raise ValueError("Data latih terlalu pendek untuk ARIMA")
    fut_dates = _future_dates(train_df['ds'].max(), horizon)
    if float(np.std(y)) < 1e-9:
        preds = np.full(horizon, max(0.0, float(y[-1])), dtype=float)
        low, up = _empirical_bounds(y, preds)
        return pd.DataFrame({'ds': fut_dates, 'yhat': preds, 'yhat_lower': low, 'yhat_upper': up})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        last_err = None
        for order in ((1, 0, 1), (1, 1, 1), (1, 0, 0)):
            try:
                model = ARIMA(
                    y,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit()
                raw_pred = np.asarray(fitted.forecast(steps=horizon), dtype=float)
                if np.all(np.isfinite(raw_pred)):
                    preds = np.clip(raw_pred, 0.0, None)
                    low, up = _empirical_bounds(y, preds)
                    return pd.DataFrame({'ds': fut_dates, 'yhat': preds, 'yhat_lower': low, 'yhat_upper': up})
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(f"ARIMA gagal konvergen: {last_err}")


def _design_matrix_lr(dates: pd.Series, start_ordinal: int) -> np.ndarray:
    t_idx = (pd.to_datetime(dates) - pd.Timestamp.fromordinal(start_ordinal)).dt.days.to_numpy(dtype=float)
    dows = pd.to_datetime(dates).dt.day_of_week.to_numpy()
    feats = [t_idx]
    for dow in range(6):
        feats.append((dows == dow).astype(float))
    return np.column_stack(feats)


def _fit_predict_linear_regression(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not HAS_SKLEARN or LinearRegression is None:
        raise RuntimeError("scikit-learn LinearRegression belum terpasang")
    if len(train_df) < 3:
        raise ValueError("Data latih terlalu pendek untuk LinearRegression")
    fut_dates = _future_dates(train_df['ds'].max(), horizon)
    start_ord = pd.to_datetime(train_df['ds'].min()).date().toordinal()
    X_train = _design_matrix_lr(train_df['ds'], start_ord)
    y_train = train_df['y'].to_numpy(dtype=float)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    X_fut = _design_matrix_lr(pd.Series(fut_dates), start_ord)
    raw_pred = np.asarray(reg.predict(X_fut), dtype=float)
    preds = np.clip(raw_pred, 0.0, None)
    low, up = _empirical_bounds(y_train, preds)
    return pd.DataFrame({'ds': fut_dates, 'yhat': preds, 'yhat_lower': low, 'yhat_upper': up})


_MODEL_FITTERS = {
    'Prophet': _fit_predict_prophet,
    'Seasonal Naive': _fit_predict_seasonal_naive,
    'MA7': _fit_predict_ma7,
    'ETS': _fit_predict_ets,
    'ARIMA': _fit_predict_arima,
    'LinearRegression': _fit_predict_linear_regression,
}


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sum_actual = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / sum_actual * 100.0) if sum_actual > 0 else None

    non_zero_mask = y_true > 0
    n_nonzero = int(np.sum(non_zero_mask))
    if np.any(non_zero_mask):
        mape = float(
            np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
        )
    else:
        mape = None

    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'mape': round(mape, 2) if mape is not None else None,
        'wape': round(wape, 2) if wape is not None else None,
        'positive_holdout_days': n_nonzero,
    }


def evaluate_forecast_candidates_df(
    daily_indexed: pd.DataFrame,
    test_days: int = 28,
    candidates: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Compare all candidate forecast models on an identical time-ordered holdout split."""
    total_days = len(daily_indexed)
    if total_days < 14:
        return {
            'success': False,
            'ready': False,
            'mae': None,
            'rmse': None,
            'mape': None,
            'wape': None,
            'best_model': None,
            'target_met': False,
            'models': [],
            'model_comparison': [],
            'skipped_models': [],
            'message': (
                f"Evaluasi multi-model memerlukan riwayat data minimal 14 hari berturut-turut untuk validasi holdout "
                f"(saat ini {total_days} hari). Tambahkan catatan transaksi untuk mengaktifkan pengujian akurasi."
            )
        }

    req_test = max(3, int(test_days))
    eff_test = req_test if total_days >= (req_test + 14) else max(3, total_days // 3)
    split_idx = total_days - eff_test
    train_df = daily_indexed.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = daily_indexed.iloc[split_idx:].copy().reset_index(drop=True)
    y_true = test_df['y'].to_numpy(dtype=float)

    n_nonzero = int(np.sum(y_true > 0))
    zero_count = int(len(y_true) - n_nonzero)
    if n_nonzero > 0 and zero_count > 0:
        mape_note = (
            f"MAPE dihitung khusus pada {n_nonzero} hari dengan pengeluaran aktif "
            f"(mengecualikan {zero_count} hari Rp0)."
        )
    elif n_nonzero == 0:
        mape_note = (
            "Seluruh hari pada periode uji memiliki pengeluaran Rp0, sehingga MAPE tidak dapat dihitung "
            "(pembagian dengan nol dicegah)."
        )
    else:
        mape_note = ""

    candidate_names = candidates or CANDIDATE_MODEL_ORDER
    comparison_rows: List[Dict[str, Any]] = []
    skipped_models: List[Dict[str, str]] = []

    for model_name in candidate_names:
        fitter = _MODEL_FITTERS.get(model_name)
        if fitter is None:
            continue
        try:
            pred_df = fitter(train_df, eff_test)
            y_pred = pred_df['yhat'].to_numpy(dtype=float)[:eff_test]
            if len(y_pred) != len(y_true):
                raise ValueError(f"Panjang prediksi {len(y_pred)} tidak cocok dengan uji {len(y_true)}")
            m = _compute_regression_metrics(y_true, y_pred)
            comparison_rows.append({
                'model': model_name,
                'mape': m['mape'],
                'mae': m['mae'],
                'rmse': m['rmse'],
                'wape': m['wape'],
                'target_met': bool(m['mape'] is not None and m['mape'] < TARGET_MAPE_THRESHOLD and n_nonzero >= 5),
                'status': 'evaluated',
            })
        except Exception as exc:
            logging.warning("Model peramalan '%s' dilewati saat evaluasi holdout: %s", model_name, exc)
            skipped_models.append({'model': model_name, 'reason': str(exc)})

    if not comparison_rows:
        return {
            'success': False,
            'ready': False,
            'mae': None,
            'rmse': None,
            'mape': None,
            'wape': None,
            'best_model': None,
            'target_met': False,
            'models': [],
            'model_comparison': [],
            'skipped_models': skipped_models,
            'message': "Seluruh kandidat model peramalan gagal dihitung pada data toko saat ini."
        }

    comparison_rows.sort(
        key=lambda row: (
            float('inf') if row['mape'] is None else row['mape'],
            row['mae'],
            row['rmse'],
        )
    )
    best = comparison_rows[0]
    best_model = best['model']
    best_mape = best['mape']
    target_met = bool(best_mape is not None and best_mape < TARGET_MAPE_THRESHOLD and n_nonzero >= 5)

    if best_mape is None:
        target_status = "target <20% belum dapat dihitung (periode uji tanpa pengeluaran aktif)"
    elif n_nonzero < 5:
        target_status = (
            f"target <20% belum dapat disimpulkan (hanya {n_nonzero} hari aktif dari {eff_test} hari uji; "
            f"MAPE terbaik {best_mape:.2f}%)"
        )
    elif best_mape < TARGET_MAPE_THRESHOLD:
        target_status = f"target <20% tercapai (MAPE terbaik {best_mape:.2f}%)"
    else:
        target_status = f"target <20% belum tercapai (MAPE terbaik {best_mape:.2f}%)"

    return {
        'success': True,
        'ready': True,
        'test_days': eff_test,
        'train_days': len(train_df),
        'positive_holdout_days': n_nonzero,
        'best_model': best_model,
        'selected_model': best_model,
        'best_mape': best_mape,
        'mae': best['mae'],
        'rmse': best['rmse'],
        'mape': best_mape,
        'wape': best['wape'],
        'target_met': target_met,
        'target_status': target_status,
        'mape_note': mape_note,
        'models': comparison_rows,
        'model_comparison': comparison_rows,
        'skipped_models': skipped_models,
    }


def _select_and_forecast_full(
    daily_indexed: pd.DataFrame,
    days_ahead: int,
    test_days: int = 28,
) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """Run holdout evaluation to select lowest-MAPE model, then refit winner on full history."""
    eval_res = evaluate_forecast_candidates_df(daily_indexed, test_days=test_days)
    ranked_candidates: List[str] = []
    if eval_res.get('success') and eval_res.get('model_comparison'):
        ranked_candidates = [row['model'] for row in eval_res['model_comparison']]

    for fallback_name in ('Seasonal Naive', 'MA7', 'Prophet', 'LinearRegression', 'ETS', 'ARIMA'):
        if fallback_name not in ranked_candidates:
            ranked_candidates.append(fallback_name)

    last_err = None
    for candidate_name in ranked_candidates:
        fitter = _MODEL_FITTERS.get(candidate_name)
        if fitter is None:
            continue
        try:
            proj_df = fitter(daily_indexed[['ds', 'y']], days_ahead)
            return candidate_name, proj_df, eval_res
        except Exception as exc:
            logging.warning("Refit penuh model '%s' gagal, lanjut kandidat berikutnya: %s", candidate_name, exc)
            last_err = exc
            continue

    raise RuntimeError(f"Semua kandidat model gagal melakukan peramalan penuh: {last_err}")


def train_and_forecast_expenses(
    token: str,
    days_ahead: int = 30,
    category_name: Optional[str] = None,
    store_name: Optional[str] = None
) -> Dict[str, Any]:
    """Compare candidate time-series models on holdout, select lowest MAPE, and forecast 3..90 days ahead."""
    validated_days = validate_forecast_horizon(days_ahead)

    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)
    shop_id = actor['toko_id']
    resolved_store_name = store_name or actor.get('toko_nama') or get_store_name(shop_id)

    readiness = get_forecast_readiness_status(token)
    if not readiness['ready']:
        return {
            'success': False,
            'readiness': readiness,
            'error': 'INSUFFICIENT_DATA',
            'message': readiness['message']
        }

    approved_expenses, target_cat_label, cat_err = _filter_expenses_by_category(
        approved_expenses, all_cats, category_name
    )
    if cat_err is not None:
        return cat_err

    latest_id = str(approved_expenses[-1].get('id', '')) if approved_expenses else ''
    cache_key = (shop_id, len(approved_expenses), latest_id, validated_days, target_cat_label)
    if cache_key in _FORECAST_CACHE:
        return _FORECAST_CACHE[cache_key]

    daily_indexed, distinct_days_count, min_d, max_d = _build_daily_expense_df(approved_expenses)
    if daily_indexed.empty:
        return {
            'success': False,
            'error': 'NO_VALID_RECORDS',
            'message': "Tidak ada catatan tanggal transaksi pengeluaran yang valid."
        }

    date_span = int((max_d - min_d).days + 1)

    try:
        chosen_model, future_forecast, eval_res = _select_and_forecast_full(daily_indexed, validated_days)

        future_forecast['yhat'] = future_forecast['yhat'].clip(lower=0.0)
        future_forecast['yhat_lower'] = future_forecast['yhat_lower'].clip(lower=0.0)
        future_forecast['yhat_upper'] = future_forecast['yhat_upper'].clip(lower=0.0)

        daily_projections = []
        for _, row in future_forecast.iterrows():
            daily_projections.append({
                'ds': pd.to_datetime(row['ds']).strftime('%Y-%m-%d'),
                'yhat': round(float(row['yhat']), 2),
                'yhat_lower': round(float(row['yhat_lower']), 2),
                'yhat_upper': round(float(row['yhat_upper']), 2)
            })

        total_predicted = float(future_forecast['yhat'].sum())
        total_lower = float(future_forecast['yhat_lower'].sum())
        total_upper = float(future_forecast['yhat_upper'].sum())
        avg_daily = float(future_forecast['yhat'].mean()) if len(future_forecast) > 0 else 0.0

        peak_row = future_forecast.loc[future_forecast['yhat'].idxmax()] if len(future_forecast) > 0 else None
        peak_info = {
            'date': pd.to_datetime(peak_row['ds']).strftime('%Y-%m-%d') if peak_row is not None else None,
            'amount': round(float(peak_row['yhat']), 2) if peak_row is not None else 0.0
        }

        hist_total = float(daily_indexed['y'].sum())
        hist_avg_daily = float(daily_indexed['y'].mean())

        result = {
            'success': True,
            'store_id': shop_id,
            'store_name': resolved_store_name,
            'days_ahead': validated_days,
            'category_filtered': target_cat_label,
            'selected_model': chosen_model,
            'best_model': chosen_model,
            'best_mape': eval_res.get('best_mape'),
            'best_mae': eval_res.get('mae'),
            'best_rmse': eval_res.get('rmse'),
            'best_wape': eval_res.get('wape'),
            'target_met': bool(eval_res.get('target_met', False)),
            'target_status': eval_res.get('target_status', ''),
            'model_comparison': eval_res.get('model_comparison', []),
            'skipped_models': eval_res.get('skipped_models', []),
            'total_predicted_expense': total_predicted,
            'total_predicted_lower': total_lower,
            'total_predicted_upper': total_upper,
            'average_daily_predicted': avg_daily,
            'historical_total_expense': hist_total,
            'historical_daily_average': hist_avg_daily,
            'peak_expense': peak_info,
            'historical_coverage': {
                'start_date': str(min_d)[:10],
                'end_date': str(max_d)[:10],
                'total_days': date_span,
                'distinct_transaction_days': distinct_days_count,
            },
            'projections': daily_projections
        }

        _FORECAST_CACHE[cache_key] = result
        return result

    except Exception as exc:
        logging.error(f"Forecast fitting error: {exc}")
        return {
            'success': False,
            'error': 'FIT_ERROR',
            'message': f"Terjadi kendala teknis saat memproses deret waktu peramalan: {str(exc)}"
        }


def compare_forecast_models(
    token: str,
    test_days: int = 28,
    category_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare all candidate forecast models on an identical time-ordered holdout split."""
    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)
    shop_id = actor['toko_id']
    store_name = actor.get('toko_nama') or get_store_name(shop_id)

    readiness = get_forecast_readiness_status(token)
    if not readiness['ready']:
        return {
            'success': False,
            'ready': False,
            'message': readiness['message']
        }

    approved_expenses, target_cat_label, cat_err = _filter_expenses_by_category(
        approved_expenses, all_cats, category_name
    )
    if cat_err is not None:
        return cat_err

    daily_indexed, _, _, _ = _build_daily_expense_df(approved_expenses)
    eval_res = evaluate_forecast_candidates_df(daily_indexed, test_days=test_days)
    if not eval_res.get('success'):
        return eval_res

    rows = eval_res['model_comparison']
    eff_test = eval_res['test_days']
    best_model = eval_res['best_model']
    best_mape = eval_res['best_mape']
    best_mae = eval_res['mae']
    best_rmse = eval_res['rmse']
    best_wape = eval_res['wape']
    target_status = eval_res['target_status']
    mape_note = eval_res.get('mape_note', '')

    table_lines = [
        "| Model | MAPE | MAE | RMSE | WAPE | Status |",
        "| :--- | :---: | ---: | ---: | :---: | :--- |"
    ]
    for r in rows:
        mape_str = f"{r['mape']:.2f}%" if r['mape'] is not None else "N/A"
        wape_str = f"{r['wape']:.2f}%" if r['wape'] is not None else "N/A"
        badge = "**Terpilih (Terbaik)**" if r['model'] == best_model else "Kandidat"
        table_lines.append(
            f"| {r['model']} | {mape_str} | {format_rupiah(r['mae'])} | {format_rupiah(r['rmse'])} | {wape_str} | {badge} |"
        )
    table_md = "\n".join(table_lines)

    cat_hdr = f" (Kategori: {target_cat_label})" if target_cat_label else ""
    msg = (
        f"### Evaluasi & Perbandingan Model Peramalan{cat_hdr} — {store_name}\n\n"
        f"- **Metode Evaluasi:** Time-Ordered Holdout Backtesting ({eff_test} hari terakhir sebagai data uji)\n"
        f"- **Model Terpilih (MAPE Terendah):** **{best_model}**\n"
        f"- **Status Target (<20% MAPE):** {target_status}\n"
        f"- **MAE Terbaik:** {format_rupiah(best_mae)} | **RMSE Terbaik:** {format_rupiah(best_rmse)}\n"
    )
    if best_mape is not None:
        msg += f"- **MAPE Terbaik ({best_model}):** {best_mape:.2f}%\n"
    if best_wape is not None:
        msg += f"- **WAPE Terbaik:** {best_wape:.2f}%\n"
    msg += f"\n{table_md}\n"
    if mape_note:
        msg += f"\n*Catatan Metrik:* {mape_note}"

    eval_res['store_name'] = store_name
    eval_res['category_filtered'] = target_cat_label
    eval_res['message'] = msg
    return eval_res


def evaluate_forecast_accuracy(token: str, test_days: int = 28) -> Dict[str, Any]:
    """Evaluate accuracy across candidate models on holdout backtesting (MAPE, MAE, RMSE, WAPE)."""
    return compare_forecast_models(token, test_days=test_days)


def _backtest_series(dates: List[date], amounts: List[float]) -> Dict[str, Any]:
    """Holdout backtesting on recurring expense series history to verify predictability."""
    n = len(dates)
    fold_indices = list(range(max(2, n - 5), n))
    kejadian = len(fold_indices)
    if kejadian == 0:
        return {
            'kejadian': 0,
            'tepat_tanggal': 0,
            'hit_rate': 0.0,
            'mape_uji': None,
        }

    tepat_tanggal = 0
    hit_apes: List[float] = []

    for k in fold_indices:
        prior_dates = dates[:k]
        prior_amounts = amounts[:k]
        gaps = [(prior_dates[i] - prior_dates[i - 1]).days for i in range(1, len(prior_dates))]
        median_gap = int(round(float(np.median(gaps))))
        predicted_date = prior_dates[-1] + timedelta(days=median_gap)

        actual_date = dates[k]
        if abs((predicted_date - actual_date).days) <= 2:
            tepat_tanggal += 1
            actual_amount = float(amounts[k])
            if actual_amount != 0.0:
                med_prior = float(np.median(prior_amounts))
                ape = abs(med_prior - actual_amount) / actual_amount * 100.0
                hit_apes.append(ape)

    hit_rate = float(tepat_tanggal / kejadian) if kejadian > 0 else 0.0
    mape_uji = float(np.mean(hit_apes)) if hit_apes else None

    return {
        'kejadian': kejadian,
        'tepat_tanggal': tepat_tanggal,
        'hit_rate': hit_rate,
        'mape_uji': mape_uji,
    }


def detect_recurring_expenses(token: str) -> Dict[str, Any]:
    """Detect recurring expense patterns (utilities, rent, salaries, subscriptions) and surface 3-day reminders."""
    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)
    cmap = {c['id']: c.get('nama', 'Lainnya') for c in all_cats}
    now_dt = get_jakarta_now()
    today = now_dt.date()

    total_approved_nominal = sum(float(t.get('nominal') or 0.0) for t in approved_expenses)

    if not approved_expenses:
        return {
            'success': True,
            'recurring_items': [],
            'upcoming_3_days_count': 0,
            'upcoming_3_days_items': [],
            'seri_layak': 0,
            'seri_lolos': 0,
            'seri_belum_teruji': 0,
            'cakupan_nominal': 0.0,
            'message': (
                "Belum ada transaksi pengeluaran disetujui yang dapat dianalisis untuk pengeluaran rutin.\n\n"
                "Cakupan prediksi: 0.0% dari total nominal pengeluaran yang dianalisis berasal dari seri yang lolos uji-mundur."
            )
        }

    # Group expenses by normalized description and category
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in approved_expenses:
        desc = (t.get('deskripsi') or '').strip().lower()
        cat_name = cmap.get(t.get('kategori_id'), 'Lainnya')
        # Clean description key
        cleaned_key = "".join([c for c in desc if c.isalnum() or c.isspace()]).strip()
        if not cleaned_key:
            cleaned_key = cat_name.lower()
        if cleaned_key not in groups:
            groups[cleaned_key] = []
        groups[cleaned_key].append(t)

    recurring_candidates = []
    known_utility_keywords = ['listrik', 'wifi', 'internet', 'sewa', 'gaji', 'air', 'pdam', 'langganan', 'retribusi', 'keamanan', 'kebersihan', 'ipl']

    seri_layak = 0
    seri_lolos = 0
    seri_belum_teruji = 0
    nominal_lolos = 0.0

    for key, items in groups.items():
        if len(items) >= 2 or any(kw in key for kw in known_utility_keywords):
            # Parse dates
            dates = []
            amounts = []
            for it in items:
                try:
                    d = date.fromisoformat(str(it['tanggal'])[:10])
                    dates.append(d)
                    amounts.append(float(it['nominal']))
                except (ValueError, TypeError):
                    continue

            if not dates:
                continue

            # Sort dates and keep amounts aligned
            sorted_pairs = sorted(zip(dates, amounts), key=lambda x: x[0])
            dates = [p[0] for p in sorted_pairs]
            amounts = [p[1] for p in sorted_pairs]

            last_date = dates[-1]
            schedule = infer_schedule(dates)
            median_interval = schedule['estimated_interval_days']

            # Skip stale series (more than 2 missed cycles)
            if median_interval and schedule['schedule_regular'] and (today - last_date).days > 2 * median_interval:
                continue

            seri_layak += 1

            historical_count = len(dates)
            if historical_count >= 4:
                bt = _backtest_series(dates, amounts)
                is_passed = bool(
                    bt['kejadian'] >= 3
                    and bt['hit_rate'] >= 0.70
                    and bt['mape_uji'] is not None
                    and bt['mape_uji'] <= 20
                )
                terbukti_uji = is_passed
                mape_uji = bt['mape_uji']
                tepat_tanggal_uji = bt['tepat_tanggal']
                kejadian = bt['kejadian']
                if is_passed:
                    seri_lolos += 1
                    nominal_lolos += sum(amounts)
            else:
                terbukti_uji = False
                mape_uji = None
                tepat_tanggal_uji = 0
                kejadian = 0
                seri_belum_teruji += 1

            cat_name = cmap.get(items[0].get('kategori_id'), 'Lainnya')
            display_desc = items[0].get('deskripsi') or key.title()

            # Project next expected date
            dates_ahead = scheduled_dates(dict(schedule, last_date=str(last_date)), today, today + timedelta(days=366))
            next_date, clipped = dates_ahead[0] if dates_ahead else (None, False)
            days_until = (next_date - today).days if next_date else None
            is_upcoming_3_days = bool(days_until is not None and 0 <= days_until <= 3)

            est_amount = float(np.median(amounts))

            pattern_desc = 'jadwal belum dapat dipastikan'
            if schedule['schedule_regular']:
                pattern_desc = 'Bulanan (tanggal kalender)' if schedule['schedule_kind'] == 'monthly' else f'Setiap {median_interval:g} hari'
            if median_interval and schedule['schedule_regular'] and median_interval <= 4:
                pattern_desc = f"Setiap {median_interval:g} hari"
            elif median_interval and schedule['schedule_regular'] and 6 <= median_interval <= 8:
                pattern_desc = "Mingguan (~7 hari)"
            elif median_interval and schedule['schedule_regular'] and 12 <= median_interval <= 16:
                pattern_desc = "Dua Mingguan (~14 hari)"

            schedule_verified = bool(schedule['schedule_regular'] and schedule.get('historical_count', 0) >= 3 and (
                items[0].get('is_verified_due_date')
                or (items[0].get('usulan') or {}).get('is_verified_due_date')
                or items[0].get('jatuh_tempo_terkonfirmasi')
            ))
            is_verified = schedule_verified and not clipped
            reminder_label = (
                "Tagihan dengan jatuh tempo terkonfirmasi"
                if is_verified
                else "Perkiraan pengeluaran rutin berdasarkan riwayat"
            )

            recurring_candidates.append({
                **schedule,
                'schedule_verified': schedule_verified,
                'key': key,
                'description': display_desc,
                'category': cat_name,
                'historical_count': len(set(dates)),
                'estimated_interval_days': median_interval,
                'pattern_description': pattern_desc,
                'estimated_amount': est_amount,
                'last_date': str(last_date),
                'next_expected_date': str(next_date) if next_date else None,
                'days_until': days_until,
                'is_upcoming_3_days': is_upcoming_3_days,
                'is_verified_due_date': is_verified,
                'reminder_type': reminder_label,
                'mape_uji': mape_uji,
                'tepat_tanggal_uji': tepat_tanggal_uji,
                'kejadian': kejadian,
                'terbukti_uji': terbukti_uji,
            })

    # Sort candidates by nearest next_expected_date
    recurring_candidates.sort(key=lambda x: x['next_expected_date'] or '9999-12-31')
    upcoming_3_days = [c for c in recurring_candidates if c['is_upcoming_3_days']]

    cakupan_nominal = float(nominal_lolos / total_approved_nominal) if total_approved_nominal > 0 else 0.0
    cakupan_pct = round(cakupan_nominal * 100, 1)
    cakupan_line = f"Cakupan prediksi: {cakupan_pct}% dari total nominal pengeluaran yang dianalisis berasal dari seri yang lolos uji-mundur."

    if not recurring_candidates:
        msg = (
            "Belum terdeteksi pengeluaran rutin yang berulang secara berkala (seperti tagihan listrik, internet, "
            "sewa, atau gaji). Sistem akan otomatis mendeteksi pola setelah transaksi tercatat beberapa kali.\n\n"
            f"{cakupan_line}"
        )
    else:
        msg = f"### Deteksi Pengeluaran Rutin & Tagihan Mendatang\n\n"
        if upcoming_3_days:
            msg += f"**Pengingat Tagihan dalam 3 Hari ke Depan ({len(upcoming_3_days)} pengeluaran):**\n"
            for item in upcoming_3_days:
                msg += (
                    f"- **{item['description']}** ({item['category']}): "
                    f"Perkiraan nominal **{format_rupiah(item['estimated_amount'])}** pada **{item['next_expected_date']}** "
                    f"(*{item['days_until']} hari lagi*) [{item['reminder_type']}]\n"
                )
            msg += "\n"

        msg += f"**Daftar Pola Pengeluaran Rutin Teridentifikasi:**\n"
        for item in recurring_candidates:
            badge = "*(Segera dipersiapkan)*" if item['is_upcoming_3_days'] else (f"({item['days_until']} hari lagi)" if item['days_until'] is not None else '')
            suffix = " • belum teruji" if not item['terbukti_uji'] else ""
            msg += (
                f"- **{item['description']}** [{item['category']}]: "
                f"Estimasi {format_rupiah(item['estimated_amount'])} • Jadwal: {item['next_expected_date'] or 'jadwal belum dapat dipastikan'} {badge} "
                f"• Pola: {item['pattern_description']} (riwayat {item['historical_count']}x){suffix}\n"
            )
        msg += f"\n{cakupan_line}\n"
        msg += "\n*Catatan: Jadwal dan nominal ini merupakan estimasi berbasis deret waktu historis, bukan tagihan resmi.*"

    return {
        'success': True,
        'recurring_items': recurring_candidates,
        'upcoming_3_days_count': len(upcoming_3_days),
        'upcoming_3_days_items': upcoming_3_days,
        'seri_layak': seri_layak,
        'seri_lolos': seri_lolos,
        'seri_belum_teruji': seri_belum_teruji,
        'cakupan_nominal': cakupan_nominal,
        'message': msg
    }


def get_forecast_for_period(
    token: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period_label: str = "Bulan Depan",
    output_mode: str = "summary",
    category_name: Optional[str] = None,
    days_ahead: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieve expense forecast for a specific future period, single day, or daily breakdown."""
    if days_ahead is not None:
        validate_forecast_horizon(days_ahead)

    actor, approved_expenses, _ = _get_eligible_expense_rows(token)
    shop_id = actor['toko_id']
    store_name = actor.get('toko_nama') or get_store_name(shop_id)

    now_dt = get_jakarta_now()
    today = now_dt.date()

    # Determine start and end date if not provided
    if not start_date or not end_date:
        if "minggu" in period_label.lower():
            start = today + timedelta(days=7 - today.weekday())
            end = start + timedelta(days=6)
        else:
            if today.month == 12:
                start = date(today.year + 1, 1, 1)
                end = date(today.year + 1, 1, 31)
            else:
                start = date(today.year, today.month + 1, 1)
                next_month_start = date(today.year, today.month + 2, 1) if today.month < 11 else date(today.year + 1, 1, 1)
                end = next_month_start - timedelta(days=1)
    else:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

    # Determine maximum historical date to compute necessary forecast horizon
    if approved_expenses:
        try:
            max_hist = max(date.fromisoformat(str(t['tanggal'])[:10]) for t in approved_expenses if t.get('tanggal'))
        except (ValueError, TypeError):
            max_hist = today
    else:
        max_hist = today

    needed_from_hist = (end - max_hist).days + 1 if end > max_hist else 1
    needed_from_today = (end - today).days + 1 if end > today else 1
    horizon_days = (
        validate_forecast_horizon(days_ahead)
        if days_ahead is not None
        else min(MAX_FORECAST_DAYS, max(MIN_FORECAST_DAYS, needed_from_hist, needed_from_today, DEFAULT_FORECAST_DAYS))
    )

    fcst = train_and_forecast_expenses(
        token,
        days_ahead=horizon_days,
        category_name=category_name,
        store_name=store_name
    )
    if not fcst.get('success'):
        return fcst

    selected_model = fcst.get('selected_model') or fcst.get('best_model')

    # Filter projections in the specific window [start, end]
    start_str = start.isoformat()
    end_str = end.isoformat()

    period_projections = [
        p for p in fcst.get('projections', [])
        if start_str <= p['ds'] <= end_str
    ]

    # Enforce strict date alignment; never relabel projections from an unrelated date range
    if not period_projections:
        last_proj_date = fcst.get('projections', [{}])[-1].get('ds') if fcst.get('projections') else None
        cov_info = f" (proyeksi model saat ini mencakup s.d. {last_proj_date})" if last_proj_date else ""
        return {
            'success': False,
            'error': 'INSUFFICIENT_COVERAGE',
            'store_name': store_name,
            'period_label': period_label,
            'start_date': start_str,
            'end_date': end_str,
            'total_predicted': 0.0,
            'total_lower': 0.0,
            'total_upper': 0.0,
            'average_daily': 0.0,
            'days_count': 0,
            'projections': [],
            'message': (
                f"Tidak tersedia proyeksi pengeluaran untuk periode **{period_label}** ({start_str} s.d. {end_str}){cov_info}. "
                f"Sistem tidak merelokasi data tanggal lain untuk menjaga integritas data keuangan."
            )
        }

    footer = get_dynamic_forecast_footer(store_name, model_name=selected_model)
    is_single_day = (start == end)

    # CASE A: Single-Day Forecast (e.g. "Besok" or specific date)
    if is_single_day and period_projections:
        day_proj = period_projections[0]
        date_val = day_proj['ds']
        pred_val = day_proj['yhat']
        low_val = day_proj['yhat_lower']
        up_val = day_proj['yhat_upper']

        if period_label.lower() in ['besok', 'esok'] or (start == today + timedelta(days=1)):
            title_period = "Besok"
        else:
            title_period = f"Tanggal {period_label}"

        model_line = f"- **Model Terpilih:** **{selected_model}**\n" if selected_model else ""
        msg = (
            f"### Prediksi Pengeluaran {title_period} — {store_name}\n\n"
            f"- **Tanggal:** {date_val}\n"
            f"{model_line}"
            f"- **Estimasi Pengeluaran:** **{format_rupiah(pred_val)}**\n"
            f"- **Rentang Ketidakpastian (80% CI):** {format_rupiah(low_val)} s.d. {format_rupiah(up_val)}\n"
            f"- **Dasar Prediksi:** Riwayat transaksi pengeluaran disetujui toko\n\n"
            f"> ℹ️ {footer}"
        )
        return {
            'success': True,
            'store_name': store_name,
            'selected_model': selected_model,
            'period_label': period_label,
            'output_mode': 'single_day',
            'start_date': start_str,
            'end_date': end_str,
            'total_predicted': round(pred_val, 2),
            'total_lower': round(low_val, 2),
            'total_upper': round(up_val, 2),
            'average_daily': round(pred_val, 2),
            'days_count': 1,
            'projections': [day_proj],
            'message': msg
        }

    total_pred = sum(p['yhat'] for p in period_projections)
    total_low = sum(p['yhat_lower'] for p in period_projections)
    total_up = sum(p['yhat_upper'] for p in period_projections)
    avg_pred = total_pred / len(period_projections) if period_projections else 0.0
    model_line = f"- **Model Terpilih:** **{selected_model}**\n" if selected_model else ""

    # CASE B: Daily Breakdown Table (e.g. "rincian 7 hari ke depan")
    normalized_mode = str(output_mode).lower().replace("forecastoutputmode.", "").strip()
    if normalized_mode in ('daily_breakdown', 'breakdown', 'daily'):
        table_rows = [
            "| Tanggal | Prediksi Pengeluaran | Rentang Estimasi (80% CI) |",
            "| :--- | :--- | :--- |"
        ]
        for p in period_projections:
            table_rows.append(
                f"| {p['ds']} | {format_rupiah(p['yhat'])} | {format_rupiah(p['yhat_lower'])} s.d. {format_rupiah(p['yhat_upper'])} |"
            )

        table_md = "\n".join(table_rows)
        msg = (
            f"### Rincian Prediksi Pengeluaran Harian ({period_label}) — {store_name}\n\n"
            f"{table_md}\n\n"
            f"{model_line}"
            f"- **Total Estimasi ({len(period_projections)} Hari):** **{format_rupiah(total_pred)}**\n"
            f"- **Rata-rata Harian:** {format_rupiah(avg_pred)}/hari\n\n"
            f"> ℹ️ {footer}"
        )
        return {
            'success': True,
            'store_name': store_name,
            'selected_model': selected_model,
            'period_label': period_label,
            'output_mode': 'daily_breakdown',
            'start_date': start_str,
            'end_date': end_str,
            'total_predicted': round(total_pred, 2),
            'total_lower': round(total_low, 2),
            'total_upper': round(total_up, 2),
            'average_daily': round(avg_pred, 2),
            'days_count': len(period_projections),
            'projections': period_projections,
            'message': msg
        }

    # CASE C: Summary Mode (Multi-day aggregate)
    msg = (
        f"### Prediksi Pengeluaran Toko ({period_label}) — {store_name}\n\n"
        f"- **Periode Proyeksi:** {start_str} s.d. {end_str} ({len(period_projections)} hari)\n"
        f"{model_line}"
        f"- **Estimasi Total Pengeluaran:** **{format_rupiah(total_pred)}**\n"
        f"- **Rentang Ketidakpastian (80% CI):** {format_rupiah(total_low)} s.d. {format_rupiah(total_up)}\n"
        f"- **Rata-rata Pengeluaran Harian:** {format_rupiah(avg_pred)}/hari\n\n"
        f"> ℹ️ {footer}"
    )

    return {
        'success': True,
        'store_name': store_name,
        'selected_model': selected_model,
        'period_label': period_label,
        'output_mode': 'summary',
        'start_date': start_str,
        'end_date': end_str,
        'total_predicted': round(total_pred, 2),
        'total_lower': round(total_low, 2),
        'total_upper': round(total_up, 2),
        'average_daily': round(avg_pred, 2),
        'days_count': len(period_projections),
        'projections': period_projections,
        'message': msg
    }


def get_next_predicted_expense(token: str) -> Dict[str, Any]:
    """Retrieve the nearest expected future expense from recurring schedules or daily forecast."""
    rec_result = detect_recurring_expenses(token)

    dated_items = [item for item in rec_result.get('recurring_items', []) if item.get('next_expected_date')]
    if dated_items:
        nearest = dated_items[0]
        days_str = "hari ini" if nearest['days_until'] == 0 else f"{nearest['days_until']} hari lagi"
        msg = (
            f"### Pengeluaran Mendatang Terdekat\n\n"
            f"- **Keterangan:** {nearest['description']} ({nearest['category']})\n"
            f"- **Estimasi Tanggal:** **{nearest['next_expected_date']}** ({days_str})\n"
            f"- **Estimasi Nominal:** **{format_rupiah(nearest['estimated_amount'])}**\n"
            f"- **Pola Riwayat:** {nearest['pattern_description']}\n\n"
            f"*Catatan: Ini adalah perkiraan pengeluaran berdasarkan pola riwayat transaksi toko.*"
        )
        return {
            'success': True,
            'type': 'recurring',
            'item': nearest,
            'message': msg
        }

    fcst = train_and_forecast_expenses(token, days_ahead=7)
    if not fcst.get('success'):
        return fcst

    selected_model = fcst.get('selected_model', 'Multi-Model Terbaik')
    projections = fcst.get('projections', [])
    if projections:
        first_day = projections[0]
        msg = (
            f"### Estimasi Pengeluaran Berikutnya ({selected_model})\n\n"
            f"- **Tanggal:** **{first_day['ds']}**\n"
            f"- **Estimasi Nominal Harian:** **{format_rupiah(first_day['yhat'])}**\n"
            f"- **Rentang Estimasi:** {format_rupiah(first_day['yhat_lower'])} - {format_rupiah(first_day['yhat_upper'])}\n\n"
            f"*Berdasarkan model peramalan terpilih {selected_model}.*"
        )
        return {
            'success': True,
            'type': 'daily_forecast',
            'selected_model': selected_model,
            'item': first_day,
            'message': msg
        }

    return {
        'success': False,
        'message': "Tidak ada proyeksi pengeluaran terdekat yang dapat dihitung."
    }


def compare_actual_vs_forecast(
    token: str,
    actual_period_label: str = "Bulan Ini",
    forecast_period_label: str = "Bulan Depan"
) -> Dict[str, Any]:
    """Compare current actual expense performance against forecasted expense for the upcoming period."""
    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)
    now_dt = get_jakarta_now()
    today = now_dt.date()

    first_this_month = date(today.year, today.month, 1)
    actual_rows = [
        t for t in approved_expenses
        if str(t.get('tanggal', '')) >= first_this_month.isoformat()
    ]
    actual_sum = sum(float(t['nominal']) for t in actual_rows)

    fcst = get_forecast_for_period(token, period_label=forecast_period_label)
    if not fcst.get('success'):
        return fcst

    selected_model = fcst.get('selected_model') or 'Multi-Model Terbaik'
    forecast_sum = fcst['total_predicted']
    diff = forecast_sum - actual_sum
    pct_diff = (diff / actual_sum * 100) if actual_sum > 0 else 0.0
    direction_word = "meningkat" if diff > 0 else "menurun"

    msg = (
        f"### Perbandingan Realisasi vs Prediksi Pengeluaran\n\n"
        f"- **Realisasi Pengeluaran ({actual_period_label} s.d. hari ini):** {format_rupiah(actual_sum)} ({len(actual_rows)} transaksi)\n"
        f"- **Prediksi Pengeluaran ({forecast_period_label}):** **{format_rupiah(forecast_sum)}**\n"
        f"- **Model Terpilih:** **{selected_model}**\n"
        f"- **Perubahan Proyeksi:** Proyeksi pengeluaran diperkirakan **{direction_word} sebesar {format_rupiah(abs(diff))}** ({abs(pct_diff):.1f}%)\n"
        f"- **Rentang Prediksi:** {format_rupiah(fcst['total_lower'])} s.d. {format_rupiah(fcst['total_upper'])}\n\n"
        f"*Analisis ini membandingkan data realisasi yang telah disetujui dengan model peramalan terpilih {selected_model}.*"
    )

    return {
        'success': True,
        'selected_model': selected_model,
        'actual_label': actual_period_label,
        'actual_amount': round(actual_sum, 2),
        'forecast_label': forecast_period_label,
        'forecast_amount': round(forecast_sum, 2),
        'difference': round(diff, 2),
        'percentage_difference': round(pct_diff, 2),
        'message': msg
    }


def get_expense_trend(token: str, days_ahead: int = 30) -> Dict[str, Any]:
    """Analyze the trajectory of store expenses (increasing, decreasing, or stable) and top category drivers."""
    validated_days = validate_forecast_horizon(days_ahead)
    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)
    cmap = {c['id']: c.get('nama', 'Lainnya') for c in all_cats}

    fcst = train_and_forecast_expenses(token, days_ahead=validated_days)
    if not fcst.get('success'):
        return fcst

    selected_model = fcst.get('selected_model', 'Multi-Model Terbaik')
    hist_avg = fcst['historical_daily_average']
    pred_avg = fcst['average_daily_predicted']

    diff = pred_avg - hist_avg
    pct_change = (diff / hist_avg * 100) if hist_avg > 0 else 0.0

    if abs(pct_change) < 5.0:
        trend = "stabil"
        trend_label = "Stabil (Relatif Konstan)"
        trend_explanation = "Rata-rata pengeluaran harian diperkirakan relatif sama dengan tren historis."
    elif diff > 0:
        trend = "meningkat"
        trend_label = "Meningkat"
        trend_explanation = f"Rata-rata pengeluaran harian diproyeksikan naik sekitar {pct_change:.1f}% dibanding rata-rata historis."
    else:
        trend = "menurun"
        trend_label = "Menurun"
        trend_explanation = f"Rata-rata pengeluaran harian diproyeksikan turun sekitar {abs(pct_change):.1f}% dibanding rata-rata historis."

    cat_totals = {}
    for t in approved_expenses:
        c_name = cmap.get(t.get('kategori_id'), 'Lainnya')
        cat_totals[c_name] = cat_totals.get(c_name, 0.0) + float(t['nominal'])

    sorted_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)[:3]
    top_cat_lines = "\n".join([f"- **{name}:** {format_rupiah(amt)}" for name, amt in sorted_cats])

    msg = (
        f"### Analisis Tren Pengeluaran Toko\n\n"
        f"- **Arah Tren:** **{trend_label}**\n"
        f"- **Model Peramalan:** **{selected_model}**\n"
        f"- **Rata-rata Historis:** {format_rupiah(hist_avg)}/hari\n"
        f"- **Estimasi Rata-rata Proyeksi:** {format_rupiah(pred_avg)}/hari\n"
        f"- **Penjelasan Tren:** {trend_explanation}\n\n"
        f"**Pos Pengeluaran Terbesar yang Mempengaruhi Tren:**\n"
        f"{top_cat_lines}\n\n"
        f"*Analisis tren didasarkan pada perbandingan matematis antara rata-rata deret waktu historis dan proyeksi model {selected_model}.*"
    )

    return {
        'success': True,
        'selected_model': selected_model,
        'trend': trend,
        'historical_daily_avg': round(hist_avg, 2),
        'predicted_daily_avg': round(pred_avg, 2),
        'percentage_change': round(pct_change, 2),
        'top_categories': sorted_cats,
        'message': msg
    }


def check_forecast_against_cash_balance(token: str, days_ahead: int = 30) -> Dict[str, Any]:
    """Compare authorized recorded net cash balance against upcoming forecasted expenses to assess cash sufficiency."""
    validated_days = validate_forecast_horizon(days_ahead)
    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)

    all_rows = period_rows(token)
    financial_metrics = metrics(all_rows, all_cats)
    cash_balance = float(financial_metrics.get('arus_kas_total', 0.0))

    fcst = train_and_forecast_expenses(token, days_ahead=validated_days)
    if not fcst.get('success'):
        return fcst

    selected_model = fcst.get('selected_model', 'Multi-Model Terbaik')
    pred_total = fcst['total_predicted_expense']
    gap = cash_balance - pred_total
    is_sufficient = gap >= 0

    if is_sufficient:
        status_label = "Saldo Kas Tercatat Mencukupi"
        status_desc = (
            f"Saldo kas bersih toko saat ini ({format_rupiah(cash_balance)}) lebih besar daripada "
            f"total estimasi pengeluaran {validated_days} hari ke depan ({format_rupiah(pred_total)}), "
            f"dengan surplus cadangan sebesar **{format_rupiah(gap)}**."
        )
    else:
        status_label = "Potensi Defisit Kas Terdeteksi"
        status_desc = (
            f"Saldo kas bersih toko saat ini ({format_rupiah(cash_balance)}) lebih kecil daripada "
            f"total estimasi pengeluaran {validated_days} hari ke depan ({format_rupiah(pred_total)}), "
            f"sehingga terdapat potensi defisit kas sebesar **{format_rupiah(abs(gap))}** jika tidak ada pemasukan baru."
        )

    disclaimer = (
        "Pemeriksaan ini membandingkan saldo kas tercatat saat ini dengan estimasi pengeluaran. "
        "Ini merupakan indikator kesiapan kas operasional, bukan jaminan likuiditas mutlak karena "
        "belum memperhitungkan potensi pemasukan baru yang akan diterima di masa depan."
    )

    msg = (
        f"### Kesiapan Saldo Kas vs Proyeksi Pengeluaran ({validated_days} Hari ke Depan)\n\n"
        f"- **Status Kesiapan:** **{status_label}**\n"
        f"- **Model Peramalan:** **{selected_model}**\n"
        f"- **Saldo Kas Bersih Saat Ini:** {format_rupiah(cash_balance)}\n"
        f"- **Estimasi Total Pengeluaran ({validated_days} Hari):** {format_rupiah(pred_total)}\n"
        f"- **Selisih Kas:** {format_rupiah(gap)}\n\n"
        f"{status_desc}\n\n"
        f"> ℹ️ *{disclaimer}*"
    )

    return {
        'success': True,
        'selected_model': selected_model,
        'cash_balance': round(cash_balance, 2),
        'predicted_expense': round(pred_total, 2),
        'difference': round(gap, 2),
        'is_sufficient': is_sufficient,
        'days_ahead': validated_days,
        'disclaimer': disclaimer,
        'message': msg
    }
