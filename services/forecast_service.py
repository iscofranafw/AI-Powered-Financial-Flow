"""Prophet Forecasting Service for AI-Powered Financial Recording & Cash Flow Assistant.

Provides store-specific, owner-only expense forecasting, recurring-expense detection,
accuracy backtesting (MAE, RMSE, MAPE), cash-sufficiency analysis, and trend evaluation
using Meta Prophet, Pandas, and NumPy.
"""

import logging
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

from services.auth_service import verify_actor, AccessError
from repositories.finance_repository import transactions, categories
from services.report_service import period_rows, metrics
from utils.ui_components import format_rupiah
from ai.financial_chat import get_jakarta_now

# In-memory forecast cache keyed by (store_id, expense_count, latest_tx_id, days_ahead, category)
_FORECAST_CACHE: Dict[Tuple[str, int, str, int, Optional[str]], Dict[str, Any]] = {}


def invalidate_forecast_cache(store_id: Optional[str] = None) -> None:
    """Invalidate in-memory forecasting cache for a specific store or globally."""
    global _FORECAST_CACHE
    if store_id:
        _FORECAST_CACHE = {k: v for k, v in _FORECAST_CACHE.items() if k[0] != store_id}
    else:
        _FORECAST_CACHE.clear()


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


def get_dynamic_forecast_footer(store_name: str) -> str:
    """Produce the dynamic disclaimer footer scoped to the authenticated store name."""
    store_display = store_name if store_name and store_name != "toko Anda" else "toko Anda"
    return (
        f"Estimasi pengeluaran {store_display} dihitung menggunakan model Prophet berdasarkan "
        f"riwayat transaksi pengeluaran yang telah disetujui pada toko tersebut. Hasil ini merupakan "
        f"perkiraan statistik, bukan kepastian pengeluaran di masa mendatang."
    )


def get_forecast_readiness_status(token: str) -> Dict[str, Any]:
    """Check if the store has sufficient historical expense transactions to run Prophet."""
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

    # Requirements for reliable Prophet forecasting:
    # 1. At least 5 approved expense transactions
    # 2. At least 5 distinct transaction days
    # 3. Date span of at least 7 days
    req_tx = 5
    req_days = 5
    req_span = 7

    is_ready = (total_count >= req_tx) and (distinct_days >= req_days) and (date_span_days >= req_span)

    if is_ready:
        msg = (
            f"**Data Keuangan Siap untuk Forecasting Prophet**\n\n"
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
            f"**Data Belum Mencukupi untuk Forecasting Prophet yang Andal**\n\n"
            f"Model Prophet memerlukan riwayat data yang memadai untuk mengenali tren dan pola pengeluaran toko. "
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


def train_and_forecast_expenses(
    token: str,
    days_ahead: int = 30,
    category_name: Optional[str] = None,
    store_name: Optional[str] = None
) -> Dict[str, Any]:
    """Train Meta Prophet model on daily aggregated approved expenses and generate forecast."""
    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)
    shop_id = actor['toko_id']
    resolved_store_name = store_name or actor.get('toko_nama') or get_store_name(shop_id)

    if not HAS_PROPHET or Prophet is None:
        return {
            'success': False,
            'error': 'PROPHET_UNAVAILABLE',
            'message': (
                "Pustaka peramalan deret waktu Meta Prophet belum terpasang di environment Python saat ini. "
                "Silakan jalankan perintah: `pip install prophet` atau aktifkan virtual environment (.venv)."
            )
        }

    readiness = get_forecast_readiness_status(token)
    if not readiness['ready']:
        return {
            'success': False,
            'readiness': readiness,
            'error': 'INSUFFICIENT_DATA',
            'message': readiness['message']
        }

    # Filter by category if requested
    target_cat_id = None
    target_cat_label = None
    if category_name:
        for c in all_cats:
            if category_name.lower() in c.get('nama', '').lower():
                target_cat_id = c['id']
                target_cat_label = c.get('nama')
                break

        if target_cat_id:
            approved_expenses = [t for t in approved_expenses if t.get('kategori_id') == target_cat_id]
            if len(approved_expenses) < 3:
                return {
                    'success': False,
                    'error': 'INSUFFICIENT_CATEGORY_DATA',
                    'message': (
                        f"Data pengeluaran untuk kategori **'{target_cat_label}'** baru memiliki {len(approved_expenses)} transaksi. "
                        f"Dibutuhkan minimal 3 transaksi pada kategori ini untuk membuat peramalan khusus."
                    )
                }
        else:
            return {
                'success': False,
                'error': 'CATEGORY_NOT_FOUND',
                'message': f"Kategori pengeluaran **'{category_name}'** tidak ditemukan di daftar kategori toko Anda."
            }

    # Cache key check
    latest_id = str(approved_expenses[-1].get('id', '')) if approved_expenses else ''
    cache_key = (shop_id, len(approved_expenses), latest_id, days_ahead, target_cat_label)
    if cache_key in _FORECAST_CACHE:
        return _FORECAST_CACHE[cache_key]

    # Build daily aggregated series
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
        return {
            'success': False,
            'error': 'NO_VALID_RECORDS',
            'message': "Tidak ada catatan tanggal transaksi pengeluaran yang valid."
        }

    df_raw = pd.DataFrame(records)
    daily = df_raw.groupby('ds')['nominal'].sum()

    # Reindex daily to fill zero-expense days between min_date and max_date
    min_d = daily.index.min()
    max_d = daily.index.max()
    full_idx = pd.date_range(start=min_d, end=max_d, freq='D')
    daily_series = daily.reindex(full_idx, fill_value=0.0)
    daily_indexed = pd.DataFrame({
        'ds': pd.to_datetime(daily_series.index),
        'y': daily_series.values.astype(float)
    })

    date_span = (max_d - min_d).days + 1
    use_weekly = bool(date_span >= 14)

    try:
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=use_weekly,
            yearly_seasonality=False,
            interval_width=0.80
        )
        model.fit(daily_indexed[['ds', 'y']])

        future = model.make_future_dataframe(periods=days_ahead, freq='D')
        forecast = model.predict(future)

        # Clip predictions at 0 because expenses cannot be negative
        forecast['yhat'] = forecast['yhat'].clip(lower=0.0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0.0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0.0)

        # Extract only future projection days
        max_dt = pd.to_datetime(max_d)
        future_mask = forecast['ds'] > max_dt
        future_forecast = forecast[future_mask].copy()

        daily_projections = []
        for _, row in future_forecast.iterrows():
            daily_projections.append({
                'ds': row['ds'].strftime('%Y-%m-%d'),
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
            'date': peak_row['ds'].strftime('%Y-%m-%d') if peak_row is not None else None,
            'amount': round(float(peak_row['yhat']), 2) if peak_row is not None else 0.0
        }

        # Historical baseline
        hist_total = float(daily_indexed['y'].sum())
        hist_avg_daily = float(daily_indexed['y'].mean())

        result = {
            'success': True,
            'store_id': shop_id,
            'store_name': resolved_store_name,
            'days_ahead': days_ahead,
            'category_filtered': target_cat_label,
            'total_predicted_expense': total_predicted,
            'total_predicted_lower': total_lower,
            'total_predicted_upper': total_upper,
            'average_daily_predicted': avg_daily,
            'historical_total_expense': hist_total,
            'historical_daily_average': hist_avg_daily,
            'peak_expense': peak_info,
            'historical_coverage': {
                'start_date': str(min_d),
                'end_date': str(max_d),
                'total_days': date_span,
                'distinct_transaction_days': len(daily)
            },
            'projections': daily_projections
        }

        _FORECAST_CACHE[cache_key] = result
        return result

    except Exception as exc:
        logging.error(f"Prophet fitting error: {exc}")
        return {
            'success': False,
            'error': 'FIT_ERROR',
            'message': f"Terjadi kendala teknis saat memproses deret waktu Prophet: {str(exc)}"
        }


def evaluate_forecast_accuracy(token: str, test_days: int = 7) -> Dict[str, Any]:
    """Perform time-ordered holdout backtesting to evaluate Prophet accuracy (MAE, RMSE, MAPE)."""
    actor, approved_expenses, _ = _get_eligible_expense_rows(token)
    shop_id = actor['toko_id']

    if not HAS_PROPHET or Prophet is None:
        return {
            'success': False,
            'ready': False,
            'message': "Pustaka Meta Prophet belum terpasang di environment Python saat ini."
        }

    readiness = get_forecast_readiness_status(token)
    if not readiness['ready']:
        return {
            'success': False,
            'ready': False,
            'message': readiness['message']
        }

    # Extract daily series
    records = []
    for t in approved_expenses:
        try:
            records.append({
                'ds': pd.Timestamp(str(t['tanggal'])[:10]),
                'nominal': float(t['nominal'])
            })
        except (ValueError, TypeError):
            continue

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

    total_days = len(daily_indexed)
    if total_days < 14:
        return {
            'success': True,
            'ready': False,
            'mae': None,
            'rmse': None,
            'mape': None,
            'target_met': False,
            'message': (
                f"Evaluasi akurasi Prophet memerlukan riwayat data minimal 14 hari berturut-turut untuk validasi holdout "
                f"(saat ini {total_days} hari). Tambahkan catatan transaksi untuk mengaktifkan pengujian akurasi."
            )
        }

    # Holdout split: last `test_days` as test set
    split_idx = total_days - test_days
    train_df = daily_indexed.iloc[:split_idx].copy()
    test_df = daily_indexed.iloc[split_idx:].copy()

    try:
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=bool(len(train_df) >= 14),
            yearly_seasonality=False,
            interval_width=0.80
        )
        model.fit(train_df[['ds', 'y']])

        future = model.make_future_dataframe(periods=test_days, freq='D')
        forecast = model.predict(future)

        # Merge actual and predicted
        test_forecast = forecast.iloc[split_idx:].copy()
        y_true = test_df['y'].to_numpy(dtype=float)
        y_pred = test_forecast['yhat'].clip(lower=0.0).to_numpy(dtype=float)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # Weighted Absolute Percentage Error (WAPE)
        sum_actual = float(np.sum(y_true))
        wape = float(np.sum(np.abs(y_true - y_pred)) / sum_actual * 100) if sum_actual > 0 else None

        # Mean Absolute Percentage Error (MAPE) computed only on non-zero actual days
        non_zero_mask = y_true > 0
        n_nonzero = int(np.sum(non_zero_mask))
        mape_note = ""
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
            zero_count = int(np.sum(~non_zero_mask))
            if zero_count > 0:
                mape_note = f"MAPE dihitung khusus pada {n_nonzero} hari dengan pengeluaran aktif (mengecualikan {zero_count} hari Rp0)."
        else:
            mape = None
            mape_note = "Seluruh hari pada periode uji memiliki pengeluaran Rp0, sehingga MAPE tidak dapat dihitung (pembagian dengan nol dicegah)."

        # Target rule: target MAPE < 20% ONLY when strictly measured
        target_met = bool(mape is not None and mape < 20.0 and n_nonzero >= 5)

        msg = (
            f"### Evaluasi Akurasi Model Peramalan Prophet\n\n"
            f"- **Metode Evaluasi:** Holdout Backtesting ({test_days} hari terakhir sebagai data uji)\n"
            f"- **MAE (Mean Absolute Error):** {format_rupiah(mae)}\n"
            f"- **RMSE (Root Mean Squared Error):** {format_rupiah(rmse)}\n"
        )
        if mape is not None:
            msg += f"- **MAPE (Mean Absolute Percentage Error):** {mape:.2f}%\n"
            if target_met:
                msg += f"- **Target (<20%):** Terpenuhi (Akurasi model tergolong sangat baik)\n"
            elif n_nonzero < 5:
                msg += f"- **Target (<20%):** Belum dapat disimpulkan (hanya {n_nonzero} hari aktif dari {test_days} hari uji — butuh minimal 5 hari aktif)\n"
            else:
                msg += f"- **Target (<20%):** Belum Terpenuhi ({mape:.2f}% ≥ 20.0%)\n"
        else:
            msg += f"- **MAPE:** Tidak dapat dihitung karena tidak ada pengeluaran pada hari uji.\n"

        if wape is not None:
            msg += f"- **WAPE (Weighted APE):** {wape:.2f}%\n"

        if mape_note:
            msg += f"\n*Catatan Metrik:* {mape_note}"

        return {
            'success': True,
            'ready': True,
            'test_days': test_days,
            'train_days': len(train_df),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2) if mape is not None else None,
            'wape': round(wape, 2) if wape is not None else None,
            'target_met': target_met,
            'mape_note': mape_note,
            'message': msg
        }

    except Exception as exc:
        logging.error(f"Prophet evaluation error: {exc}")
        return {
            'success': False,
            'error': str(exc),
            'message': f"Gagal menghitung metrik evaluasi model Prophet: {str(exc)}"
        }


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
            if len(dates) >= 2:
                intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
                median_interval = int(np.median(intervals))
            else:
                # Default monthly for known utilities with 1 observation
                median_interval = 30

            # Skip stale series (more than 2 missed cycles)
            if (today - last_date).days > 2 * median_interval:
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
                if not is_passed:
                    continue
                terbukti_uji = True
                mape_uji = bt['mape_uji']
                tepat_tanggal_uji = bt['tepat_tanggal']
                kejadian = bt['kejadian']
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
            next_date = last_date + timedelta(days=median_interval)
            # Roll forward if in the past
            while next_date < today:
                next_date += timedelta(days=median_interval)

            days_until = (next_date - today).days
            is_upcoming_3_days = bool(0 <= days_until <= 3)

            est_amount = float(np.median(amounts))

            pattern_desc = "Bulanan (~30 hari)"
            if median_interval <= 4:
                pattern_desc = f"Setiap {median_interval} hari"
            elif 6 <= median_interval <= 8:
                pattern_desc = "Mingguan (~7 hari)"
            elif 12 <= median_interval <= 16:
                pattern_desc = "Dua Mingguan (~14 hari)"

            is_verified = bool(
                items[0].get('is_verified_due_date')
                or (items[0].get('usulan') or {}).get('is_verified_due_date')
                or items[0].get('jatuh_tempo_terkonfirmasi')
            )
            reminder_label = (
                "Tagihan dengan jatuh tempo terkonfirmasi"
                if is_verified
                else "Perkiraan pengeluaran rutin berdasarkan riwayat"
            )

            recurring_candidates.append({
                'key': key,
                'description': display_desc,
                'category': cat_name,
                'historical_count': len(dates),
                'estimated_interval_days': median_interval,
                'pattern_description': pattern_desc,
                'estimated_amount': est_amount,
                'last_date': str(last_date),
                'next_expected_date': str(next_date),
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
    recurring_candidates.sort(key=lambda x: x['next_expected_date'])
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
            badge = "*(Segera dipersiapkan)*" if item['is_upcoming_3_days'] else f"({item['days_until']} hari lagi)"
            suffix = " • belum teruji" if not item['terbukti_uji'] else ""
            msg += (
                f"- **{item['description']}** [{item['category']}]: "
                f"Estimasi {format_rupiah(item['estimated_amount'])} • Jadwal: {item['next_expected_date']} {badge} "
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
    category_name: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve Prophet expense forecast for a specific future period, single day, or daily breakdown."""
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
            # Default to next calendar month
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
    days_ahead = max(needed_from_hist, needed_from_today, 30)

    fcst = train_and_forecast_expenses(
        token,
        days_ahead=days_ahead,
        category_name=category_name,
        store_name=store_name
    )
    if not fcst.get('success'):
        return fcst

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

    footer = get_dynamic_forecast_footer(store_name)
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

        msg = (
            f"### Prediksi Pengeluaran {title_period} — {store_name}\n\n"
            f"- **Tanggal:** {date_val}\n"
            f"- **Estimasi Pengeluaran:** **{format_rupiah(pred_val)}**\n"
            f"- **Rentang Ketidakpastian (80% CI):** {format_rupiah(low_val)} s.d. {format_rupiah(up_val)}\n"
            f"- **Dasar Prediksi:** Riwayat transaksi pengeluaran disetujui toko\n\n"
            f"> ℹ️ {footer}"
        )
        return {
            'success': True,
            'store_name': store_name,
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
            f"- **Total Estimasi ({len(period_projections)} Hari):** **{format_rupiah(total_pred)}**\n"
            f"- **Rata-rata Harian:** {format_rupiah(avg_pred)}/hari\n\n"
            f"> ℹ️ {footer}"
        )
        return {
            'success': True,
            'store_name': store_name,
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
        f"- **Estimasi Total Pengeluaran:** **{format_rupiah(total_pred)}**\n"
        f"- **Rentang Ketidakpastian (80% CI):** {format_rupiah(total_low)} s.d. {format_rupiah(total_up)}\n"
        f"- **Rata-rata Pengeluaran Harian:** {format_rupiah(avg_pred)}/hari\n\n"
        f"> ℹ️ {footer}"
    )

    return {
        'success': True,
        'store_name': store_name,
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
    """Retrieve the nearest expected future expense from recurring schedules and Prophet daily peaks."""
    # Check recurring expenses first
    rec_result = detect_recurring_expenses(token)
    now_dt = get_jakarta_now()
    today = now_dt.date()

    if rec_result.get('recurring_items'):
        nearest = rec_result['recurring_items'][0]
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

    # Fallback to Prophet 7-day forecast
    fcst = train_and_forecast_expenses(token, days_ahead=7)
    if not fcst.get('success'):
        return fcst

    projections = fcst.get('projections', [])
    if projections:
        first_day = projections[0]
        msg = (
            f"### Estimasi Pengeluaran Berikutnya (Prophet)\n\n"
            f"- **Tanggal:** **{first_day['ds']}**\n"
            f"- **Estimasi Nominal Harian:** **{format_rupiah(first_day['yhat'])}**\n"
            f"- **Rentang Estimasi:** {format_rupiah(first_day['yhat_lower'])} - {format_rupiah(first_day['yhat_upper'])}\n\n"
            f"*Berdasarkan model time-series Prophet harian.*"
        )
        return {
            'success': True,
            'type': 'daily_forecast',
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
    cmap = {c['id']: c for c in all_cats}
    now_dt = get_jakarta_now()
    today = now_dt.date()

    # Actual current month expenses
    first_this_month = date(today.year, today.month, 1)
    actual_rows = [
        t for t in approved_expenses
        if str(t.get('tanggal', '')) >= first_this_month.isoformat()
    ]
    actual_sum = sum(float(t['nominal']) for t in actual_rows)

    # Forecast for next month
    fcst = get_forecast_for_period(token, period_label=forecast_period_label)
    if not fcst.get('success'):
        return fcst

    forecast_sum = fcst['total_predicted']
    diff = forecast_sum - actual_sum
    pct_diff = (diff / actual_sum * 100) if actual_sum > 0 else 0.0

    direction_word = "meningkat" if diff > 0 else "menurun"

    msg = (
        f"### Perbandingan Realisasi vs Prediksi Pengeluaran\n\n"
        f"- **Realisasi Pengeluaran ({actual_period_label} s.d. hari ini):** {format_rupiah(actual_sum)} ({len(actual_rows)} transaksi)\n"
        f"- **Prediksi Pengeluaran ({forecast_period_label}):** **{format_rupiah(forecast_sum)}**\n"
        f"- **Perubahan Proyeksi:** Proyeksi pengeluaran diperkirakan **{direction_word} sebesar {format_rupiah(abs(diff))}** ({abs(pct_diff):.1f}%)\n"
        f"- **Rentang Prediksi:** {format_rupiah(fcst['total_lower'])} s.d. {format_rupiah(fcst['total_upper'])}\n\n"
        f"*Analisis ini membandingkan data realisasi yang telah disetujui dengan model deret waktu Meta Prophet.*"
    )

    return {
        'success': True,
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
    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)
    cmap = {c['id']: c.get('nama', 'Lainnya') for c in all_cats}

    fcst = train_and_forecast_expenses(token, days_ahead=days_ahead)
    if not fcst.get('success'):
        return fcst

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

    # Identify top expense categories from historical data
    cat_totals = {}
    for t in approved_expenses:
        c_name = cmap.get(t.get('kategori_id'), 'Lainnya')
        cat_totals[c_name] = cat_totals.get(c_name, 0.0) + float(t['nominal'])

    sorted_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)[:3]
    top_cat_lines = "\n".join([f"- **{name}:** {format_rupiah(amt)}" for name, amt in sorted_cats])

    msg = (
        f"### Analisis Tren Pengeluaran Toko\n\n"
        f"- **Arah Tren:** **{trend_label}**\n"
        f"- **Rata-rata Historis:** {format_rupiah(hist_avg)}/hari\n"
        f"- **Estimasi Rata-rata Proyeksi:** {format_rupiah(pred_avg)}/hari\n"
        f"- **Penjelasan Tren:** {trend_explanation}\n\n"
        f"**Pos Pengeluaran Terbesar yang Mempengaruhi Tren:**\n"
        f"{top_cat_lines}\n\n"
        f"*Analisis tren didasarkan pada perbandingan matematis antara rata-rata deret waktu historis dan proyeksi model Prophet.*"
    )

    return {
        'success': True,
        'trend': trend,
        'historical_daily_avg': round(hist_avg, 2),
        'predicted_daily_avg': round(pred_avg, 2),
        'percentage_change': round(pct_change, 2),
        'top_categories': sorted_cats,
        'message': msg
    }


def check_forecast_against_cash_balance(token: str, days_ahead: int = 30) -> Dict[str, Any]:
    """Compare authorized recorded net cash balance against upcoming forecasted expenses to assess cash sufficiency."""
    actor, approved_expenses, all_cats = _get_eligible_expense_rows(token)

    # Get authoritative financial numbers
    all_rows = period_rows(token)
    financial_metrics = metrics(all_rows, all_cats)
    cash_balance = float(financial_metrics.get('arus_kas_total', 0.0))

    # Get expense forecast
    fcst = train_and_forecast_expenses(token, days_ahead=days_ahead)
    if not fcst.get('success'):
        return fcst

    pred_total = fcst['total_predicted_expense']
    gap = cash_balance - pred_total
    is_sufficient = gap >= 0

    if is_sufficient:
        status_label = "Saldo Kas Tercatat Mencukupi"
        status_desc = (
            f"Saldo kas bersih toko saat ini ({format_rupiah(cash_balance)}) lebih besar daripada "
            f"total estimasi pengeluaran {days_ahead} hari ke depan ({format_rupiah(pred_total)}), "
            f"dengan surplus cadangan sebesar **{format_rupiah(gap)}**."
        )
    else:
        status_label = "Potensi Defisit Kas Terdeteksi"
        status_desc = (
            f"Saldo kas bersih toko saat ini ({format_rupiah(cash_balance)}) lebih kecil daripada "
            f"total estimasi pengeluaran {days_ahead} hari ke depan ({format_rupiah(pred_total)}), "
            f"sehingga terdapat potensi defisit kas sebesar **{format_rupiah(abs(gap))}** jika tidak ada pemasukan baru."
        )

    disclaimer = (
        "Pemeriksaan ini membandingkan saldo kas tercatat saat ini dengan estimasi pengeluaran. "
        "Ini merupakan indikator kesiapan kas operasional, bukan jaminan likuiditas mutlak karena "
        "belum memperhitungkan potensi pemasukan baru yang akan diterima di masa depan."
    )

    msg = (
        f"### Kesiapan Saldo Kas vs Proyeksi Pengeluaran ({days_ahead} Hari ke Depan)\n\n"
        f"- **Status Kesiapan:** **{status_label}**\n"
        f"- **Saldo Kas Bersih Saat Ini:** {format_rupiah(cash_balance)}\n"
        f"- **Estimasi Total Pengeluaran ({days_ahead} Hari):** {format_rupiah(pred_total)}\n"
        f"- **Selisih Kas:** {format_rupiah(gap)}\n\n"
        f"{status_desc}\n\n"
        f"> ℹ️ *{disclaimer}*"
    )

    return {
        'success': True,
        'cash_balance': round(cash_balance, 2),
        'predicted_expense': round(pred_total, 2),
        'difference': round(gap, 2),
        'is_sufficient': is_sufficient,
        'days_ahead': days_ahead,
        'disclaimer': disclaimer,
        'message': msg
    }
