"""Unified Measurement Script: Classification Accuracy & Multi-Model Forecast Benchmark.

Evaluates against Project Plan DB15-I023 revisi 1 quantitative targets:
1. RQ1: Classification Accuracy >= 85.0%
2. RQ2: Expense Forecast MAPE <= 20.0% (+ MAE/RMSE/WAPE) across candidate models:
   Prophet, Seasonal Naive, MA7, ETS, ARIMA, LinearRegression.

Usage:
    python scripts/measure_accuracy_and_mape.py
    python scripts/measure_accuracy_and_mape.py --days 28
    python scripts/measure_accuracy_and_mape.py --forecast-only --shop-id <store-uuid>
    python scripts/measure_accuracy_and_mape.py --forecast-only --synthetic-csv <path-to-csv>
    python scripts/measure_accuracy_and_mape.py --forecast-only --tune
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress cmdstanpy and prophet console log spam
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from prophet import Prophet
    HAS_PROPHET = True
except (ImportError, ModuleNotFoundError):
    Prophet = None
    HAS_PROPHET = False

from ai.classifier import classify
from config.settings import get_settings
from database.client import privileged_client
from services.forecast_service import (
    CANDIDATE_MODEL_ORDER,
    _MODEL_FITTERS,
    evaluate_forecast_candidates_df,
)
from tests.test_accuracy_evaluation import (
    GROUND_TRUTH_DATASET,
    _get_eval_categories,
    calculate_metrics,
    run_legacy_naive_baseline,
)


def run_classification_accuracy_benchmark() -> Dict[str, Any]:
    print("\n" + "=" * 68)
    print(" 1. TRANSACTION CLASSIFICATION ACCURACY BENCHMARK")
    print("=" * 68)

    y_true = [item[3] for item in GROUND_TRUTH_DATASET]
    legacy_preds, legacy_abstentions = run_legacy_naive_baseline(GROUND_TRUTH_DATASET)
    legacy_metrics = calculate_metrics(y_true, legacy_preds, legacy_abstentions)

    new_preds = []
    new_abstentions = 0

    for domain, tp, desc, expected, _ in GROUND_TRUTH_DATASET:
        cats = _get_eval_categories(domain)
        cat_lookup = {c['id']: c['nama'] for c in cats}

        res = classify(tp, desc, cats, business_type=domain)
        cid = res.get('kategori_id')
        predicted_name = cat_lookup.get(cid, 'Others')

        if res.get('status_kategori') == 'perlu_tinjauan':
            new_abstentions += 1

        new_preds.append(predicted_name)

    new_metrics = calculate_metrics(y_true, new_preds, new_abstentions)
    acc_pct = new_metrics['accuracy'] * 100
    macro_f1_pct = new_metrics['macro_f1'] * 100
    target_met = acc_pct >= 85.0

    print(f"Dataset Evaluated           : Benchmark Ground Truth ({len(GROUND_TRUTH_DATASET)} samples)")
    print(f"Domains Covered             : Retail, Bakery, Warung Makan")
    print(f"Legacy Baseline Accuracy    : {legacy_metrics['accuracy'] * 100:.2f}% (Macro-F1: {legacy_metrics['macro_f1'] * 100:.2f}%)")
    print(f"New Store-Aware Accuracy    : {acc_pct:.2f}% (Macro-F1: {macro_f1_pct:.2f}%)")
    print(f"Accuracy Lift Over Baseline : +{(new_metrics['accuracy'] - legacy_metrics['accuracy']) * 100:.2f}%")
    print(f"Abstention / Review Rate    : {new_metrics['abstention_rate'] * 100:.2f}%")
    print(f"Target Project Plan DB15-I023 revisi 1 (RQ1 >= 85.0%) : {'[PASS] ' + f'{acc_pct:.2f}% >= 85%' if target_met else '[FAIL] Below 85%'}")
    print("-" * 68)
    print("Per-Category Breakdown (New Store-Aware Engine):")
    for cat_name, metrics in sorted(new_metrics['per_category'].items()):
        print(f" - {cat_name:26}: Precision {metrics['precision']*100:5.1f}% | Recall {metrics['recall']*100:5.1f}% | F1 {metrics['f1']*100:5.1f}%")

    return {
        'total_samples': len(GROUND_TRUTH_DATASET),
        'accuracy_pct': acc_pct,
        'macro_f1_pct': macro_f1_pct,
        'target_met': target_met,
    }


def _fetch_approved_expenses(client, shop_id):
    rows, offset = [], 0
    while True:
        page = (client.table('transaksi').select('id,tanggal,nominal,tipe,status_persetujuan')
                .eq('toko_id', shop_id).eq('tipe', 'pengeluaran')
                .eq('status_persetujuan', 'disetujui').order('tanggal').order('id')
                .range(offset, offset + 999).execute().data)
        if page is None:
            raise RuntimeError('Supabase returned no transaction result')
        rows.extend(page)
        if len(page) < 1000:
            return rows
        offset += 1000


def get_real_supabase_transactions(shop_id: Optional[str] = None) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Retrieve real approved expense transactions from the configured Supabase project.

    Fails loudly if Supabase is unreachable, credentials are missing, or no store
    has adequate transaction data.
    """
    try:
        settings = get_settings()
        supabase_url = settings.get('SUPABASE_URL', '')
        client = privileged_client()
    except Exception as exc:
        print(f"\n[FATAL ERROR] Gagal menginisialisasi client Supabase: {exc}", file=sys.stderr)
        print("Pastikan berkas .env telah dikonfigurasi dengan SUPABASE_URL dan SUPABASE_SERVICE_ROLE_KEY.", file=sys.stderr)
        sys.exit(1)

    try:
        stores_resp = client.table('toko').select('id, nama').execute()
        stores = stores_resp.data or []
    except Exception as exc:
        print(f"\n[FATAL ERROR] Gagal melakukan query ke tabel 'toko' di Supabase ({supabase_url}): {exc}", file=sys.stderr)
        print("Pastikan database Supabase / PostgreSQL aktif dan seluruh migrasi (001-010) telah diterapkan.", file=sys.stderr)
        sys.exit(1)

    if not stores:
        print(f"\n[FATAL ERROR] Tidak ada data toko pada proyek Supabase ({supabase_url}).", file=sys.stderr)
        print("Pastikan toko telah dibuat/disetujui di database sebelum mengukur MAPE.", file=sys.stderr)
        sys.exit(1)

    selected_shop_id = shop_id
    selected_shop_name = ""

    if selected_shop_id:
        matched = [s for s in stores if s['id'] == selected_shop_id]
        if not matched:
            print(f"\n[FATAL ERROR] Toko dengan ID '{selected_shop_id}' tidak ditemukan di Supabase.", file=sys.stderr)
            sys.exit(1)
        selected_shop_name = matched[0]['nama']
    else:
        # Auto-detect store with the most approved expense transactions
        candidates = []
        for s in stores:
            try:
                tx_data = _fetch_approved_expenses(client, s['id'])
                if len(tx_data) >= 5:
                    candidates.append((s, len(tx_data), tx_data))
            except Exception as exc:
                print(f"\n[FATAL ERROR] Gagal memeriksa transaksi toko '{s.get('nama')}': {exc}", file=sys.stderr)
                sys.exit(1)

        if not candidates:
            print(
                "\n[FATAL ERROR] Tidak ditemukan toko dengan transaksi pengeluaran disetujui (minimal 5 transaksi) "
                f"pada proyek Supabase ({supabase_url}).",
                file=sys.stderr
            )
            print("Silakan impor/catat transaksi riil toko terlebih dahulu sebelum mengukur MAPE.", file=sys.stderr)
            sys.exit(1)

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_store, _, _ = candidates[0]
        selected_shop_id = best_store['id']
        selected_shop_name = best_store['nama']

    # Fetch all approved expense transactions for the chosen shop
    try:
        approved_expenses = _fetch_approved_expenses(client, selected_shop_id)
    except Exception as exc:
        print(f"\n[FATAL ERROR] Gagal mengambil transaksi pengeluaran toko '{selected_shop_name}': {exc}", file=sys.stderr)
        sys.exit(1)

    return selected_shop_id, selected_shop_name, approved_expenses


def print_daily_breakdown(dates, actual, predicted):
    """Show every held-out day; zeros affect MAE/RMSE/WAPE, never MAPE."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    errors = np.abs(actual - predicted)
    active = actual > 0
    apes = errors[active] / actual[active] * 100
    print("| Date | Actual Rp | Predicted Rp | Absolute error Rp | APE % |")
    print("|---|---:|---:|---:|---|")
    for ds, a, pred, err in zip(dates, actual, predicted, errors):
        ape = f"{err / a * 100:.2f}" if a > 0 else "EXCLUDED: actual Rp0 (division by zero)"
        print(f"| {pd.Timestamp(ds):%Y-%m-%d} | {a:.2f} | {pred:.2f} | {err:.2f} | {ape} |")
    print(f"Holdout days: {len(actual)}; active days: {int(active.sum())}; zero days: {int((~active).sum())}")
    if len(apes):
        worst_idx = np.flatnonzero(active)[int(np.argmax(apes))]
        worst = float(max(apes))
        contribution = worst / len(apes)
        top_two_share = float(np.sort(apes)[-2:].sum() / apes.sum() * 100) if apes.sum() else 0.0
        print(f"Worst day: {pd.Timestamp(list(dates)[worst_idx]):%Y-%m-%d}; APE {worst:.2f}%; contribution to mean MAPE {contribution:.2f} percentage points")
        print(f"Top two days contribute {top_two_share:.2f}% of summed APE; "
              + ("errors dominated by 1-2 outlier days (>50%)." if top_two_share > 50 else "errors spread beyond 1-2 days (top two <=50%)."))


def calculate_forecast_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual/predicted must be aligned, non-empty and finite")
    error = np.abs(actual - predicted)
    active = actual > 0
    apes = error[active] / actual[active] * 100
    return {
        'mape': float(np.mean(apes)) if len(apes) else None,
        'median_ape': float(np.median(apes)) if len(apes) else None,
        'wape': float(error.sum() / actual.sum() * 100) if actual.sum() > 0 else None,
        'mae': float(error.mean()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
    }


def predict_prophet(train, dates, overrides=None, drop_zeros=False):
    # Same point-forecast configuration as forecast_service.py. No changes to
    # production settings; experimental variants are benchmark-only.
    config = dict(daily_seasonality=False, weekly_seasonality=bool(len(train) >= 14),
                  yearly_seasonality=False, interval_width=0.80)
    config.update(overrides or {})
    fit_data = train[train['y'] > 0] if drop_zeros else train
    if len(fit_data) < 2:
        raise ValueError("Insufficient training observations")
    model = Prophet(**config)
    model.fit(fit_data[['ds', 'y']])
    # Predict exact held-out dates, including when zero training days are dropped.
    return model.predict(pd.DataFrame({'ds': list(dates)}))['yhat'].clip(lower=0).to_numpy(dtype=float)


def load_synthetic_csv_transactions(csv_path: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Load expense rows from a synthetic benchmark CSV (explicitly labeled synthetic)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic CSV not found: {path}")
    df = pd.read_csv(path)
    if 'tanggal' not in df.columns or 'nominal' not in df.columns:
        raise ValueError("Synthetic CSV must include 'tanggal' and 'nominal' columns")
    if 'tipe' in df.columns:
        df = df[df['tipe'].astype(str).str.strip().str.lower() == 'pengeluaran'].copy()
    expenses = [
        {
            'tanggal': str(row['tanggal'])[:10],
            'nominal': float(row['nominal']),
            'tipe': 'pengeluaran',
            'status_persetujuan': 'disetujui',
        }
        for _, row in df.iterrows()
    ]
    return 'SYNTHETIC_CSV', f'DATA SINTETIS ({path.name})', expenses


def predict_candidates(train: pd.DataFrame, horizon: int) -> Tuple[Dict[str, np.ndarray], List[Dict[str, str]]]:
    """Evaluate all Revisi 3 candidate forecast fitters on the exact same training split."""
    train_reset = train[['ds', 'y']].copy().reset_index(drop=True)
    predictions: Dict[str, np.ndarray] = {}
    skipped: List[Dict[str, str]] = []
    for model_name in CANDIDATE_MODEL_ORDER:
        fitter = _MODEL_FITTERS.get(model_name)
        if fitter is None:
            continue
        try:
            pred_df = fitter(train_reset, horizon)
            yhat = np.asarray(pred_df['yhat'], dtype=float)[:horizon]
            if len(yhat) != horizon:
                raise ValueError(f"Panjang prediksi {len(yhat)} tidak sama dengan horizon {horizon}")
            predictions[model_name] = np.clip(yhat, 0.0, None)
        except Exception as exc:
            skipped.append({'model': model_name, 'reason': str(exc)})
    return predictions, skipped


def weekly_evaluation(dates, actual, predicted):
    daily = pd.DataFrame({'actual': np.asarray(actual, dtype=float), 'predicted': np.asarray(predicted, dtype=float)}, index=pd.DatetimeIndex(dates))
    weekly = daily.resample('W-SUN').sum()
    counts = daily.resample('W-SUN').size()
    return calculate_forecast_metrics(weekly['actual'], weekly['predicted']), list(counts)


def pct(value):
    return f"{value:.2f}%" if value is not None else 'N/A'


def print_metric_table(results, best_model: Optional[str] = None):
    print('| Model | Daily MAPE | WAPE | Median APE | MAE Rp | RMSE Rp | Status |')
    print('|---|---:|---:|---:|---:|---:|---|')
    for name, item in results.items():
        m = item['metrics']
        status = 'TERPILIH (terbaik)' if best_model and name == best_model else 'Kandidat'
        print(f"| {name} | {pct(m['mape'])} | {pct(m['wape'])} | {pct(m['median_ape'])} | {m['mae']:.2f} | {m['rmse']:.2f} | {status} |")


def _print_target_summary(horizons):
    print('\nFinal summary: <=20% comparison per percentage metric (Project Plan DB15-I023 revisi 1 RQ2 headline = selected best model MAPE, 28 days).')
    print('WAPE/median/weekly metrics are diagnostics alongside daily MAPE and MAE.')
    for days, result in horizons.items():
        best_model = result.get('best_model')
        print(f'--- Holdout {days}d | Selected Best Model: {best_model} ---')
        for name, item in result['models'].items():
            for metric in ('mape', 'wape', 'median_ape'):
                value = item['metrics'][metric]
                status = 'PASS' if value is not None and value <= 20 else 'FAIL' if value is not None else 'SKIPPED'
                print(f'{days}d {name} {metric}: {pct(value)} [{status}]')
            for metric in ('mape', 'wape'):
                value = item['weekly'][metric]
                status = 'PASS' if value is not None and value <= 20 else 'FAIL' if value is not None else 'SKIPPED'
                print(f'{days}d {name} weekly {metric}: {pct(value)} [{status}]')


def run_forecast_mape_benchmark(shop_id: Optional[str] = None, test_days: int = 28,
                                tune: bool = False, synthetic_csv: Optional[str] = None) -> Dict[str, Any]:
    is_synthetic = bool(synthetic_csv)
    if is_synthetic:
        print("\n" + '=' * 68)
        print("*** SYNTHETIC DATASET BENCHMARK — NOT REAL STORE DATA ***")
        print("*** DATA SINTETIS — BUKAN DATA TOKO ASLI ***")
        print('=' * 68)
        target_shop_id, shop_name, expenses = load_synthetic_csv_transactions(synthetic_csv)
    else:
        print("\n" + '=' * 68 + "\nMULTI-MODEL FORECAST BENCHMARK: REAL SUPABASE, FIXED-ORIGIN HOLDOUT\n" + '=' * 68)
        target_shop_id, shop_name, expenses = get_real_supabase_transactions(shop_id)

    records = [{'ds': pd.Timestamp(str(t['tanggal'])[:10]), 'nominal': float(t['nominal'])} for t in expenses]
    if not records:
        raise ValueError(f'No approved expenses for {shop_name}')
    raw = pd.DataFrame(records)
    if raw['ds'].isna().any() or not np.isfinite(raw['nominal']).all() or (raw['nominal'] <= 0).any():
        raise ValueError('Invalid transaction data; no records silently discarded')
    daily = raw.groupby('ds')['nominal'].sum()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq='D'), fill_value=0.0)
    frame = pd.DataFrame({'ds': daily.index, 'y': daily.values.astype(float)})
    data_label = 'SYNTHETIC (NOT REAL STORE DATA)' if is_synthetic else 'REAL SUPABASE STORE'
    print(f'Data Source: {data_label}')
    print(f'Store: {shop_name} ({target_shop_id}); expense records: {len(expenses)}')
    print(f"Range: {daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d}; {len(frame)} days; {int((daily > 0).sum())} active days")
    print('Missing calendar days between first/last transaction treated as zero, matching production.')
    print('MAPE/median APE exclude actual Rp0; MAE/RMSE/WAPE include those days. No epsilon or smoothing.')

    horizons = {}
    for days in sorted({7, 28, test_days}):
        if days < 1 or len(frame) - days < 14:
            raise ValueError(f'{shop_name}: need at least 14 training days before the {days}-day holdout')
        train, test = frame.iloc[:-days].copy(), frame.iloc[-days:].copy()
        print(f"\nHOLDOUT {days}d: {test['ds'].iloc[0]:%Y-%m-%d} to {test['ds'].iloc[-1]:%Y-%m-%d}; train {len(train)} days")

        eval_service = evaluate_forecast_candidates_df(frame, test_days=days)
        pred_map, skipped = predict_candidates(train, days)
        if not pred_map:
            raise RuntimeError(f"All candidate models failed on {days}d holdout: {skipped}")

        models: Dict[str, Any] = {}
        for name, values in pred_map.items():
            weekly, counts = weekly_evaluation(test['ds'], test['y'], values)
            models[name] = {
                'metrics': calculate_forecast_metrics(test['y'], values),
                'weekly': weekly,
                'predictions': values.tolist(),
            }
        # Sort rows consistently by MAPE ascending, then MAE ascending
        models = dict(sorted(
            models.items(),
            key=lambda kv: (
                float('inf') if kv[1]['metrics']['mape'] is None else kv[1]['metrics']['mape'],
                kv[1]['metrics']['mae'],
                kv[1]['metrics']['rmse'],
            ),
        ))
        best_model = eval_service.get('best_model') or next(iter(models))
        best_pred = pred_map.get(best_model)
        if best_pred is not None:
            print(f"Daily breakdown for selected best model [{best_model}]:")
            print_daily_breakdown(test['ds'], test['y'], best_pred)
        if skipped:
            print("Skipped candidate models:", skipped)

        print_metric_table(models, best_model=best_model)
        print(f'Weekly totals: calendar Mon-Sun, held-out dates ONLY; bin day counts {counts} (edge bins may be partial).')
        print('| Model | Weekly MAPE | Weekly WAPE | Weekly MAE Rp |')
        print('|---|---:|---:|---:|')
        for name, item in models.items():
            m = item['weekly']
            print(f"| {name} | {pct(m['mape'])} | {pct(m['wape'])} | {m['mae']:.2f} |")
        horizons[days] = {
            'models': models,
            'train_days': len(train),
            'best_model': best_model,
            'target_status': eval_service.get('target_status', ''),
            'skipped_models': skipped,
        }

    headline_28 = horizons[28]
    best_28_name = headline_28['best_model']
    best_28 = headline_28['models'][best_28_name]['metrics']
    prophet_28 = headline_28['models'].get('Prophet', {}).get('metrics')
    baselines = [headline_28['models'][name]['metrics'] for name in ('MA7', 'Seasonal Naive') if name in headline_28['models']]
    if prophet_28 and baselines:
        beats_baselines = all(
            prophet_28['mape'] is not None and b['mape'] is not None and prophet_28['mape'] < b['mape']
            for b in baselines
        )
        print(f'Prophet beats BOTH naive baselines on 28d MAPE: {beats_baselines}')

    variants = {}
    if tune:
        if not HAS_PROPHET or Prophet is None:
            raise RuntimeError('Meta Prophet is not installed for --tune')
        print('Experimental one-change-at-a-time Prophet variants; same outer holdout, NOT independent confirmation.')
        train, test = frame.iloc[:-28].copy(), frame.iloc[-28:].copy()
        specs = [('changepoint=0.05', {'changepoint_prior_scale': 0.05}, False),
                 ('changepoint=0.5', {'changepoint_prior_scale': 0.5}, False),
                 ('weekly=on', {'weekly_seasonality': True}, False),
                 ('weekly=off', {'weekly_seasonality': False}, False),
                 ('mode=additive', {'seasonality_mode': 'additive'}, False),
                 ('mode=multiplicative', {'seasonality_mode': 'multiplicative'}, False),
                 ('zeros=keep', {}, False), ('zeros=drop', {}, True)]
        for name, config, drop in specs:
            try:
                pred = predict_prophet(train, test['ds'], config, drop)
                variants[name] = {'metrics': calculate_forecast_metrics(test['y'], pred)}
            except Exception as exc:
                print(f'{name}: FAILED {exc}')
        print_metric_table(variants)
        if prophet_28 and baselines:
            eligible = [name for name, item in variants.items()
                        if all(item['metrics']['mape'] is not None and ref['mape'] is not None
                               and item['metrics']['mape'] < ref['mape'] for ref in [prophet_28, *baselines])]
            print(f'Variants beating Prophet AND both baselines on 28d MAPE alone: {eligible or "NONE"}')
            robust_eligible = [name for name in eligible if all(
                variants[name]['metrics'][metric] < ref[metric]
                for metric in ('mape', 'wape', 'median_ape') for ref in [prophet_28, *baselines])]
            print(f'Variants improving ALL THREE diagnostics over Prophet and both baselines: {robust_eligible or "NONE"}')

    _print_target_summary(horizons)
    if best_28['mape'] is None or best_28['mape'] >= 20:
        print(f"Known limitation: MAPE harian pada data transaksi jarang/tidak teratur belum memenuhi target <=20%; "
              f"model terbaik 28d={best_28_name} (MAPE {pct(best_28['mape'])}, MAE Rp {best_28['mae']:.2f}); "
              f"MAPE mingguan {best_28_name} {pct(headline_28['models'][best_28_name]['weekly']['mape'])}.")
    else:
        print(f"RQ2 Target (<= 20.0% MAPE) TERCAPAI: Model terbaik 28d={best_28_name} "
              f"(MAPE {pct(best_28['mape'])}, MAE Rp {best_28['mae']:.2f}).")

    selected_horizon = horizons[test_days]
    selected_best = selected_horizon['best_model']
    selected_metrics = selected_horizon['models'][selected_best]['metrics']
    return {
        'success': True,
        'is_synthetic': is_synthetic,
        'shop_id': target_shop_id,
        'shop_name': shop_name,
        'best_model': selected_best,
        'test_days': test_days,
        'train_days': selected_horizon['train_days'],
        **selected_metrics,
        'target_met': selected_metrics['mape'] is not None and selected_metrics['mape'] <= 20,
        'horizons': horizons,
        'variants': variants,
    }


def main():
    parser = argparse.ArgumentParser(description='Measure classification accuracy and multi-model forecast diagnostics.')
    parser.add_argument('--days', type=int, default=28, help='Headline holdout days (default 28; 7 remains available).')
    parser.add_argument('--shop-id', default=None, help='Supabase store UUID.')
    parser.add_argument('--synthetic-csv', default=None, help='Optional CSV path for explicitly labeled SYNTHETIC forecast benchmarking.')
    parser.add_argument('--forecast-only', action='store_true', help='Skip the unrelated classification benchmark.')
    parser.add_argument('--tune', action='store_true', help='Run the limited Prophet diagnostic grid.')
    args = parser.parse_args()
    acc_res = None if args.forecast_only else run_classification_accuracy_benchmark()
    result = run_forecast_mape_benchmark(args.shop_id, args.days, args.tune, synthetic_csv=args.synthetic_csv)
    if acc_res is not None:
        print(f"Classification: {acc_res['accuracy_pct']:.2f}% [{'PASS' if acc_res['target_met'] else 'FAIL'}]")
    label = "DATA SINTETIS — BUKAN DATA TOKO ASLI" if result.get('is_synthetic') else "Real Supabase Store"
    print(f"Requested headline [{label}]: {args.days}d best model={result['best_model']} MAPE {pct(result['mape'])}, MAE Rp {result['mae']:.2f}; default RQ2 headline remains 28d.")
    sys.exit(0 if result['target_met'] and (acc_res is None or acc_res['target_met']) else 1)


if __name__ == '__main__':
    main()
