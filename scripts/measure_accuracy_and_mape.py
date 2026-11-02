"""Unified Measurement Script: Classification Accuracy & Forecast MAPE.

Evaluates against PRD quantitative targets:
1. Classification Accuracy >= 85.0% (Evaluated on independent 63 ground-truth samples)
2. Forecast MAPE < 20.0% (Evaluated on real Supabase store transaction series)

Usage:
    python scripts/measure_accuracy_and_mape.py
    python scripts/measure_accuracy_and_mape.py --days 7
    python scripts/measure_accuracy_and_mape.py --forecast-only --tune
    python scripts/measure_accuracy_and_mape.py --shop-id <store-uuid>
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
    print(f"PRD Target (>= 85.0%)       : {'[PASS] ' + f'{acc_pct:.2f}% >= 85%' if target_met else '[FAIL] Below 85%'}")
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


def predict_baselines(train, horizon):
    # Fixed-origin multi-step forecasts: recurse on predictions, never on test actuals.
    history = list(train['y'].astype(float))
    if len(history) < 7:
        raise ValueError("Baselines require at least 7 training days")
    moving, seasonal = history.copy(), history.copy()
    for _ in range(horizon):
        moving.append(float(np.mean(moving[-7:])))
        seasonal.append(seasonal[-7])
    return {'MA7 recursive': np.asarray(moving[-horizon:]),
            'Seasonal naive': np.asarray(seasonal[-horizon:])}


def weekly_evaluation(dates, actual, predicted):
    daily = pd.DataFrame({'actual': np.asarray(actual, dtype=float), 'predicted': np.asarray(predicted, dtype=float)}, index=pd.DatetimeIndex(dates))
    weekly = daily.resample('W-SUN').sum()
    counts = daily.resample('W-SUN').size()
    return calculate_forecast_metrics(weekly['actual'], weekly['predicted']), list(counts)


def pct(value):
    return f"{value:.2f}%" if value is not None else 'N/A'


def print_metric_table(results):
    print('| Model | Daily MAPE | WAPE | Median APE | MAE Rp | RMSE Rp |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, item in results.items():
        m = item['metrics']
        print(f"| {name} | {pct(m['mape'])} | {pct(m['wape'])} | {pct(m['median_ape'])} | {m['mae']:.2f} | {m['rmse']:.2f} |")


def _print_target_summary(horizons):
    print('\nFinal summary: <20% comparison per percentage metric (PRD headline = daily Prophet MAPE, 28 days).')
    print('WAPE/median/weekly/baselines are diagnostics, not replacements for the PRD headline.')
    for days, result in horizons.items():
        for name, item in result['models'].items():
            for metric in ('mape', 'wape', 'median_ape'):
                value = item['metrics'][metric]
                status = 'PASS' if value is not None and value < 20 else 'FAIL' if value is not None else 'SKIPPED'
                print(f'{days}d {name} {metric}: {pct(value)} [{status}]')
            for metric in ('mape', 'wape'):
                value = item['weekly'][metric]
                status = 'PASS' if value is not None and value < 20 else 'FAIL' if value is not None else 'SKIPPED'
                print(f'{days}d {name} weekly {metric}: {pct(value)} [{status}]')


def run_forecast_mape_benchmark(shop_id: Optional[str] = None, test_days: int = 28,
                                tune: bool = False) -> Dict[str, Any]:
    print("\n" + '=' * 68 + "\nPROPHET BENCHMARK: REAL SUPABASE, FIXED-ORIGIN HOLDOUT\n" + '=' * 68)
    if not HAS_PROPHET or Prophet is None:
        raise RuntimeError('Meta Prophet is not installed')
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
    print(f'Store: {shop_name} ({target_shop_id}); approved expenses: {len(expenses)}')
    print(f"Range: {daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d}; {len(frame)} days; {int((daily > 0).sum())} active days")
    print('Missing calendar days between first/last transaction treated as zero, matching production; completeness not independently verified.')
    print('MAPE/median APE exclude actual Rp0; MAE/RMSE/WAPE include those days. No epsilon or smoothing.')
    horizons = {}
    for days in sorted({7, 28, test_days}):
        if days < 1 or len(frame) - days < 14:
            raise ValueError(f'{shop_name}: need at least 14 training days before the {days}-day holdout')
        train, test = frame.iloc[:-days].copy(), frame.iloc[-days:].copy()
        print(f"\nHOLDOUT {days}d: {test['ds'].iloc[0]:%Y-%m-%d} to {test['ds'].iloc[-1]:%Y-%m-%d}; train {len(train)} days")
        predicted = predict_prophet(train, test['ds'])
        print_daily_breakdown(test['ds'], test['y'], predicted)
        models = {}
        for name, values in {'Prophet current': predicted, **predict_baselines(train, days)}.items():
            weekly, counts = weekly_evaluation(test['ds'], test['y'], values)
            models[name] = {'metrics': calculate_forecast_metrics(test['y'], values),
                            'weekly': weekly, 'predictions': values.tolist()}
        print_metric_table(models)
        print(f'Weekly totals: calendar Mon-Sun, held-out dates ONLY; bin day counts {counts} (edge bins may be partial).')
        print('| Model | Weekly MAPE | Weekly WAPE | Weekly MAE Rp |')
        print('|---|---:|---:|---:|')
        for name, item in models.items():
            m = item['weekly']
            print(f"| {name} | {pct(m['mape'])} | {pct(m['wape'])} | {m['mae']:.2f} |")
        horizons[days] = {'models': models, 'train_days': len(train)}
    current = horizons[28]['models']['Prophet current']['metrics']
    baselines = [horizons[28]['models'][name]['metrics'] for name in ('MA7 recursive', 'Seasonal naive')]
    beats_baselines = all(current['mape'] is not None and b['mape'] is not None and current['mape'] < b['mape'] for b in baselines)
    print(f'Prophet beats BOTH naive baselines on 28d MAPE: {beats_baselines}')
    # Steps 1-4 must be inspected before explicitly requesting the small tuning grid.
    variants = {}
    if tune:
        print('Experimental one-change-at-a-time variants; same outer holdout, NOT independent confirmation.')
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
        eligible = [name for name, item in variants.items()
                    if all(item['metrics']['mape'] is not None and ref['mape'] is not None
                           and item['metrics']['mape'] < ref['mape'] for ref in [current, *baselines])]
        print(f'Variants beating current AND both baselines on 28d MAPE alone: {eligible or "NONE"}')
        robust_eligible = [name for name in eligible if all(
            variants[name]['metrics'][metric] < ref[metric]
            for metric in ('mape', 'wape', 'median_ape') for ref in [current, *baselines])]
        print(f'Variants improving ALL THREE diagnostics over current and both baselines: {robust_eligible or "NONE"}')
        print('No automatic production adoption: selection on this holdout requires separate validation and cross-store checks.')
    _print_target_summary(horizons)
    if current['mape'] is None or current['mape'] >= 20:
        best = min([current, *(v['metrics'] for v in variants.values())], key=lambda m: float('inf') if m['mape'] is None else m['mape'])
        print(f"Known limitation: MAPE harian pada data UMKM jarang/tidak teratur masih gagal target <20%; "
              f"MAPE Prophet saat ini {pct(current['mape'])}, terbaik terukur {pct(best['mape'])}; "
              f"MAPE mingguan saat ini {pct(horizons[28]['models']['Prophet current']['weekly']['mape'])}. "
              'Agregasi mingguan hanya diagnostik; headline harian dan target PRD tidak diganti.')
    selected = horizons[test_days]['models']['Prophet current']['metrics']
    return {'success': True, 'shop_id': target_shop_id, 'shop_name': shop_name,
            'test_days': test_days, 'train_days': horizons[test_days]['train_days'],
            **selected, 'target_met': selected['mape'] is not None and selected['mape'] < 20,
            'horizons': horizons, 'variants': variants}


def main():
    parser = argparse.ArgumentParser(description='Measure classification accuracy and real forecast diagnostics.')
    parser.add_argument('--days', type=int, default=28, help='Headline holdout days (default 28; 7 remains available).')
    parser.add_argument('--shop-id', default=None, help='Supabase store UUID.')
    parser.add_argument('--forecast-only', action='store_true', help='Skip the unrelated classification benchmark.')
    parser.add_argument('--tune', action='store_true', help='Run the limited diagnostic grid AFTER reviewing steps 1-4.')
    args = parser.parse_args()
    acc_res = None if args.forecast_only else run_classification_accuracy_benchmark()
    result = run_forecast_mape_benchmark(args.shop_id, args.days, args.tune)
    if acc_res is not None:
        print(f"Classification: {acc_res['accuracy_pct']:.2f}% [{'PASS' if acc_res['target_met'] else 'FAIL'}]")
    print(f"Requested headline: {args.days}d daily Prophet MAPE {pct(result['mape'])}; default PRD headline remains 28d.")
    sys.exit(0 if result['target_met'] and (acc_res is None or acc_res['target_met']) else 1)


if __name__ == '__main__':
    main()
