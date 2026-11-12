"""Recompute real forecasts on isolated local test data; no tenant DB access."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services import forecast_service as forecast


def run(label):
    rows = [dict(id=f'test-{i}', tanggal=str(date(2026, 6, 1)+timedelta(days=i)),
                 nominal=100000+i*250+(i % 7)*1250, kategori_id='ops', tipe='pengeluaran')
            for i in range(120)]
    actor = dict(toko_id='display-only-new-local-test-shop', toko_nama='Display Test')
    cats = [dict(id='ops', nama='Operasional', tipe='pengeluaran')]
    np.random.seed(42)
    forecast.invalidate_forecast_cache(actor['toko_id'])
    with patch.object(forecast, '_get_eligible_expense_rows', return_value=(actor, rows, cats)), \
         patch.object(forecast, 'get_forecast_readiness_status', return_value={'ready': True}):
        result = forecast.train_and_forecast_expenses('test-only', days_ahead=30)
    assert result['success'], result
    out = ROOT / 'evidence/display_simplification'
    (out/f'forecast_{label}.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    if label == 'after':
        before = json.loads((out/'forecast_before.json').read_text(encoding='utf-8'))
        assert before == result, 'Recomputed forecast differs from the before result'
    print(label, result['selected_model'], len(result['projections']), result['total_predicted_expense'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
