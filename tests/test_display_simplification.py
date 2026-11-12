"""Display-only regressions; diagnostics remain in the original payloads."""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def evaluate_display_assignment(path, name, scope):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def test_import_100_rows_hide_diagnostics_and_preserve_payload():
    from services.import_category_policy import preview_category_label
    rows = [dict(tanggal='2026-11-12', deskripsi=f'Baris {i}', nominal=100+i,
                 tipe='pengeluaran', kategori_id='ops' if i % 2 else None,
                 status_kategori='dikonfirmasi' if i % 2 else 'perlu_tinjauan',
                 is_unresolved=not bool(i % 2), skor=.45, source='llm',
                 usulan={'llm_confidence': 85, 'similarity_diag': {'top_score': .45}})
            for i in range(100)]
    before = copy.deepcopy(rows)
    display = evaluate_display_assignment(ROOT/'pages/import_view.py', 'preview_data',
        dict(suggested=rows, cat_name_map={'ops': 'Operasional'}, preview_category_label=preview_category_label))
    assert set(display[0]) == {'Baris', 'Tanggal', 'Deskripsi', 'Nominal', 'Jenis', 'Kategori', 'Status'}
    assert len(display) == 50  # Existing preview limit; all 100 underlying rows remain.
    assert display[0]['Status'] == 'Perlu Tinjauan' and display[1]['Status'] == 'Terpetakan'
    assert display[0]['Kategori'] == 'Belum Terpetakan (Perlu Tinjauan)'
    assert rows == before


def test_others_preview_hides_diagnostics_keeps_review_status_and_payload():
    from utils.ui_components import format_rupiah
    groups = [dict(deskripsi='contoh', tipe='pengeluaran', count=2, total_nominal=100,
                   suggested_kategori_id=cid, suggested_kategori_nama='Operasional' if cid else 'Perlu Tinjauan',
                   skor=.87, source='similarity') for cid in ('ops', None)]
    preview = {'groups': groups}
    before = copy.deepcopy(preview)
    display = evaluate_display_assignment(ROOT/'pages/finance.py', 'table_rows',
                                         dict(preview=preview, format_rupiah=format_rupiah))
    assert set(display[0]) == {'Deskripsi', 'Jenis', 'Jumlah', 'Total Nominal', 'Usulan Kategori', 'Status'}
    assert [r['Status'] for r in display] == ['Terpetakan', 'Perlu Tinjauan']
    assert preview == before


def test_quick_entry_never_displays_numeric_diagnostics_or_model_message():
    payload = dict(tanggal='2026-11-12', nominal=100, deskripsi='contoh', tipe='pengeluaran',
                   kategori_id='ops', status_kategori='dikonfirmasi', skor=.87654, source='similarity',
                   message='Kemiripan 0.87654 margin 0.12345', usulan={'llm_confidence': 99})
    original = copy.deepcopy(payload)
    with patch('services.finance_service.list_categories', return_value=[dict(id='ops', nama='Operasional', tipe='pengeluaran')]):
        app = AppTest.from_string("import streamlit as st\nfrom pages.ai_tools import quick_entry\nquick_entry('test')")
        app.session_state['quick_preview'] = payload
        app.run()
    assert not app.exception
    text = '\n'.join(str(e.value) for e in [*app.markdown, *app.info])
    assert 'Terpetakan' in text
    assert '0.87654' not in text and 'Keyakinan' not in text and 'Kemiripan' not in text
    assert app.session_state['quick_preview'] == original


def main_elements(node):
    if getattr(node, 'type', '') == 'expander' and node.label == 'Detail teknis':
        return
    yield node
    for child in getattr(node, 'children', {}).values():
        yield from main_elements(child)


def test_forecast_technical_details_collapsed_and_daily_results_unchanged():
    forecast = dict(success=True, selected_model='ETS', total_predicted_expense=300,
        total_predicted_lower=250, total_predicted_upper=350, average_daily_predicted=100,
        target_status='MAPE 12.34%', target_met=True, best_mape=12.34,
        projections=[dict(ds=f'2026-11-{i}', yhat=100., yhat_lower=80., yhat_upper=120.) for i in (13, 14, 15)])
    evaluation = dict(success=True, ready=True, test_days=28, best_model='ETS', message='MAPE 12.34%',
        model_comparison=[dict(model='ETS', mape=12.34, mae=1, rmse=2, wape=3)])
    original = copy.deepcopy(forecast)
    numbers = dict(pemasukan=0, pengeluaran=0, omzet=0, arus_kas_operasional=0, arus_kas_total=0,
                   bulanan={}, pengeluaran_per_kategori={})
    with patch('services.report_service.period_rows', return_value=[]), \
         patch('pages.report_view.list_categories', return_value=[]), \
         patch('services.report_service.metrics', return_value=numbers), \
         patch('services.report_service.export_csv', return_value=b''), \
         patch('services.report_service.export_excel', return_value=b''), \
         patch('services.forecast_service.detect_recurring_expenses', return_value={}), \
         patch('services.forecast_service.get_forecast_readiness_status', return_value={'ready': True}), \
         patch('services.forecast_service.train_and_forecast_expenses', return_value=forecast) as train, \
         patch('services.forecast_service.evaluate_forecast_accuracy', return_value=evaluation) as evaluate:
        app = AppTest.from_string("from pages.report_view import render\nrender('test')").run()
        app.button(key='btn_calculate_forecast').click().run()
    assert not app.exception and not app.error
    details = next(e for e in app.expander if e.label == 'Detail teknis')
    assert details.proto.expanded is False
    outside = '\n'.join(str(getattr(e, 'value', ''))+' '+str(getattr(e, 'label', ''))
                        for e in main_elements(app.main))
    assert not any(word in outside for word in ('ETS', 'Prophet', 'ARIMA', 'MAPE', 'MAE', 'RMSE', 'WAPE'))
    assert any('ETS' in str(e.value) for e in details.markdown)
    tables = [e.value for e in main_elements(app.main) if hasattr(getattr(e, 'value', None), 'columns')]
    assert len(tables) == 1
    assert list(tables[0].columns) == ['Tanggal', 'Estimasi Pengeluaran (Rp)']
    assert tables[0].to_dict('records') == [{'Tanggal': r['ds'], 'Estimasi Pengeluaran (Rp)': r['yhat']} for r in forecast['projections']]
    charts = app.get('plotly_chart')
    assert len(charts) == 1 and 'ETS' not in json.loads(charts[0].proto.spec)['layout']['title']['text']
    assert forecast == original
    train.assert_called_once_with('test', days_ahead=30)
    evaluate.assert_called_once_with('test')
