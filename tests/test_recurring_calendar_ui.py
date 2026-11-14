from copy import deepcopy
from datetime import date, timedelta
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from utils.recurring_calendar import infer_schedule
from utils.ui_components import format_rupiah


def test_calendar_detail_horizon_button_and_owner_isolation():
    projections = [dict(ds=str(date(2026, 11, 6) + timedelta(days=i)), yhat=100. + i)
                   for i in range(30)]
    forecast = dict(success=True, projections=projections, total_predicted_expense=3435,
                    average_daily_predicted=114.5, total_predicted_lower=3000, total_predicted_upper=4000)
    bills = [dict(infer_schedule(['2026-10-22', '2026-10-29', '2026-11-05']),
                  last_date='2026-11-05', estimated_amount=350000, description=name,
                  is_verified_due_date=True) for name in ['Internet', 'Air']]
    bills.append(dict(description='Tidak teratur', estimated_amount=40, schedule_regular=False))
    numbers = dict(pemasukan=0, pengeluaran=0, omzet=0, arus_kas_operasional=0,
                   arus_kas_total=0, bulanan={}, pengeluaran_per_kategori={})
    original = deepcopy(forecast)
    with patch('services.report_service.period_rows', return_value=[]), \
         patch('pages.report_view.list_categories', return_value=[]), \
         patch('services.report_service.metrics', return_value=numbers), \
         patch('services.report_service.export_csv', return_value=b''), \
         patch('services.report_service.export_excel', return_value=b''), \
         patch('services.forecast_service.detect_recurring_expenses', return_value={'recurring_items': bills}), \
         patch('services.forecast_service.get_forecast_readiness_status', return_value={'ready': True}), \
         patch('services.forecast_service.train_and_forecast_expenses', return_value=forecast) as train, \
         patch('services.forecast_service.evaluate_forecast_accuracy', return_value={}):
        app = AppTest.from_string("import streamlit as st\nfrom pages.report_view import render\nrender(st.session_state.get('owner', 'first'))").run()
        assert not app.dataframe
        app.button(key='btn_calculate_forecast').click().run()
        assert len(app.dataframe) == 2
        calendar = app.dataframe[1].value
        assert len(calendar) == 8
        assert set(calendar['Jadwal']) == {'Terkonfirmasi'}
        app.selectbox[0].select('2026-11-12').run()
        text = '\n'.join(str(element.value) for element in [*app.markdown, *app.caption])
        assert 'Internet — nominal berdasarkan riwayat' in text
        assert 'Air — nominal berdasarkan riwayat' in text
        assert format_rupiah(106) in text
        assert 'jadwal belum dapat dipastikan' in text
        app.selectbox[0].select('2026-11-13').run()
        assert any('Tidak ada tagihan rutin jatuh tempo hari ini' in element.value for element in app.info)
        assert any(format_rupiah(107) in element.value for element in app.markdown)
        app.slider[0].set_value(15).run()
        assert not app.dataframe
        train.assert_called_once_with('first', days_ahead=30)
        app.slider[0].set_value(30).run()
        assert not app.dataframe
        app.button(key='btn_calculate_forecast').click().run()
        assert len(app.dataframe) == 2
        assert train.call_count == 2
        app.session_state['owner'] = 'second'
        app.run()
        assert not app.dataframe
        assert not app.exception and not app.error
    assert forecast == original
