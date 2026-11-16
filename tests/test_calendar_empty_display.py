from contextlib import ExitStack
from copy import deepcopy
import hashlib
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from utils.recurring_calendar import infer_schedule
from utils.ui_components import format_rupiah


@pytest.fixture
def calendar_app():
    with ExitStack() as stack:
        numbers = dict(pemasukan=0, pengeluaran=0, omzet=0, arus_kas_operasional=0,
                       arus_kas_total=0, bulanan={}, pengeluaran_per_kategori={})
        for target, value in {
            'services.report_service.period_rows': [],
            'pages.report_view.list_categories': [],
            'services.report_service.metrics': numbers,
            'services.report_service.export_csv': b'',
            'services.report_service.export_excel': b'',
            'services.forecast_service.evaluate_forecast_accuracy': {},
        }.items():
            stack.enter_context(patch(target, return_value=value))
        recurring = stack.enter_context(patch('services.forecast_service.detect_recurring_expenses'))
        readiness = stack.enter_context(patch('services.forecast_service.get_forecast_readiness_status'))
        train = stack.enter_context(patch('services.forecast_service.train_and_forecast_expenses'))

        def render(items, ready=True):
            forecast = dict(success=True, projections=[dict(ds='2026-11-12', yhat=400000),
                                                       dict(ds='2026-11-13', yhat=250000)],
                            total_predicted_expense=650000, average_daily_predicted=325000,
                            total_predicted_lower=600000, total_predicted_upper=700000)
            recurring.return_value = {'recurring_items': items}
            readiness.return_value = dict(ready=ready, message='Belum cukup riwayat toko uji baru.')
            train.return_value = forecast
            original = deepcopy(forecast)
            app = AppTest.from_string("from pages.report_view import render\nrender('new-local-empty-display-shop')").run()
            if ready:
                app.button(key='btn_calculate_forecast').click().run()
            assert not app.error and not app.exception
            assert forecast == original
            return app, train

        yield render


def visible_text(app):
    return '\n'.join(str(element.value) for element in [*app.markdown, *app.caption, *app.info])


def uncertain_item():
    return dict(description='Barang tanpa tanggal uji', estimated_amount=80000, schedule_regular=False)


@pytest.mark.parametrize('items', [[], [uncertain_item()]], ids=['empty-calendar', 'uncertain-only'])
def test_calendar_empty_and_uncertain_items_render_no_noise(calendar_app, items):
    app, train = calendar_app(items)
    text = visible_text(app)
    assert len(app.dataframe) == 1
    assert list(app.dataframe[0].value['Estimasi Pengeluaran (Rp)']) == [400000, 250000]
    assert 'Kalender Tagihan' in text
    assert 'Tanggal berasal dari pola riwayat toko' in text
    assert 'Tidak ada tagihan rutin' not in text
    assert 'jadwal belum dapat dipastikan' not in text
    assert 'Barang tanpa tanggal uji' not in text
    assert any(metric.value == format_rupiah(650000) for metric in app.metric)
    owner_key = hashlib.sha256(b'new-local-empty-display-shop').hexdigest()
    assert app.session_state[f'forecast_display_{owner_key}']['recurring_items'] == items
    train.assert_called_once_with('new-local-empty-display-shop', days_ahead=30)
    print(f'AppTest calendar={"uncertain-only" if items else "empty"}: calendar_rows=0; daily_rows=2; total={format_rupiah(650000)}; empty_message=False; uncertain_list=False')


def test_confirmed_bill_and_empty_day_detail_keep_amount_and_titles(calendar_app):
    bill = dict(infer_schedule(['2026-10-22', '2026-10-29', '2026-11-05']),
                description='Internet toko uji', last_date='2026-11-05',
                estimated_amount=350000, is_verified_due_date=True)
    items = [bill, uncertain_item()]
    app, train = calendar_app(items)
    assert app.dataframe[1].value.to_dict('records') == [{
        'Tanggal': '2026-11-12', 'Tagihan': 'Internet toko uji',
        'Nominal berdasarkan riwayat': format_rupiah(350000), 'Jadwal': 'Terkonfirmasi'}]
    assert 'Barang tanpa tanggal uji' not in visible_text(app)
    app.selectbox[0].select('2026-11-13').run()
    text = visible_text(app)
    assert 'Detail pengeluaran 2026-11-13' in text
    assert 'Jadwal tagihan dari data toko' in text
    assert format_rupiah(250000) in text
    assert 'Sumber jadwal: deteksi pengeluaran rutin' in text
    assert 'Tidak ada tagihan rutin' not in text
    assert not app.error and not app.exception
    train.assert_called_once()
    print(f'AppTest confirmed: date=2026-11-12; name=Internet toko uji; amount={format_rupiah(350000)}; label=Terkonfirmasi')
    print(f'AppTest empty-day: date=2026-11-13; daily_total={format_rupiah(250000)}; heading=True; caption=True; empty_message=False')


def test_new_empty_shop_keeps_existing_readiness_guard(calendar_app):
    app, train = calendar_app([], ready=False)
    assert not app.dataframe
    assert any('Belum cukup riwayat toko uji baru' in element.value for element in app.info)
    assert not any(button.key == 'btn_calculate_forecast' for button in app.button)
    train.assert_not_called()
    print('AppTest truly-empty-new-shop: readiness_guard=True; forecast_not_called=True; no fabricated prediction')
