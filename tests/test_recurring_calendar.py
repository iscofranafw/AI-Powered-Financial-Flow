from datetime import datetime
from unittest.mock import patch

import pytest

from services import forecast_service
from utils.forecast_details import daily_expense_detail
from utils.recurring_calendar import build_bill_calendar, infer_schedule


def item(dates, name='Internet', verified=False):
    return dict(infer_schedule(dates), description=name, estimated_amount=350000,
                last_date=max(dates), is_verified_due_date=verified)


def test_weekly_all_cycles_and_median_amount_with_two_bills_and_empty_day():
    bill = item(['2026-10-22', '2026-10-29', '2026-11-05'], verified=True)
    calendar = build_bill_calendar([bill], '2026-11-06', '2026-12-05')
    assert [entry['tanggal'] for entry in calendar] == ['2026-11-12', '2026-11-19', '2026-11-26', '2026-12-03']
    assert all(entry['nominal'] == 350000 and entry['terkonfirmasi'] for entry in calendar)
    other = item(['2026-10-22', '2026-10-29', '2026-11-05'], name='Air')
    calendar = build_bill_calendar([bill, other], '2026-11-06', '2026-12-05')
    projections = [{'ds': '2026-11-12', 'yhat': 123.45}, {'ds': '2026-11-13', 'yhat': 456.78}]
    for row in projections:
        detail = daily_expense_detail(projections, [], row['ds'], calendar=calendar)
        assert detail['estimasi_total'] == row['yhat']
        assert len(detail['tagihan']) == (2 if row['ds'] == '2026-11-12' else 0)
    assert projections[0]['yhat'] == 123.45


@pytest.mark.parametrize('dates,start,end,expected', [
    (['2026-09-01', '2026-10-01', '2026-11-01'], '2026-11-02', '2027-01-01', ['2026-12-01', '2027-01-01']),
    (['2026-10-31', '2026-11-30', '2026-12-31'], '2027-01-01', '2027-03-31', ['2027-01-31', '2027-02-28', '2027-03-31']),
    (['2023-11-30', '2023-12-31', '2024-01-31'], '2024-02-01', '2024-03-31', ['2024-02-29', '2024-03-31']),
])
def test_monthly_anchor_survives_short_month(dates, start, end, expected):
    bill = item(dates, verified=True)
    assert bill['schedule_kind'] == 'monthly'
    calendar = build_bill_calendar([bill], start, end)
    assert [entry['tanggal'] for entry in calendar] == expected
    assert all(not entry['terkonfirmasi'] for entry in calendar if entry['tanggal'].endswith(('02-28', '02-29')))
    assert calendar[-1]['terkonfirmasi']


def test_irregular_single_and_duplicate_dates_never_invent_schedule():
    for dates in [['2026-10-01', '2026-10-08', '2026-10-15', '2026-11-05'],
                  ['2026-11-05'], ['2026-11-05', '2026-11-05']]:
        bill = item(dates, verified=True)
        assert not bill['schedule_regular']
        assert build_bill_calendar([bill], '2026-11-06', '2026-12-05') == []


def test_two_occurrences_estimate_even_when_verified_and_horizon_bounds():
    bill = item(['2026-10-29', '2026-11-05'], verified=True)
    calendar = build_bill_calendar([bill], '2026-11-12', '2026-11-19')
    assert len(calendar) == 2
    assert not any(entry['terkonfirmasi'] for entry in calendar)
    assert build_bill_calendar([bill], '2026-11-13', '2026-11-18') == []


def test_detector_uses_real_history_median_and_keeps_irregular_without_date():
    rows = [dict(tanggal=day, nominal=amount, deskripsi=name, kategori_id='ops')
            for name, observations in [('Internet', [('2026-10-22', 300000), ('2026-10-29', 350000), ('2026-11-05', 800000)]),
                                       ('Air', [('2026-10-01', 1), ('2026-10-08', 2), ('2026-10-15', 3), ('2026-11-05', 4)])]
            for day, amount in observations]
    with patch.object(forecast_service, '_get_eligible_expense_rows', return_value=({'toko_id': 'new-empty-local-fixture'}, rows, [])), \
         patch.object(forecast_service, 'get_jakarta_now', return_value=datetime(2026, 11, 6)):
        result = forecast_service.detect_recurring_expenses('local-fixture')
    bills = {bill['description']: bill for bill in result['recurring_items']}
    assert bills['Internet']['estimated_amount'] == 350000
    assert bills['Internet']['estimated_interval_days'] == 7
    assert bills['Air']['next_expected_date'] is None
    assert bills['Air']['pattern_description'] == 'jadwal belum dapat dipastikan'
    assert not bills['Air']['is_upcoming_3_days']


def test_detector_keeps_source_verification_after_clipped_next_month():
    rows = [dict(tanggal=day, nominal=350000, deskripsi='Sewa', kategori_id='ops', is_verified_due_date=True)
            for day in ['2026-11-30', '2026-12-31', '2027-01-31']]
    with patch.object(forecast_service, '_get_eligible_expense_rows', return_value=({'toko_id': 'local'}, rows, [])), \
         patch.object(forecast_service, 'get_jakarta_now', return_value=datetime(2027, 2, 1)):
        bill = forecast_service.detect_recurring_expenses('local')['recurring_items'][0]
    assert not bill['is_verified_due_date']
    calendar = build_bill_calendar([bill], '2027-02-01', '2027-03-31')
    assert [entry['terkonfirmasi'] for entry in calendar] == [False, True]
