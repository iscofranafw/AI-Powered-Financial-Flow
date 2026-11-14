"""History-derived schedules, kept separate from model expenditure totals."""
from calendar import monthrange
from datetime import date, timedelta
from statistics import median


def infer_schedule(history_dates):
    dates = sorted({date.fromisoformat(str(value)[:10]) for value in history_dates})
    if len(dates) < 2:
        return {'schedule_regular': False, 'estimated_interval_days': None}
    gaps = [(right - left).days for left, right in zip(dates, dates[1:])]
    interval = median(gaps)
    months = [value.year * 12 + value.month for value in dates]
    anchor = max(value.day for value in dates)
    monthly = (all(right - left == 1 for left, right in zip(months, months[1:]))
               and all(value.day == min(anchor, monthrange(value.year, value.month)[1]) for value in dates))
    # Calendar-month variation is evaluated by its day anchor; other gaps allow at most one day or 20% jitter.
    regular = monthly or all(abs(gap - interval) <= max(1, interval * .2) for gap in gaps)
    return {'schedule_regular': regular, 'estimated_interval_days': interval,
            'schedule_kind': 'monthly' if monthly else 'days', 'anchor_day': anchor,
            'historical_dates': [str(value) for value in dates], 'historical_count': len(dates)}


def scheduled_dates(item, start, end):
    if not item.get('schedule_regular') or not item.get('last_date'):
        return []
    start, end = date.fromisoformat(str(start)[:10]), date.fromisoformat(str(end)[:10])
    last = date.fromisoformat(item['last_date'])
    result = []
    step = 1
    while True:
        clipped = False
        if item.get('schedule_kind') == 'monthly':
            month_index = last.year * 12 + last.month - 1 + step
            year, month = divmod(month_index, 12)
            month += 1
            day = min(item['anchor_day'], monthrange(year, month)[1])
            clipped = day != item['anchor_day']
            due = date(year, month, day)
        else:
            interval = item.get('estimated_interval_days')
            if not interval or interval <= 0:
                return []
            due = last + timedelta(days=round(interval * step))
        if due > end:
            return result
        if due >= start:
            result.append((due, clipped))
        step += 1


def build_bill_calendar(items, start, end):
    entries = []
    for item in items:
        confirmed = bool(item.get('schedule_verified', item.get('is_verified_due_date')) and item.get('historical_count', 0) >= 3)
        for due, clipped in scheduled_dates(item, start, end):
            verified = confirmed and not clipped
            entries.append({'nama': item['description'], 'nominal': item['estimated_amount'],
                            'tanggal': str(due), 'terkonfirmasi': verified,
                            'sumber': ('Jadwal terkonfirmasi' if verified else 'Perkiraan jadwal')
                            + '; nominal berdasarkan riwayat toko'})
    return sorted(entries, key=lambda entry: (entry['tanggal'], entry['nama']))
