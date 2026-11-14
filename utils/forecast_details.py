"""Join displayed daily totals to existing recurring schedules, without forecasting items."""


def daily_expense_detail(projections, recurring_items, selected_date, calendar=None):
    row = next((r for r in projections if r['ds'] == selected_date), None)
    if row is None:
        raise ValueError('Tanggal tidak tersedia pada hasil prediksi.')
    bills = [entry for entry in calendar if entry['tanggal'] == selected_date] if calendar is not None else [{
        'nama': item['description'],
        'nominal': item['estimated_amount'],
        'tanggal': item['next_expected_date'],
        'terkonfirmasi': bool(item.get('is_verified_due_date')),
        'sumber': 'Jadwal terkonfirmasi; nominal berdasarkan riwayat toko'
                  if item.get('is_verified_due_date') else 'Perkiraan jadwal dan nominal dari riwayat toko',
    } for item in recurring_items if str(item.get('next_expected_date', ''))[:10] == selected_date]
    return {'tanggal': selected_date, 'estimasi_total': row['yhat'], 'tagihan': bills}
