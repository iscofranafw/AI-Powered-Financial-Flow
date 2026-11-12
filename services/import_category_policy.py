"""Shared import approval and preview rules, independent of Streamlit."""

def missing_category_types(rows, existing, reviewed):
    present = {r.get('tipe') for r in rows if not r.get('error')}
    usable = existing + reviewed
    return [tp for tp in ('pengeluaran', 'pemasukan') if tp in present and not any(
        c.get('tipe') == tp and str(c.get('nama') or '').strip()
        and not c.get('sistem') and not c.get('bukan_operasional')
        and c.get('aktif', True) is not False
        and str(c.get('nama')).strip().casefold() not in ('others', 'tambahan modal')
        for c in usable)]


def preview_category_label(row, names):
    if not row.get('kategori_id') or row.get('is_unresolved'):
        return 'Belum Terpetakan (Perlu Tinjauan)'
    return names.get(row['kategori_id'], 'Belum Terpetakan (Perlu Tinjauan)')


def missing_allowed_types(rows):
    return sorted({r.get('tipe', 'transaksi') for r in rows
                   if (r.get('reason_code') or (r.get('usulan') or {}).get('reason_code')) == 'NO_ALLOWED_CATEGORIES'})
