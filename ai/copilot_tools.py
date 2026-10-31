"""Authorized Native Python Tool Registry for the Owner Financial Copilot.

Maps explicit, validated tool intents to existing backend services and database repositories
with strict role verification, tenant isolation, and deterministic financial math.
"""

from decimal import Decimal
from datetime import date
from typing import Dict, Any, List, Optional

from database.client import privileged_client
from services.auth_service import verify_actor, AccessError
from services import finance_service as finance
from services import cashier_service
from services.report_service import period_rows, metrics
from utils.ui_components import format_rupiah
from ai.copilot_schema import ToolName, PlannerDecision
from ai.copilot_knowledge import (
    get_feature_explanation,
    detect_feature_topic,
    get_forecast_explanation,
)


# =====================================================================
# 1. STORE CAPABILITIES
# =====================================================================

def get_store_profile(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    shop = privileged_client().table('toko').select('id,nama,bidang_usaha,created_at').eq('id', actor['toko_id']).single().execute().data
    created_date = str(shop.get('created_at', ''))[:10]
    msg = (
        f"### Profil Toko Terdaftar\n\n"
        f"- **Nama Usaha:** {shop['nama']}\n"
        f"- **Bidang Usaha:** {shop['bidang_usaha']}\n"
        f"- **Pengguna Aktif:** {actor['nama']} (`{actor['email']}`) [{actor['role']}]\n"
        f"- **Terdaftar Sejak:** {created_date}\n"
        f"- **Status Operasional:** Aktif"
    )
    return {'tool': ToolName.GET_STORE_PROFILE, 'success': True, 'data': shop, 'message': msg}


def get_store_business_type(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    shop = privileged_client().table('toko').select('id,nama,bidang_usaha').eq('id', actor['toko_id']).single().execute().data
    msg = f"Toko **{shop['nama']}** terdaftar dalam bidang usaha **{shop['bidang_usaha']}**."
    return {'tool': ToolName.GET_STORE_BUSINESS_TYPE, 'success': True, 'data': shop, 'message': msg}


def get_store_operational_status(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    return {
        'tool': ToolName.GET_STORE_OPERATIONAL_STATUS,
        'success': True,
        'data': {'status': 'aktif', 'toko_id': actor['toko_id']},
        'message': "Status operasional toko saat ini adalah **Aktif** dan siap melayani pencatatan transaksi harian."
    }


# =====================================================================
# 2. CASHIER CAPABILITIES
# =====================================================================

def count_cashiers(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    is_owner = actor['role'] == 'pemilik_toko'
    if not is_owner:
        return {
            'tool': ToolName.COUNT_CASHIERS,
            'success': False,
            'message': "Informasi jumlah akun kasir hanya dapat diakses oleh **Pemilik Toko**."
        }

    cashiers = cashier_service.list_cashiers(token)
    total_count = len(cashiers)
    active_count = len([c for c in cashiers if c.get('status') == 'aktif'])

    if total_count == 0:
        msg = "Saat ini belum ada akun kasir yang terdaftar untuk toko Anda. Anda dapat mendaftarkan kasir baru melalui menu **Akun Kasir**."
    else:
        msg = (
            f"Toko Anda saat ini memiliki total **{total_count} akun kasir terdaftar** "
            f"(**{active_count} aktif**).\n\n"
            f"Anda dapat melihat rincian nama atau menambah staf baru melalui menu **Akun Kasir**."
        )

    return {
        'tool': ToolName.COUNT_CASHIERS,
        'success': True,
        'data': {'total_count': total_count, 'active_count': active_count},
        'message': msg
    }


def list_cashiers(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    cashiers = cashier_service.list_cashiers(token)
    if not cashiers:
        return {
            'tool': ToolName.LIST_CASHIERS,
            'success': True,
            'data': [],
            'message': "Belum ada akun kasir terdaftar untuk toko ini. Buka menu **Akun Kasir** untuk mendaftarkan staf baru."
        }

    lines = [f"Daftar **{len(cashiers)} akun kasir terdaftar** di toko Anda:\n"]
    for idx, c in enumerate(cashiers, start=1):
        status = "Aktif" if c.get('status') == 'aktif' else c.get('status', '-')
        lines.append(f"{idx}. **{c['nama']}** (`{c['email']}`) &bull; Status: **{status}**")
    lines.append("\nPengelolaan kasir dapat dilakukan melalui menu **Akun Kasir**.")

    return {'tool': ToolName.LIST_CASHIERS, 'success': True, 'data': cashiers, 'message': "\n".join(lines)}


def get_cashier_activity(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    cashiers = cashier_service.list_cashiers(token)
    if not cashiers:
        return {
            'tool': ToolName.GET_CASHIER_ACTIVITY,
            'success': True,
            'data': {},
            'message': "Belum ada akun kasir yang terdaftar di toko Anda."
        }

    txs = finance.list_transactions(token)
    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"

    if start:
        txs = [t for t in txs if t['tanggal'] >= str(start)]
    if end:
        txs = [t for t in txs if t['tanggal'] <= str(end)]

    counts = {c['id']: {'nama': c['nama'], 'email': c['email'], 'total': 0, 'disetujui': 0, 'pending': 0, 'nominal': Decimal(0)} for c in cashiers}
    for t in txs:
        creator = t.get('dicatat_oleh')
        if creator in counts:
            counts[creator]['total'] += 1
            counts[creator]['nominal'] += Decimal(str(t.get('nominal', 0)))
            st_tx = t.get('status_persetujuan')
            if st_tx == 'pending':
                counts[creator]['pending'] += 1
            elif st_tx == 'disetujui':
                counts[creator]['disetujui'] += 1

    lines = []
    # Identify most active cashier
    most_active = max(cashiers, key=lambda c: (counts[c['id']]['total'], counts[c['id']]['nominal']))
    m_stats = counts[most_active['id']]
    if m_stats['total'] > 0:
        lines.append(
            f"Kasir yang paling aktif pada periode **{period_label}** adalah **{most_active['nama']}** "
            f"dengan mencatat **{m_stats['total']} transaksi** "
            f"(total nilai **{format_rupiah(m_stats['nominal'])}**; "
            f"{m_stats['disetujui']} disetujui, {m_stats['pending']} menunggu persetujuan).\n"
        )
    else:
        lines.append(f"Belum ada aktivitas pencatatan transaksi oleh kasir pada periode **{period_label}**.\n")

    lines.append(f"Aktivitas akun kasir toko pada periode **{period_label}** ({len(cashiers)} kasir terdaftar):\n")
    for idx, c in enumerate(cashiers, start=1):
        c_stats = counts.get(c['id'], {'total': 0, 'disetujui': 0, 'pending': 0, 'nominal': Decimal(0)})
        status_label = "Aktif" if c.get('status') == 'aktif' else c.get('status', '-')
        lines.append(
            f"{idx}. **{c['nama']}** (`{c['email']}`) &bull; Status: **{status_label}**\n"
            f"   - Total Transaksi Dicatat: **{c_stats['total']}** "
            f"({c_stats['disetujui']} disetujui, {c_stats['pending']} menunggu persetujuan) "
            f"&bull; Akumulasi: **{format_rupiah(c_stats['nominal'])}**"
        )

    lines.append("\nPendaftaran dan verifikasi staf kasir dapat dilakukan di menu **Akun Kasir**.")
    return {'tool': ToolName.GET_CASHIER_ACTIVITY, 'success': True, 'data': counts, 'message': "\n".join(lines)}


def get_cashier_transaction_summary(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    return get_cashier_activity(token, decision)


# =====================================================================
# 3. CATEGORY CAPABILITIES
# =====================================================================

def _format_cat_line(cat: Dict[str, Any]) -> str:
    tag = "Kustom"
    if cat.get('sistem'):
        tag = "Sistem (Others / Fallback)"
    elif cat.get('bukan_operasional'):
        tag = "Non-Operasional (Tambahan Modal)"
    desc = f" - *{cat['deskripsi']}*" if cat.get('deskripsi') else ""
    return f"- **{cat['nama']}** `[{tag}]`{desc}"


def list_income_categories(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    verify_actor(token, {'pemilik_toko', 'kasir'})
    categories = finance.list_categories(token)
    income_cats = [c for c in categories if c['tipe'] == 'pemasukan']

    lines = [f"### Kategori Pemasukan Toko ({len(income_cats)}/5)\n"]
    if income_cats:
        for c in income_cats:
            lines.append(_format_cat_line(c))
    else:
        lines.append("- *(Belum ada kategori pemasukan)*")

    lines.append("\n**Keterangan:**")
    lines.append("- Kuota maksimal: 5 kategori pemasukan per toko.")
    lines.append("- Kelola pos pemasukan melalui menu **Kategori Pembukuan**.")
    return {'tool': ToolName.LIST_INCOME_CATEGORIES, 'success': True, 'data': income_cats, 'message': "\n".join(lines)}


def list_expense_categories(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    verify_actor(token, {'pemilik_toko', 'kasir'})
    categories = finance.list_categories(token)
    expense_cats = [c for c in categories if c['tipe'] == 'pengeluaran']

    lines = [f"### Kategori Pengeluaran Toko ({len(expense_cats)}/10)\n"]
    if expense_cats:
        for c in expense_cats:
            lines.append(_format_cat_line(c))
    else:
        lines.append("- *(Belum ada kategori pengeluaran)*")

    lines.append("\n**Keterangan:**")
    lines.append("- Kuota maksimal: 10 kategori pengeluaran per toko.")
    lines.append("- Kelola pos biaya melalui menu **Kategori Pembukuan**.")
    return {'tool': ToolName.LIST_EXPENSE_CATEGORIES, 'success': True, 'data': expense_cats, 'message': "\n".join(lines)}


def list_all_categories(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    verify_actor(token, {'pemilik_toko', 'kasir'})
    categories = finance.list_categories(token)
    income_cats = [c for c in categories if c['tipe'] == 'pemasukan']
    expense_cats = [c for c in categories if c['tipe'] == 'pengeluaran']

    lines = [
        f"### Kategori Pemasukan ({len(income_cats)}/5)\n" +
        ("\n".join(_format_cat_line(c) for c in income_cats) if income_cats else "- *(Kosong)*"),
        f"\n### Kategori Pengeluaran ({len(expense_cats)}/10)\n" +
        ("\n".join(_format_cat_line(c) for c in expense_cats) if expense_cats else "- *(Kosong)*"),
        "\n**Struktur Pos Anggaran:**",
        "- **Kustom:** Pos harian operasional sesuai kebutuhan toko Anda.",
        "- **Sistem (Others):** Kategori sistem wajib sebagai penampung transaksi yang belum terklasifikasi.",
        "- **Non-Operasional (Tambahan Modal):** Khusus setoran modal yang tidak dihitung sebagai omzet penjualan.",
        "\nPengelolaan kategori dapat dilakukan di menu **Kategori Pembukuan**."
    ]
    return {'tool': ToolName.LIST_ALL_CATEGORIES, 'success': True, 'data': categories, 'message': "\n".join(lines)}


def rank_expense_categories(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"

    cats = finance.list_categories(token)
    rows = period_rows(token, start=start, end=end)
    m = metrics(rows, cats)

    cat_expenses = m['pengeluaran_per_kategori']
    total_expense = m['pengeluaran']

    if not cat_expenses or total_expense <= 0:
        return {
            'tool': ToolName.RANK_EXPENSE_CATEGORIES,
            'success': True,
            'data': {},
            'message': f"Belum ada data transaksi pengeluaran yang disetujui pada periode **{period_label}**."
        }

    sorted_cats = sorted(cat_expenses.items(), key=lambda x: x[1], reverse=True)
    lines = [f"Peringkat kategori pengeluaran terbanyak pada periode **{period_label}** (Total: **{format_rupiah(total_expense)}**):\n"]
    for idx, (cat_name, amount) in enumerate(sorted_cats, start=1):
        pct = (amount / total_expense * Decimal(100)) if total_expense > 0 else Decimal(0)
        lines.append(f"{idx}. **{cat_name}** &bull; **{format_rupiah(amount)}** ({pct:.1f}%)")

    lines.append(f"\nPos pengeluaran terbesar yang paling banyak menyerap kas adalah **{sorted_cats[0][0]}**.")
    return {'tool': ToolName.RANK_EXPENSE_CATEGORIES, 'success': True, 'data': dict(sorted_cats), 'message': "\n".join(lines)}


def rank_income_categories(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"

    cats = finance.list_categories(token)
    cmap = {c['id']: c['nama'] for c in cats}
    rows = [r for r in period_rows(token, start=start, end=end) if r['tipe'] == 'pemasukan']

    by_cat = {}
    total_income = Decimal(0)
    for r in rows:
        cname = cmap.get(r['kategori_id'], 'Others')
        amt = Decimal(str(r['nominal']))
        by_cat[cname] = by_cat.get(cname, Decimal(0)) + amt
        total_income += amt

    if not by_cat:
        return {
            'tool': ToolName.RANK_INCOME_CATEGORIES,
            'success': True,
            'data': {},
            'message': f"Belum ada data pemasukan yang tercatat pada periode **{period_label}**."
        }

    sorted_cats = sorted(by_cat.items(), key=lambda x: x[1], reverse=True)
    lines = [f"Peringkat sumber pemasukan terbesar pada periode **{period_label}** (Total: **{format_rupiah(total_income)}**):\n"]
    for idx, (cat_name, amount) in enumerate(sorted_cats, start=1):
        pct = (amount / total_income * Decimal(100)) if total_income > 0 else Decimal(0)
        lines.append(f"{idx}. **{cat_name}** &bull; **{format_rupiah(amount)}** ({pct:.1f}%)")

    return {'tool': ToolName.RANK_INCOME_CATEGORIES, 'success': True, 'data': dict(sorted_cats), 'message': "\n".join(lines)}


# =====================================================================
# 4. TRANSACTION & SUMMARY CAPABILITIES
# =====================================================================

def get_transaction_details(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    is_owner = actor['role'] == 'pemilik_toko'

    raw_tx = finance.list_transactions(token)
    if not is_owner:
        raw_tx = [t for t in raw_tx if t['dicatat_oleh'] == actor['id']]

    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"
    req_type = decision.filter_params.transaction_type

    if start:
        raw_tx = [t for t in raw_tx if t['tanggal'] >= str(start)]
    if end:
        raw_tx = [t for t in raw_tx if t['tanggal'] <= str(end)]
    if req_type in ('pemasukan', 'pengeluaran'):
        raw_tx = [t for t in raw_tx if t['tipe'] == req_type]

    if not raw_tx:
        type_str = f" {req_type}" if req_type in ('pemasukan', 'pengeluaran') else ""
        return {
            'tool': ToolName.GET_TRANSACTION_DETAILS,
            'success': True,
            'data': [],
            'message': f"Tidak ditemukan rincian transaksi{type_str} pada periode **{period_label}**."
        }

    cats = finance.list_categories(token)
    cmap = {c['id']: c['nama'] for c in cats}
    total_val = sum(Decimal(str(t['nominal'])) for t in raw_tx)

    limit = decision.filter_params.limit or 10
    display_rows = raw_tx[:limit]

    type_title = req_type.capitalize() if req_type in ('pemasukan', 'pengeluaran') else "Transaksi"
    lines = [
        f"Ditemukan **{len(raw_tx)} transaksi {type_title}** pada periode **{period_label}** "
        f"dengan total akumulasi **{format_rupiah(total_val)}**:\n"
    ]

    for idx, t in enumerate(display_rows, start=1):
        cat_name = cmap.get(t['kategori_id'], 'Others')
        tp_label = "Pemasukan" if t['tipe'] == 'pemasukan' else "Pengeluaran"
        status_label = "Disetujui" if t.get('status_persetujuan') == 'disetujui' else "Menunggu Persetujuan"
        desc = t.get('deskripsi') or '-'
        lines.append(
            f"{idx}. **{t['tanggal']}** &bull; {tp_label} &bull; **{format_rupiah(t['nominal'])}** &bull; "
            f"*{desc}* (Kategori: {cat_name}, Status: {status_label})"
        )

    if len(raw_tx) > limit:
        lines.append(f"\n*(Menampilkan {limit} dari {len(raw_tx)} transaksi. Buka menu **Transaksi** untuk riwayat lengkap.)*")

    return {'tool': ToolName.GET_TRANSACTION_DETAILS, 'success': True, 'data': raw_tx, 'message': "\n".join(lines)}


def list_transactions(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    return get_transaction_details(token, decision)


def get_cashier_transactions(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    is_owner = actor['role'] == 'pemilik_toko'

    raw_tx = finance.list_transactions(token)
    cashiers = cashier_service.list_cashiers(token) if is_owner else []
    cashier_ids = {c['id'] for c in cashiers}

    # Cashier-submitted transactions
    if is_owner:
        cashier_tx = [t for t in raw_tx if t.get('dicatat_oleh') in cashier_ids]
    else:
        cashier_tx = [t for t in raw_tx if t.get('dicatat_oleh') == actor['id']]

    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Hari Ini"

    if start:
        cashier_tx = [t for t in cashier_tx if t['tanggal'] >= str(start)]
    if end:
        cashier_tx = [t for t in cashier_tx if t['tanggal'] <= str(end)]

    if not cashier_tx:
        return {
            'tool': ToolName.GET_CASHIER_TRANSACTIONS,
            'success': True,
            'data': [],
            'message': f"Tidak ada transaksi yang dicatat oleh kasir pada periode **{period_label}**."
        }

    cats = finance.list_categories(token)
    cmap = {c['id']: c['nama'] for c in cats}
    cmap_kasir = {c['id']: c['nama'] for c in cashiers}

    approved_count = len([t for t in cashier_tx if t.get('status_persetujuan') == 'disetujui'])
    pending_count = len([t for t in cashier_tx if t.get('status_persetujuan') == 'pending'])
    total_amt = sum(Decimal(str(t['nominal'])) for t in cashier_tx)

    lines = [
        f"Ditemukan **{len(cashier_tx)} transaksi kasir** pada periode **{period_label}** "
        f"(**{approved_count} disetujui**, **{pending_count} menunggu persetujuan**) "
        f"dengan total nilai **{format_rupiah(total_amt)}**:\n"
    ]

    for idx, t in enumerate(cashier_tx[:10], start=1):
        cat_name = cmap.get(t['kategori_id'], 'Others')
        cashier_name = cmap_kasir.get(t.get('dicatat_oleh'), 'Kasir')
        tp_label = "Pemasukan" if t['tipe'] == 'pemasukan' else "Pengeluaran"
        status_label = "Disetujui" if t.get('status_persetujuan') == 'disetujui' else "Menunggu Persetujuan"
        lines.append(
            f"{idx}. **{t['tanggal']}** &bull; {tp_label} &bull; **{format_rupiah(t['nominal'])}** &bull; "
            f"*{t.get('deskripsi', '-')}* (Kategori: {cat_name}, Kasir: {cashier_name}, Status: **{status_label}**)"
        )

    if is_owner and pending_count > 0:
        lines.append("\nUntuk menyetujui transaksi yang berstatus pending, buka menu **Persetujuan Kasir**.")

    return {'tool': ToolName.GET_CASHIER_TRANSACTIONS, 'success': True, 'data': cashier_tx, 'message': "\n".join(lines)}


def get_pending_approvals(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    raw_tx = finance.list_transactions(token)
    pending = [t for t in raw_tx if t.get('status_persetujuan') == 'pending']

    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Tanggal"

    if start:
        pending = [t for t in pending if t['tanggal'] >= str(start)]
    if end:
        pending = [t for t in pending if t['tanggal'] <= str(end)]

    if not pending:
        return {
            'tool': ToolName.GET_PENDING_APPROVALS,
            'success': True,
            'data': [],
            'message': f"Tidak ada transaksi kasir yang menunggu persetujuan pada periode **{period_label}**. Semua transaksi telah diverifikasi atau belum ada catatan baru."
        }

    cats = finance.list_categories(token)
    cmap = {c['id']: c['nama'] for c in cats}
    cashiers = cashier_service.list_cashiers(token)
    cmap_kasir = {c['id']: c['nama'] for c in cashiers}

    total_amt = sum(Decimal(str(t['nominal'])) for t in pending)
    lines = [
        f"Ditemukan **{len(pending)} transaksi kasir** yang **menunggu persetujuan** "
        f"pada periode **{period_label}** dengan total nilai **{format_rupiah(total_amt)}**:\n"
    ]

    for idx, t in enumerate(pending[:10], start=1):
        cat_name = cmap.get(t['kategori_id'], 'Others')
        cashier_name = cmap_kasir.get(t.get('dicatat_oleh'), 'Kasir')
        tp_label = "Pemasukan" if t['tipe'] == 'pemasukan' else "Pengeluaran"
        lines.append(
            f"{idx}. **{t['tanggal']}** &bull; {tp_label} &bull; **{format_rupiah(t['nominal'])}** &bull; "
            f"*{t.get('deskripsi', '-')}* (Kategori: {cat_name}, Kasir: {cashier_name})"
        )

    if len(pending) > 10:
        lines.append(f"\n*(Menampilkan 10 dari {len(pending)} transaksi. Buka menu **Persetujuan Kasir** untuk verifikasi massal.)*")
    else:
        lines.append("\nAnda dapat meninjau dan melakukan verifikasi massal melalui menu **Persetujuan Kasir**.")

    return {'tool': ToolName.GET_PENDING_APPROVALS, 'success': True, 'data': pending, 'message': "\n".join(lines)}


def count_pending_approvals(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    return get_pending_approvals(token, decision)


def get_income_summary(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"

    cats = finance.list_categories(token)
    rows = period_rows(token, start=start, end=end)
    m = metrics(rows, cats)

    income_count = len([r for r in rows if r['tipe'] == 'pemasukan'])
    msg = (
        f"Total pemasukan untuk periode **{period_label}** adalah **{format_rupiah(m['pemasukan'])}** "
        f"dari total **{income_count} transaksi pemasukan** yang telah disetujui."
    )
    return {'tool': ToolName.GET_INCOME_SUMMARY, 'success': True, 'data': {'pemasukan': m['pemasukan'], 'count': income_count}, 'message': msg}


def get_expense_summary(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"

    cats = finance.list_categories(token)
    rows = period_rows(token, start=start, end=end)
    m = metrics(rows, cats)

    expense_count = len([r for r in rows if r['tipe'] == 'pengeluaran'])
    msg = (
        f"Total pengeluaran untuk periode **{period_label}** adalah **{format_rupiah(m['pengeluaran'])}** "
        f"dari total **{expense_count} transaksi pengeluaran** yang telah disetujui."
    )
    return {'tool': ToolName.GET_EXPENSE_SUMMARY, 'success': True, 'data': {'pengeluaran': m['pengeluaran'], 'count': expense_count}, 'message': msg}


def get_cash_flow_summary(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"

    cats = finance.list_categories(token)
    rows = period_rows(token, start=start, end=end)
    m = metrics(rows, cats)

    all_approved = period_rows(token, None, None)
    all_m = metrics(all_approved, cats)

    msg = (
        f"Ringkasan Arus Kas Toko (Periode: **{period_label}**):\n\n"
        f"- **Pemasukan:** {format_rupiah(m['pemasukan'])}\n"
        f"- **Pengeluaran:** {format_rupiah(m['pengeluaran'])}\n"
        f"- **Arus Kas Operasional:** {format_rupiah(m['arus_kas_operasional'])}\n"
        f"- **Arus Kas Bersih Total (Periode ini):** {format_rupiah(m['arus_kas_total'])}\n"
        f"- **Saldo Kas Kumulatif Toko:** **{format_rupiah(all_m['arus_kas_total'])}**\n\n"
        f"*Catatan:* Saldo kas dihitung dari transaksi yang telah disetujui."
    )
    return {'tool': ToolName.GET_CASH_FLOW_SUMMARY, 'success': True, 'data': m, 'message': msg}


def get_largest_expenses(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"

    cats = finance.list_categories(token)
    rows = period_rows(token, start=start, end=end)
    expense_rows = [r for r in rows if r['tipe'] == 'pengeluaran']

    if not expense_rows:
        return {
            'tool': ToolName.GET_LARGEST_EXPENSES,
            'success': True,
            'data': [],
            'message': f"Tidak ada catatan transaksi pengeluaran pada periode **{period_label}**."
        }

    sorted_exp = sorted(expense_rows, key=lambda x: Decimal(str(x['nominal'])), reverse=True)
    max_tx = sorted_exp[0]
    cmap = {c['id']: c['nama'] for c in cats}
    cat_name = cmap.get(max_tx['kategori_id'], 'Others')

    m = metrics(rows, cats)
    top_cat_part = ""
    if m['pengeluaran_per_kategori']:
        top_cat, top_cat_amt = max(m['pengeluaran_per_kategori'].items(), key=lambda x: x[1])
        top_cat_part = f"\n\nBerdasarkan akumulasi pos anggaran, pos pengeluaran terbanyak adalah **{top_cat}** ({format_rupiah(top_cat_amt)})."

    msg = (
        f"Pengeluaran terbesar pada periode **{period_label}** adalah transaksi **'{max_tx.get('deskripsi', '-')}'** "
        f"senilai **{format_rupiah(max_tx['nominal'])}** pada tanggal **{max_tx['tanggal']}** "
        f"(Kategori: **{cat_name}**).{top_cat_part}"
    )
    return {'tool': ToolName.GET_LARGEST_EXPENSES, 'success': True, 'data': sorted_exp[:5], 'message': msg}


def get_profit_analysis(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    start = decision.date_params.start_date
    end = decision.date_params.end_date
    period_label = decision.date_params.period_label or "Semua Periode"

    cats = finance.list_categories(token)
    rows = period_rows(token, start=start, end=end)
    m = metrics(rows, cats)

    omzet = m['omzet']
    pengeluaran = m['pengeluaran']
    arus_kas_ops = m['arus_kas_operasional']

    msg = (
        f"Analisis Arus Kas Operasional Toko periode **{period_label}**:\n\n"
        f"- **Omzet Penjualan (Pemasukan Operasional):** {format_rupiah(omzet)}\n"
        f"- **Total Beban Pengeluaran:** {format_rupiah(pengeluaran)}\n"
        f"- **Arus Kas Operasional Bersih:** **{format_rupiah(arus_kas_ops)}**\n\n"
        f"**Penjelasan Finansial Mengenai Laba Bersih:**\n"
        f"Dalam sistem pencatatan berbasis kas (*cash basis*) UMKM saat ini, angka di atas merepresentasikan "
        f"**Arus Kas Operasional Bersih (Net Operating Cash Flow)**, **bukan Laba Bersih Akuntansi (Net Profit)**.\n\n"
        f"Keterbatasan pencatatan kas saat ini:\n"
        f"1. **Harga Pokok Penjualan (HPP / COGS):** Belum memperhitungkan mutasi stok persediaan bahan.\n"
        f"2. **Penyusutan Aset (Depresiasi):** Belum memperhitungkan beban penurunan nilai mesin/peralatan.\n"
        f"3. **Prinsip Akrual:** Belum mencatat utang atau piutang yang belum cair secara tunai.\n\n"
        f"Untuk likuiditas uang tunai toko, sisa kas operasional pada periode ini adalah **{format_rupiah(arus_kas_ops)}**."
    )
    return {'tool': ToolName.GET_PROFIT_ANALYSIS, 'success': True, 'data': m, 'message': msg}


def compare_financial_periods(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko'})
    # Resolve P1 and P2
    from ai.financial_chat import resolve_comparison_periods
    p1_s, p1_e, p1_lbl, p2_s, p2_e, p2_lbl = resolve_comparison_periods(
        decision.raw_query or "", previous_period=decision.date_params.period_label
    )

    cats = finance.list_categories(token)
    r1 = period_rows(token, start=p1_s, end=p1_e)
    r2 = period_rows(token, start=p2_s, end=p2_e)
    m1 = metrics(r1, cats)
    m2 = metrics(r2, cats)

    def _diff_str(v1, v2):
        diff = v1 - v2
        if v2 > 0:
            pct = (diff / v2) * Decimal(100)
            pct_s = f"{abs(pct):.1f}%"
        else:
            pct_s = "N/A"
        if diff > 0:
            return f"naik **{format_rupiah(diff)}** (+{pct_s})"
        elif diff < 0:
            return f"turun **{format_rupiah(abs(diff))}** (-{pct_s})"
        return "sama persis (0%)"

    metric_type = decision.filter_params.transaction_type or "all"
    lines = [f"Perbandingan Keuangan Toko (**{p1_lbl}** vs **{p2_lbl}**):\n"]

    if metric_type == "pemasukan":
        lines.append(f"- **Pemasukan {p1_lbl}:** {format_rupiah(m1['pemasukan'])}")
        lines.append(f"- **Pemasukan {p2_lbl}:** {format_rupiah(m2['pemasukan'])}")
        lines.append(f"- **Perubahan:** Pemasukan {_diff_str(m1['pemasukan'], m2['pemasukan'])}.")
    elif metric_type == "pengeluaran":
        lines.append(f"- **Pengeluaran {p1_lbl}:** {format_rupiah(m1['pengeluaran'])}")
        lines.append(f"- **Pengeluaran {p2_lbl}:** {format_rupiah(m2['pengeluaran'])}")
        lines.append(f"- **Perubahan:** Pengeluaran {_diff_str(m1['pengeluaran'], m2['pengeluaran'])}.")
    else:
        lines.append(f"1. **Pemasukan:** {format_rupiah(m1['pemasukan'])} vs {format_rupiah(m2['pemasukan'])} &rarr; {_diff_str(m1['pemasukan'], m2['pemasukan'])}")
        lines.append(f"2. **Pengeluaran:** {format_rupiah(m1['pengeluaran'])} vs {format_rupiah(m2['pengeluaran'])} &rarr; {_diff_str(m1['pengeluaran'], m2['pengeluaran'])}")
        lines.append(f"3. **Arus Kas Bersih:** {format_rupiah(m1['arus_kas_total'])} vs {format_rupiah(m2['arus_kas_total'])} &rarr; {_diff_str(m1['arus_kas_total'], m2['arus_kas_total'])}")

    return {'tool': ToolName.COMPARE_FINANCIAL_PERIODS, 'success': True, 'data': {'m1': m1, 'm2': m2}, 'message': "\n".join(lines)}


# =====================================================================
# 5. APPLICATION GUIDANCE & FORECAST
# =====================================================================

def explain_feature(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    topic = decision.feature_topic or detect_feature_topic(decision.raw_query or "")
    msg = get_feature_explanation(topic, is_owner=(actor['role'] == 'pemilik_toko'))
    return {'tool': ToolName.EXPLAIN_FEATURE, 'success': True, 'data': {'topic': topic}, 'message': msg}


def explain_owner_approval_workflow(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    msg = get_feature_explanation('persetujuan', is_owner=(actor['role'] == 'pemilik_toko'))
    return {'tool': ToolName.EXPLAIN_OWNER_APPROVAL_WORKFLOW, 'success': True, 'data': {'topic': 'persetujuan'}, 'message': msg}


def explain_csv_import(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    msg = get_feature_explanation('impor', is_owner=(actor['role'] == 'pemilik_toko'))
    return {'tool': ToolName.EXPLAIN_CSV_IMPORT, 'success': True, 'data': {'topic': 'impor'}, 'message': msg}


def explain_ai_classification(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    msg = get_feature_explanation('klasifikasi_ai', is_owner=(actor['role'] == 'pemilik_toko'))
    return {'tool': ToolName.EXPLAIN_AI_CLASSIFICATION, 'success': True, 'data': {'topic': 'klasifikasi_ai'}, 'message': msg}


def get_available_forecast(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    msg = get_forecast_explanation(is_owner=(actor['role'] == 'pemilik_toko'))
    return {'tool': ToolName.GET_AVAILABLE_FORECAST, 'success': True, 'data': {}, 'message': msg}


# =====================================================================
# TOOL REGISTRY DISPATCH TABLE
# =====================================================================

TOOL_REGISTRY = {
    ToolName.GET_STORE_PROFILE: get_store_profile,
    ToolName.GET_STORE_BUSINESS_TYPE: get_store_business_type,
    ToolName.GET_STORE_OPERATIONAL_STATUS: get_store_operational_status,

    ToolName.COUNT_CASHIERS: count_cashiers,
    ToolName.LIST_CASHIERS: list_cashiers,
    ToolName.GET_CASHIER_ACTIVITY: get_cashier_activity,
    ToolName.GET_CASHIER_TRANSACTION_SUMMARY: get_cashier_transaction_summary,

    ToolName.LIST_INCOME_CATEGORIES: list_income_categories,
    ToolName.LIST_EXPENSE_CATEGORIES: list_expense_categories,
    ToolName.LIST_ALL_CATEGORIES: list_all_categories,
    ToolName.RANK_EXPENSE_CATEGORIES: rank_expense_categories,
    ToolName.RANK_INCOME_CATEGORIES: rank_income_categories,

    ToolName.GET_TRANSACTION_DETAILS: get_transaction_details,
    ToolName.LIST_TRANSACTIONS: list_transactions,
    ToolName.GET_CASHIER_TRANSACTIONS: get_cashier_transactions,
    ToolName.GET_PENDING_APPROVALS: get_pending_approvals,
    ToolName.COUNT_PENDING_APPROVALS: count_pending_approvals,
    ToolName.GET_INCOME_SUMMARY: get_income_summary,
    ToolName.GET_EXPENSE_SUMMARY: get_expense_summary,
    ToolName.GET_CASH_FLOW_SUMMARY: get_cash_flow_summary,
    ToolName.COMPARE_FINANCIAL_PERIODS: compare_financial_periods,
    ToolName.GET_LARGEST_EXPENSES: get_largest_expenses,
    ToolName.GET_PROFIT_ANALYSIS: get_profit_analysis,

    ToolName.EXPLAIN_FEATURE: explain_feature,
    ToolName.EXPLAIN_NAVIGATION: explain_feature,
    ToolName.EXPLAIN_OWNER_APPROVAL_WORKFLOW: explain_owner_approval_workflow,
    ToolName.EXPLAIN_CSV_IMPORT: explain_csv_import,
    ToolName.EXPLAIN_AI_CLASSIFICATION: explain_ai_classification,
    ToolName.GET_AVAILABLE_FORECAST: get_available_forecast,
}


def dispatch_tool(token: str, decision: PlannerDecision) -> Dict[str, Any]:
    tool_func = TOOL_REGISTRY.get(decision.tool_name)
    if not tool_func:
        return {
            'tool': decision.tool_name,
            'success': False,
            'message': f"Kapabilitas '{decision.tool_name}' belum didukung pada registry sistem."
        }
    try:
        return tool_func(token, decision)
    except AccessError as exc:
        return {
            'tool': decision.tool_name,
            'success': False,
            'message': f"Akses ditolak: {exc}"
        }
    except Exception as exc:
        return {
            'tool': decision.tool_name,
            'success': False,
            'message': f"Gagal mengeksekusi layanan: {exc}"
        }
