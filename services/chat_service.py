import logging
from datetime import date

logger = logging.getLogger(__name__)
from decimal import Decimal
import re

from database.client import privileged_client
from services.auth_service import verify_actor, AccessError
from services import finance_service as finance
from services import cashier_service
from services.report_service import period_rows, metrics
from utils.ui_components import format_rupiah
from ai.financial_chat import (
    Intent,
    detect_chat_intent,
    detect_query_metric,
    resolve_date_range_indonesian,
    resolve_comparison_periods,
    has_date_specifier,
    build_transaction_proposal,
    get_previous_context,
    get_jakarta_now,
)
from ai.classifier import detect_type
from ai.copilot_schema import (
    ToolName,
    PlannerDecision,
    ConversationContext,
    DateRangeParams,
    FilterParams,
)
from ai.copilot_planner import plan_intent
from ai.copilot_tools import dispatch_tool
from ai.copilot_knowledge import (
    get_feature_explanation,
    detect_feature_topic,
    get_forecast_explanation,
)
from ai.ollama_client import ask_text, LLMUnavailable

TOOL_TO_INTENT = {
    ToolName.GET_STORE_PROFILE: Intent.QUERY_STORE_PROFILE,
    ToolName.GET_STORE_BUSINESS_TYPE: Intent.QUERY_STORE_PROFILE,
    ToolName.GET_STORE_OPERATIONAL_STATUS: Intent.QUERY_STORE_PROFILE,

    ToolName.COUNT_CASHIERS: Intent.QUERY_CASHIER_ACTIVITY,
    ToolName.LIST_CASHIERS: Intent.QUERY_CASHIER_ACTIVITY,
    ToolName.GET_CASHIER_ACTIVITY: Intent.QUERY_CASHIER_ACTIVITY,
    ToolName.GET_CASHIER_TRANSACTION_SUMMARY: Intent.QUERY_CASHIER_ACTIVITY,

    ToolName.LIST_INCOME_CATEGORIES: Intent.LIST_INCOME_CATEGORIES,
    ToolName.LIST_EXPENSE_CATEGORIES: Intent.LIST_EXPENSE_CATEGORIES,
    ToolName.LIST_ALL_CATEGORIES: Intent.LIST_ALL_CATEGORIES,
    ToolName.RANK_EXPENSE_CATEGORIES: Intent.QUERY_FINANCIAL_INSIGHTS,
    ToolName.RANK_INCOME_CATEGORIES: Intent.QUERY_FINANCIAL_INSIGHTS,

    ToolName.GET_TRANSACTION_DETAILS: Intent.QUERY_TRANSACTIONS,
    ToolName.LIST_TRANSACTIONS: Intent.QUERY_TRANSACTIONS,
    ToolName.GET_CASHIER_TRANSACTIONS: Intent.QUERY_TRANSACTIONS,
    ToolName.GET_PENDING_APPROVALS: Intent.QUERY_PENDING_APPROVALS,
    ToolName.COUNT_PENDING_APPROVALS: Intent.QUERY_PENDING_APPROVALS,

    ToolName.GET_INCOME_SUMMARY: Intent.QUERY_INCOME,
    ToolName.GET_EXPENSE_SUMMARY: Intent.QUERY_EXPENSE,
    ToolName.GET_CASH_FLOW_SUMMARY: Intent.QUERY_CASH_FLOW,
    ToolName.COMPARE_FINANCIAL_PERIODS: Intent.QUERY_FINANCIAL_COMPARISON,
    ToolName.GET_LARGEST_EXPENSES: Intent.QUERY_FINANCIAL_INSIGHTS,
    ToolName.GET_PROFIT_ANALYSIS: Intent.QUERY_PROFIT,

    ToolName.EXPLAIN_FEATURE: Intent.EXPLAIN_APP_FEATURE,
    ToolName.GET_BUSINESS_GUIDANCE: Intent.QUERY_BUSINESS_GUIDANCE,
    ToolName.EXPLAIN_NAVIGATION: Intent.EXPLAIN_APP_FEATURE,
    ToolName.EXPLAIN_OWNER_APPROVAL_WORKFLOW: Intent.EXPLAIN_APP_FEATURE,
    ToolName.EXPLAIN_CSV_IMPORT: Intent.EXPLAIN_APP_FEATURE,
    ToolName.EXPLAIN_AI_CLASSIFICATION: Intent.EXPLAIN_APP_FEATURE,
    ToolName.GET_AVAILABLE_FORECAST: Intent.QUERY_FORECAST,
    ToolName.GET_EXPENSE_FORECAST: Intent.QUERY_FORECAST,
    ToolName.GET_FORECAST_FOR_PERIOD: Intent.QUERY_FORECAST,
    ToolName.GET_UPCOMING_RECURRING_EXPENSES: Intent.QUERY_FORECAST,
    ToolName.GET_NEXT_PREDICTED_EXPENSE: Intent.QUERY_FORECAST,
    ToolName.COMPARE_ACTUAL_VS_FORECAST: Intent.QUERY_FORECAST,
    ToolName.GET_FORECAST_ACCURACY: Intent.QUERY_FORECAST,
    ToolName.GET_EXPENSE_TREND: Intent.QUERY_FORECAST,
    ToolName.GET_FORECAST_READINESS: Intent.QUERY_FORECAST,
    ToolName.CHECK_FORECAST_AGAINST_CASH_BALANCE: Intent.QUERY_FORECAST,

    ToolName.CREATE_TRANSACTION_PROPOSAL: Intent.CREATE_INCOME,
    ToolName.UNSUPPORTED_MUTATION: Intent.UNSUPPORTED_MUTATION,
    ToolName.GENERAL_HELP: Intent.GENERAL_HELP,
    ToolName.CLARIFICATION_NEEDED: Intent.CLARIFICATION_NEEDED,
    ToolName.UNKNOWN: Intent.UNKNOWN,
}


def _angka_rp(teks):
    """Extract Rp amounts as integers, e.g. 'Rp 200.000' -> {200000}."""
    return {int(re.sub(r"\D", "", m)) for m in re.findall(r"Rp\s?-?([\d.]+)", teks) if re.sub(r"\D", "", m)}


def _is_numeric_consistent(fact_text, candidate_text):
    if not candidate_text:
        return False
    # Extract clean numeric tokens from facts and candidate
    fact_nums = set(re.findall(r'\b\d+(?:[\.,]\d+)*\b', fact_text))
    # Normalized digits
    fact_digits = set(re.findall(r'\d+', fact_text.replace('.', '').replace(',', '')))
    cand_digits = set(re.findall(r'\d+', candidate_text.replace('.', '').replace(',', '')))

    # Reject if candidate introduces multi-digit numbers not grounded in verified facts
    hallucinated = [n for n in cand_digits if len(n) >= 2 and n not in fact_digits]
    if hallucinated:
        return False

    # Check percentages: any percentage in candidate must appear in fact
    cand_pcts = re.findall(r'\b\d+(?:[\.,]\d+)?\s*%', candidate_text)
    fact_pcts = {p.replace(' ', '') for p in re.findall(r'\b\d+(?:[\.,]\d+)?\s*%', fact_text)}
    for p in cand_pcts:
        p_clean = p.replace(' ', '')
        if p_clean not in fact_pcts:
            return False

    # Rp amounts in the candidate must be a subset of the Rp amounts in the facts.
    # This closes the hole where a bare digit string appears in the facts as a
    # count (e.g. "500 transaksi") while "Rp 500" is invented in the candidate.
    if not _angka_rp(candidate_text) <= _angka_rp(fact_text):
        return False

    return True


def _handle_category_listing(token, intent, actor):
    categories = finance.list_categories(token)
    if not categories:
        return {
            'intent': intent,
            'type': 'message',
            'message': (
                "Belum ada kategori pembukuan yang terdaftar untuk toko Anda. "
                "Pemilik toko dapat menyiapkan kategori dasar otomatis di menu **Kelola Kategori** (Kategori Pembukuan)."
            )
        }

    income_cats = [c for c in categories if c['tipe'] == 'pemasukan']
    expense_cats = [c for c in categories if c['tipe'] == 'pengeluaran']

    def _fmt_cat(cat):
        tag = "Kustom"
        if cat.get('sistem'):
            tag = "Sistem (Others / Fallback)"
        elif cat.get('bukan_operasional'):
            tag = "Non-Operasional (Tambahan Modal)"
        desc = f" - *{cat['deskripsi']}*" if cat.get('deskripsi') else ""
        return f"- **{cat['nama']}** `[{tag}]`{desc}"

    lines = []
    if intent in (Intent.LIST_ALL_CATEGORIES, Intent.LIST_INCOME_CATEGORIES):
        lines.append(f"### Kategori Pemasukan ({len(income_cats)})")
        if income_cats:
            for c in income_cats:
                lines.append(_fmt_cat(c))
        else:
            lines.append("- *(Belum ada kategori pemasukan)*")
        lines.append("")

    if intent in (Intent.LIST_ALL_CATEGORIES, Intent.LIST_EXPENSE_CATEGORIES):
        lines.append(f"### Kategori Pengeluaran ({len(expense_cats)})")
        if expense_cats:
            for c in expense_cats:
                lines.append(_fmt_cat(c))
        else:
            lines.append("- *(Belum ada kategori pengeluaran)*")
        lines.append("")

    lines.append("**Keterangan Pos Anggaran:**")
    lines.append("- **Kustom:** Pos penerimaan/pengeluaran operasional toko.")
    lines.append("- **Sistem (Others):** Kategori sistem wajib sebagai penampung transaksi yang belum teridentifikasi atau berkeyakinan rendah.")
    lines.append("- **Non-Operasional (Tambahan Modal):** Khusus setoran modal yang tidak dimasukkan ke dalam perhitungan omzet penjualan.")
    lines.append("\nPengelolaan kategori (tambah, ubah, hapus, atau gabung) dapat dilakukan di menu **Kelola Kategori** (Kategori Pembukuan).")

    return {
        'intent': intent,
        'type': 'message',
        'message': "\n".join(lines).strip()
    }


def _handle_pending_approvals(token, raw_text, actor):
    is_owner = actor['role'] == 'pemilik_toko'
    if not is_owner:
        return {
            'intent': Intent.QUERY_PENDING_APPROVALS,
            'type': 'message',
            'message': (
                "Informasi persetujuan transaksi kasir dan kewenangan verifikasi hanya dapat diakses oleh **Pemilik Toko**. "
                "Sebagai kasir, Anda dapat melihat status transaksi yang Anda catat sendiri di menu **Transaksi**."
            )
        }

    raw_tx = finance.list_transactions(token)
    pending = [t for t in raw_tx if t.get('status_persetujuan') == 'pending']

    filter_by_date = has_date_specifier(raw_text)
    start, end, period_label = resolve_date_range_indonesian(raw_text)

    if filter_by_date:
        if start:
            pending = [t for t in pending if t['tanggal'] >= str(start)]
        if end:
            pending = [t for t in pending if t['tanggal'] <= str(end)]
    else:
        period_label = 'Semua Tanggal'

    if not pending:
        return {
            'intent': Intent.QUERY_PENDING_APPROVALS,
            'type': 'message',
            'message': (
                f"Tidak ada transaksi kasir yang menunggu persetujuan pada periode **{period_label}**. "
                "Semua transaksi telah diverifikasi atau belum ada entri kasir baru."
            )
        }

    cats = finance.list_categories(token)
    cmap = {c['id']: c['nama'] for c in cats}

    cashier_map = {}
    try:
        cashiers = cashier_service.list_cashiers(token)
        cashier_map = {c['id']: c['nama'] for c in cashiers}
    except Exception as exc:
        logger.warning("Could not resolve cashier names; using default Kasir labels: %s", exc)
        cashier_map = {}

    total_nominal = sum(Decimal(str(t['nominal'])) for t in pending)
    lines = [
        f"Ditemukan **{len(pending)} transaksi kasir** yang **menunggu persetujuan** "
        f"pada periode **{period_label}** dengan total nilai **{format_rupiah(total_nominal)}**:\n"
    ]

    for idx, t in enumerate(pending[:10], start=1):
        cat_name = cmap.get(t['kategori_id'], 'Others')
        cashier_name = cashier_map.get(t['dicatat_oleh'], 'Kasir')
        tp_label = "Pemasukan" if t['tipe'] == 'pemasukan' else "Pengeluaran"
        lines.append(
            f"{idx}. **{t['tanggal']}** &bull; {tp_label} &bull; **{format_rupiah(t['nominal'])}** &bull; "
            f"*{t.get('deskripsi', '-')}* (Kategori: {cat_name}, Kasir: {cashier_name})"
        )

    if len(pending) > 10:
        lines.append(f"\n*(Menampilkan 10 dari {len(pending)} transaksi. Buka menu **Persetujuan Kasir** untuk verifikasi massal.)*")
    else:
        lines.append("\nAnda dapat meninjau dan melakukan verifikasi massal melalui menu **Persetujuan Kasir**.")

    return {
        'intent': Intent.QUERY_PENDING_APPROVALS,
        'type': 'message',
        'message': "\n".join(lines)
    }


def _handle_cashier_activity(token, raw_text, actor):
    is_owner = actor['role'] == 'pemilik_toko'
    if not is_owner:
        return {
            'intent': Intent.QUERY_CASHIER_ACTIVITY,
            'type': 'message',
            'message': "Daftar akun kasir dan pengelolaan staf hanya dapat diakses oleh **Pemilik Toko**."
        }

    try:
        cashiers = cashier_service.list_cashiers(token)
    except Exception as exc:
        return {
            'intent': Intent.QUERY_CASHIER_ACTIVITY,
            'type': 'message',
            'message': f"Gagal mengambil daftar kasir: {exc}"
        }

    if not cashiers:
        return {
            'intent': Intent.QUERY_CASHIER_ACTIVITY,
            'type': 'message',
            'message': (
                "Belum ada akun kasir terdaftar untuk toko Anda. "
                "Pemilik toko dapat mendaftarkan kasir baru melalui menu **Akun Kasir**."
            )
        }

    txs = finance.list_transactions(token)
    filter_by_date = has_date_specifier(raw_text)
    start, end, period_label = resolve_date_range_indonesian(raw_text)

    if filter_by_date:
        filtered_tx = [t for t in txs if (not start or t.get('tanggal', '') >= str(start)) and (not end or t.get('tanggal', '') <= str(end))]
    else:
        filtered_tx = txs
        period_label = 'Seluruh Waktu'

    counts = {c['id']: {'total': 0, 'disetujui': 0, 'pending': 0, 'nominal': Decimal(0)} for c in cashiers}
    for t in filtered_tx:
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
    s = raw_text.lower()
    asks_count = bool(
        re.search(r'\b(?:berapa|jumlah|total)\s+(?:semua\s+)?kasir\b', s) or
        re.search(r'\bada\s+berapa\s+kasir\b', s) or
        re.search(r'\bberapa\s+(?:banyak\s+)?kasir\b', s) or
        re.search(r'\bkasir\s+(?:ada\s+)?berapa\b', s)
    )
    if asks_count:
        active_c = len([c for c in cashiers if c.get('status') == 'aktif'])
        lines.append(
            f"Toko Anda saat ini memiliki total **{len(cashiers)} akun kasir terdaftar** "
            f"(**{active_c} aktif**).\n"
        )

    asks_most_active = any(k in s for k in ['paling aktif', 'kasir mana', 'terbanyak', 'teraktif'])

    if asks_most_active:
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
            lines.append(f"Belum ada transaksi yang dicatat oleh kasir pada periode **{period_label}**.\n")

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

    lines.append("\nPendaftaran dan pengelolaan akun kasir dapat dilakukan di menu **Akun Kasir**.")
    return {
        'intent': Intent.QUERY_CASHIER_ACTIVITY,
        'type': 'message',
        'message': "\n".join(lines),
        'metadata': {'intent': Intent.QUERY_CASHIER_ACTIVITY, 'metric': 'aktivitas_kasir', 'period': period_label}
    }


def _handle_profit_query(token, raw_text, actor):
    is_owner = actor['role'] == 'pemilik_toko'
    if not is_owner:
        return {
            'intent': Intent.QUERY_PROFIT,
            'type': 'message',
            'message': "Informasi analisis laba dan arus kas operasional toko hanya dapat diakses oleh **Pemilik Toko**."
        }

    start, end, period_label = resolve_date_range_indonesian(raw_text)
    all_cats = finance.list_categories(token)
    rows = period_rows(token, start=start, end=end)
    m = metrics(rows, all_cats)

    omzet = m['omzet']
    pengeluaran = m['pengeluaran']
    arus_kas_ops = m['arus_kas_operasional']

    lines = [
        f"Analisis Arus Kas Operasional Toko periode **{period_label}**:\n",
        f"- **Omzet Penjualan (Pemasukan Operasional):** {format_rupiah(omzet)}",
        f"- **Total Beban Pengeluaran:** {format_rupiah(pengeluaran)}",
        f"- **Arus Kas Operasional Bersih:** **{format_rupiah(arus_kas_ops)}**\n",
        "**Penjelasan Finansial Mengenai Laba Bersih:**",
        "Dalam sistem pencatatan berbasis kas (*cash basis*) UMKM saat ini, angka di atas merepresentasikan **Arus Kas Operasional Bersih (Net Operating Cash Flow)**, **bukan Laba Bersih Akuntansi (Net Profit)**.",
        "Keterbatasan pencatatan kas:",
        "1. **Harga Pokok Penjualan (HPP / COGS):** Belum memperhitungkan mutasi stok awal dan sisa persediaan bahan baku.",
        "2. **Penyusutan Aset (Depresiasi):** Belum memperhitungkan penurunan nilai ekonomis mesin atau peralatan toko.",
        "3. **Prinsip Akrual:** Belum mencatat utang atau piutang yang belum cair menjadi uang tunai.",
        f"\nUntuk kebutuhan likuiditas kas operasional, saldo kelebihan uang kas operasional toko pada periode ini adalah **{format_rupiah(arus_kas_ops)}**."
    ]

    return {
        'intent': Intent.QUERY_PROFIT,
        'type': 'message',
        'message': "\n".join(lines),
        'metadata': {'intent': Intent.QUERY_PROFIT, 'metric': 'arus_kas_operasional', 'period': period_label}
    }


def _handle_financial_comparison(token, raw_text, history, actor):
    is_owner = actor['role'] == 'pemilik_toko'
    if not is_owner:
        return {
            'intent': Intent.QUERY_FINANCIAL_COMPARISON,
            'type': 'message',
            'message': "Analisis dan perbandingan keuangan toko hanya dapat diakses oleh **Pemilik Toko**."
        }

    prev_ctx = get_previous_context(history)
    s = raw_text.lower()

    if any(k in s for k in ['pemasukan', 'omzet', 'pendapatan', 'penjualan']):
        metric_type = 'pemasukan'
    elif any(k in s for k in ['pengeluaran', 'biaya', 'belanja']):
        metric_type = 'pengeluaran'
    elif any(k in s for k in ['saldo', 'arus kas', 'cash flow', 'cashflow']):
        metric_type = 'arus_kas_total'
    elif prev_ctx.get('metric') in ('pemasukan', 'pengeluaran', 'saldo', 'arus_kas_total', 'arus_kas_operasional'):
        metric_type = prev_ctx['metric']
    else:
        metric_type = 'ringkasan_umum'

    p1_start, p1_end, p1_label, p2_start, p2_end, p2_label = resolve_comparison_periods(
        raw_text, previous_period=prev_ctx.get('period')
    )

    cats = finance.list_categories(token)
    r1 = period_rows(token, start=p1_start, end=p1_end)
    r2 = period_rows(token, start=p2_start, end=p2_end)
    m1 = metrics(r1, cats)
    m2 = metrics(r2, cats)

    def _calc_diff(val1, val2):
        diff = val1 - val2
        if val2 > 0:
            pct = (diff / val2) * Decimal(100)
            pct_str = f"{abs(pct):.1f}%"
        elif val2 < 0 and val1 >= 0:
            pct_str = "Membaik"
        else:
            pct_str = "N/A"
        if diff > 0:
            trend = f"naik **{format_rupiah(diff)}** (+{pct_str})"
        elif diff < 0:
            trend = f"turun **{format_rupiah(abs(diff))}** (-{pct_str})"
        else:
            trend = "sama persis (0%)"
        return diff, trend

    lines = [f"Perbandingan Keuangan Toko (**{p1_label}** vs **{p2_label}**):\n"]

    if metric_type == 'pemasukan':
        v1 = m1['pemasukan']
        v2 = m2['pemasukan']
        _, trend = _calc_diff(v1, v2)
        lines.append(f"- **Pemasukan {p1_label}:** {format_rupiah(v1)}")
        lines.append(f"- **Pemasukan {p2_label}:** {format_rupiah(v2)}")
        lines.append(f"- **Perubahan:** Pemasukan {trend}.")

    elif metric_type == 'pengeluaran':
        v1 = m1['pengeluaran']
        v2 = m2['pengeluaran']
        _, trend = _calc_diff(v1, v2)
        lines.append(f"- **Pengeluaran {p1_label}:** {format_rupiah(v1)}")
        lines.append(f"- **Pengeluaran {p2_label}:** {format_rupiah(v2)}")
        lines.append(f"- **Perubahan:** Pengeluaran {trend}.")

    elif metric_type in ('saldo', 'arus_kas_total'):
        v1 = m1['arus_kas_total']
        v2 = m2['arus_kas_total']
        _, trend = _calc_diff(v1, v2)
        lines.append(f"- **Arus Kas Bersih {p1_label}:** {format_rupiah(v1)}")
        lines.append(f"- **Arus Kas Bersih {p2_label}:** {format_rupiah(v2)}")
        lines.append(f"- **Perubahan:** Arus kas {trend}.")

    else:
        _, trend_inc = _calc_diff(m1['pemasukan'], m2['pemasukan'])
        _, trend_exp = _calc_diff(m1['pengeluaran'], m2['pengeluaran'])
        _, trend_cf = _calc_diff(m1['arus_kas_total'], m2['arus_kas_total'])

        lines.append(f"1. **Pemasukan:** {format_rupiah(m1['pemasukan'])} (vs {format_rupiah(m2['pemasukan'])}) &rarr; {trend_inc}")
        lines.append(f"2. **Pengeluaran:** {format_rupiah(m1['pengeluaran'])} (vs {format_rupiah(m2['pengeluaran'])}) &rarr; {trend_exp}")
        lines.append(f"3. **Arus Kas Bersih:** {format_rupiah(m1['arus_kas_total'])} (vs {format_rupiah(m2['arus_kas_total'])}) &rarr; {trend_cf}")

    return {
        'intent': Intent.QUERY_FINANCIAL_COMPARISON,
        'type': 'message',
        'message': "\n".join(lines),
        'metadata': {'intent': Intent.QUERY_FINANCIAL_COMPARISON, 'metric': metric_type, 'period': p1_label}
    }


def _handle_financial_insights(token, raw_text, actor):
    is_owner = actor['role'] == 'pemilik_toko'
    if not is_owner:
        return {
            'intent': Intent.QUERY_FINANCIAL_INSIGHTS,
            'type': 'message',
            'message': "Informasi analisis keuangan toko hanya dapat diakses oleh **Pemilik Toko**."
        }

    start, end, period_label = resolve_date_range_indonesian(raw_text)
    all_cats = finance.list_categories(token)
    rows = period_rows(token, start=start, end=end)
    data_metrics = metrics(rows, all_cats)

    s = raw_text.lower()
    is_top_category = (
        'kategori' in s and ('paling banyak' in s or 'terbesar' in s or 'boros' in s or 'menghabiskan' in s or 'peringkat' in s or 'ranking' in s)
    )

    if is_top_category:
        cat_expenses = data_metrics['pengeluaran_per_kategori']
        if not cat_expenses:
            return {
                'intent': Intent.QUERY_FINANCIAL_INSIGHTS,
                'type': 'message',
                'message': f"Belum ada data pengeluaran yang tercatat pada periode **{period_label}**."
            }
        sorted_cats = sorted(cat_expenses.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_cat_amt = sorted_cats[0]
        total_exp = data_metrics['pengeluaran']
        pct = (top_cat_amt / total_exp * 100) if total_exp > 0 else Decimal(0)

        if any(w in s for w in ['apa saja', 'daftar', 'peringkat', 'ranking', 'semua']):
            lines = [f"Peringkat kategori pengeluaran terbanyak pada periode **{period_label}** (Total: **{format_rupiah(total_exp)}**):\n"]
            for idx, (cname, camt) in enumerate(sorted_cats, start=1):
                cpct = (camt / total_exp * 100) if total_exp > 0 else Decimal(0)
                lines.append(f"{idx}. **{cname}** &bull; **{format_rupiah(camt)}** ({cpct:.1f}%)")
            lines.append(f"\nPos pengeluaran terbesar yang paling banyak menyerap kas adalah **{top_cat}**.")
            msg = "\n".join(lines)
        else:
            msg = (
                f"Kategori yang paling banyak menghabiskan uang pada periode **{period_label}** adalah **{top_cat}** "
                f"dengan akumulasi pengeluaran sebesar **{format_rupiah(top_cat_amt)}** "
                f"(sekitar **{pct:.1f}%** dari total pengeluaran)."
            )

        return {
            'intent': Intent.QUERY_FINANCIAL_INSIGHTS,
            'type': 'message',
            'message': msg,
            'metadata': {'intent': Intent.QUERY_FINANCIAL_INSIGHTS, 'metric': 'kategori_terbesar', 'period': period_label}
        }

    # Top single expense transaction
    expense_rows = [r for r in rows if r['tipe'] == 'pengeluaran']
    if not expense_rows:
        return {
            'intent': Intent.QUERY_FINANCIAL_INSIGHTS,
            'type': 'message',
            'message': f"Tidak ada catatan transaksi pengeluaran pada periode **{period_label}**."
        }

    max_tx = max(expense_rows, key=lambda x: Decimal(str(x['nominal'])))
    cmap = {c['id']: c['nama'] for c in all_cats}
    cat_name = cmap.get(max_tx['kategori_id'], 'Others')

    top_cat_part = ""
    if data_metrics['pengeluaran_per_kategori']:
        top_cat, top_cat_amt = max(data_metrics['pengeluaran_per_kategori'].items(), key=lambda x: x[1])
        top_cat_part = f"\n\nBerdasarkan akumulasi pos anggaran, pos pengeluaran terbanyak adalah **{top_cat}** ({format_rupiah(top_cat_amt)})."

    return {
        'intent': Intent.QUERY_FINANCIAL_INSIGHTS,
        'type': 'message',
        'message': (
            f"Pengeluaran terbesar pada periode **{period_label}** adalah transaksi **'{max_tx['deskripsi']}'** "
            f"senilai **{format_rupiah(max_tx['nominal'])}** pada tanggal **{max_tx['tanggal']}** "
            f"(Kategori: **{cat_name}**).{top_cat_part}"
        ),
        'metadata': {'intent': Intent.QUERY_FINANCIAL_INSIGHTS, 'metric': 'pengeluaran_terbesar', 'period': period_label}
    }


def _handle_store_profile(token, actor):
    try:
        shop = privileged_client().table('toko').select('id,nama,bidang_usaha,created_at').eq('id', actor['toko_id']).single().execute().data
    except Exception as exc:
        return {
            'intent': Intent.QUERY_STORE_PROFILE,
            'type': 'message',
            'message': f"Gagal mengambil informasi profil toko: {exc}"
        }

    created_date = str(shop.get('created_at', ''))[:10]
    return {
        'intent': Intent.QUERY_STORE_PROFILE,
        'type': 'message',
        'message': (
            f"### Profil Toko Terdaftar\n\n"
            f"- **Nama Usaha:** {shop['nama']}\n"
            f"- **Bidang Usaha:** {shop['bidang_usaha']}\n"
            f"- **Pemilik Toko:** {actor['nama']} (`{actor['email']}`)\n"
            f"- **Terdaftar Sejak:** {created_date}\n"
            f"- **Status Operasional:** Aktif"
        )
    }


def _handle_cashier_transactions(token, raw_text, actor):
    is_owner = actor['role'] == 'pemilik_toko'
    raw_tx = finance.list_transactions(token)
    cashiers = cashier_service.list_cashiers(token) if is_owner else []
    cashier_ids = {c['id'] for c in cashiers}

    if is_owner:
        cashier_tx = [t for t in raw_tx if t.get('dicatat_oleh') in cashier_ids]
    else:
        cashier_tx = [t for t in raw_tx if t.get('dicatat_oleh') == actor['id']]

    start, end, period_label = resolve_date_range_indonesian(raw_text)
    if has_date_specifier(raw_text):
        if start:
            cashier_tx = [t for t in cashier_tx if t.get('tanggal', '') >= str(start)]
        if end:
            cashier_tx = [t for t in cashier_tx if t.get('tanggal', '') <= str(end)]
    else:
        today_iso = str(get_jakarta_now().date())
        if 'hari ini' in raw_text.lower():
            cashier_tx = [t for t in cashier_tx if t.get('tanggal', '') == today_iso]
            period_label = 'Hari Ini'
        else:
            period_label = 'Seluruh Waktu'

    if not cashier_tx:
        return {
            'intent': Intent.QUERY_TRANSACTIONS,
            'type': 'message',
            'message': f"Tidak ada transaksi yang dicatat oleh kasir pada periode **{period_label}**."
        }

    cats = finance.list_categories(token)
    cmap = {c['id']: c['nama'] for c in cats}
    cmap_kasir = {c['id']: c['nama'] for c in cashiers}

    approved_count = len([t for t in cashier_tx if t.get('status_persetujuan') == 'disetujui'])
    pending_count = len([t for t in cashier_tx if t.get('status_persetujuan') == 'pending'])
    total_amt = sum(Decimal(str(t.get('nominal', 0))) for t in cashier_tx)

    lines = [
        f"Ditemukan **{len(cashier_tx)} transaksi kasir** pada periode **{period_label}** "
        f"(**{approved_count} disetujui**, **{pending_count} menunggu persetujuan**) "
        f"dengan total nilai **{format_rupiah(total_amt)}**:\n"
    ]

    for idx, t in enumerate(cashier_tx[:10], start=1):
        cat_name = cmap.get(t.get('kategori_id'), 'Others')
        cashier_name = cmap_kasir.get(t.get('dicatat_oleh'), 'Kasir')
        tp_label = "Pemasukan" if t.get('tipe') == 'pemasukan' else "Pengeluaran"
        status_label = "Disetujui" if t.get('status_persetujuan') == 'disetujui' else "Menunggu Persetujuan"
        desc = t.get('deskripsi', '-')
        nom = t.get('nominal', 0)
        lines.append(
            f"{idx}. **{t.get('tanggal', '-')}** &bull; {tp_label} &bull; **{format_rupiah(nom)}** &bull; "
            f"*{desc}* (Kategori: {cat_name}, Kasir: {cashier_name}, Status: **{status_label}**)"
        )

    if len(cashier_tx) > 10:
        lines.append(f"\n*(Menampilkan 10 dari {len(cashier_tx)} transaksi. Buka menu **Transaksi** untuk riwayat lengkap.)*")
    elif is_owner and pending_count > 0:
        lines.append("\nUntuk menyetujui transaksi yang berstatus pending, buka menu **Persetujuan Kasir**.")

    return {
        'intent': Intent.QUERY_TRANSACTIONS,
        'type': 'message',
        'message': "\n".join(lines),
        'metadata': {'intent': Intent.QUERY_TRANSACTIONS, 'metric': 'transaksi_kasir', 'period': period_label}
    }


def _handle_transaction_details(token, raw_text, actor):
    is_owner = actor['role'] == 'pemilik_toko'
    raw_tx = finance.list_transactions(token)
    if not is_owner:
        raw_tx = [t for t in raw_tx if t.get('dicatat_oleh') == actor['id']]

    start, end, period_label = resolve_date_range_indonesian(raw_text)
    s = raw_text.lower()
    req_type = None
    if any(w in s for w in ['pengeluaran', 'biaya', 'belanja']):
        req_type = 'pengeluaran'
    elif any(w in s for w in ['pemasukan', 'pendapatan', 'omzet']):
        req_type = 'pemasukan'

    if start:
        raw_tx = [t for t in raw_tx if t.get('tanggal', '') >= str(start)]
    if end:
        raw_tx = [t for t in raw_tx if t.get('tanggal', '') <= str(end)]
    if req_type:
        raw_tx = [t for t in raw_tx if t.get('tipe') == req_type]

    type_str = f" {req_type.capitalize()}" if req_type else ""
    if not raw_tx:
        return {
            'intent': Intent.QUERY_TRANSACTIONS,
            'type': 'message',
            'message': f"Tidak ditemukan transaksi{type_str} pada periode **{period_label}**."
        }

    cats = finance.list_categories(token)
    cmap = {c['id']: c['nama'] for c in cats}
    total_val = sum(Decimal(str(t.get('nominal', 0))) for t in raw_tx)

    lines = [
        f"Ditemukan **{len(raw_tx)} transaksi{type_str}** pada periode **{period_label}** "
        f"dengan total akumulasi **{format_rupiah(total_val)}**:\n"
    ]

    for idx, t in enumerate(raw_tx[:10], start=1):
        cat_name = cmap.get(t.get('kategori_id'), 'Others')
        tp_label = "Pemasukan" if t.get('tipe') == 'pemasukan' else "Pengeluaran"
        status_label = "Disetujui" if t.get('status_persetujuan') == 'disetujui' else "Menunggu Persetujuan"
        desc = t.get('deskripsi', '-')
        nom = t.get('nominal', 0)
        lines.append(
            f"{idx}. **{t.get('tanggal', '-')}** &bull; {tp_label} &bull; **{format_rupiah(nom)}** &bull; "
            f"*{desc}* (Kategori: {cat_name}, Status: {status_label})"
        )

    if len(raw_tx) > 10:
        lines.append(f"\n*(Menampilkan 10 dari {len(raw_tx)} transaksi. Buka menu **Transaksi** untuk riwayat lengkap.)*")

    return {
        'intent': Intent.QUERY_TRANSACTIONS,
        'type': 'message',
        'message': "\n".join(lines),
        'metadata': {'intent': Intent.QUERY_TRANSACTIONS, 'metric': req_type or 'transaksi', 'period': period_label}
    }


def handle_chat_message(token, message_text, history=None, page_context=None):
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    is_owner = actor['role'] == 'pemilik_toko'
    raw_text = str(message_text or '').strip()

    if not raw_text:
        return {
            'intent': Intent.UNKNOWN,
            'type': 'message',
            'message': 'Silakan masukkan pertanyaan atau instruksi pencatatan keuangan Anda.'
        }

    # Contextual inquiry about current page
    s_lower = raw_text.lower()
    if page_context and any(w in s_lower for w in ['halaman ini', 'menu ini', 'layar ini', 'di sini']):
        page_explanations = {
            "Ringkasan Bisnis": "Halaman **Ringkasan Bisnis** menyajikan visualisasi arus kas, metrik omzet, perbandingan pengeluaran, dan peramalan pengeluaran multi-model (Prophet, ETS, ARIMA, Seasonal Naive, MA7, LinearRegression) toko Anda.",
            "Transaksi": "Halaman **Transaksi** menampilkan seluruh riwayat transaksi toko dengan status persetujuan, filter tanggal, pencarian, dan opsi hapus.",
            "Pencatatan Cepat": "Halaman **Pencatatan Cepat** memungkinkan entri transaksi kilat dengan bantuan klasifikasi otomatis AI.",
            "Persetujuan Kasir": "Halaman **Persetujuan Kasir** digunakan oleh Pemilik Toko untuk meninjau dan menyetujui transaksi yang dicatat kasir sebelum masuk ke laporan resmi.",
            "Kelola Kategori": "Halaman **Kelola Kategori** digunakan untuk mengatur pos-pos anggaran pemasukan dan pengeluaran toko Anda.",
            "Impor Data CSV": "Halaman **Impor Data CSV** memungkinkan impor massal berkas transaksi lama dengan pemetaan kolom otomatis dan penemuan kategori cerdas.",
            "Panduan Usaha": "Halaman **Panduan Usaha** menyediakan referensi operasional sesuai bidang usaha toko Anda (stok, HPP, SOP harian, pembukuan, dan legalitas) lengkap dengan rujukan sumber.",
            "Akun Kasir": "Halaman **Akun Kasir** digunakan untuk mendaftarkan dan memantau akun kasir yang bertugas di toko Anda.",
            "Status Sistem": "Halaman **Status Sistem** menyajikan diagnosa koneksi database Supabase dan kesiapan model AI Ollama.",
        }
        expl = page_explanations.get(page_context)
        if expl:
            return {
                'intent': Intent.EXPLAIN_APP_FEATURE,
                'tool': ToolName.EXPLAIN_FEATURE.value,
                'type': 'message',
                'message': f"Anda saat ini berada di halaman **{page_context}**.\n\n{expl}\n\nApakah ada hal spesifik yang ingin Anda tanyakan atau lakukan di halaman ini?"
            }

    # Stage 1: Build context from conversation history
    prev_ctx = get_previous_context(history)
    conv_context = ConversationContext(
        last_tool=prev_ctx.get('tool') or prev_ctx.get('intent'),
        last_metric=prev_ctx.get('metric'),
        last_period_label=prev_ctx.get('period'),
        last_start_date=prev_ctx.get('start_date'),
        last_end_date=prev_ctx.get('end_date'),
        last_forecast_period=prev_ctx.get('period'),
        last_forecast_start_date=prev_ctx.get('start_date'),
        last_forecast_end_date=prev_ctx.get('end_date'),
        last_forecast_days=prev_ctx.get('days_ahead'),
        last_forecast_category=prev_ctx.get('category'),
        last_forecast_output_mode=prev_ctx.get('output_mode'),
        store_id=actor.get('toko_id'),
        user_id=actor.get('id')
    )

    # Stage 2: Intent Planning (Stage A: Deterministic Regex/Tokens, Stage B: Ollama Structured JSON)
    decision = plan_intent(raw_text, context=conv_context)

    # Map decision tool to legacy Intent for backward-compatibility with UI and tests
    intent = TOOL_TO_INTENT.get(decision.tool_name, Intent.UNKNOWN)

    # 0. Clarification Needed / Inventory Withdrawal Inquiry
    if decision.tool_name == ToolName.CLARIFICATION_NEEDED:
        clarif_msg = decision.clarification_question or (
            "Pengambilan barang dagangan (seperti barang stok etalase) memerlukan konfirmasi konteks: "
            "apakah pengambilan ini untuk **keperluan operasional toko**, **prive (keperluan pribadi pemilik)**, "
            "atau **barang rusak/hilang**? Aktivitas ini adalah mutasi fisik persediaan barang dagangan, bukan mutasi kas langsung."
        )
        return {
            'intent': Intent.CLARIFICATION_NEEDED,
            'tool': decision.tool_name.value,
            'type': 'message',
            'message': clarif_msg,
        }

    # 1. Transaction Creation Proposal
    if decision.tool_name == ToolName.CREATE_TRANSACTION_PROPOSAL:
        tp_intent = detect_chat_intent(raw_text, history=history, is_owner=is_owner)
        if tp_intent not in (Intent.CREATE_INCOME, Intent.CREATE_EXPENSE):
            tp_intent = Intent.CREATE_INCOME if detect_type(raw_text) == 'pemasukan' else Intent.CREATE_EXPENSE

        categories = finance.list_categories(token)
        try:
            proposal = build_transaction_proposal(
                raw_text, tp_intent, categories, toko_id=actor['toko_id']
            )
        except ValueError as exc:
            return {
                'intent': tp_intent,
                'tool': decision.tool_name.value,
                'type': 'message',
                'message': f"Perintah belum lengkap: {exc}"
            }
        except Exception as exc:
            return {
                'intent': tp_intent,
                'tool': decision.tool_name.value,
                'type': 'message',
                'message': f"Gagal memproses instruksi transaksi: {exc}"
            }

        approval_note = (
            "Transaksi akan **langsung disetujui** secara permanen."
            if is_owner else
            "Transaksi akan disimpan berstatus **Menunggu Persetujuan** untuk ditinjau oleh Pemilik Toko."
        )

        return {
            'intent': tp_intent,
            'tool': decision.tool_name.value,
            'type': 'proposal',
            'proposal': proposal,
            'approval_note': approval_note,
            'message': (
                "Saya telah menyiapkan draf transaksi berikut berdasarkan instruksi Anda. "
                "Mohon periksa rincian di bawah ini sebelum data disimpan:"
            ),
            'metadata': {'intent': tp_intent, 'metric': proposal['tipe'], 'period': proposal['tanggal']}
        }

    # 2. Unsupported Mutation
    if decision.tool_name == ToolName.UNSUPPORTED_MUTATION:
        return {
            'intent': Intent.UNSUPPORTED_MUTATION,
            'tool': decision.tool_name.value,
            'type': 'message',
            'message': (
                "Perintah pengubahan atau penghapusan transaksi tidak dapat dilakukan melalui percakapan ini "
                "demi menjaga integritas dan keamanan pembukuan toko. Silakan gunakan menu **Transaksi** secara langsung "
                "jika Anda memiliki hak akses."
            )
        }

    # 3. General Help
    if decision.tool_name == ToolName.GENERAL_HELP:
        from ai.store_context import get_store_business_context
        store_ctx = get_store_business_context(actor.get('toko_id', ''))
        store_name = store_ctx.get('nama') or store_ctx.get('nama_toko') or 'Toko Anda'

        help_text = (
            f"Halo! Saya **AI Financial Chat** (Asisten Keuangan AI), asisten dan copilot keuangan **{store_name}**. Berikut kapabilitas yang dapat saya bantu:\n\n"
            "1. **Mencatat Transaksi Bahasa Alami:**\n"
            "   - *'Catat pemasukan penjualan roti Rp150.000 hari ini.'*\n"
            "   - *'Beli tepung terigu 2 karung Rp180.000 kemarin.'*\n\n"
            "2. **Tanya Kategori & Operasional:**\n"
            "   - *'Ada kategori pemasukan dan pengeluaran apa saja di aplikasi ini?'*\n"
            "   - *'Ada persetujuan transaksi kasir hari ini yang belum dikonfirmasi?'*\n"
            "   - *'Siapa saja kasir yang terdaftar?'*\n\n"
            "3. **Tanya Data Finansial & Perbandingan (Khusus Pemilik):**\n"
            "   - *'Berapa total pemasukan bulan ini?'*\n"
            "   - *'Berapa saldo kas toko saat ini?'*\n"
            "   - *'Bandingkan pemasukan bulan ini dengan bulan lalu.'*\n"
            "   - *'Pengeluaran terbesar minggu ini apa?'*\n\n"
            "4. **Panduan Fitur & Peramalan:**\n"
            "   - *'Bagaimana cara menambah kategori baru?'*\n"
            "   - *'Bagaimana cara kerja forecasting Prophet?'*\n\n"
            "Setiap transaksi baru memerlukan konfirmasi Anda sebelum disimpan ke pembukuan."
        )
        return {
            'intent': Intent.GENERAL_HELP,
            'tool': decision.tool_name.value,
            'type': 'message',
            'message': help_text
        }

    # 4. Category Listings
    if decision.tool_name in (ToolName.LIST_ALL_CATEGORIES, ToolName.LIST_INCOME_CATEGORIES, ToolName.LIST_EXPENSE_CATEGORIES):
        res = _handle_category_listing(token, intent, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 5. Pending Cashier Approvals
    if decision.tool_name in (ToolName.GET_PENDING_APPROVALS, ToolName.COUNT_PENDING_APPROVALS):
        res = _handle_pending_approvals(token, raw_text, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 6. Cashier Accounts and Activity
    if decision.tool_name in (ToolName.COUNT_CASHIERS, ToolName.LIST_CASHIERS, ToolName.GET_CASHIER_ACTIVITY, ToolName.GET_CASHIER_TRANSACTION_SUMMARY):
        res = _handle_cashier_activity(token, raw_text, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 7. Cashier Transactions (both approved and pending)
    if decision.tool_name == ToolName.GET_CASHIER_TRANSACTIONS:
        res = _handle_cashier_transactions(token, raw_text, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 8. Transaction Details & History
    if decision.tool_name in (ToolName.GET_TRANSACTION_DETAILS, ToolName.LIST_TRANSACTIONS):
        res = _handle_transaction_details(token, raw_text, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 9. Profit & Operating Cash Limitations
    if decision.tool_name == ToolName.GET_PROFIT_ANALYSIS:
        res = _handle_profit_query(token, raw_text, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 10. Financial Period Comparison
    if decision.tool_name == ToolName.COMPARE_FINANCIAL_PERIODS:
        res = _handle_financial_comparison(token, raw_text, history, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 11. Financial Insights (Ranked Categories, Top Expense)
    if decision.tool_name in (ToolName.RANK_EXPENSE_CATEGORIES, ToolName.RANK_INCOME_CATEGORIES, ToolName.GET_LARGEST_EXPENSES):
        res = _handle_financial_insights(token, raw_text, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 12. Store Profile
    if decision.tool_name in (ToolName.GET_STORE_PROFILE, ToolName.GET_STORE_BUSINESS_TYPE, ToolName.GET_STORE_OPERATIONAL_STATUS):
        res = _handle_store_profile(token, actor)
        res['tool'] = decision.tool_name.value
        return res

    # 13. Forecast (Multi-Model)
    forecast_tools = {
        ToolName.GET_AVAILABLE_FORECAST,
        ToolName.GET_EXPENSE_FORECAST,
        ToolName.GET_FORECAST_FOR_PERIOD,
        ToolName.GET_UPCOMING_RECURRING_EXPENSES,
        ToolName.GET_NEXT_PREDICTED_EXPENSE,
        ToolName.COMPARE_ACTUAL_VS_FORECAST,
        ToolName.GET_FORECAST_ACCURACY,
        ToolName.GET_EXPENSE_TREND,
        ToolName.GET_FORECAST_READINESS,
        ToolName.CHECK_FORECAST_AGAINST_CASH_BALANCE,
    }
    if decision.tool_name in forecast_tools:
        if not is_owner:
            return {
                'intent': Intent.QUERY_FORECAST,
                'tool': decision.tool_name.value,
                'type': 'message',
                'message': "Fitur peramalan pengeluaran (forecasting multi-model) hanya dapat diakses oleh **Pemilik Toko** demi menjaga kerahasiaan proyeksi bisnis."
            }
        if decision.tool_name == ToolName.GET_AVAILABLE_FORECAST:
            return {
                'intent': Intent.QUERY_FORECAST,
                'tool': decision.tool_name.value,
                'type': 'message',
                'message': get_forecast_explanation(is_owner=is_owner)
            }
        tool_res = dispatch_tool(token, decision)
        out_mode = decision.filter_params.output_mode
        out_mode_val = out_mode.value if hasattr(out_mode, 'value') else str(out_mode or 'summary')
        meta = {
            'intent': decision.tool_name.value,
            'tool': decision.tool_name.value,
            'period': decision.date_params.period_label,
            'start_date': decision.date_params.start_date,
            'end_date': decision.date_params.end_date,
            'days_ahead': decision.filter_params.days_ahead,
            'category': decision.filter_params.category_name,
            'output_mode': out_mode_val,
        }
        return {
            'intent': Intent.QUERY_FORECAST,
            'tool': decision.tool_name.value,
            'type': 'message',
            'message': tool_res.get('message', ''),
            'data': tool_res.get('data', {}),
            'metadata': meta
        }

    # 14. App Feature Explanations
    if decision.tool_name in (ToolName.EXPLAIN_FEATURE, ToolName.EXPLAIN_NAVIGATION, ToolName.EXPLAIN_OWNER_APPROVAL_WORKFLOW, ToolName.EXPLAIN_CSV_IMPORT, ToolName.EXPLAIN_AI_CLASSIFICATION):
        topic = decision.feature_topic or detect_feature_topic(raw_text)
        return {
            'intent': Intent.EXPLAIN_APP_FEATURE,
            'tool': decision.tool_name.value,
            'type': 'message',
            'message': get_feature_explanation(topic, is_owner=is_owner)
        }

    # 14b. Isolated Business Guidance RAG (Revisi 2)
    if decision.tool_name == ToolName.GET_BUSINESS_GUIDANCE:
        tool_res = dispatch_tool(token, decision)
        res_data = tool_res.get('data') or {}
        return {
            'intent': Intent.QUERY_BUSINESS_GUIDANCE,
            'tool': decision.tool_name.value,
            'type': 'message',
            'message': tool_res.get('message', ''),
            'data': res_data,
            'citations': res_data.get('citations', []),
            'metadata': {
                'intent': decision.tool_name.value,
                'tool': decision.tool_name.value,
                'jenis_usaha': res_data.get('jenis_usaha'),
                'found': bool(res_data.get('found')),
            },
        }

    # 15. Financial Summaries (Income, Expense, Cash Flow)
    if decision.tool_name in (ToolName.GET_INCOME_SUMMARY, ToolName.GET_EXPENSE_SUMMARY, ToolName.GET_CASH_FLOW_SUMMARY):
        if not is_owner:
            return {
                'intent': intent,
                'tool': decision.tool_name.value,
                'type': 'message',
                'message': (
                    "Informasi rekapitulasi keuangan dan saldo toko hanya dapat diakses oleh Pemilik Toko. "
                    "Sebagai kasir, Anda dapat mencatat transaksi harian atau melihat daftar transaksi yang Anda catat sendiri."
                )
            }

        start, end, period_label = resolve_date_range_indonesian(raw_text)
        metric = detect_query_metric(raw_text)
        all_cats = finance.list_categories(token)
        rows = period_rows(token, start=start, end=end)
        data_metrics = metrics(rows, all_cats)

        fact_text = ""
        if decision.tool_name == ToolName.GET_CASH_FLOW_SUMMARY or metric in ('saldo', 'arus_kas_total'):
            all_approved = period_rows(token, None, None)
            all_metrics = metrics(all_approved, all_cats)
            saldo = all_metrics['arus_kas_total']
            pemasukan = all_metrics['pemasukan']
            pengeluaran = all_metrics['pengeluaran']
            fact_text = (
                f"Saldo kas toko saat ini adalah **{format_rupiah(saldo)}**.\n\n"
                f"- Total Pemasukan: **{format_rupiah(pemasukan)}**\n"
                f"- Total Pengeluaran: **{format_rupiah(pengeluaran)}**\n"
                f"- Arus Kas Bersih: **{format_rupiah(saldo)}**\n"
                f"- Jumlah Transaksi Disetujui: **{all_metrics['jumlah_disetujui']} transaksi**"
            )

        elif decision.tool_name == ToolName.GET_INCOME_SUMMARY or metric == 'pemasukan':
            pemasukan = data_metrics['pemasukan']
            count = len([r for r in rows if r['tipe'] == 'pemasukan'])
            fact_text = (
                f"Total pemasukan untuk periode **{period_label}** adalah **{format_rupiah(pemasukan)}** "
                f"dari total **{count} transaksi pemasukan** yang telah disetujui."
            )

        elif decision.tool_name == ToolName.GET_EXPENSE_SUMMARY or metric == 'pengeluaran':
            pengeluaran = data_metrics['pengeluaran']
            count = len([r for r in rows if r['tipe'] == 'pengeluaran'])
            fact_text = (
                f"Total pengeluaran untuk periode **{period_label}** adalah **{format_rupiah(pengeluaran)}** "
                f"dari total **{count} transaksi pengeluaran** yang telah disetujui."
            )

        else:
            fact_text = (
                f"Ringkasan keuangan toko periode **{period_label}**:\n\n"
                f"- Pemasukan: **{format_rupiah(data_metrics['pemasukan'])}**\n"
                f"- Pengeluaran: **{format_rupiah(data_metrics['pengeluaran'])}**\n"
                f"- Arus Kas Operasional: **{format_rupiah(data_metrics['arus_kas_operasional'])}**\n"
                f"- Arus Kas Bersih Total: **{format_rupiah(data_metrics['arus_kas_total'])}**\n"
                f"- Jumlah Transaksi Disetujui: **{data_metrics['jumlah_disetujui']} transaksi**"
            )

        # Enhance presentation via Ollama if available without mutating verified facts
        final_message = fact_text
        try:
            system_prompt = (
                "Kamu adalah asisten keuangan UMKM profesional. Sajikan ringkasan data keuangan berikut "
                "dalam bahasa Indonesia yang ramah, ringkas, dan jelas. "
                "JANGAN mengubah angka nominal sama sekali dan jangan menambahkan angka baru."
            )
            llm_text = ask_text(system_prompt, fact_text)
            if llm_text and len(llm_text) <= 600 and _is_numeric_consistent(fact_text, llm_text):
                final_message = llm_text
        except Exception as exc:
            logger.warning("Chat summary generation failed; returning deterministic financial summary: %s", exc)
            final_message = fact_text

        return {
            'intent': intent,
            'tool': decision.tool_name.value,
            'type': 'message',
            'message': final_message,
            'metadata': {'intent': intent, 'metric': metric, 'period': period_label}
        }

    # 16. Fallback / Unknown
    return {
        'intent': Intent.UNKNOWN,
        'tool': ToolName.UNKNOWN.value,
        'type': 'message',
        'message': decision.clarification_question or (
            "Maaf, saya belum memahami instruksi tersebut. Anda dapat memerintahkan pencatatan transaksi "
            "(misal: *'Catat beli telur Rp50.000'*), menanyakan kategori (*'Ada kategori apa saja?'*), "
            "persetujuan kasir (*'Ada transaksi kasir hari ini?'*), atau ringkasan keuangan toko."
        )
    }


_CONFIRMING_PROPOSALS = set()
_CONFIRMED_PROPOSALS = {}


def confirm_proposal(token, proposal_dict):
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})

    # Prevent cross-store execution
    if proposal_dict.get('toko_id') and proposal_dict['toko_id'] != actor['toko_id']:
        raise AccessError('Transaksi ini dibuat untuk toko lain dan tidak dapat dikonfirmasi.')

    prop_id = str(proposal_dict.get('id') or '')
    if not prop_id:
        return {'success': False, 'message': 'ID proposal wajib tersedia untuk konfirmasi.'}
    cache_key = (str(actor['toko_id']), prop_id)

    # Check session/in-memory confirmed cache
    if proposal_dict.get('status') == 'confirmed':
        res = dict(proposal_dict.get('result') or {})
        res['message'] = 'Transaksi ini sudah dikonfirmasi sebelumnya.'
        res['already_confirmed'] = True
        return res

    if prop_id and cache_key in _CONFIRMED_PROPOSALS:
        proposal_dict['status'] = 'confirmed'
        res = dict(_CONFIRMED_PROPOSALS[cache_key])
        res['message'] = 'Transaksi ini sudah dikonfirmasi sebelumnya.'
        res['already_confirmed'] = True
        proposal_dict['result'] = res
        return res

    if proposal_dict.get('status') == 'cancelled':
        return {'success': False, 'message': 'Transaksi ini sudah dibatalkan sebelumnya.'}

    if prop_id and cache_key in _CONFIRMING_PROPOSALS:
        return {'success': False, 'message': 'Transaksi sedang diproses, harap tunggu.'}

    if prop_id:
        _CONFIRMING_PROPOSALS.add(cache_key)

    try:
        tanggal = proposal_dict.get('tanggal')
        deskripsi = proposal_dict.get('deskripsi')
        nominal = proposal_dict.get('nominal')
        tipe = proposal_dict.get('tipe')
        kategori_id = proposal_dict.get('kategori_id')
        status_kategori = proposal_dict.get('status_kategori', 'dikonfirmasi')
        skor = proposal_dict.get('skor')
        usulan = dict(proposal_dict.get('usulan') or {})
        if prop_id:
            usulan['proposal_id'] = prop_id

        # Strict structural validation before database mutation
        if tipe not in ('pemasukan', 'pengeluaran'):
            return {'success': False, 'message': 'Jenis transaksi tidak valid.'}

        try:
            nom = Decimal(str(nominal))
            if nom <= 0:
                return {'success': False, 'message': 'Nominal transaksi harus lebih besar dari 0.'}
        except Exception:
            return {'success': False, 'message': 'Nominal transaksi tidak valid.'}

        if not str(deskripsi or '').strip():
            return {'success': False, 'message': 'Keterangan transaksi tidak boleh kosong.'}

        already_persisted = False
        try:
            created = finance.add_transaction(
                token,
                tanggal=tanggal,
                deskripsi=deskripsi,
                nominal=nominal,
                tipe=tipe,
                kategori_id=kategori_id,
                status_kategori=status_kategori,
                skor=skor,
                usulan=usulan
            )
        except Exception as exc:
            # The unique proposal index arbitrates concurrent workers. Only a
            # uniqueness violation can be resolved by loading a persisted row.
            if str(getattr(exc, 'code', '')) != '23505':
                logger.error("Gagal menambahkan transaksi untuk proposal %s: %s", prop_id, exc)
                raise
            try:
                existing_tx = (privileged_client().table('transaksi')
                               .select('*').eq('toko_id', actor['toko_id'])
                               .contains('usulan', {'proposal_id': prop_id})
                               .limit(1).execute().data)
            except Exception as db_err:
                logger.error("Gagal memeriksa idempotensi proposal %s di database: %s", prop_id, db_err)
                raise RuntimeError(f"Pemeriksaan idempotensi proposal ke database gagal: {db_err}") from db_err

            if not existing_tx:
                logger.error("Konflik proposal %s tidak dapat diverifikasi dari basis data: %s", prop_id, exc)
                raise RuntimeError('Konflik penyimpanan proposal tidak dapat diverifikasi') from exc
            created = existing_tx
            already_persisted = True

        record = created[0] if isinstance(created, list) and created else created
        if not isinstance(record, dict) or not record.get('id'):
            raise RuntimeError('Database tidak mengonfirmasi penyimpanan transaksi')
        status_persetujuan = record.get('status_persetujuan', 'disetujui')

        res_msg = (
            f"Transaksi **{tipe.capitalize()}** senilai **{format_rupiah(nominal)}** berhasil dicatat!\n"
            f"- Status: **{'Disetujui' if status_persetujuan == 'disetujui' else 'Menunggu Persetujuan Pemilik'}**\n"
            f"- Tanggal: {tanggal}\n"
            f"- Kategori: {proposal_dict.get('kategori_nama', 'Others')}"
        )

        result_payload = {
            'success': True,
            'record': record,
            'already_persisted': already_persisted,
            'status_persetujuan': status_persetujuan,
            'message': res_msg
        }
        proposal_dict['status'] = 'confirmed'
        proposal_dict['result'] = result_payload
        if prop_id:
            _CONFIRMED_PROPOSALS[cache_key] = result_payload
        return result_payload

    except Exception as exc:
        err_payload = {
            'success': False,
            'message': f"Gagal menyimpan transaksi ke database: {exc}"
        }
        proposal_dict['result'] = err_payload
        return err_payload
    finally:
        if prop_id:
            _CONFIRMING_PROPOSALS.discard(cache_key)


def cancel_proposal(proposal_dict):
    proposal_dict['status'] = 'cancelled'
    result_payload = {
        'success': False,
        'message': 'Pencatatan transaksi dibatalkan oleh pengguna. Tidak ada data yang disimpan ke pembukuan.'
    }
    proposal_dict['result'] = result_payload
    return result_payload

