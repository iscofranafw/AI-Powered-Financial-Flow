"""Hybrid Intent Planner for the AI Financial Copilot.

Stage A: Deterministic resolution for explicit, well-defined patterns, dates, and commands.
Stage B: Ollama LLM structured interpretation with Pydantic validation for flexible natural-language queries.
"""

import re
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple

from ai.copilot_schema import (
    ToolName,
    PlannerDecision,
    DateRangeParams,
    FilterParams,
    ConversationContext,
)
from ai.financial_chat import (
    resolve_date_range_indonesian,
    resolve_date_indonesian,
    clean_transaction_description,
    has_date_specifier,
    get_jakarta_now,
    MONTH_MAP,
)
from ai.text_normalizer import extract_money
from ai.copilot_knowledge import detect_feature_topic
from ai.ollama_client import ask_json, LLMUnavailable


def _extract_date_params(text: str) -> DateRangeParams:
    start, end, label = resolve_date_range_indonesian(text)
    return DateRangeParams(
        start_date=start.isoformat() if start else None,
        end_date=end.isoformat() if end else None,
        period_label=label or "Semua Periode"
    )


def plan_stage_a_deterministic(text: str, context: Optional[ConversationContext] = None) -> Optional[PlannerDecision]:
    s = text.lower().strip()

    # 0. Prompt injection, jailbreak, and privilege escalation attempts
    if any(w in s for w in ['superadmin', 'jailbreak', 'ignore previous', 'system prompt', 'you are now', 'bypass']):
        return PlannerDecision(
            tool_name=ToolName.UNKNOWN,
            confidence=0.0,
            raw_query=text,
            clarification_question="Instruksi tidak dapat diproses demi keamanan sistem."
        )

    # 1. Reject unsupported mutation commands
    if re.search(r'\b(hapus|delete|edit|ubah|ganti)\s+(?:transaksi|data|catatan)\b', s) or re.search(r'\b(batalkan|hapus)\s+(?:transaksi|catatan)\b', s):
        return PlannerDecision(tool_name=ToolName.UNSUPPORTED_MUTATION, raw_query=text)
    if re.search(r'\b(drop\s+table|delete\s+from|update\s+set|truncate\s+table)\b', s):
        return PlannerDecision(tool_name=ToolName.UNSUPPORTED_MUTATION, raw_query=text)

    # 2. General help and greetings
    if any(w in s for w in ['halo', 'hai', 'bantuan', 'help', 'menu apa', 'bisa apa', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam', 'panduan penggunaan']):
        return PlannerDecision(tool_name=ToolName.GENERAL_HELP, raw_query=text)

    # 3. Forecast / Proyeksi (Prophet)
    if any(w in s for w in ['forecast', 'forecasting', 'peramalan', 'prediksi', 'proyeksi', 'prophet']):
        return PlannerDecision(tool_name=ToolName.GET_AVAILABLE_FORECAST, raw_query=text)

    # 4. App Feature Guides & How-To
    is_feature_q = bool(re.search(
        r'\b(bagaimana\s+cara|cara\s+(?:menambah|membuat|mengubah|menghapus|menyetujui|approve|impor|import|kerja|pendaftaran)|fitur\s+apa|bisa\s+apa\s+saja\s+aplikasi|panduan\s+fitur|cara\s+kerja)\b',
        s
    ))
    is_fin_state_q = bool(re.search(r'bagaimana\s+(?:arus\s+kas|saldo|pemasukan|pengeluaran|kondisi\s+keuangan|keuntungan|laba)', s))
    if is_feature_q and not is_fin_state_q:
        topic = detect_feature_topic(s)
        if topic == 'persetujuan':
            return PlannerDecision(tool_name=ToolName.EXPLAIN_OWNER_APPROVAL_WORKFLOW, feature_topic=topic, raw_query=text)
        elif topic == 'impor':
            return PlannerDecision(tool_name=ToolName.EXPLAIN_CSV_IMPORT, feature_topic=topic, raw_query=text)
        elif topic == 'klasifikasi_ai':
            return PlannerDecision(tool_name=ToolName.EXPLAIN_AI_CLASSIFICATION, feature_topic=topic, raw_query=text)
        return PlannerDecision(tool_name=ToolName.EXPLAIN_FEATURE, feature_topic=topic, raw_query=text)

    # 5. Profit / Keuntungan Analysis (Distinguishing Net Cash Flow vs Accounting Profit)
    if any(k in s for k in ['keuntungan bersih', 'laba bersih', 'profit bersih', 'keuntungan toko', 'laba toko', 'untung bersih', 'keuntungan saya', 'laba saya', 'profit saya']):
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_PROFIT_ANALYSIS, date_params=date_p, raw_query=text)

    # 6. Cashier Count Queries (CASE 2 & 3: Ensure "total kasir" or "berapa kasir" does NOT hit financial summary!)
    is_cashier_count = bool(
        re.search(r'\b(?:berapa|brp|jumlah|total)\s+(?:semua\s+)?kasir\b', s) or
        re.search(r'\b(?:ada\s+berapa|ada\s+brp|berapa|brp)\s+(?:banyak\s+)?kasir\b', s) or
        re.search(r'\b(?:berapa|brp)\s+kasir\b', s) or
        re.search(r'\bkasir\s+(?:ada\s+)?(?:berapa|brp)\b', s)
    )
    if is_cashier_count:
        return PlannerDecision(tool_name=ToolName.COUNT_CASHIERS, raw_query=text)

    # 7. Cashier Listing Queries: "siapa saja kasir", "kasir siapa saja yang aktif", "daftar kasir"
    is_cashier_list = bool(
        re.search(r'\b(?:kasir\s+siapa|siapa\s+(?:saja\s+)?kasir|daftar\s+kasir|akun\s+kasir|kasir\s+terdaftar)\b', s) or
        re.search(r'\bkasir\s+siapa\s+saja\b', s) or
        ('kasir' in s and 'siapa' in s) or
        ('kasir' in s and 'aktif' in s and not any(k in s for k in ['paling', 'teraktif', 'transaksi', 'kinerja', 'minggu', 'bulan']))
    )
    if is_cashier_list:
        return PlannerDecision(tool_name=ToolName.LIST_CASHIERS, raw_query=text)

    # 8. Cashier Activity & Most Active Cashier
    is_cashier_activity = bool(
        re.search(r'\bkasir\b.*\b(?:paling\s+aktif|paling\s+rajin|terbanyak|teraktif|kinerja|performa|aktivitas)\b', s) or
        re.search(r'\bkasir\s+mana\b', s) or
        re.search(r'\baktivitas\s+kasir\b', s)
    )
    if is_cashier_activity:
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_CASHIER_ACTIVITY, date_params=date_p, raw_query=text)

    # 9. Cashier Transactions vs Pending Approvals (CASE 5 & 6)
    has_approval_word = bool(
        any(w in s for w in ['persetujuan', 'approval', 'menunggu persetujuan', 'belum disetujui', 'belum dikonfirmasi', 'antrean kasir', 'transaksi pending']) or
        ('kasir' in s and any(k in s for k in ['belum', 'pending', 'menunggu']))
    )
    if has_approval_word:
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_PENDING_APPROVALS, date_params=date_p, raw_query=text)

    # Case 5: "ada transaksi kasir hari ini?" (all cashier transactions today, not pending-only)
    is_cashier_tx_inquiry = bool(
        re.search(r'\btransaksi\s+kasir\b', s) or
        ('kasir' in s and 'transaksi' in s and any(w in s for w in ['ada', 'apa saja', 'daftar', 'lihat', 'hari ini', 'kemarin', 'minggu']))
    )
    if is_cashier_tx_inquiry:
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_CASHIER_TRANSACTIONS, date_params=date_p, raw_query=text)

    # 10. Category Ranking (CASE 1: "kategori pengeluaran apa saja yang paling banyak menurut data toko saya?")
    is_cat_ranking = bool(
        any(k in s for k in ['kategori', 'pos']) and
        any(k in s for k in ['paling banyak', 'terbanyak', 'terbesar', 'paling besar', 'boros', 'menghabiskan', 'peringkat', 'ranking'])
    )
    if is_cat_ranking:
        date_p = _extract_date_params(s)
        if any(w in s for w in ['pemasukan', 'pendapatan', 'omzet']) and not any(w in s for w in ['pengeluaran', 'biaya', 'belanja']):
            return PlannerDecision(tool_name=ToolName.RANK_INCOME_CATEGORIES, date_params=date_p, raw_query=text)
        return PlannerDecision(tool_name=ToolName.RANK_EXPENSE_CATEGORIES, date_params=date_p, raw_query=text)

    # 11. Category Taxonomies (CASE 7: "ada kategori pemasukan dan pengeluaran apa saja?")
    has_cat_word = any(w in s for w in ['kategori', 'pos belanja', 'pos anggaran', 'pos pemasukan', 'pos pengeluaran', 'pos biaya', 'pos pendapatan'])
    has_cat_listing = any(w in s for w in ['apa saja', 'daftar', 'tampilkan', 'lihat', 'ada apa', 'ada kategori', 'kategori apa', 'pos apa', 'pos anggaran'])
    if has_cat_word and (has_cat_listing or 'ada' in s or 'apa' in s):
        has_inc = any(w in s for w in ['pemasukan', 'pendapatan', 'omzet'])
        has_exp = any(w in s for w in ['pengeluaran', 'biaya', 'belanja'])
        if has_inc and not has_exp:
            return PlannerDecision(tool_name=ToolName.LIST_INCOME_CATEGORIES, raw_query=text)
        if has_exp and not has_inc:
            return PlannerDecision(tool_name=ToolName.LIST_EXPENSE_CATEGORIES, raw_query=text)
        return PlannerDecision(tool_name=ToolName.LIST_ALL_CATEGORIES, raw_query=text)

    # 12. Period Comparisons (including conversational follow-ups)
    has_comp_word = any(w in s for w in ['bandingkan', 'perbandingan', 'dibandingkan', 'komparasi'])
    if has_comp_word:
        filter_p = FilterParams()
        if any(w in s for w in ['pemasukan', 'omzet', 'pendapatan']):
            filter_p.transaction_type = 'pemasukan'
        elif any(w in s for w in ['pengeluaran', 'biaya', 'belanja']):
            filter_p.transaction_type = 'pengeluaran'
        elif context and context.last_metric in ('pemasukan', 'pengeluaran'):
            filter_p.transaction_type = context.last_metric
        return PlannerDecision(tool_name=ToolName.COMPARE_FINANCIAL_PERIODS, filter_params=filter_p, raw_query=text)

    # Conversational follow-up: e.g. "Bandingkan dengan bulan lalu" or "Bagaimana dengan bulan lalu?"
    if context and (s.startswith('bagaimana dengan') or 'bandingkan' in s):
        if any(p in s for p in ['bulan lalu', 'bulan kemarin', 'minggu lalu', 'minggu kemarin', 'kemarin']):
            filter_p = FilterParams(transaction_type=context.last_transaction_type or context.last_metric or 'all')
            return PlannerDecision(
                tool_name=ToolName.COMPARE_FINANCIAL_PERIODS,
                filter_params=filter_p,
                follow_up_resolved=True,
                raw_query=text
            )

    # 13. Transaction Details & History (CASE 4: "bisa beri detail pengeluaran bulan kemarin?")
    is_tx_detail_req = bool(
        re.search(r'\b(?:detail|rincian|daftar|tampilkan|lihat|riwayat)\s+(?:semua\s+)?(?:transaksi|pengeluaran|pemasukan)\b', s) or
        re.search(r'\btransaksi\s+tanggal\b', s) or
        re.search(r'\briwayat\s+transaksi\b', s) or
        ('detail' in s and any(w in s for w in ['pengeluaran', 'pemasukan', 'transaksi']))
    )
    if is_tx_detail_req:
        date_p = _extract_date_params(s)
        filter_p = FilterParams()
        if any(w in s for w in ['pemasukan', 'pendapatan', 'omzet']) and not any(w in s for w in ['pengeluaran', 'biaya']):
            filter_p.transaction_type = 'pemasukan'
        elif any(w in s for w in ['pengeluaran', 'biaya', 'belanja']) and not any(w in s for w in ['pemasukan', 'pendapatan']):
            filter_p.transaction_type = 'pengeluaran'
        return PlannerDecision(tool_name=ToolName.GET_TRANSACTION_DETAILS, date_params=date_p, filter_params=filter_p, raw_query=text)

    # 14. Transaction Creation (Commands or money with action)
    is_create_cmd = bool(re.search(r'^(?:tolong\s+|mohon\s+|bisa\s+)?(?:catat(?:lah)?|tambah(?:kan)?|masukkan|input)\b', s))
    has_money = bool(re.search(r'(?<!\w)(?:rp\.?\s*)?\d[\d.,]*\s*(?:juta|jt|ribu|rb|k|rupiah)?\b', s))
    is_explicit_query = bool(re.search(r'\b(berapa|apa|tampilkan|lihat|cek|daftar|siapa|bagaimana|rincian|detail)\b', s))

    if (is_create_cmd or has_money) and not is_explicit_query:
        return PlannerDecision(tool_name=ToolName.CREATE_TRANSACTION_PROPOSAL, raw_query=text)

    # 15. Store Profile (strictly matching store profile / business identity queries)
    if any(w in s for w in ['profil toko', 'bidang usaha', 'nama toko saya', 'tentang toko ini', 'identitas toko']):
        return PlannerDecision(tool_name=ToolName.GET_STORE_PROFILE, raw_query=text)

    # 16. Largest Expense
    if any(k in s for k in ['pengeluaran terbesar', 'biaya terbesar', 'transaksi terbesar']):
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_LARGEST_EXPENSES, date_params=date_p, raw_query=text)

    # 17. Fine-grained Financial Summaries
    has_sum_q = any(w in s for w in ['berapa', 'total', 'rekap', 'jumlah', 'bagaimana'])
    has_inc_w = any(w in s for w in ['pemasukan', 'pendapatan', 'omzet'])
    has_exp_w = any(w in s for w in ['pengeluaran', 'biaya', 'belanja'])

    if has_inc_w and has_sum_q and not has_exp_w:
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_INCOME_SUMMARY, date_params=date_p, raw_query=text)

    if has_exp_w and has_sum_q and not has_inc_w:
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_EXPENSE_SUMMARY, date_params=date_p, raw_query=text)

    if any(k in s for k in ['saldo', 'sisa kas', 'uang kas', 'arus kas', 'cash flow', 'cashflow']):
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_CASH_FLOW_SUMMARY, date_params=date_p, raw_query=text)

    # General Financial Summary
    fin_terms = ['pemasukan', 'pengeluaran', 'omzet', 'transaksi', 'biaya', 'rekap', 'ringkasan']
    if any(term in s for term in fin_terms) and any(w in s for w in ['berapa', 'total', 'rekap', 'ringkasan', 'kondisi']):
        date_p = _extract_date_params(s)
        return PlannerDecision(tool_name=ToolName.GET_CASH_FLOW_SUMMARY, date_params=date_p, raw_query=text)

    return None


def plan_stage_b_llm(text: str, context: Optional[ConversationContext] = None) -> Optional[PlannerDecision]:
    """Stage B: Ollama LLM structured interpretation with Pydantic validation."""
    allowlisted_tools = [t.value for t in ToolName]

    system_prompt = (
        "Kamu adalah planner AI untuk asisten keuangan UMKM. "
        "Tugasmu adalah menganalisis pertanyaan pengguna dan memetakannya ke salah satu nama tool yang diizinkan.\n\n"
        "Daftar tool yang diizinkan:\n"
        "- rank_expense_categories: peringkat pos pengeluaran terbanyak/terbesar\n"
        "- count_cashiers: menanyakan jumlah/total kasir terdaftar\n"
        "- list_cashiers: daftar nama kasir toko\n"
        "- get_cashier_activity: aktivitas pencatatan kasir atau kasir paling aktif\n"
        "- get_cashier_transactions: transaksi yang dicatat oleh kasir\n"
        "- get_pending_approvals: transaksi kasir yang belum disetujui/pending/belum dikonfirmasi\n"
        "- list_all_categories: daftar semua kategori pembukuan\n"
        "- list_income_categories: daftar kategori pemasukan\n"
        "- list_expense_categories: daftar kategori pengeluaran\n"
        "- get_transaction_details: detail/rincian riwayat transaksi\n"
        "- get_income_summary: total pemasukan toko\n"
        "- get_expense_summary: total pengeluaran toko\n"
        "- get_cash_flow_summary: saldo kas atau arus kas toko\n"
        "- get_profit_analysis: keuntungan bersih, laba bersih, atau laba toko\n"
        "- compare_financial_periods: membandingkan periode keuangan (misal bulan ini vs bulan lalu)\n"
        "- get_store_profile: informasi profil toko, nama toko, atau bidang usaha\n"
        "- explain_feature: panduan cara menggunakan fitur aplikasi\n"
        "- get_available_forecast: peramalan atau prediksi Prophet\n"
        "- unknown: jika instruksi tidak jelas\n\n"
        "Kembalikan JSON dengan format:\n"
        "{\"tool_name\": string, \"confidence\": float (0.0-1.0), \"period_label\": string}"
    )

    schema = {
        'type': 'object',
        'properties': {
            'tool_name': {'type': 'string', 'enum': allowlisted_tools},
            'confidence': {'type': 'number'},
            'period_label': {'type': 'string'},
            'transaction_type': {'type': 'string', 'enum': ['pemasukan', 'pengeluaran', 'all']}
        },
        'required': ['tool_name', 'confidence']
    }

    try:
        raw_json = ask_json(system_prompt, text, schema=schema)
        tool_str = raw_json.get('tool_name')
        conf = float(raw_json.get('confidence', 0.8))

        if tool_str in allowlisted_tools and conf >= 0.6:
            date_p = _extract_date_params(text)
            if raw_json.get('period_label') and not date_p.start_date:
                date_p.period_label = raw_json['period_label']

            filter_p = FilterParams(
                transaction_type=raw_json.get('transaction_type', 'all')
            )

            return PlannerDecision(
                tool_name=ToolName(tool_str),
                confidence=conf,
                date_params=date_p,
                filter_params=filter_p,
                raw_query=text
            )
    except (LLMUnavailable, Exception):
        pass

    return None


def plan_intent(text: str, context: Optional[ConversationContext] = None) -> PlannerDecision:
    """Hybrid intent planner: Stage A (deterministic) followed by Stage B (LLM-constrained)."""
    # Stage A: Deterministic resolution
    decision = plan_stage_a_deterministic(text, context=context)
    if decision:
        return decision

    # Stage B: LLM structured interpretation
    decision_llm = plan_stage_b_llm(text, context=context)
    if decision_llm and decision_llm.tool_name != ToolName.UNKNOWN:
        return decision_llm

    # Fallback when both stages are uncertain
    return PlannerDecision(
        tool_name=ToolName.UNKNOWN,
        confidence=0.0,
        raw_query=text,
        clarification_question=(
            "Maaf, saya belum memahami instruksi tersebut. Anda dapat memerintahkan pencatatan transaksi "
            "(misal: *'Catat beli telur Rp50.000'*), menanyakan jumlah kasir (*'Berapa total kasir?'*), "
            "peringkat pengeluaran (*'Kategori pengeluaran apa saja yang paling banyak?'*), "
            "atau transaksi kasir (*'Ada transaksi kasir hari ini?'*)."
        )
    )
