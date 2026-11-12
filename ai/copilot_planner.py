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
    ForecastOutputMode,
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

    # Explicit guide starters must not be mistaken for a financial cash-flow
    # query just because a real guide heading contains "Cash Flow Harian".
    if s.startswith('menurut panduan usaha ') and _is_business_guidance_query(s):
        return PlannerDecision(tool_name=ToolName.GET_BUSINESS_GUIDANCE, raw_query=text)

    # 2. General help and greetings
    if any(w in s for w in ['halo', 'hai', 'bantuan', 'help', 'menu apa', 'bisa apa', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam', 'panduan penggunaan']):
        return PlannerDecision(tool_name=ToolName.GENERAL_HELP, raw_query=text)

    # 3. Forecast / Proyeksi (Prophet) Capabilities
    is_forecast_word = any(w in s for w in [
        'forecast', 'forecasting', 'peramalan', 'ramal', 'ramalan', 'prediksi', 'proyeksi',
        'prophet', 'perkiraan', 'kiraan', 'diperkirakan', 'estimasi'
    ])

    # 3.1 Accuracy and evaluation metrics (MAPE, MAE, RMSE)
    if any(w in s for w in ['mape', 'mae', 'rmse', 'wape', 'seberapa akurat', 'akurasi prediksi', 'akurasi peramalan', 'akurasi prophet', 'keandalan model', 'evaluasi model', 'backtest']):
        return PlannerDecision(tool_name=ToolName.GET_FORECAST_ACCURACY, raw_query=text)

    # 3.2 Forecast readiness / data sufficiency
    if (is_forecast_word or 'data toko' in s) and any(w in s for w in ['cukup untuk', 'apakah bisa', 'syarat', 'kesiapan', 'readiness', 'mencukupi untuk']):
        return PlannerDecision(tool_name=ToolName.GET_FORECAST_READINESS, raw_query=text)

    # 3.3 Cash balance sufficiency against forecast
    if any(w in s for w in ['saldo', 'uang kas', 'kas saat ini', 'saldo kas']) and any(w in s for w in ['cukup', 'mencukupi', 'aman']) and any(w in s for w in ['prediksi', 'pengeluaran', 'beban', 'ke depan']):
        days = 30
        if '7 hari' in s or 'tujuh hari' in s or 'seminggu' in s:
            days = 7
        return PlannerDecision(
            tool_name=ToolName.CHECK_FORECAST_AGAINST_CASH_BALANCE,
            filter_params=FilterParams(days_ahead=days),
            raw_query=text
        )

    # 3.4 Actual vs Forecast Comparison (direct or contextual follow-up)
    has_comp_word = ('bandingkan' in s or 'perbandingan' in s or 'komparasi' in s)
    has_forecast_comp = has_comp_word and (is_forecast_word or ('bulan depan' in s and ('bulan ini' in s or 'bulan lalu' in s)))
    is_follow_up_comp = (
        context is not None and
        context.last_tool in [
            ToolName.GET_EXPENSE_FORECAST.value,
            ToolName.GET_FORECAST_FOR_PERIOD.value,
            ToolName.COMPARE_ACTUAL_VS_FORECAST.value,
            ToolName.GET_NEXT_PREDICTED_EXPENSE.value,
        ] and (
            has_comp_word or
            (s.startswith('bagaimana dengan') and ('bulan ini' in s or 'bulan lalu' in s or 'realisasi' in s))
        )
    )
    if has_forecast_comp or is_follow_up_comp:
        return PlannerDecision(
            tool_name=ToolName.COMPARE_ACTUAL_VS_FORECAST,
            follow_up_resolved=is_follow_up_comp,
            raw_query=text
        )

    # 3.5 Expense Trend and Drivers
    if (is_forecast_word or any(w in s for w in ['pengeluaran', 'biaya', 'tren'])) and any(w in s for w in ['meningkat', 'naik', 'turun', 'menurun', 'arah tren', 'tren pengeluaran', 'kenapa prediksi', 'mengapa prediksi', 'kenapa pengeluaran naik', 'mengapa pengeluaran naik']):
        return PlannerDecision(tool_name=ToolName.GET_EXPENSE_TREND, raw_query=text)

    # 3.6 Next Predicted Expense
    if re.search(r'\bkapan\s+(?:harus\s+bayar|jadwal\s+bayar|pengeluaran|tagihan)\b', s) or any(w in s for w in ['kapan pengeluaran berikutnya', 'kapan tagihan berikutnya', 'kapan jadwal bayar', 'kapan bayar tagihan']):
        return PlannerDecision(tool_name=ToolName.GET_NEXT_PREDICTED_EXPENSE, raw_query=text)

    # 3.7 Recurring Expenses and 3-Day Upcoming Reminders
    is_pure_short_horizon_forecast = (
        is_forecast_word
        and not any(w in s for w in ['rutin', 'tagihan', 'cicilan', 'berkala', 'langganan'])
    )
    if (
        any(w in s for w in ['pengeluaran rutin', 'biaya rutin', 'tagihan rutin', 'pembayaran rutin', 'rutin apa saja'])
        or (any(w in s for w in ['tiga hari ke depan', '3 hari ke depan']) and not is_pure_short_horizon_forecast)
        or (re.search(r'\btagihan\b', s) and any(w in s for w in ['persiapkan', 'mendatang', 'akan datang', 'jadwal']))
    ):
        return PlannerDecision(tool_name=ToolName.GET_UPCOMING_RECURRING_EXPENSES, raw_query=text)

    # 3.8 Conceptual Prophet questions (e.g. "bagaimana cara kerja prophet", "mekanisme peramalan", "bagaimana peramalan pengeluaran", "prediksi pengeluaran berikutnya")
    is_conceptual = (
        (re.search(r'\b(?:bagaimana\s+cara\s+kerja|mekanisme|apa\s+itu|cara\s+kerja)\b', s) and any(w in s for w in ['prophet', 'peramalan', 'forecasting'])) or
        ('bagaimana' in s and any(w in s for w in ['peramalan', 'prophet', 'forecasting']) and not has_date_specifier(s)) or
        (s.rstrip('?.') in ['bagaimana prediksi pengeluaran', 'prediksi pengeluaran berikutnya', 'bagaimana prediksi pengeluaran berikutnya'])
    )
    if is_conceptual:
        return PlannerDecision(tool_name=ToolName.GET_AVAILABLE_FORECAST, raw_query=text)

    # 3.9 Forecast for Specific Future Period or Horizon or Category
    if is_forecast_word:
        explicit_days = _extract_explicit_horizon_days(s)
        if explicit_days is not None and (explicit_days < 3 or explicit_days > 90):
            return PlannerDecision(
                tool_name=ToolName.UNKNOWN,
                clarification_question=(
                    f"Horizon peramalan pengeluaran hanya mendukung rentang **3 hingga 90 hari ke depan** "
                    f"(permintaan Anda: {explicit_days} hari). Silakan gunakan jumlah hari antara 3 sampai 90."
                ),
                raw_query=text,
            )

        found_category = None
        for cat_kw in ['listrik', 'wifi', 'internet', 'sewa', 'gaji', 'air', 'pdam', 'bahan baku', 'operasional', 'belanja']:
            if cat_kw in s:
                found_category = cat_kw
                break

        is_breakdown = bool(re.search(r'\b(rincian|rincikan|detail|per\s+hari|setiap\s+hari|tabel|harian|breakdown)\b', s))
        output_m = ForecastOutputMode.DAILY_BREAKDOWN if is_breakdown else ForecastOutputMode.SUMMARY

        date_p = _extract_date_params(s)
        has_period = (
            date_p.start_date is not None or
            date_p.period_label in ['Bulan Depan', 'Minggu Depan', 'Besok'] or
            'hari ke depan' in s or
            'seminggu' in s or
            'minggu depan' in s or
            'bulan depan' in s or
            'besok' in s or
            'esok' in s
        )
        if has_period:
            return PlannerDecision(
                tool_name=ToolName.GET_FORECAST_FOR_PERIOD,
                date_params=date_p,
                filter_params=FilterParams(
                    output_mode=output_m,
                    category_name=found_category
                ),
                raw_query=text
            )

        days = explicit_days if (explicit_days is not None and 3 <= explicit_days <= 90) else 30
        if explicit_days is None:
            if any(w in s for w in ['7 hari', 'tujuh hari', 'seminggu']):
                days = 7
            elif any(w in s for w in ['14 hari', 'dua minggu']):
                days = 14
            elif any(w in s for w in ['60 hari', 'dua bulan']):
                days = 60
            elif any(w in s for w in ['90 hari', 'tiga bulan']):
                days = 90

        return PlannerDecision(
            tool_name=ToolName.GET_EXPENSE_FORECAST,
            date_params=date_p,
            filter_params=FilterParams(
                days_ahead=days,
                category_name=found_category,
                output_mode=output_m
            ),
            raw_query=text
        )

    # 4. App Feature Guides & How-To
    is_feature_q = bool(re.search(
        r'\b(bagaimana\s+cara|cara\s+(?:menambah|membuat|mengubah|menghapus|menyetujui|approve|impor|import|kerja|pendaftaran)|fitur\s+apa|bisa\s+apa\s+saja\s+aplikasi|panduan\s+fitur|cara\s+kerja)\b',
        s
    ))
    is_fin_state_q = bool(re.search(r'bagaimana\s+(?:arus\s+kas|saldo|pemasukan|pengeluaran|kondisi\s+keuangan|keuntungan|laba)', s))
    if is_feature_q and not is_fin_state_q and not _is_business_guidance_query(s):
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

    # Conversational follow-up: Requesting breakdown / details of previous forecast
    # e.g. "Tampilkan rinciannya", "Bisa berikan rinciannya?", "Rinciannya", "Detailnya"
    has_prev_forecast = (
        context is not None and (
            context.last_tool in [
                ToolName.GET_FORECAST_FOR_PERIOD.value,
                ToolName.GET_EXPENSE_FORECAST.value,
                "query_forecast"
            ] or
            context.last_forecast_period is not None or
            context.last_forecast_start_date is not None
        )
    )
    is_breakdown_follow_up = (
        has_prev_forecast and
        any(w in s for w in ['rincian', 'rinciannya', 'rincikan', 'detailnya', 'tabelnya', 'per harinya', 'pecah per hari', 'harian']) and
        not any(w in s for w in ['transaksi', 'pemasukan', 'bulan lalu', 'kemarin', 'kasir'])
    )
    if is_breakdown_follow_up:
        start_d = context.last_forecast_start_date or context.last_start_date
        end_d = context.last_forecast_end_date or context.last_end_date
        period_lbl = context.last_forecast_period or context.last_period_label or "7 Hari ke Depan"
        days_cnt = context.last_forecast_days or 7

        date_p = DateRangeParams(
            start_date=start_d,
            end_date=end_d,
            period_label=period_lbl
        )
        filter_p = FilterParams(
            output_mode=ForecastOutputMode.DAILY_BREAKDOWN,
            days_ahead=days_cnt,
            category_name=context.last_forecast_category
        )
        return PlannerDecision(
            tool_name=ToolName.GET_FORECAST_FOR_PERIOD,
            date_params=date_p,
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

    # 13b. Inventory withdrawal / stock taking ("ambil 5 bungkus indomie", "ambil beras", "ambil mie instan")
    is_inventory_withdrawal = bool(
        re.search(r'\b(?:ambil|mengambil|pakai|menggunakan)\s+(?:\d+\s*(?:bungkus|bks|pcs|pack|sak|kg|dus|karton|botol|kaleng|butir|lembar|buah)\s+)?(?:stok|barang|etalase|mie|indomie|beras|telur|sabun|rokok|kopi|minyak)\b', s)
        or (s.startswith('ambil ') and not any(k in s for k in ['kasir', 'kembalian', 'uang', 'modal', 'laba']))
    )
    if is_inventory_withdrawal and not any(w in s for w in ['beli', 'bayar', 'jual', 'omzet', 'penjualan']):
        return PlannerDecision(
            tool_name=ToolName.CLARIFICATION_NEEDED,
            raw_query=text,
            clarification_question=(
                "Pengambilan barang dagangan memerlukan keterangan konteks: apakah untuk keperluan operasional toko, "
                "prive (keperluan pribadi pemilik), atau barang rusak/hilang? Pencatatan ini adalah mutasi persediaan, bukan mutasi kas langsung."
            )
        )

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

    # Operational Business Guidance (Isolated Per-Business-Type Knowledge RAG)
    if _is_business_guidance_query(s):
        return PlannerDecision(
            tool_name=ToolName.GET_BUSINESS_GUIDANCE,
            raw_query=text,
        )

    return None


def _extract_explicit_horizon_days(s: str) -> Optional[int]:
    '''Extract explicit multi-day horizon requests (e.g. '2 hari', '45 hari ke depan', '120 hari', 'setahun').'''
    m_num = re.search(r'\b(\d+)\s*hari(?:\s*ke\s*depan)?\b', s)
    if m_num:
        return int(m_num.group(1))
    word_days = {
        'satu hari': 1,
        'dua hari': 2,
        'tiga hari': 3,
        'empat hari': 4,
        'lima hari': 5,
        'enam hari': 6,
        'tujuh hari': 7,
        'sepuluh hari': 10,
        'empat belas hari': 14,
        'tiga puluh hari': 30,
        'enam puluh hari': 60,
        'sembilan puluh hari': 90,
        'empat bulan': 120,
        'enam bulan': 180,
        'setahun': 365,
        'satu tahun': 365,
    }
    for phrase, n_val in word_days.items():
        if phrase in s:
            return n_val
    return None


def _is_business_guidance_query(s: str) -> bool:
    app_only_terms = (
        'fitur', 'aplikasi', 'impor', 'import', 'csv', 'klasifikasi',
        'persetujuan', 'approval', 'pencatatan cepat', 'halaman ',
    )
    if any(t in s for t in app_only_terms):
        return False

    guidance_phrases = (
        'panduan usaha', 'panduan operasional', 'panduan bisnis', 'tips usaha', 'tips bisnis',
        'panduan ', 'tips ', 'strategi pemasaran', 'tips pemasaran', 'strategi promosi',
        'cara mengelola stok', 'pengelolaan stok', 'kelola stok', 'stock opname', 'overstock',
        'dead stock', 'fifo', 'fefo', 'par stock', 'fast moving', 'slow moving',
        'cara menghitung hpp', 'hitung hpp', 'perhitungan hpp', 'food cost', 'cogm', 'cogs',
        'penetapan harga', 'harga jual', 'margin keuntungan', 'multi pricing', 'multi-pricing',
        'legalitas', 'perizinan', 'nib', 'oss rba', 'oss-rba', 'pirt', 'spp-irt',
        'sertifikasi halal', 'sertifikat halal', 'bpjph', 'kbli', 'limbah oli', 'limbah b3',
        'sop harian', 'sop toko', 'sop buka', 'sop tutup', 'work order', 'nota servis',
        'komisi mekanik', 'model usaha', 'standar kualitas', 'kelola supplier',
        'manajemen supplier', 'stok bahan baku', 'roti sisa', 'barang sisa',
        'pisahkan rekening', 'rekening pribadi', 'kas pribadi',
    )
    return any(phrase in s for phrase in guidance_phrases)


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
        "- get_business_guidance: panduan operasional, manajemen stok, HPP, SOP harian, legalitas (NIB/PIRT/Halal), dan pemasaran sesuai bidang usaha toko\n"
        "- get_available_forecast: penjelasan konsep dan cara kerja peramalan multi-model (Prophet, ETS, ARIMA, Seasonal Naive, MA7, LinearRegression)\n"
        "- get_expense_forecast: proyeksi pengeluaran umum toko atau kategori tertentu untuk horizon 3-90 hari\n"
        "- get_forecast_for_period: prediksi pengeluaran untuk periode tertentu (besok, bulan depan, minggu depan)\n"
        "- get_upcoming_recurring_expenses: deteksi pengeluaran/tagihan rutin mendatang dan pengingat 3 hari\n"
        "- get_next_predicted_expense: estimasi jadwal dan nominal pengeluaran berikutnya\n"
        "- compare_actual_vs_forecast: perbandingan realisasi pengeluaran saat ini vs proyeksi mendatang\n"
        "- get_forecast_accuracy: perbandingan akurasi kandidat model peramalan (MAPE, MAE, RMSE, WAPE)\n"
        "- get_expense_trend: analisis arah tren pengeluaran (meningkat/menurun) dan faktor penggerak\n"
        "- get_forecast_readiness: kesiapan dan kecukupan data historis toko untuk peramalan pengeluaran\n"
        "- check_forecast_against_cash_balance: pengecekan apakah saldo kas saat ini mencukupi prediksi pengeluaran\n"
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
        # Safe fallback: if LLM planning fails, return None so plan_intent falls back to fallback decision
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
