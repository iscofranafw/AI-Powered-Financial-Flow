from datetime import datetime, date, timedelta
from decimal import Decimal
import re
import uuid

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from ai.text_normalizer import extract_money, clean_text_for_storage
from ai.classifier import classify, detect_type
import calendar


def _safe_date(year, month, day):
    """Clamp day to the last valid day of the month (31 Feb -> 28/29 Feb)."""
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day, last_day))


MONTH_MAP = {
    'januari': 1, 'jan': 1,
    'februari': 2, 'feb': 2,
    'maret': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'mei': 5,
    'juni': 6, 'jun': 6,
    'juli': 7, 'jul': 7,
    'agustus': 8, 'agu': 8, 'agt': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'oktober': 10, 'okt': 10,
    'november': 11, 'nov': 11,
    'desember': 12, 'des': 12,
}


class Intent:
    CREATE_INCOME = 'CREATE_INCOME'
    CREATE_EXPENSE = 'CREATE_EXPENSE'
    LIST_INCOME_CATEGORIES = 'LIST_INCOME_CATEGORIES'
    LIST_EXPENSE_CATEGORIES = 'LIST_EXPENSE_CATEGORIES'
    LIST_ALL_CATEGORIES = 'LIST_ALL_CATEGORIES'
    QUERY_INCOME = 'QUERY_INCOME'
    QUERY_EXPENSE = 'QUERY_EXPENSE'
    QUERY_CASH_FLOW = 'QUERY_CASH_FLOW'
    QUERY_TRANSACTIONS = 'QUERY_TRANSACTIONS'
    QUERY_TRANSACTION_HISTORY = 'QUERY_TRANSACTIONS'
    QUERY_PENDING_APPROVALS = 'QUERY_PENDING_APPROVALS'
    QUERY_CASHIER_ACTIVITY = 'QUERY_CASHIER_ACTIVITY'
    QUERY_FINANCIAL_COMPARISON = 'QUERY_FINANCIAL_COMPARISON'
    QUERY_FINANCIAL_INSIGHTS = 'QUERY_FINANCIAL_INSIGHTS'
    QUERY_PROFIT = 'QUERY_PROFIT'
    QUERY_FORECAST = 'QUERY_FORECAST'
    QUERY_STORE_PROFILE = 'QUERY_STORE_PROFILE'
    EXPLAIN_APP_FEATURE = 'EXPLAIN_APP_FEATURE'
    QUERY_BUSINESS_GUIDANCE = 'QUERY_BUSINESS_GUIDANCE'
    UNSUPPORTED_MUTATION = 'UNSUPPORTED_MUTATION'
    CLARIFICATION_NEEDED = 'CLARIFICATION_NEEDED'
    GENERAL_HELP = 'GENERAL_HELP'
    UNKNOWN = 'UNKNOWN'
    # Retained for legacy backward-compatibility
    QUERY_FINANCIAL_SUMMARY = 'QUERY_FINANCIAL_SUMMARY'


def get_jakarta_now():
    if ZoneInfo:
        try:
            return datetime.now(ZoneInfo('Asia/Jakarta'))
        except Exception:
            # Safe fallback: if IANA tzdata is missing for Asia/Jakarta, use fixed UTC+7 timezone
            pass
    from datetime import timezone
    return datetime.now(timezone(timedelta(hours=7)))


def has_date_specifier(text):
    s = text.lower()
    return (
        any(w in s for w in [
            'hari ini', 'hr ini', 'kemarin', 'kemaren', 'kmrn', 'lusa', 'besok', 'esok',
            'minggu ini', 'mggu ini', 'minggu lalu', 'minggu kemarin', 'minggu depan', 'mggu depan',
            'bulan ini', 'bln ini', 'bulan lalu', 'bulan kemarin', 'bulan depan', 'bln depan',
            'tahun ini', 'tahun lalu', 'tahun kemarin', 'hari ke depan', 'seminggu ke depan'
        ])
        or bool(re.search(r'\b(?:tanggal|tgl)\b', s))
        or bool(re.search(r'\b\d{4}-\d{2}-\d{2}\b', s))
        or bool(re.search(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b', s))
        or any(m in s for m in MONTH_MAP)
    )


def resolve_date_indonesian(text, base_dt=None):
    base = base_dt or get_jakarta_now()
    s = text.lower()

    if 'kemarin lusa' in s or '2 hari lalu' in s:
        return (base - timedelta(days=2)).date()
    if 'kemarin' in s or '1 hari lalu' in s:
        return (base - timedelta(days=1)).date()
    if 'lusa' in s:
        return (base + timedelta(days=2)).date()
    if 'besok' in s or 'esok' in s:
        return (base + timedelta(days=1)).date()
    if 'hari ini' in s:
        return base.date()

    m_days = re.search(r'\b(\d+)\s*hari\s*lalu\b', s)
    if m_days:
        return (base - timedelta(days=int(m_days.group(1)))).date()

    months_re = '|'.join(sorted(MONTH_MAP.keys(), key=len, reverse=True))
    m_full = re.search(rf'(?:tanggal|tgl)?\s*(\d{{1,2}})\s+({months_re})(?:\s+(\d{{4}}))?', s)
    if m_full:
        day = int(m_full.group(1))
        month = MONTH_MAP[m_full.group(2)]
        year = int(m_full.group(3)) if m_full.group(3) else base.year
        return _safe_date(year, month, day)

    m_iso = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', s)
    if m_iso:
        return _safe_date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))

    m_slash = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', s)
    if m_slash:
        return _safe_date(int(m_slash.group(3)), int(m_slash.group(2)), int(m_slash.group(1)))

    m_day_only = re.search(r'\b(?:tanggal|tgl)\s+(\d{1,2})\b', s)
    if m_day_only:
        day = int(m_day_only.group(1))
        return _safe_date(base.year, base.month, day)

    return base.date()


def resolve_date_range_indonesian(text, base_dt=None):
    base = base_dt or get_jakarta_now()
    today = base.date()
    s = text.lower()

    months_re = '|'.join(sorted(MONTH_MAP.keys(), key=len, reverse=True))
    m_range = re.search(
        rf'(?:tanggal|tgl)?\s*(\d{{1,2}})\s*(?:sampai|s\.?d\.?|hingga|-)\s*(\d{{1,2}})\s+({months_re})(?:\s+(\d{{4}}))?',
        s
    )
    if m_range:
        d1 = int(m_range.group(1))
        d2 = int(m_range.group(2))
        month = MONTH_MAP[m_range.group(3)]
        year = int(m_range.group(4)) if m_range.group(4) else today.year
        start = _safe_date(year, month, min(d1, d2))
        end = _safe_date(year, month, max(d1, d2))
        return start, end, f"{start.day} s.d. {end.day} {m_range.group(3).capitalize()} {year}"

    if any(k in s for k in ['bulan depan', 'bln depan']):
        if today.month == 12:
            start = date(today.year + 1, 1, 1)
            end = date(today.year + 1, 1, 31)
        else:
            start = date(today.year, today.month + 1, 1)
            next_next = date(today.year, today.month + 2, 1) if today.month < 11 else date(today.year + 1, 1, 1)
            end = next_next - timedelta(days=1)
        return start, end, 'Bulan Depan'

    if any(k in s for k in ['minggu depan', 'mggu depan']):
        start = today + timedelta(days=7 - today.weekday())
        end = start + timedelta(days=6)
        return start, end, 'Minggu Depan'

    m_future_days = re.search(r'\b(\d+|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|tiga puluh)\s*hari\s*ke\s*depan\b', s)
    if m_future_days:
        word_map = {'satu': 1, 'dua': 2, 'tiga': 3, 'empat': 4, 'lima': 5, 'enam': 6, 'tujuh': 7, 'delapan': 8, 'sembilan': 9, 'sepuluh': 10, 'tiga puluh': 30}
        val_str = m_future_days.group(1)
        n_days = word_map[val_str] if val_str in word_map else int(val_str)
        start = today + timedelta(days=1)
        end = today + timedelta(days=n_days)
        return start, end, f'{n_days} Hari ke Depan'

    if re.search(r'\b(?:seminggu|1\s*minggu|satu\s*minggu)\s*(?:ke\s*depan)?\b', s) and any(w in s for w in ['prediksi', 'peramalan', 'proyeksi', 'forecast', 'depan']):
        start = today + timedelta(days=1)
        end = today + timedelta(days=7)
        return start, end, '7 Hari ke Depan'

    if any(k in s for k in ['besok', 'esok']):
        tomorrow = today + timedelta(days=1)
        return tomorrow, tomorrow, 'Besok'

    if any(k in s for k in ['bulan lalu', 'bulan kemarin', 'bln lalu', 'bln kemarin']):
        first_this = date(today.year, today.month, 1)
        end = first_this - timedelta(days=1)
        start = date(end.year, end.month, 1)
        return start, end, 'Bulan Lalu'

    if any(k in s for k in ['minggu lalu', 'minggu kemarin', 'mggu lalu', 'mggu kemarin']):
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return start, end, 'Minggu Lalu'

    if any(k in s for k in ['tahun lalu', 'tahun kemarin', 'thn lalu', 'thn kemarin']):
        start = date(today.year - 1, 1, 1)
        end = date(today.year - 1, 12, 31)
        return start, end, 'Tahun Lalu'

    if any(k in s for k in ['bulan ini', 'bln ini', 'buln ini']):
        start = date(today.year, today.month, 1)
        if today.month == 12:
            end = date(today.year, 12, 31)
        else:
            end = date(today.year, today.month + 1, 1) - timedelta(days=1)
        return start, end, 'Bulan Ini'

    if any(k in s for k in ['minggu ini', 'mggu ini']):
        start = today - timedelta(days=today.weekday())
        return start, today, 'Minggu Ini'

    if any(k in s for k in ['tahun ini', 'thn ini']):
        start = date(today.year, 1, 1)
        end = date(today.year, 12, 31)
        return start, end, 'Tahun Ini'

    if any(k in s for k in ['hari ini', 'hr ini']):
        return today, today, 'Hari Ini'

    if any(k in s for k in ['kemarin lusa', '2 hari lalu']):
        d = today - timedelta(days=2)
        return d, d, 'Kemarin Lusa'

    if re.search(r'(?<!bulan\s)(?<!minggu\s)(?<!tahun\s)(?<!bln\s)(?<!mggu\s)\b(?:kemarin|kemaren|kmrn|1 hari lalu)\b', s):
        yest = today - timedelta(days=1)
        return yest, yest, 'Kemarin'

    m_single = re.search(rf'(?:tanggal|tgl)?\s*(\d{{1,2}})\s+({months_re})(?:\s+(\d{{4}}))?', s)
    if m_single:
        day = int(m_single.group(1))
        month = MONTH_MAP[m_single.group(2)]
        year = int(m_single.group(3)) if m_single.group(3) else today.year
        d = _safe_date(year, month, day)
        month_label = m_single.group(2).capitalize()
        return d, d, f"{day} {month_label} {year}"

    m_iso = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', s)
    if m_iso:
        d = _safe_date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        return d, d, d.isoformat()

    return None, None, 'Semua Periode'


def resolve_comparison_periods(text, previous_period=None, base_dt=None):
    base = base_dt or get_jakarta_now()
    today = base.date()
    s = text.lower()

    if any(k in s for k in ['minggu lalu', 'minggu kemarin', 'mggu lalu', 'mggu kemarin']) or (previous_period and 'minggu' in str(previous_period).lower() and 'bulan' not in s):
        p1_start = today - timedelta(days=today.weekday())
        p1_end = today
        p1_label = 'Minggu Ini'

        p2_start = today - timedelta(days=today.weekday() + 7)
        p2_end = p2_start + timedelta(days=6)
        p2_label = 'Minggu Lalu'
        return p1_start, p1_end, p1_label, p2_start, p2_end, p2_label

    p1_start = date(today.year, today.month, 1)
    if today.month == 12:
        p1_end = date(today.year, 12, 31)
    else:
        p1_end = date(today.year, today.month + 1, 1) - timedelta(days=1)
    p1_label = 'Bulan Ini'

    first_this = date(today.year, today.month, 1)
    p2_end = first_this - timedelta(days=1)
    p2_start = date(p2_end.year, p2_end.month, 1)
    p2_label = 'Bulan Lalu'
    return p1_start, p1_end, p1_label, p2_start, p2_end, p2_label


def clean_transaction_description(text):
    s = text.strip()

    # Strip monetary expressions
    s = re.sub(
        r'(?i)(?:dengan\s+total|dengan\s+nilai|dengan|sebesar|senilai|total|seharga)\s*(?:rp\.?\s*)?\d[\d.,]*\s*(?:juta|jt|ribu|rb|k|rupiah)?',
        ' ', s
    )
    s = re.sub(r'(?i)(?:rp\.?\s*)?\d[\d.,]*\s*(?:juta|jt|ribu|rb|k|rupiah)\b', ' ', s)
    s = re.sub(r'(?i)(?:rp\.?\s*)\d[\d.,]*\b', ' ', s)

    # Strip relative and explicit dates
    s = re.sub(r'(?i)\b(hari ini|kemarin lusa|kemarin|lusa|\d+\s*hari lalu)\b', ' ', s)
    months_re = '|'.join(sorted(MONTH_MAP.keys(), key=len, reverse=True))
    s = re.sub(rf'(?i)(?:tanggal|tgl)?\s*\d{{1,2}}\s+(?:{months_re})(?:\s+\d{{4}})?', ' ', s)
    s = re.sub(r'(?i)(?:tanggal|tgl)\s*\d{1,2}\b', ' ', s)

    # Strip transaction command verbs
    s = re.sub(r'(?i)^\s*(?:tolong\s+|mohon\s+|bisa\s+)?(?:catat(?:lah)?|tambah(?:kan)?|masukkan|input)\s+', ' ', s)
    s = re.sub(r'(?i)^\s*(?:saya\s+)?(?:membeli|beli|membayar|bayar)\s+', ' ', s)
    s = re.sub(r'(?i)^\s*(?:saya\s+)', ' ', s)
    s = re.sub(r'(?i)\b(?:pemasukan|pengeluaran)\s+(?:dari|untuk)?\s*', ' ', s)

    cleaned = ' '.join(s.split()).strip(' .,:;-')
    return cleaned[:200].capitalize() if cleaned else ''


def get_previous_context(history):
    if not history:
        return {}
    for msg in reversed(history):
        if msg.get('role') == 'assistant' and msg.get('metadata'):
            return msg['metadata']
        if msg.get('metadata'):
            return msg['metadata']
    for msg in reversed(history):
        if msg.get('role') == 'user':
            prev_text = msg.get('content', '')
            prev_intent = detect_chat_intent(prev_text, history=None)
            prev_metric = detect_query_metric(prev_text)
            p_start, p_end, prev_period = resolve_date_range_indonesian(prev_text)
            days = (p_end - p_start).days + 1 if (p_start and p_end) else None
            return {
                'intent': prev_intent,
                'metric': prev_metric,
                'period': prev_period,
                'start_date': p_start.isoformat() if p_start else None,
                'end_date': p_end.isoformat() if p_end else None,
                'days_ahead': days,
                'raw_text': prev_text
            }
    return {}


def detect_chat_intent(text, history=None, is_owner=True):
    s = text.lower().strip()

    # 1. Reject unsupported database modifications via chat
    if re.search(r'\b(hapus|delete|edit|ubah|ganti)\s+(?:transaksi|data|catatan)\b', s) or re.search(r'\b(batalkan|hapus)\s+(?:transaksi|catatan)\b', s):
        return Intent.UNSUPPORTED_MUTATION
    if re.search(r'\b(drop\s+table|delete\s+from|update\s+set|truncate\s+table)\b', s):
        return Intent.UNSUPPORTED_MUTATION

    # 2. General help and pleasantries
    if any(w in s for w in ['halo', 'hai', 'bantuan', 'help', 'menu apa', 'bisa apa', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam', 'panduan penggunaan']):
        return Intent.GENERAL_HELP

    # 3. Application feature guides and how-to inquiries
    is_feature_query = bool(re.search(
        r'\b(bagaimana\s+cara|cara\s+(?:menambah|membuat|mengubah|menghapus|menyetujui|impor|import|kerja|pendaftaran)|fitur\s+apa|bisa\s+apa\s+saja\s+aplikasi|panduan\s+fitur|cara\s+kerja)\b',
        s
    ))
    # Exclude questions inquiring financial states like "bagaimana arus kas" or "bagaimana pemasukan"
    is_financial_state_q = bool(re.search(r'bagaimana\s+(?:arus\s+kas|saldo|pemasukan|pengeluaran|kondisi\s+keuangan)', s))
    if is_feature_query and not is_financial_state_q:
        return Intent.EXPLAIN_APP_FEATURE

    # 4. Forecast and future expense projections
    if any(w in s for w in ['forecast', 'forecasting', 'peramalan', 'prediksi', 'proyeksi', 'prophet']):
        return Intent.QUERY_FORECAST

    # 5. Store profile and business identity
    is_store_profile_query = (
        any(w in s for w in ['profil toko', 'bidang usaha', 'tentang toko ini', 'identitas toko'])
        or bool(re.search(r'\bnama\s+toko(?:\s+saya)?\b', s))
        or s in ['data toko saya', 'data toko', 'informasi toko']
    )
    if is_store_profile_query and not any(k in s for k in ['kategori', 'pengeluaran', 'pemasukan', 'transaksi', 'kasir', 'keuntungan', 'laba']):
        return Intent.QUERY_STORE_PROFILE

    # 6. Cashier accounts and activity
    is_cashier_count = bool(
        re.search(r'\b(?:berapa|jumlah|total)\s+(?:semua\s+)?kasir\b', s) or
        re.search(r'\bada\s+berapa\s+kasir\b', s) or
        re.search(r'\bberapa\s+(?:banyak\s+)?kasir\b', s) or
        re.search(r'\bkasir\s+(?:ada\s+)?berapa\b', s)
    )
    is_cashier_activity = (
        is_cashier_count
        or any(w in s for w in ['siapa saja kasir', 'daftar kasir', 'akun kasir', 'kasir aktif', 'aktivitas kasir', 'siapa kasir', 'kasir terdaftar', 'kasir mana'])
        or ('kasir' in s and any(k in s for k in ['aktif', 'aktivitas', 'kinerja', 'catat', 'rajin', 'terbanyak', 'performa', 'mana', 'siapa']))
    )
    if is_cashier_activity and not (any(w in s for w in ['persetujuan', 'approval', 'belum disetujui', 'belum dikonfirmasi', 'pending']) or 'belum' in s):
        return Intent.QUERY_CASHIER_ACTIVITY

    # 7. Cashier transaction approvals
    has_approval_word = any(w in s for w in ['persetujuan', 'approval', 'menunggu persetujuan', 'belum disetujui', 'belum dikonfirmasi', 'antrean kasir', 'transaksi pending'])
    has_cashier_pending = 'kasir' in s and any(k in s for k in ['belum', 'pending', 'menunggu'])
    if has_approval_word or has_cashier_pending:
        return Intent.QUERY_PENDING_APPROVALS

    # Cashier transactions (all, not pending-only, e.g. "ada transaksi kasir hari ini?")
    is_cashier_tx = bool(
        re.search(r'\btransaksi\s+kasir\b', s) or
        ('kasir' in s and 'transaksi' in s and any(w in s for w in ['ada', 'apa saja', 'daftar', 'lihat', 'hari ini', 'kemarin', 'minggu']))
    )
    if is_cashier_tx:
        return Intent.QUERY_TRANSACTIONS

    # 8. Profit and operational cash limitations
    if any(k in s for k in ['keuntungan bersih', 'laba bersih', 'profit bersih', 'keuntungan toko', 'laba toko', 'untung bersih', 'keuntungan saya', 'laba saya', 'profit saya']):
        return Intent.QUERY_PROFIT

    # 8. Financial insights (e.g. largest expenses, top spending categories)
    if any(k in s for k in ['pengeluaran terbesar', 'biaya terbesar', 'transaksi terbesar', 'paling banyak menghabiskan', 'paling boros', 'kategori terbesar', 'kategori paling banyak', 'menghabiskan uang']) or \
       (any(k in s for k in ['kategori', 'pos']) and any(k in s for k in ['paling banyak', 'terbanyak', 'terbesar', 'boros', 'menghabiskan', 'peringkat', 'ranking'])):
        return Intent.QUERY_FINANCIAL_INSIGHTS

    # 9. Category taxonomy queries
    has_cat_word = any(w in s for w in ['kategori', 'pos belanja', 'pos anggaran', 'pos pemasukan', 'pos pengeluaran', 'pos biaya', 'pos pendapatan'])
    has_cat_listing = any(w in s for w in ['apa saja', 'daftar', 'tampilkan', 'lihat', 'ada apa', 'ada kategori', 'kategori apa', 'pos apa'])
    has_income_word = any(w in s for w in ['pemasukan', 'pendapatan', 'omzet'])
    has_expense_word = any(w in s for w in ['pengeluaran', 'biaya', 'belanja'])

    if has_cat_word and (has_cat_listing or 'ada' in s or 'apa' in s):
        if has_income_word and has_expense_word:
            return Intent.LIST_ALL_CATEGORIES
        if has_income_word and not has_expense_word:
            return Intent.LIST_INCOME_CATEGORIES
        if has_expense_word and not has_income_word:
            return Intent.LIST_EXPENSE_CATEGORIES
        return Intent.LIST_ALL_CATEGORIES

    # 9. Period comparisons (including conversational follow-ups)
    has_comparison_word = any(w in s for w in ['bandingkan', 'perbandingan', 'dibandingkan', 'komparasi'])
    if has_comparison_word:
        return Intent.QUERY_FINANCIAL_COMPARISON

    # 10. Check conversational follow-ups from prior turn
    prev_ctx = get_previous_context(history)
    if prev_ctx:
        if 'bandingkan' in s or (s.startswith('bagaimana dengan') and any(p in s for p in ['bulan lalu', 'minggu lalu', 'kemarin'])):
            return Intent.QUERY_FINANCIAL_COMPARISON
        if s.startswith('bagaimana dengan') and has_income_word:
            return Intent.QUERY_INCOME
        if s.startswith('bagaimana dengan') and has_expense_word:
            return Intent.QUERY_EXPENSE

    # 10b. Check for inventory withdrawal / stock taking ("ambil 5 bungkus indomie", "ambil mie instan", "ambil beras")
    is_inventory_withdrawal = bool(
        re.search(r'\b(?:ambil|mengambil|pakai|menggunakan)\s+(?:\d+\s*(?:bungkus|bks|pcs|pack|sak|kg|dus|karton|botol|kaleng|butir|lembar|buah)\s+)?(?:stok|barang|etalase|mie|indomie|beras|telur|sabun|rokok|kopi|minyak)\b', s)
        or (s.startswith('ambil ') and not any(k in s for k in ['kasir', 'kembalian', 'uang', 'modal', 'laba']))
    )
    if is_inventory_withdrawal and not any(w in s for w in ['beli', 'bayar', 'jual', 'omzet', 'penjualan']):
        return Intent.CLARIFICATION_NEEDED

    # 11. Transaction creation intents
    is_command = bool(re.search(r'^(?:tolong\s+|mohon\s+|bisa\s+)?(?:catat(?:lah)?|tambah(?:kan)?|masukkan|input)\b', s))
    has_money = bool(re.search(r'(?<!\w)(?:rp\.?\s*)?\d[\d.,]*\s*(?:juta|jt|ribu|rb|k|rupiah)?\b', s))
    is_explicit_query = bool(re.search(r'\b(berapa|apa|tampilkan|lihat|cek|daftar|siapa|bagaimana)\b', s))

    if (is_command or has_money) and not is_explicit_query:
        income_score = sum(k in s for k in ['pemasukan', 'penjualan', 'omzet', 'pendapatan', 'setor', 'modal'])
        expense_score = sum(k in s for k in ['pengeluaran', 'beli', 'bayar', 'belanja', 'gaji', 'listrik', 'biaya', 'sewa', 'ongkir', 'tepung', 'telur', 'bensin', 'stok'])
        if income_score > 0 and income_score >= expense_score:
            return Intent.CREATE_INCOME
        if expense_score > 0 and expense_score > income_score:
            return Intent.CREATE_EXPENSE
        if any(w in s for w in ['beli', 'bayar', 'belanja']):
            return Intent.CREATE_EXPENSE
        return Intent.CREATE_INCOME

    # 12. Transaction history & detail queries
    is_tx_detail_req = bool(
        re.search(r'\b(detail|rincian|tampilkan|lihat|daftar|riwayat)\s+(?:semua\s+)?(?:transaksi|pengeluaran|pemasukan)\b', s) or
        re.search(r'\btransaksi\s+tanggal\b', s) or
        re.search(r'\briwayat\s+transaksi\b', s) or
        ('detail' in s and any(w in s for w in ['pengeluaran', 'pemasukan', 'transaksi']))
    )
    if is_tx_detail_req:
        return Intent.QUERY_TRANSACTIONS

    # 13. Financial insights (e.g. largest expenses, top categories)
    if any(k in s for k in ['pengeluaran terbesar', 'biaya terbesar', 'transaksi terbesar', 'paling banyak menghabiskan', 'paling boros', 'kategori terbesar', 'kategori paling banyak']):
        return Intent.QUERY_FINANCIAL_INSIGHTS

    # 14. Fine-grained financial metric queries
    has_summary_q = any(w in s for w in ['berapa', 'total', 'rekap', 'jumlah', 'bagaimana', 'apa', 'sisa'])
    if has_income_word and has_summary_q and not has_expense_word:
        return Intent.QUERY_INCOME
    if has_expense_word and has_summary_q and not has_income_word:
        return Intent.QUERY_EXPENSE
    if any(k in s for k in ['saldo', 'sisa kas', 'uang kas', 'arus kas', 'cash flow', 'cashflow']):
        return Intent.QUERY_CASH_FLOW

    # 15. General financial overview
    fin_terms = ['pemasukan', 'pengeluaran', 'uang', 'saldo', r'\bkas\b', 'omzet', 'transaksi', 'biaya', 'laba', 'rugi', 'rekap', 'ringkasan']
    has_fin_term = any(bool(re.search(t, s)) if t.startswith(r'\b') else (t in s) for t in fin_terms)
    if has_fin_term and any(w in s for w in ['berapa', 'apa', 'total', 'saldo', 'sisa', 'arus', 'rekap', 'ringkasan', 'kondisi']):
        return Intent.QUERY_FINANCIAL_SUMMARY

    return Intent.UNKNOWN


def detect_query_metric(text):
    s = text.lower()
    if 'pengeluaran terbesar' in s or 'biaya terbesar' in s:
        return 'pengeluaran_terbesar'
    if 'kategori' in s and ('paling banyak' in s or 'terbesar' in s or 'boros' in s or 'menghabiskan' in s):
        return 'kategori_terbesar'
    if 'saldo' in s or 'sisa kas' in s or 'uang kas' in s:
        return 'saldo'
    if 'arus kas operasional' in s:
        return 'arus_kas_operasional'
    if 'arus kas' in s or 'cash flow' in s or 'cashflow' in s:
        return 'arus_kas_total'
    if 'pemasukan' in s or 'omzet' in s or 'pendapatan' in s or 'penjualan' in s:
        return 'pemasukan'
    if 'pengeluaran' in s or 'biaya' in s or 'belanja' in s:
        return 'pengeluaran'
    return 'ringkasan_umum'


def build_transaction_proposal(text, intent, categories, toko_id=None, base_dt=None):
    tp = 'pemasukan' if intent == Intent.CREATE_INCOME else 'pengeluaran'

    # Extract exact monetary value
    amount = extract_money(text)

    # Resolve transaction date in Asia/Jakarta timezone
    tx_date = resolve_date_indonesian(text, base_dt=base_dt)

    # Clean description
    desc = clean_transaction_description(text)
    if not desc:
        raise ValueError('Keterangan transaksi belum jelas. Mohon sertakan untuk apa transaksi ini dilakukan.')

    # Classify against store categories with verified store business context
    from ai.store_context import get_store_business_context
    b_type = get_store_business_context(toko_id).get('bidang_usaha', 'Umum') if toko_id else 'Umum'
    from ai.ollama_client import LLMUnavailable
    try:
        classification = classify(tp, desc, categories, toko_id=toko_id, business_type=b_type)
    except LLMUnavailable as exc:
        classification = {'kategori_id': None, 'status_kategori': 'perlu_tinjauan', 'skor': None,
                          'usulan': {'ai_pending': True, 'ai_error': str(exc)[:300]}}

    cat_id = classification.get('kategori_id')
    cat_name = 'Others'

    if cat_id:
        matched = next((c for c in categories if c['id'] == cat_id), None)
        if matched:
            cat_name = matched['nama']
    elif not (classification.get('usulan') or {}).get('ai_pending'):
        other_cat = next((c for c in categories if c['tipe'] == tp and c['nama'].casefold() == 'others'), None)
        cat_id = other_cat['id'] if other_cat else None
        cat_name = other_cat['nama'] if other_cat else 'Others'

    proposal_id = str(uuid.uuid4())
    return {
        'id': proposal_id,
        'toko_id': toko_id,
        'tipe': tp,
        'nominal': str(amount),
        'nominal_decimal': amount,
        'tanggal': tx_date.isoformat(),
        'deskripsi': desc,
        'kategori_id': cat_id,
        'kategori_nama': cat_name,
        'status_kategori': classification.get('status_kategori', 'perlu_tinjauan'),
        'skor': classification.get('skor'),
        'usulan': classification.get('usulan', {}),
        'status': 'pending_confirmation',
        'result': None
    }
