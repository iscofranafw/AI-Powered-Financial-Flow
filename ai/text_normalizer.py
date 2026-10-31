import re
from decimal import Decimal

ABBREVIATIONS = {
    'bl': 'beli',
    'bli': 'beli',
    'byr': 'bayar',
    'dibyr': 'dibayar',
    'brg': 'barang',
    'bhn': 'bahan',
    'bku': 'baku',
    'gji': 'gaji',
    'lstrk': 'listrik',
    'ongkir': 'ongkos kirim',
    'ongk': 'ongkos',
    'tgl': 'tanggal',
    'trf': 'transfer',
    'tf': 'transfer',
    'pmskn': 'pemasukan',
    'pnglrn': 'pengeluaran',
    'pjl': 'penjualan',
    'penj': 'penjualan',
    'pemb': 'pembelian',
    'pbl': 'pembelian',
    'ops': 'operasional',
    'oprasional': 'operasional',
    'pmbyrn': 'pembayaran',
    'pgw': 'pegawai',
    'krywn': 'karyawan',
    'mdl': 'modal',
    'pdptn': 'pendapatan',
    'sup': 'supplier',
    'supl': 'supplier',
    'cust': 'pelanggan',
    'plggn': 'pelanggan',
    'wrg': 'warung',
    'tk': 'toko',
    'bkl': 'bengkel',
    'srv': 'servis',
    'serv': 'servis',
    'spart': 'suku cadang',
    'sparepart': 'suku cadang',
    'kmps': 'kampas',
    'adm': 'administrasi',
    'bbm': 'bensin',
    'omset': 'omzet',
}

MONEY_RE = re.compile(
    r'(?<!\w)(?:rp\.?\s*)?(\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*(juta|jt|ribu|rb|k|rupiah)?\b',
    re.I
)

CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff]')


def clean_text_for_storage(text):
    if text is None:
        return ''
    cleaned = CONTROL_CHARS_RE.sub('', str(text))
    return ' '.join(cleaned.split())


def normalize(text):
    text = str(text or '').lower().strip()
    text = re.sub(r'(?<!\w)(?:rp\.?\s*)?\d[\d.,]*\s*(?:juta|jt|ribu|rb|k|rupiah)?\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(ABBREVIATIONS.get(word, word) for word in text.split())


def extract_money(text):
    s = str(text or '').lower()
    matches = []
    for m in MONEY_RE.finditer(s):
        raw = m.group(1)
        suffix = (m.group(2) or '').lower()
        prefix = s[max(0, m.start() - 3):m.start()]
        explicit = bool(suffix) or bool(re.match(r'\s*rp\.?\s*', m.group(0))) or bool(re.search(r'rp\.?\s*$', prefix))
        if not explicit:
            continue
        if suffix in ('juta', 'jt'):
            factor = 1_000_000
        elif suffix in ('ribu', 'rb', 'k'):
            factor = 1000
        else:
            factor = 1

        # Disambiguate European vs Indonesian decimal and thousands separators
        if '.' in raw and ',' in raw:
            if raw.find('.') < raw.find(','):
                whole = raw.replace('.', '').replace(',', '.')
            else:
                whole = raw.replace(',', '')
        elif suffix and re.search(r'[.,]\d{1,2}$', raw):
            whole = raw.replace(',', '.')
        elif re.search(r'[.,]\d{3}$', raw) and not suffix:
            whole = raw.replace('.', '').replace(',', '')
        else:
            whole = raw.replace(',', '.')

        try:
            matches.append(Decimal(whole) * factor)
        except Exception:
            continue

    if len(matches) != 1 or matches[0] <= 0:
        raise ValueError('Nominal ambigu. Gunakan misalnya Rp250000 atau 250rb (satu nominal).')
    return matches[0]
