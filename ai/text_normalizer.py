import re
from decimal import Decimal

ABBREVIATIONS = {
    'brg': 'barang', 'byr': 'bayar', 'bl': 'beli', 'lstrk': 'listrik',
    'gji': 'gaji', 'ongkir': 'ongkos kirim', 'tgl': 'tanggal'
}
MONEY_RE = re.compile(r'(?<!\w)(?:rp\.?\s*)?(\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*(juta|jt|ribu|rb|k)?\b', re.I)

def normalize(text):
    text = str(text or '').lower().strip()
    text = re.sub(r'(?<!\w)(?:rp\.?\s*)?\d[\d.,]*\s*(?:juta|jt|ribu|rb|k)?\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(ABBREVIATIONS.get(word, word) for word in text.split())

def extract_money(text):
    s = str(text or '').lower()
    matches = []
    for m in MONEY_RE.finditer(s):
        raw, suffix = m.group(1), (m.group(2) or '').lower()
        prefix = s[max(0, m.start()-3):m.start()]
        explicit = bool(suffix) or bool(re.match(r'\s*rp\.?\s*', m.group(0))) or bool(re.search(r'rp\.?\s*$', prefix))
        if not explicit:
            continue
        if suffix:
            factor = 1_000_000 if suffix in ('juta', 'jt') else 1000
            if '.' in raw and ',' in raw:
                if raw.find('.') < raw.find(','):
                    whole = raw.replace('.', '').replace(',', '.')
                else:
                    whole = raw.replace(',', '')
            elif re.search(r'[.,]\d{1,2}$', raw):
                whole = raw.replace(',', '.')
            elif re.search(r'[.,]\d{3}$', raw):
                whole = raw.replace('.', '').replace(',', '')
            else:
                whole = raw.replace(',', '.')
        else:
            factor = 1
            if '.' in raw and ',' in raw:
                if raw.find('.') < raw.find(','):
                    whole = raw.replace('.', '').replace(',', '.')
                else:
                    whole = raw.replace(',', '')
            elif re.search(r'[.,]\d{3}(?:[.,]\d{3})*$', raw):
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
