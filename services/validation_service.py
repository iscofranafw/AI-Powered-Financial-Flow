from datetime import date
from decimal import Decimal, InvalidOperation
from ai.text_normalizer import clean_text_for_storage

MAX_AMOUNT = Decimal('9999999999999999.99')
VALID_TRANSACTION_TYPES = {'pemasukan', 'pengeluaran'}


def validate_date(value):
    if value is None:
        raise ValueError('Tanggal tidak valid')
    if isinstance(value, date):
        parsed_date = value
    else:
        try:
            parsed_date = date.fromisoformat(str(value).strip())
        except (ValueError, TypeError):
            raise ValueError('Tanggal tidak valid')

    # Bounded calendar range prevents corrupted or non-standard historical entries
    if parsed_date.year < 2000 or parsed_date.year > 2099:
        raise ValueError('Tahun transaksi harus antara 2000 dan 2099')

    return parsed_date.isoformat()


def validate_amount(value):
    try:
        n = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError('Nominal tidak valid')

    # Exponent check ensures financial precision does not exceed fractional cents
    if not n.is_finite() or n <= 0 or n > MAX_AMOUNT or n.as_tuple().exponent < -2:
        raise ValueError('Nominal harus positif, maksimal 2 desimal')

    return format(n, 'f')


def validate_type(value):
    normalized_type = str(value or '').strip().lower()
    if normalized_type not in VALID_TRANSACTION_TYPES:
        raise ValueError('Jenis transaksi tidak valid')
    return normalized_type


def validate_description(value, max_length=500):
    cleaned = clean_text_for_storage(value)
    if not cleaned or len(cleaned) > max_length:
        raise ValueError(f'Keterangan wajib (maksimal {max_length} karakter)')
    return cleaned


def validate_transaction_payload(tanggal, deskripsi, nominal, tipe, kategori_id=None):
    valid_date = validate_date(tanggal)
    valid_type = validate_type(tipe)
    valid_amount = validate_amount(nominal)
    valid_desc = validate_description(deskripsi)

    payload = {
        'tanggal': valid_date,
        'deskripsi': valid_desc,
        'nominal': valid_amount,
        'tipe': valid_type,
    }
    if kategori_id is not None:
        payload['kategori_id'] = str(kategori_id).strip()
    return payload
