import re
import warnings
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, InvalidOperation

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from ai.text_normalizer import clean_text_for_storage

MAX_AMOUNT = Decimal('9999999999999999.99')
VALID_TRANSACTION_TYPES = {'pemasukan', 'pengeluaran'}
JAKARTA_TIMEZONE = ZoneInfo('Asia/Jakarta') if ZoneInfo else timezone(timedelta(hours=7))


def get_jakarta_today() -> date:
    """Return current calendar date in Asia/Jakarta (WIB, UTC+7) timezone."""
    try:
        if ZoneInfo:
            return datetime.now(ZoneInfo('Asia/Jakarta')).date()
    except Exception:
        # Safe fallback: if IANA tzdata is missing for Asia/Jakarta, use fixed UTC+7 timezone
        pass
    return datetime.now(timezone(timedelta(hours=7))).date()


def parse_monetary_value(value):
    """Parse monetary representation to Decimal, standardizing Indonesian and international notations."""
    if value is None:
        raise ValueError('Nominal tidak valid')
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        dec = Decimal(str(value))
    else:
        raw_s = str(value).strip()
        if not raw_s:
            raise ValueError('Nominal tidak valid')
        if re.match(r'^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$', raw_s):
            try:
                dec = Decimal(raw_s)
            except Exception:
                raise ValueError('Nominal tidak valid')
        else:
            has_currency_prefix = bool(re.search(r'^(?:rp|idr)\.?\s*', raw_s, flags=re.IGNORECASE))
            s = re.sub(r'^(?:rp|idr)\.?\s*', '', raw_s, flags=re.IGNORECASE).strip()
            s = s.replace(' ', '').replace('+', '')
            if s.startswith('-') or s.startswith('('):
                raise ValueError('Nominal harus positif (bukan minus)')
            if '.' in s and ',' in s:
                dot_idx = s.rfind('.')
                comma_idx = s.rfind(',')
                if dot_idx < comma_idx:
                    s = s.replace('.', '').replace(',', '.')
                else:
                    s = s.replace(',', '')
            elif ',' in s:
                parts = s.split(',')
                if len(parts) == 2 and len(parts[1]) in (1, 2):
                    s = s.replace(',', '.')
                else:
                    s = s.replace(',', '')
            elif '.' in s:
                parts = s.split('.')
                if len(parts) > 2:
                    s = s.replace('.', '')
                elif len(parts) == 2:
                    if has_currency_prefix:
                        s = s.replace('.', '')
                    elif len(parts[1]) == 3 and parts[1] in ('000', '500', '250', '750', '100', '200', '300', '400', '600', '700', '800', '900'):
                        s = s.replace('.', '')
                    elif len(parts[1]) in (1, 2):
                        pass
                    # If single dot, no prefix, and len(parts[1]) > 2 (e.g. 1.999, 100.123):
                    # Keep as decimal, which will trigger exponent < -2 validation error below.
            try:
                dec = Decimal(s)
            except (InvalidOperation, ValueError):
                raise ValueError('Nominal tidak valid')

    if not dec.is_finite() or dec <= 0 or dec > MAX_AMOUNT or dec.as_tuple().exponent < -2:
        raise ValueError('Nominal harus positif, maksimal 2 desimal')
    return dec


def validate_date(value, is_scheduled: bool = False):
    """Validate date value, enforcing calendar bounds and disallowing future actual transactions.

    Args:
        value: Date/datetime object or ISO formatted date string.
        is_scheduled: Flag indicating whether this is a planned/scheduled entry.
                      If False (default), dates beyond today (Asia/Jakarta) are rejected.
                      If True, future dates within the valid calendar range emit a warning.
    """
    if value is None:
        raise ValueError('Tanggal tidak valid')
    if isinstance(value, datetime):
        parsed_date = value.date()
    elif isinstance(value, date):
        parsed_date = value
    else:
        try:
            parsed_date = date.fromisoformat(str(value).strip())
        except (ValueError, TypeError):
            raise ValueError('Tanggal tidak valid')

    # Bounded calendar range prevents corrupted or non-standard historical entries
    if parsed_date.year < 2000 or parsed_date.year > 2099:
        raise ValueError('Tahun transaksi harus antara 2000 dan 2099')

    today_wib = get_jakarta_today()
    if parsed_date > today_wib:
        if not is_scheduled:
            raise ValueError(
                f'Tanggal transaksi ({parsed_date.isoformat()}) tidak boleh melebihi hari ini '
                f'({today_wib.isoformat()} WIB)'
            )
        warnings.warn(
            f'Transaksi bertanggal {parsed_date.isoformat()} dicatat sebagai transaksi terjadwal/rencana di masa depan.',
            UserWarning,
            stacklevel=2,
        )

    return parsed_date.isoformat()


def validate_amount(value):
    n = parse_monetary_value(value)
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


def validate_transaction_payload(tanggal, deskripsi, nominal, tipe, kategori_id=None, is_scheduled: bool = False):
    valid_date = validate_date(tanggal, is_scheduled=is_scheduled)
    valid_type = validate_type(tipe)
    valid_amount = validate_amount(nominal)
    valid_desc = validate_description(deskripsi)

    payload = {
        'tanggal': valid_date,
        'deskripsi': valid_desc,
        'nominal': valid_amount,
        'tipe': valid_type,
    }
    if is_scheduled:
        payload['is_scheduled'] = True
    if kategori_id is not None:
        payload['kategori_id'] = str(kategori_id).strip()
    return payload
