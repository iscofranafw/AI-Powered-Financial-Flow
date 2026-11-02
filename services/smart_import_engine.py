import logging
import csv
import hashlib
import json
from collections import Counter
import io
import re
import uuid
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai.text_normalizer import clean_text_for_storage
from database.client import privileged_client
from services.auth_service import verify_actor
from services.finance_service import _validate_category
from services.validation_service import (
    MAX_AMOUNT,
    VALID_TRANSACTION_TYPES,
    parse_monetary_value,
    validate_amount,
    validate_date,
    validate_description,
    validate_type,
)

logger = logging.getLogger(__name__)

# Supported encodings in probing priority
SUPPORTED_ENCODINGS = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1']

# Standard delimiters to probe
SUPPORTED_DELIMITERS = [',', ';', '\t', '|']

# Header aliases for automatic matching
HEADER_ALIASES: Dict[str, List[str]] = {
    'tanggal': [
        'tanggal', 'date', 'transaction_date', 'trans_date', 'tgl', 'dt',
        'waktu_transaksi', 'tgl_transaksi', 'waktu'
    ],
    'deskripsi': [
        'deskripsi', 'description', 'keterangan', 'uraian', 'item', 'nama_barang',
        'rincian', 'transaksi', 'detail', 'keperluan', 'kegiatan'
    ],
    'nominal': [
        'nominal', 'amount', 'total', 'nilai', 'harga', 'jumlah', 'debit',
        'kredit', 'pengeluaran', 'pemasukan', 'biaya', 'subtotal', 'grand_total'
    ],
    'tipe': [
        'tipe', 'type', 'jenis', 'jenis_transaksi', 'transaction_type', 'arah',
        'kategori_arus', 'status_tipe'
    ],
    'source_id': [
        'no transaction', 'no_transaksi', 'transaction_id', 'transaction_number',
        'id', 'id_transaksi', 'no_ref', 'reference', 'invoice', 'nomor'
    ],
    'staff': [
        'staff', 'cashier', 'kasir', 'pegawai', 'operator', 'user', 'pencatat',
        'created_by'
    ],
    'time': [
        'time', 'jam', 'waktu', 'transaction_time', 'trans_time', 'pukul'
    ],
    'notes': [
        'notes', 'catatan', 'keterangan_tambahan', 'memo', 'remark', 'note',
        'komentar'
    ],
    'kategori': [
        'kategori', 'category', 'pos', 'rubrik', 'nama_kategori'
    ],
}

KNOWN_TYPE_VALUES = {
    'pemasukan': {'pemasukan', 'income', 'masuk', 'in', 'cr', 'kredit', 'credit'},
    'pengeluaran': {'pengeluaran', 'expense', 'keluar', 'out', 'dr', 'debit'},
}


def detect_csv_dialect(content: bytes) -> Tuple[str, str, str]:
    """Detect encoding, delimiter, and return decoded text sample."""
    if len(content) > 25_000_000:
        raise ValueError('Ukuran berkas CSV maksimal 25 MB')

    decoded_text = None
    selected_encoding = None

    for enc in SUPPORTED_ENCODINGS:
        try:
            decoded_text = content.decode(enc)
            selected_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None or selected_encoding is None:
        raise ValueError('Format encoding CSV tidak didukung. Harap simpan berkas dengan encoding UTF-8.')

    # Delimiter detection using first non-empty lines
    lines = [ln for ln in decoded_text.splitlines() if ln.strip()][:20]
    if not lines:
        raise ValueError('Berkas CSV kosong')

    best_delimiter = ','
    best_count = -1

    for delim in SUPPORTED_DELIMITERS:
        counts = [ln.count(delim) for ln in lines]
        # Valid delimiter should appear consistently and more than 0 times
        if counts and all(c > 0 for c in counts) and len(set(counts)) == 1:
            if counts[0] > best_count:
                best_count = counts[0]
                best_delimiter = delim

    if best_count <= 0:
        # Fallback to csv.Sniffer on first 5KB
        sample = '\n'.join(lines[:10])
        try:
            sniffed = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            best_delimiter = sniffed.delimiter
        except Exception as exc:
            logger.warning('CSV delimiter detection failed; using comma: %s', exc)
            best_delimiter = ','

    return selected_encoding, best_delimiter, decoded_text


def normalize_amount(raw_val: Any) -> Decimal:
    """Safely convert Indonesian or international monetary representation to Decimal."""
    return parse_monetary_value(raw_val)


def normalize_date(raw_val: Any) -> str:
    """Normalize date strings in ISO, Indonesian, or datetime format to YYYY-MM-DD."""
    if raw_val is None:
        raise ValueError('Tanggal kosong')

    if isinstance(raw_val, (date, datetime)):
        dt = raw_val.date() if isinstance(raw_val, datetime) else raw_val
        return validate_date(dt)

    s = str(raw_val).strip()
    if not s:
        raise ValueError('Tanggal kosong')

    # Strip time component if present
    s_date = re.split(r'[\sT]', s)[0].strip()

    formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%Y/%m/%d',
        '%d.%m.%Y',
        '%Y.%m.%d',
    ]

    parsed_dt = None
    for fmt in formats:
        try:
            parsed_dt = datetime.strptime(s_date, fmt).date()
            break
        except ValueError:
            continue

    if parsed_dt is None:
        raise ValueError(f'Format tanggal tidak dikenali: {raw_val}')

    return validate_date(parsed_dt)


def normalize_description(raw_val: Any, notes_val: Any = None) -> str:
    """Clean and optionally combine description with meaningful notes."""
    if raw_val is None:
        raise ValueError('Deskripsi tidak valid')
    s = clean_text_for_storage(str(raw_val) or '')
    s = s.lstrip('.').strip()
    if not s or s.lower() == 'nan':
        raise ValueError('Deskripsi tidak valid')

    if notes_val:
        note_clean = clean_text_for_storage(str(notes_val) or '').strip()
        # Discard stray punctuation or single letter noise
        if note_clean and len(note_clean) > 1 and note_clean not in (',', ';', '.', '-') and note_clean.lower() != 'nan':
            if note_clean.lower() not in s.lower():
                s = f'{s} ({note_clean})'

    return validate_description(s)


class CsvSchemaEngine:
    """Smart CSV schema inspection, value profiling, and automatic column mapping."""

    def __init__(self, sample_df: pd.DataFrame):
        self.df = sample_df
        self.headers = [str(c).strip() for c in sample_df.columns]

    def _match_header_alias(self, field_name: str) -> Optional[str]:
        aliases = HEADER_ALIASES.get(field_name, [])
        for h in self.headers:
            clean_h = re.sub(r'[\s_\-]+', ' ', h.lower()).strip()
            if clean_h in aliases or h.lower() in aliases:
                return h
        return None

    def _inspect_type_column_semantics(self, col: str) -> bool:
        """Return True if column actually contains transaction direction keywords."""
        sample_vals = [
            str(v).strip().lower() for v in self.df[col].dropna().head(30)
            if str(v).strip()
        ]
        if not sample_vals:
            return False

        all_type_keywords = KNOWN_TYPE_VALUES['pemasukan'].union(KNOWN_TYPE_VALUES['pengeluaran'])
        match_count = sum(1 for v in sample_vals if v in all_type_keywords)
        # If over 60% of non-empty sample values match income/expense keywords, it is a direction column
        return (match_count / len(sample_vals)) >= 0.6

    def _inspect_id_prefix_direction(self, col: str) -> bool:
        """Return True if column values consistently use I- (income) and E- (expense) prefixes."""
        sample_vals = [
            str(v).strip().upper() for v in self.df[col].dropna().head(30)
            if str(v).strip()
        ]
        if not sample_vals or len(sample_vals) < 3:
            return False

        has_i_or_e = [v for v in sample_vals if v.startswith('I-') or v.startswith('E-')]
        return len(has_i_or_e) == len(sample_vals)

    def analyze_schema(self) -> Dict[str, Any]:
        """Perform header matching, value profiling, and generate mapping recommendations."""
        mappings: Dict[str, Optional[str]] = {
            'tanggal': None,
            'deskripsi': None,
            'nominal': None,
            'tipe': None,
            'source_id': None,
            'staff': None,
            'time': None,
            'notes': None,
            'kategori': None,
        }
        confidence: Dict[str, str] = {}
        warnings: List[str] = []
        type_inference_mode = 'column'

        # 1. Match source_id
        source_id_col = self._match_header_alias('source_id')
        if source_id_col:
            mappings['source_id'] = source_id_col
            confidence['source_id'] = 'tinggi'

        # 2. Match staff & time & notes & kategori
        for opt_fld in ('staff', 'time', 'notes', 'kategori'):
            col = self._match_header_alias(opt_fld)
            if col and col not in mappings.values():
                mappings[opt_fld] = col
                confidence[opt_fld] = 'tinggi'

        # 3. Match tanggal
        date_col = self._match_header_alias('tanggal')
        if date_col:
            mappings['tanggal'] = date_col
            confidence['tanggal'] = 'tinggi'
        else:
            # Value-based fallback for date
            for col in self.headers:
                if col in mappings.values():
                    continue
                valid_count = 0
                sample = self.df[col].dropna().head(10)
                for val in sample:
                    try:
                        normalize_date(val)
                        valid_count += 1
                    except Exception as exc:
                        logger.warning('CSV date sample validation failed; excluding sample from schema confidence: %s', exc)
                        continue
                if len(sample) > 0 and valid_count >= len(sample) * 0.7:
                    mappings['tanggal'] = col
                    confidence['tanggal'] = 'sedang'
                    break

        # 4. Match nominal
        amount_col = self._match_header_alias('nominal')
        if amount_col:
            mappings['nominal'] = amount_col
            confidence['nominal'] = 'tinggi'
        else:
            # Value-based fallback for nominal
            for col in self.headers:
                if col in mappings.values():
                    continue
                valid_count = 0
                sample = self.df[col].dropna().head(10)
                for val in sample:
                    try:
                        normalize_amount(val)
                        valid_count += 1
                    except Exception as exc:
                        logger.warning('CSV monetary sample validation failed; excluding sample from schema confidence: %s', exc)
                        continue
                if len(sample) > 0 and valid_count >= len(sample) * 0.7:
                    mappings['nominal'] = col
                    confidence['nominal'] = 'sedang'
                    break

        # 5. Match deskripsi
        desc_col = self._match_header_alias('deskripsi')
        if desc_col:
            mappings['deskripsi'] = desc_col
            confidence['deskripsi'] = 'tinggi'

        # 6. Match tipe (check if 'type' column is actually a description when no explicit description column exists)
        type_candidate = self._match_header_alias('tipe')
        if type_candidate:
            if not desc_col and not self._inspect_type_column_semantics(type_candidate):
                # 'type' is not income/expense and there is no dedicated description column: it is an item description
                warnings.append(
                    f"Kolom '{type_candidate}' tidak berisi tipe pemasukan/pengeluaran, "
                    f"melainkan nama/keterangan barang. Dipetakan sebagai deskripsi transaksi."
                )
                mappings['deskripsi'] = type_candidate
                confidence['deskripsi'] = 'tinggi'
            else:
                mappings['tipe'] = type_candidate
                confidence['tipe'] = 'tinggi'

        # If deskripsi still not mapped, pick first unmapped non-numeric column
        if not mappings['deskripsi']:
            for col in self.headers:
                if col not in mappings.values():
                    mappings['deskripsi'] = col
                    confidence['deskripsi'] = 'sedang'
                    break

        # 7. Check I- / E- transaction ID prefix derivation for 'tipe'
        if not mappings['tipe'] and mappings['source_id']:
            if self._inspect_id_prefix_direction(mappings['source_id']):
                type_inference_mode = 'id_prefix'
                confidence['tipe'] = 'tinggi'
                warnings.append(
                    f"Jenis transaksi otomatis disimpulkan dari kode transaksi pada kolom '{mappings['source_id']}' "
                    f"(I- = pemasukan, E- = pengeluaran)."
                )

        if not mappings['tipe'] and type_inference_mode != 'id_prefix':
            # Check if auto_detect by description classifier is needed or default pengeluaran
            type_inference_mode = 'auto_detect'
            confidence['tipe'] = 'ambigu'
            warnings.append(
                "Jenis transaksi tidak ditemukan pada kolom khusus. "
                "Sistem akan mengklasifikasikan otomatis atau Anda dapat memilih jenis transaksi."
            )

        return {
            'headers': self.headers,
            'mappings': mappings,
            'confidence': confidence,
            'type_inference_mode': type_inference_mode,
            'warnings': warnings,
        }


def parse_and_validate_dataframe(
    df: pd.DataFrame,
    mappings: Dict[str, Optional[str]],
    type_inference_mode: str = 'column',
    default_type: str = 'pengeluaran',
    append_notes: bool = True,
    start_row_index: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Transform raw DataFrame rows into normalized records and rejected row reports."""
    date_col = mappings.get('tanggal')
    desc_col = mappings.get('deskripsi')
    amount_col = mappings.get('nominal')
    type_col = mappings.get('tipe')
    notes_col = mappings.get('notes')
    id_col = mappings.get('source_id')
    staff_col = mappings.get('staff')
    time_col = mappings.get('time')

    if not date_col or not desc_col or not amount_col:
        raise ValueError('Pemetaan kolom wajib: tanggal, deskripsi, dan nominal harus ditentukan')

    valid_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    for idx, raw_record in enumerate(df.to_dict(orient='records'), start=start_row_index):
        raw_date = raw_record.get(date_col)
        raw_desc = raw_record.get(desc_col)
        raw_amount = raw_record.get(amount_col)
        raw_type = raw_record.get(type_col) if type_col else None
        raw_notes = raw_record.get(notes_col) if (notes_col and append_notes) else None
        source_id = str(raw_record.get(id_col) or '').strip() if id_col else ''
        staff = str(raw_record.get(staff_col) or '').strip() if staff_col else ''
        tx_time = str(raw_record.get(time_col) or '').strip() if time_col else ''

        # 1. Validate date
        try:
            norm_date = normalize_date(raw_date)
        except Exception as exc:
            rejected_rows.append({
                'baris': idx,
                'alasan': f'Tanggal tidak valid: {exc}',
                'kolom': 'tanggal',
                'nilai': str(raw_date),
            })
            continue

        # 2. Validate amount
        try:
            norm_amount_dec = normalize_amount(raw_amount)
            norm_amount_str = format(norm_amount_dec, 'f')
        except Exception as exc:
            rejected_rows.append({
                'baris': idx,
                'alasan': f'Nominal tidak valid: {exc}',
                'kolom': 'nominal',
                'nilai': str(raw_amount),
            })
            continue

        # 3. Validate description
        try:
            norm_desc = normalize_description(raw_desc, raw_notes)
        except Exception as exc:
            rejected_rows.append({
                'baris': idx,
                'alasan': f'Deskripsi tidak valid: {exc}',
                'kolom': 'deskripsi',
                'nilai': str(raw_desc),
            })
            continue

        # 4. Resolve transaction type
        resolved_type = None
        if type_col and raw_type is not None and str(raw_type).strip():
            t_str = str(raw_type).strip().lower()
            if t_str in KNOWN_TYPE_VALUES['pemasukan']:
                resolved_type = 'pemasukan'
            elif t_str in KNOWN_TYPE_VALUES['pengeluaran']:
                resolved_type = 'pengeluaran'
            else:
                rejected_rows.append({
                    'baris': idx,
                    'alasan': 'Jenis transaksi tidak valid',
                    'kolom': 'tipe',
                    'nilai': str(raw_type),
                })
                continue
        elif type_inference_mode == 'id_prefix' and source_id:
            if source_id.upper().startswith('I-'):
                resolved_type = 'pemasukan'
            elif source_id.upper().startswith('E-'):
                resolved_type = 'pengeluaran'

        if not resolved_type:
            resolved_type = default_type if default_type in VALID_TRANSACTION_TYPES else 'pengeluaran'

        metadata: Dict[str, Any] = {}
        if source_id:
            metadata['source_id'] = source_id
        if staff:
            metadata['source_staff'] = staff
        if tx_time:
            metadata['source_time'] = tx_time
        if raw_notes:
            metadata['original_notes'] = str(raw_notes).strip()

        # Deterministic canonical fingerprint for duplicate detection
        canonical_amt = format(Decimal(norm_amount_str), '.2f')
        fingerprint = f"{norm_date}|{norm_desc.lower()}|{canonical_amt}|{resolved_type}"

        valid_rows.append({
            'baris': idx,
            'tanggal': norm_date,
            'deskripsi': norm_desc,
            'nominal': norm_amount_str,
            'tipe': resolved_type,
            'usulan': metadata,
            'fingerprint': fingerprint,
            'source_id': source_id,
            'error': '',
        })

    return valid_rows, rejected_rows


def _source_id(row):
    return str(row.get('source_id') or (row.get('usulan') or {}).get('source_id') or '').strip().lower()


def _fingerprint(row):
    return '|'.join((validate_date(row['tanggal']),
                     clean_text_for_storage(row['deskripsi']).lower(),
                     format(Decimal(str(validate_amount(row['nominal']))), '.2f'),
                     validate_type(row['tipe'])))


def prepare_import_rows(shop_id, rows, import_id=None):
    """Assign retry-stable identities before filtering or splitting into batches.

    Without an explicit key, identical datasets represent retries. Supply a new
    import_id for a distinct import of identical source-less transactions. Keep
    returned row metadata when retrying only the failed tail of an import.
    """
    fingerprints = [_fingerprint(row) for row in rows]
    identities = sorted((_source_id(row), fp) for row, fp in zip(rows, fingerprints))
    key = str(import_id) if import_id is not None else hashlib.sha256(
        json.dumps(identities, ensure_ascii=False).encode('utf-8')).hexdigest()
    prior_keys = {(row.get('usulan') or {}).get('import_id') for row in rows}
    if import_id is None and len(prior_keys) == 1 and None not in prior_keys:
        key = prior_keys.pop()
    occurrences = Counter()
    prepared = []
    for row, fp in zip(rows, fingerprints):
        occurrences[fp] += 1
        meta = dict(row.get('usulan') or {})
        sid = _source_id(row)
        if sid:
            meta['source_id'] = sid
        if not meta.get('import_id') or (import_id is not None and meta['import_id'] != key):
            meta.update(import_id=key, import_occurrence=occurrences[fp])
        meta['fingerprint'] = fp
        prepared.append({**row, 'source_id': sid, 'fingerprint': fp, 'usulan': meta})
    fresh, duplicates = detect_duplicates(shop_id, prepared, write_layer=True)
    return fresh, duplicates, key


def detect_duplicates(
    shop_id: str,
    valid_rows: List[Dict[str, Any]],
    existing_transactions: Optional[List[Dict[str, Any]]] = None,
    *, write_layer: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate new transactions from intra-file and existing database duplicates.
    
    Uses paginated transaction retrieval to support stores with >1,000 historical transactions,
    respects explicit source_id identities, and applies multiset frequency matching
    so legitimate repeated transactions are preserved while re-imports are prevented.
    """
    if existing_transactions is None:
        try:
            from repositories.finance_repository import transactions as fetch_shop_transactions
            existing_transactions = fetch_shop_transactions(shop_id)
        except Exception as exc:
            logger.warning('Duplicate preflight unavailable for shop %s; using empty advisory history and database write-layer protection: %s', shop_id, exc)
            existing_transactions = []

    db_import_keys = set()
    db_source_ids = set()
    db_fp_counts = Counter()

    for tx in existing_transactions:
        meta = tx.get('usulan') or {}
        if isinstance(meta, dict) and meta.get('import_id'):
            db_import_keys.add((meta['import_id'], meta.get('fingerprint'), meta.get('import_occurrence')))
        if isinstance(meta, dict) and meta.get('source_id'):
            db_source_ids.add(str(meta['source_id']).strip().lower())

        d = str(tx.get('tanggal') or '')
        desc = clean_text_for_storage(tx.get('deskripsi') or '').lower()
        amt = format(Decimal(str(tx.get('nominal') or 0)), '.2f')
        tp = str(tx.get('tipe') or '').lower()
        fp = f"{d}|{desc}|{amt}|{tp}"
        db_fp_counts[fp] += 1

    seen_file_source_ids = set()
    file_fp_counts = Counter()

    new_rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []

    for row in valid_rows:
        sid = _source_id(row)
        d = str(row.get('tanggal') or '')
        desc = clean_text_for_storage(row.get('deskripsi') or '').lower()
        amt = format(Decimal(str(row.get('nominal') or 0)), '.2f') if row.get('nominal') else ''
        tp = str(row.get('tipe') or '').lower()
        fp = f"{d}|{desc}|{amt}|{tp}" if (d and desc and amt) else row.get('fingerprint', '')

        is_dup = False
        dup_reason = ''

        if sid:
            if sid in db_source_ids:
                is_dup = True
                dup_reason = f"ID transaksi '{row.get('source_id') or sid}' sudah tersimpan di database toko"
            elif sid in seen_file_source_ids:
                is_dup = True
                dup_reason = f"ID transaksi '{row.get('source_id') or sid}' muncul lebih dari satu kali di dalam berkas"
            else:
                seen_file_source_ids.add(sid)
                file_fp_counts[fp] += 1
        elif write_layer:
            meta = row['usulan']
            identity = (meta['import_id'], meta['fingerprint'], meta['import_occurrence'])
            is_dup = identity in db_import_keys
            db_import_keys.add(identity)
            dup_reason = 'Identitas baris impor sudah tersimpan'
        else:
            file_fp_counts[fp] += 1
            if file_fp_counts[fp] <= db_fp_counts.get(fp, 0):
                is_dup = True
                dup_reason = "Tanggal, deskripsi, nominal, dan jenis transaksi sama persis dengan data di database"

        if is_dup:
            duplicate_rows.append({**row, 'duplikat': True, 'alasan_duplikat': dup_reason})
        else:
            new_rows.append(row)

    return new_rows, duplicate_rows


def execute_scalable_import(
    token: str,
    rows: List[Dict[str, Any]],
    batch_size: int = 500,
    progress_callback: Optional[Any] = None,
    import_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform resource-aware chunked database import using atomic RPC batches."""
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    shop_id = actor['toko_id']
    role = actor.get('role', 'pemilik_toko')

    submitted_count = len(rows)
    rows, duplicates, import_id = prepare_import_rows(shop_id, rows, import_id)
    duplicate_count = len(duplicates)
    processed_count = 0

    # New production saves use their UUID verbatim. Legacy direct callers can
    # still supply content hashes/labels as metadata keys; p_batch is a SQL UUID.
    try:
        batch_uuid = str(uuid.UUID(str(import_id)))
    except (ValueError, TypeError):
        batch_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(import_id)))

    # Atomic DB function accepts between 1 and 1000 items
    safe_batch_size = max(1, min(batch_size, 1000))
    total_rows = len(rows)
    total_imported = 0
    batch_ids: List[str] = []
    failed_batches: List[Dict[str, Any]] = []

    client = privileged_client()

    # Audit count before import
    count_before = 0
    try:
        cnt_res = client.table('transaksi').select('id', count='exact').eq('toko_id', shop_id).execute()
        count_before = cnt_res.count if hasattr(cnt_res, 'count') and cnt_res.count is not None else 0
    except Exception as exc:
        logger.warning('Pre-import count unavailable for shop %s; using advisory count 0: %s', shop_id, exc)
        count_before = 0

    # Pre-fetch store categories once in-memory to eliminate per-row network roundtrips
    try:
        from repositories.finance_repository import categories
        raw_cats = categories(shop_id)
        store_cats = {str(c['id']): c for c in raw_cats}
    except Exception as exc:
        logger.warning('Category prefetch failed for shop %s; using per-row category validation: %s', shop_id, exc)
        store_cats = {}

    for start_idx in range(0, total_rows, safe_batch_size):
        chunk = rows[start_idx:start_idx + safe_batch_size]

        try:
            batch_payload = []
            for i, r in enumerate(chunk, 1):
                tp = validate_type(r['tipe'])
                cat_id_str = str(r.get('kategori_id') or '')
                if not cat_id_str and (r.get('usulan') or {}).get('ai_pending') is True:
                    cat_id_str = None
                elif store_cats:
                    cat = store_cats.get(cat_id_str)
                    if not cat or cat.get('tipe') != tp:
                        raise ValueError(f"Kategori baris {start_idx + i} tidak sesuai jenis transaksi/toko")
                else:
                    _validate_category(shop_id, cat_id_str, tp)
                desc = clean_text_for_storage(r.get('deskripsi', ''))
                batch_payload.append({
                    'tanggal': validate_date(r['tanggal']),
                    'deskripsi': desc,
                    'nominal': validate_amount(r['nominal']),
                    'tipe': tp,
                    'kategori_id': cat_id_str,
                    'status_kategori': r.get('status_kategori', 'dikonfirmasi'),
                    'skor': r.get('skor'),
                    'usulan': r.get('usulan') or {},
                })

            res = client.rpc('import_transactions_atomic', {
                'p_actor': actor['id'],
                'p_shop': shop_id,
                'p_rows': batch_payload,
                'p_batch': batch_uuid,
            }).execute().data

            total_imported += res if res is not None else 0
            duplicate_count += max(0, len(batch_payload) - (res or 0))
            processed_count += len(chunk)
            batch_ids.append(batch_uuid)

        except Exception as exc:
            failed_batches.append({
                'batch_index': len(batch_ids) + 1,
                'start_row': start_idx + 1,
                'end_row': min(start_idx + safe_batch_size, total_rows),
                'error': str(exc),
                'failed_rows': chunk,
            })
            # Halt subsequent batches on database level constraint violation to avoid cascading corruption
            break

        if progress_callback:
            try:
                progress_callback(min(1.0, (start_idx + len(chunk)) / total_rows), len(batch_ids))
            except TypeError:
                progress_callback(min(1.0, (start_idx + len(chunk)) / total_rows))

    cleanup_warnings = []
    # Invalidate forecasting cache if approved expenses were imported by shop owner
    if total_imported > 0 and role == 'pemilik_toko':
        try:
            from services.forecast_service import invalidate_forecast_cache
            invalidate_forecast_cache(shop_id)
        except Exception as exc:
            logger.warning('Forecast cache invalidation failed after import for shop %s; keeping committed count %s: %s', shop_id, total_imported, exc)
            cleanup_warnings.append('Forecast cache invalidation failed; imported rows remain committed.')

    # Audit count after import
    count_after = count_before + total_imported
    try:
        cnt_res2 = client.table('transaksi').select('id', count='exact').eq('toko_id', shop_id).execute()
        if hasattr(cnt_res2, 'count') and cnt_res2.count is not None:
            count_after = cnt_res2.count
    except Exception as exc:
        logger.warning('Post-import count unavailable for shop %s; using estimated count: %s', shop_id, exc)
        count_after = count_before + total_imported

    all_failed_rows = rows[processed_count:] if failed_batches else []
    failed_count = len(all_failed_rows)
    completion_status = 'completed' if not failed_batches else ('partial_failure' if total_imported > 0 else 'failed')

    return {
        'total_imported': total_imported,
        'total_submitted': submitted_count,
        'attempted_count': submitted_count,
        'duplicate_count': duplicate_count,
        'inserted_count': total_imported,
        'failed_count': failed_count,
        'failed_rows': all_failed_rows,
        'import_id': import_id,
        'completion_status': completion_status,
        'batches_completed': len(batch_ids),
        'batch_ids': batch_ids,
        'failed_batches': failed_batches,
        'count_before': count_before,
        'count_after': count_after,
        **({'warnings': cleanup_warnings} if cleanup_warnings else {}),
    }
