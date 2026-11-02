import csv
import io
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai.text_normalizer import clean_text_for_storage
from database.client import privileged_client
from services.auth_service import verify_actor
from services.finance_service import _date, _money, _type, _validate_category
from services.smart_import_engine import (
    CsvSchemaEngine,
    detect_csv_dialect,
    detect_duplicates,
    execute_scalable_import,
    normalize_amount,
    normalize_date,
    normalize_description,
    parse_and_validate_dataframe,
    prepare_import_rows,
)

MAX_IMPORT_FILE_SIZE = 25_000_000  # 25 MB safeguard
MAX_IMPORT_ROW_LIMIT = 50_000     # 50,000 rows scalable safeguard


def analyze_csv_upload(content: bytes) -> Dict[str, Any]:
    """Inspect uploaded CSV bytes, detect dialect, schema, and preliminary preview."""
    if len(content) > MAX_IMPORT_FILE_SIZE:
        raise ValueError(f'Ukuran berkas melebihi batas maksimal {MAX_IMPORT_FILE_SIZE // 1_000_000} MB')

    encoding, delimiter, text = detect_csv_dialect(content)
    df = pd.read_csv(io.StringIO(text), sep=delimiter, nrows=50, dtype=str, on_bad_lines='skip', keep_default_na=False)
    if df.empty:
        raise ValueError('Berkas CSV kosong')

    engine = CsvSchemaEngine(df)
    schema_info = engine.analyze_schema()

    # Estimate total rows efficiently
    line_count = text.count('\n') + (1 if not text.endswith('\n') else 0) - 1

    return {
        'encoding': encoding,
        'delimiter': delimiter,
        'estimated_rows': max(0, line_count),
        'sample_df': df,
        'schema_analysis': schema_info,
        'raw_text': text,
    }


def preview_csv(content: bytes, max_rows: int = MAX_IMPORT_ROW_LIMIT) -> List[Dict[str, Any]]:
    """Parse and validate CSV content, returning rows with validation status."""
    if len(content) > MAX_IMPORT_FILE_SIZE:
        raise ValueError(f'CSV maksimal {MAX_IMPORT_FILE_SIZE // 1_000_000} MB')

    encoding, delimiter, text = detect_csv_dialect(content)

    # Read up to max_rows + 2 to detect if file exceeds row limit
    df = pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        nrows=max_rows + 2,
        dtype=str,
        keep_default_na=False,
        on_bad_lines='skip',
    )

    if df.empty:
        raise ValueError('CSV kosong')

    if len(df) > max_rows:
        raise ValueError(f'Maksimal {max_rows} baris per impor')

    raw_headers = set(str(c).strip().lower() for c in df.columns)
    has_standard_headers = {'tanggal', 'deskripsi', 'nominal'}.issubset(raw_headers)

    # Use schema engine for mapping
    engine = CsvSchemaEngine(df.head(50))
    analysis = engine.analyze_schema()
    mappings = analysis['mappings']

    # For legacy compliance, if standard headers are not present and cannot map mandatory fields, raise ValueError
    if not has_standard_headers:
        if not mappings.get('tanggal') or not mappings.get('deskripsi') or not mappings.get('nominal'):
            raise ValueError('Kolom wajib: tanggal, deskripsi, nominal; opsional: tipe')

    valid_rows, rejected_rows = parse_and_validate_dataframe(
        df=df,
        mappings=mappings,
        type_inference_mode=analysis['type_inference_mode'],
        default_type='pengeluaran',
        append_notes=True,
        start_row_index=2,
    )

    # Combine valid rows and rejected rows maintaining sequential order
    combined: List[Dict[str, Any]] = []
    rejected_by_baris = {r['baris']: r for r in rejected_rows}
    valid_by_baris = {r['baris']: r for r in valid_rows}

    total_len = len(df)
    for i in range(2, total_len + 2):
        if i in valid_by_baris:
            combined.append(valid_by_baris[i])
        elif i in rejected_by_baris:
            rej = rejected_by_baris[i]
            combined.append({
                'baris': rej['baris'],
                'tanggal': rej['nilai'] if rej['kolom'] == 'tanggal' else '',
                'deskripsi': rej['nilai'] if rej['kolom'] == 'deskripsi' else '',
                'nominal': rej['nilai'] if rej['kolom'] == 'nominal' else '',
                'tipe': '',
                'error': rej['alasan'],
            })

    if not combined:
        raise ValueError('CSV kosong')

    return combined


def enrich_rows(token: str, rows: List[Dict[str, Any]], progress: Optional[Any] = None, categories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Categorize transactions using approved store categories and hybrid classification."""
    from services.ai_service import bulk_classify
    if categories is not None:
        return bulk_classify(token, rows, progress=progress, categories=categories)
    return bulk_classify(token, rows, progress=progress)


class PartialImportError(ValueError):
    """Raised when an import partially commits some batches but fails on subsequent rows."""
    def __init__(self, message: str, result: Dict[str, Any]):
        super().__init__(message)
        self.result = result
        self.inserted_count = result.get('inserted_count', result.get('total_imported', 0))
        self.attempted_count = result.get('attempted_count', result.get('total_submitted', 0))
        self.failed_count = result.get('failed_count', 0)
        self.failed_rows = result.get('failed_rows', [])
        self.import_id = result.get('import_id', '')
        self.completion_status = result.get('completion_status', 'partial_failure')


def execute_structured_import(
    token: str,
    rows: List[Dict[str, Any]],
    duplicate_count: int = 0,
    *,
    import_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a new save, or retry one by reusing its returned import_id."""
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    total_rows = len(rows)
    import_id = str(uuid.uuid4()) if import_id is None else str(import_id)

    if not rows:
        return {
            'attempted_count': 0,
            'inserted_count': 0,
            'duplicate_count': duplicate_count,
            'failed_count': 0,
            'failed_rows': [],
            'import_id': import_id,
            'completion_status': 'completed',
        }

    if len(rows) > 1000:
        res = execute_scalable_import(token, rows, batch_size=500, import_id=import_id)
        res['duplicate_count'] = res.get('duplicate_count', 0) + duplicate_count
        return res

    try:
        inserted = atomic_import(token, rows, import_id=import_id)
        return {
            'attempted_count': total_rows,
            'inserted_count': inserted,
            'duplicate_count': duplicate_count + total_rows - inserted,
            'failed_count': 0,
            'failed_rows': [],
            'import_id': import_id,
            'completion_status': 'completed',
        }
    except PartialImportError as pie:
        res = dict(pie.result)
        res['duplicate_count'] = res.get('duplicate_count', 0) + duplicate_count
        return res
    except Exception as exc:
        return {
            'attempted_count': total_rows,
            'inserted_count': 0,
            'duplicate_count': duplicate_count,
            'failed_count': total_rows,
            'failed_rows': rows,
            'import_id': import_id,
            'completion_status': 'failed',
            'error': str(exc),
        }


def atomic_import(token: str, rows: List[Dict[str, Any]], import_id: Optional[str] = None) -> int:
    """Execute import. Single batch if <= 1000 items, or scalable batching if > 1000 items."""
    actor = verify_actor(token, {'pemilik_toko', 'kasir'})
    if not rows:
        raise ValueError('Jumlah baris harus 1–1000' if len(rows) == 0 else 'Data kosong')

    # If rows exceed 1000, use scalable chunked import
    if len(rows) > 1000:
        result = execute_scalable_import(token, rows, batch_size=500, import_id=import_id)
        if result['failed_batches']:
            first_err = result['failed_batches'][0]['error']
            raise PartialImportError(
                f"Impor sebagian berhasil ({result['total_imported']} tersimpan, {result['failed_count']} gagal pada batch {result['failed_batches'][0]['batch_index']}): {first_err}",
                result=result
            )
        return result['total_imported']

    rows, _, import_id = prepare_import_rows(actor['toko_id'], rows, import_id)
    if not rows:
        return 0

    # Standard single batch execution with in-memory pre-fetched categories
    try:
        from repositories.finance_repository import categories
        store_cats = {str(c['id']): c for c in categories(actor['toko_id'])}
    except Exception:
        store_cats = {}

    batch = []
    for i, r in enumerate(rows, 1):
        tp = _type(r.get('tipe'))
        cat_id_str = str(r.get('kategori_id') or '')
        if not cat_id_str and (r.get('usulan') or {}).get('ai_pending') is True:
            cat_id_str = None
        elif store_cats:
            cat = store_cats.get(cat_id_str)
            if not cat or cat.get('tipe') != tp:
                raise ValueError(f"Kategori baris {i} tidak sesuai jenis transaksi/toko")
        else:
            _validate_category(actor['toko_id'], cat_id_str, tp)
        desc = clean_text_for_storage(r.get('deskripsi', ''))
        if not desc or len(desc) > 500:
            raise ValueError(f'Deskripsi baris {i} tidak valid')
        batch.append({
            'tanggal': _date(r['tanggal']),
            'deskripsi': desc,
            'nominal': _money(r['nominal']),
            'tipe': tp,
            'kategori_id': cat_id_str,
            'status_kategori': r.get('status_kategori', 'dikonfirmasi'),
            'skor': r.get('skor'),
            'usulan': r.get('usulan') or {},
        })

    # New production saves use their UUID verbatim. Legacy direct callers can
    # still supply content hashes/labels as metadata keys; p_batch is a SQL UUID.
    try:
        batch_id = str(uuid.UUID(import_id))
    except ValueError:
        batch_id = str(uuid.uuid5(uuid.NAMESPACE_URL, import_id))

    count = privileged_client().rpc('import_transactions_atomic', {
        'p_actor': actor['id'],
        'p_shop': actor['toko_id'],
        'p_rows': batch,
        'p_batch': batch_id,
    }).execute().data

    # Invalidate forecasting cache and refresh semantic indexes
    shop_id = actor['toko_id']
    try:
        from ai.hybrid_classifier import clear_classification_cache
        clear_classification_cache(shop_id)
        from ai.rag_engine import clear_memory_cache
        clear_memory_cache(shop_id)
    except Exception:
        # Safe post-commit cleanup: cache invalidation failure should not abort import result
        pass

    if actor.get('role') == 'pemilik_toko':
        try:
            from services.forecast_service import invalidate_forecast_cache
            invalidate_forecast_cache(shop_id)
        except Exception:
            # Safe post-commit cleanup: forecast cache invalidation failure should not abort import result
            pass

    return count


def import_rows(token: str, rows: List[Dict[str, Any]], category_map: Dict[str, str]) -> Tuple[int, List[str]]:
    """Helper to map default category IDs by type and execute import."""
    try:
        mapped = [{**r, 'kategori_id': category_map[r['tipe']]} for r in rows]
        return atomic_import(token, mapped), []
    except PartialImportError as exc:
        return exc.inserted_count, [str(exc)]
    except Exception as exc:
        return 0, [str(exc)]
