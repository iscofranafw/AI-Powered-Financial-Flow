import csv
import io
import os
from decimal import Decimal
from services.validation_service import (
    validate_date,
    validate_amount,
    validate_type,
    validate_description,
)

REQUIRED_HEADERS = {'tanggal', 'deskripsi', 'nominal', 'tipe'}
DEFAULT_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples')


def audit_csv_content(content, filename='<memory>'):
    if isinstance(content, str):
        raw = content
    elif isinstance(content, bytes):
        try:
            raw = content.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            return {
                'file_name': filename,
                'status': 'FAILED',
                'errors': [f'Encoding error: {exc}'],
                'total_rows': 0,
            }
    else:
        return {
            'file_name': filename,
            'status': 'FAILED',
            'errors': ['Unsupported content format'],
            'total_rows': 0,
        }

    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames:
        return {
            'file_name': filename,
            'status': 'FAILED',
            'errors': ['Empty CSV header'],
            'total_rows': 0,
        }

    header_set = set(reader.fieldnames)
    if not REQUIRED_HEADERS.issubset(header_set):
        missing = REQUIRED_HEADERS - header_set
        return {
            'file_name': filename,
            'status': 'FAILED',
            'errors': [f"Missing required columns: {', '.join(sorted(missing))}"],
            'total_rows': 0,
        }

    rows = []
    errors = []
    dates = []
    amounts = []
    income_count = 0
    expense_count = 0
    vocab = set()

    for idx, row in enumerate(reader, start=2):
        row_errors = []
        try:
            valid_date = validate_date(row.get('tanggal'))
            dates.append(valid_date)
        except Exception as exc:
            row_errors.append(f'Row {idx} date error: {exc}')

        try:
            desc = validate_description(row.get('deskripsi'))
            for word in desc.lower().split():
                vocab.add(word)
        except Exception as exc:
            row_errors.append(f'Row {idx} description error: {exc}')

        try:
            amt = validate_amount(row.get('nominal'))
            amounts.append(Decimal(amt))
        except Exception as exc:
            row_errors.append(f'Row {idx} amount error: {exc}')

        try:
            tp = validate_type(row.get('tipe'))
            if tp == 'pemasukan':
                income_count += 1
            else:
                expense_count += 1
        except Exception as exc:
            row_errors.append(f'Row {idx} type error: {exc}')

        if row_errors:
            errors.extend(row_errors)
        else:
            rows.append(row)

    total_valid = len(rows)
    return {
        'file_name': filename,
        'status': 'PASSED' if not errors and total_valid > 0 else 'FAILED',
        'total_rows': total_valid,
        'error_count': len(errors),
        'errors': errors,
        'income_count': income_count,
        'expense_count': expense_count,
        'date_range': (min(dates), max(dates)) if dates else (None, None),
        'amount_range': (min(amounts), max(amounts)) if amounts else (None, None),
        'vocabulary_size': len(vocab),
    }


def audit_reference_datasets(samples_dir=None):
    target_dir = samples_dir or DEFAULT_SAMPLES_DIR
    if not os.path.exists(target_dir):
        return {
            'status': 'FAILED',
            'error': f'Directory not found: {target_dir}',
            'files': [],
        }

    dataset_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.csv')])
    file_results = []
    all_passed = True
    total_rows = 0

    for fname in dataset_files:
        fpath = os.path.join(target_dir, fname)
        with open(fpath, 'rb') as fh:
            res = audit_csv_content(fh.read(), filename=fname)
        file_results.append(res)
        if res['status'] != 'PASSED':
            all_passed = False
        total_rows += res.get('total_rows', 0)

    return {
        'status': 'PASSED' if all_passed and file_results else 'FAILED',
        'samples_directory': target_dir,
        'total_files_audited': len(file_results),
        'total_rows_audited': total_rows,
        'files': file_results,
    }
