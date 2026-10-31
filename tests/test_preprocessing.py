from decimal import Decimal
import pytest
from ai.text_normalizer import clean_text_for_storage, normalize, extract_money
from services.validation_service import (
    validate_date,
    validate_amount,
    validate_type,
    validate_description,
    validate_transaction_payload,
)
from services.dataset_service import audit_reference_datasets, audit_csv_content


def test_reference_datasets_pass_audit():
    result = audit_reference_datasets()
    assert result['status'] == 'PASSED'
    assert result['total_files_audited'] == 4
    assert result['total_rows_audited'] == 1600
    for f in result['files']:
        assert f['status'] == 'PASSED'
        assert f['error_count'] == 0
        assert f['total_rows'] > 0
        assert f['income_count'] > 0
        assert f['expense_count'] > 0


def test_audit_csv_content_catches_invalid_structure():
    invalid_header = 'col1,col2,col3\n1,2,3'
    res = audit_csv_content(invalid_header)
    assert res['status'] == 'FAILED'
    assert any('Missing required columns' in err for err in res['errors'])

    corrupted_data = 'tanggal,deskripsi,nominal,tipe\nnot-a-date,beli bahan,-5000,invalid_type'
    res_corrupt = audit_csv_content(corrupted_data)
    assert res_corrupt['status'] == 'FAILED'
    assert res_corrupt['error_count'] >= 3


def test_clean_text_for_storage():
    assert clean_text_for_storage(None) == ''
    assert clean_text_for_storage('   ') == ''
    assert clean_text_for_storage('  beli   tepung   terigu  ') == 'beli tepung terigu'
    # Strips non-printable ASCII control codes while preserving standard characters
    assert clean_text_for_storage('beli\x00\x07 beras\x1b') == 'beli beras'
    assert clean_text_for_storage('Beli\u200b tepung \ufeffterigu') == 'Beli tepung terigu'


def test_normalize_abbreviation_expansion():
    # Expands common Indonesian informal abbreviations and typos found in UMKM entries
    assert 'beli' in normalize('bli tepung terigu')
    assert 'dibayar' in normalize('dibyr pesanan kue')
    assert 'bayar' in normalize('byr listrik oven')
    assert 'gaji' in normalize('gji karyawan')
    assert 'bahan' in normalize('bhn baku roti')
    assert 'transfer' in normalize('trf pelanggan')
    assert 'modal' in normalize('mdl awal')


def test_normalize_strips_monetary_values_for_nlp():
    # Strips currency figures and numeric quantities so category classification focuses on semantic terms
    normalized = normalize('beli telur 3 tray 250rb')
    assert '250' not in normalized
    assert 'rb' not in normalized
    assert 'beli telur' in normalized
    assert 'tray' in normalized


def test_extract_money_indonesian_formats():
    assert extract_money('beli telur 250rb') == Decimal('250000')
    assert extract_money('beli telur 250k') == Decimal('250000')
    assert extract_money('beli telur 250ribu') == Decimal('250000')
    assert extract_money('omzet kasir 1.5jt') == Decimal('1500000')
    assert extract_money('omzet kasir 1,5 jt') == Decimal('1500000')
    assert extract_money('omzet kasir 1.5 juta') == Decimal('1500000')
    assert extract_money('bayar sewa Rp 250.000') == Decimal('250000')
    assert extract_money('bayar sewa rp250000') == Decimal('250000')
    assert extract_money('bayar sewa 250000 rupiah') == Decimal('250000')


def test_extract_money_rejects_ambiguous():
    with pytest.raises(ValueError):
        extract_money('beli telur tanpa nominal')
    with pytest.raises(ValueError):
        extract_money('beli 250rb dan jual 300rb')


def test_validate_date():
    assert validate_date('2026-05-15') == '2026-05-15'
    # Leap year handling
    assert validate_date('2024-02-29') == '2024-02-29'
    assert validate_date('2000-02-29') == '2000-02-29'

    with pytest.raises(ValueError):
        validate_date('2023-02-29')
    with pytest.raises(ValueError):
        validate_date('2100-02-29')
    with pytest.raises(ValueError):
        validate_date('2026-04-31')
    with pytest.raises(ValueError):
        validate_date('1990-01-01')
    with pytest.raises(ValueError):
        validate_date('invalid-date')


def test_validate_amount():
    assert validate_amount(15000) == '15000'
    assert validate_amount('250000.50') == '250000.50'
    assert validate_amount(Decimal('100.25')) == '100.25'
    assert validate_amount('1e2') == '100'
    assert validate_amount('10000.00') == '10000.00'

    with pytest.raises(ValueError):
        validate_amount(0)
    with pytest.raises(ValueError):
        validate_amount(-500)
    with pytest.raises(ValueError):
        validate_amount('abc')
    with pytest.raises(ValueError):
        validate_amount('100.123')
    with pytest.raises(ValueError):
        validate_amount('1e-5')
    with pytest.raises(ValueError):
        validate_amount('NaN')
    with pytest.raises(ValueError):
        validate_amount('Infinity')


def test_validate_type():
    assert validate_type('PEMASUKAN') == 'pemasukan'
    assert validate_type('pengeluaran ') == 'pengeluaran'

    with pytest.raises(ValueError):
        validate_type('unknown')


def test_validate_description():
    assert validate_description(' Beli telur ') == 'Beli telur'

    with pytest.raises(ValueError):
        validate_description('')
    with pytest.raises(ValueError):
        validate_description('   ')
    with pytest.raises(ValueError):
        validate_description('a' * 501)


def test_validate_transaction_payload():
    payload = validate_transaction_payload(
        tanggal='2026-10-31',
        deskripsi='  Beli bahan kue  ',
        nominal='125000.00',
        tipe='pengeluaran',
        kategori_id='cat-uuid-123'
    )
    assert payload == {
        'tanggal': '2026-10-31',
        'deskripsi': 'Beli bahan kue',
        'nominal': '125000.00',
        'tipe': 'pengeluaran',
        'kategori_id': 'cat-uuid-123',
    }
