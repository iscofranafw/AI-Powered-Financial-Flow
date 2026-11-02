from decimal import Decimal
import pytest
from services.finance_service import _money,_date,_type, LIMITS
from services.import_service import preview_csv

def test_money_valid(): assert _money('125000.50')=='125000.50'
@pytest.mark.parametrize('x',['-1','0','nan','abc','1.999'])
def test_money_invalid(x):
 with pytest.raises(ValueError): _money(x)
def test_date(): assert _date('2026-10-27')=='2026-10-27'
def test_date_invalid():
 with pytest.raises(ValueError): _date('27/10/2026')
def test_type_invalid():
 with pytest.raises(ValueError): _type('modal')
def test_limits(): assert LIMITS=={'pengeluaran':7,'pemasukan':7}
def test_csv_preview():
 data=b'tanggal,deskripsi,nominal,tipe\n2026-10-27,Beli gula,10000,pengeluaran\n'
 data=data.replace(b'\\n',b'\n')
 rows=preview_csv(data)
 assert len(rows)==1 and rows[0]['error']==''
def test_csv_missing_column():
 with pytest.raises(ValueError): preview_csv(b'tanggal,nominal\n2026-10-27,100\n')
def test_csv_rejects_bad_type():
 rows=preview_csv(b'tanggal,deskripsi,nominal,tipe\n2026-10-27,A,100,foo\n')
 assert rows[0]['error']
