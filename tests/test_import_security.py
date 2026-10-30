from unittest.mock import patch,MagicMock
import pytest
from services.import_service import atomic_import


def test_import_checks_category_before_database_write():
    with patch('services.import_service.verify_actor',return_value={'id':'a','toko_id':'A'}),\
         patch('services.import_service._validate_category',side_effect=ValueError('wrong shop')):
        with pytest.raises(ValueError,match='wrong shop'):
            atomic_import('token',[{'tanggal':'2026-10-27','deskripsi':'test','nominal':100,'tipe':'pengeluaran','kategori_id':'OTHER'}])

def test_import_uses_single_atomic_rpc():
    fake=MagicMock();fake.rpc.return_value.execute.return_value.data=2
    with patch('services.import_service.verify_actor',return_value={'id':'a','toko_id':'A'}),\
         patch('services.import_service._validate_category'),\
         patch('services.import_service.privileged_client',return_value=fake):
        result=atomic_import('token',[{'tanggal':'2026-10-27','deskripsi':'test','nominal':100,
            'tipe':'pengeluaran','kategori_id':'SAME'}])
    assert result==2
    assert fake.rpc.call_args.args[0]=='import_transactions_atomic'
