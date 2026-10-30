import pytest
from unittest.mock import patch, MagicMock
from services.cashier_service import create_cashier, list_cashiers
from services.admin_service import list_owners, change_owner_status, shop_summary, list_all_users
from services.auth_service import AccessError


def test_create_cashier_input_validations():
    with patch('services.cashier_service.verify_actor', return_value={'id': 'o1', 'role': 'pemilik_toko', 'toko_id': 's1'}):
        with pytest.raises(ValueError, match='Nama/email wajib; password minimal 8 karakter'):
            create_cashier('tok', '', 'kasir@toko.com', 'password123')

        with pytest.raises(ValueError, match='Nama/email wajib; password minimal 8 karakter'):
            create_cashier('tok', 'Budi', 'kasirtokocom', 'password123')

        with pytest.raises(ValueError, match='Nama/email wajib; password minimal 8 karakter'):
            create_cashier('tok', 'Budi', 'kasir@toko.com', 'short')


def test_list_cashiers():
    with patch('services.cashier_service.verify_actor', return_value={'id': 'o1', 'role': 'pemilik_toko', 'toko_id': 's1'}), \
         patch('services.cashier_service.cashier_profiles', return_value=[{'id': 'k1', 'nama': 'Siti'}]), \
         patch('services.cashier_service.privileged_client'):
        res = list_cashiers('tok')
        assert len(res) == 1
        assert res[0]['nama'] == 'Siti'


def test_change_owner_status_success_confirms_email():
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
        'role': 'pemilik_toko', 'status': 'pending'
    }
    with patch('services.admin_service.verify_actor', return_value={'id': 'admin1', 'role': 'superadmin'}), \
         patch('services.admin_service.privileged_client', return_value=client):
        change_owner_status('tok', 'owner123', 'aktif')
        client.auth.admin.update_user_by_id.assert_called_once_with('owner123', {'email_confirm': True})
        client.table.return_value.update.assert_called_once_with({'status': 'aktif'})


def test_shop_summary():
    client = MagicMock()
    client.rpc.return_value.execute.return_value.data = [
        {'nama_toko': 'Toko Roti Enak', 'jumlah_transaksi': 42}
    ]
    with patch('services.admin_service.verify_actor', return_value={'id': 'admin1', 'role': 'superadmin'}), \
         patch('services.admin_service.privileged_client', return_value=client):
        res = shop_summary('tok')
        assert len(res) == 1
        assert res[0]['nama_toko'] == 'Toko Roti Enak'


def test_list_all_users_pagination():
    client = MagicMock()
    batch1 = [{'id': f'u{i}', 'nama': f'User {i}'} for i in range(1000)]
    batch2 = [{'id': f'u{1000+i}', 'nama': f'User {1000+i}'} for i in range(5)]
    client.table.return_value.select.return_value.order.return_value.range.return_value.execute.side_effect = [
        MagicMock(data=batch1),
        MagicMock(data=batch2)
    ]
    with patch('services.admin_service.verify_actor', return_value={'id': 'admin1', 'role': 'superadmin'}), \
         patch('services.admin_service.privileged_client', return_value=client):
        all_users = list_all_users('tok')
        assert len(all_users) == 1005
