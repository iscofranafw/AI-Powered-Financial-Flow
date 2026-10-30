import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock
from services.auth_service import AccessError
import services.finance_service as fs


def test_list_categories():
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 'shop1', 'role': 'pemilik_toko'}), \
         patch('services.finance_service.categories', return_value=[{'id': 'c1', 'nama': 'Gaji'}]):
        res = fs.list_categories('tok')
        assert len(res) == 1
        assert res[0]['nama'] == 'Gaji'


def test_create_category_validations():
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 'shop1', 'role': 'pemilik_toko'}), \
         patch('services.finance_service.categories', return_value=[]):
        with pytest.raises(ValueError, match='Nama kategori wajib diisi'):
            fs.create_category('tok', '   ', 'pengeluaran')

        fake_cats = [{'id': f'c{i}', 'nama': f'Cat{i}', 'tipe': 'pengeluaran'} for i in range(10)]
        with patch('services.finance_service.categories', return_value=fake_cats):
            with pytest.raises(ValueError, match='Batas kategori tercapai'):
                fs.create_category('tok', 'NewCat', 'pengeluaran')

        with patch('services.finance_service.categories', return_value=[{'id': 'c1', 'nama': 'Listrik', 'tipe': 'pengeluaran'}]):
            with pytest.raises(ValueError, match='Kategori sudah ada'):
                fs.create_category('tok', 'listrik', 'pengeluaran')

        with patch('services.finance_service.categories', return_value=[]):
            with pytest.raises(ValueError, match='Others adalah kategori sistem'):
                fs.create_category('tok', 'Others', 'pengeluaran', system_seed=False)

        with patch('services.finance_service.categories', return_value=[]):
            with pytest.raises(ValueError, match='Tambahan Modal harus pemasukan'):
                fs.create_category('tok', 'Tambahan Modal', 'pengeluaran')


def test_rename_category_validations():
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 'shop1', 'role': 'pemilik_toko'}):
        with patch('services.finance_service.category_by_id', return_value=None):
            with pytest.raises(ValueError, match='Kategori tidak ditemukan'):
                fs.rename_category('tok', 'c1', 'New Name')

        with patch('services.finance_service.category_by_id', return_value={'id': 'c1', 'sistem': True, 'bukan_operasional': False}):
            with pytest.raises(ValueError, match='Kategori sistem/modal tidak dapat diubah'):
                fs.rename_category('tok', 'c1', 'New Name')

        with patch('services.finance_service.category_by_id', return_value={'id': 'c1', 'sistem': False, 'bukan_operasional': True}):
            with pytest.raises(ValueError, match='Kategori sistem/modal tidak dapat diubah'):
                fs.rename_category('tok', 'c1', 'New Name')


def test_delete_category_validations():
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 'shop1', 'role': 'pemilik_toko'}):
        with patch('services.finance_service.category_by_id', return_value={'id': 'c1', 'sistem': True, 'bukan_operasional': False}):
            with pytest.raises(ValueError, match='Kategori terlindungi'):
                fs.delete_category('tok', 'c1')

        client = MagicMock()
        client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [{'id': 'tx1'}]
        with patch('services.finance_service.category_by_id', return_value={'id': 'c1', 'sistem': False, 'bukan_operasional': False}), \
             patch('services.finance_service.privileged_client', return_value=client):
            with pytest.raises(ValueError, match='Kategori sudah dipakai'):
                fs.delete_category('tok', 'c1')


def test_merge_category_validations():
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 'shop1', 'role': 'pemilik_toko'}):
        with patch('services.finance_service.category_by_id', side_effect=lambda shop, cid: {'id': cid, 'tipe': 'pengeluaran', 'sistem': False, 'bukan_operasional': False}):
            with pytest.raises(ValueError, match='Kategori sumber/tujuan tidak sesuai'):
                fs.merge_category('tok', 'c1', 'c1')

        def get_diff(shop, cid):
            if cid == 'c1':
                return {'id': 'c1', 'tipe': 'pengeluaran', 'sistem': False, 'bukan_operasional': False}
            return {'id': 'c2', 'tipe': 'pemasukan', 'sistem': False, 'bukan_operasional': False}
        with patch('services.finance_service.category_by_id', side_effect=get_diff):
            with pytest.raises(ValueError, match='Kategori sumber/tujuan tidak sesuai'):
                fs.merge_category('tok', 'c1', 'c2')


def test_list_transactions_rbac():
    txs = [
        {'id': 't1', 'dicatat_oleh': 'user_owner'},
        {'id': 't2', 'dicatat_oleh': 'user_cashier'}
    ]
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko', 'id': 'user_owner'}), \
         patch('services.finance_service.transactions', return_value=txs):
        assert len(fs.list_transactions('tok')) == 2

    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'kasir', 'id': 'user_cashier'}), \
         patch('services.finance_service.transactions', return_value=txs):
        res = fs.list_transactions('tok')
        assert len(res) == 1
        assert res[0]['id'] == 't2'


def test_add_transaction_status():
    client = MagicMock()
    client.table.return_value.insert.return_value.execute.return_value.data = [{'id': 'tx_new'}]

    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko', 'id': 'o1'}), \
         patch('services.finance_service._validate_category'), \
         patch('services.finance_service.privileged_client', return_value=client):
        fs.add_transaction('tok', '2026-10-30', 'Beli telur', 50000, 'pengeluaran', 'c1')
        insert_payload = client.table.return_value.insert.call_args.args[0]
        assert insert_payload['status_persetujuan'] == 'disetujui'
        assert insert_payload['disetujui_oleh'] == 'o1'

    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'kasir', 'id': 'k1'}), \
         patch('services.finance_service._validate_category'), \
         patch('services.finance_service.privileged_client', return_value=client):
        fs.add_transaction('tok', '2026-10-30', 'Beli beras', 75000, 'pengeluaran', 'c1')
        insert_payload = client.table.return_value.insert.call_args.args[0]
        assert insert_payload['status_persetujuan'] == 'pending'
        assert insert_payload['disetujui_oleh'] is None


def test_update_transaction_cashier_rules():
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'kasir', 'id': 'k1'}):
        with patch('services.finance_service.transaction_by_id', return_value={'id': 'tx1', 'dicatat_oleh': 'k2', 'status_persetujuan': 'pending'}):
            with pytest.raises(AccessError):
                fs.update_transaction('tok', 'tx1', '2026-10-30', 'Edit', 10000, 'pengeluaran', 'c1')

        with patch('services.finance_service.transaction_by_id', return_value={'id': 'tx1', 'dicatat_oleh': 'k1', 'status_persetujuan': 'disetujui'}):
            with pytest.raises(AccessError):
                fs.update_transaction('tok', 'tx1', '2026-10-30', 'Edit', 10000, 'pengeluaran', 'c1')


def test_approve_transactions_validation():
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko', 'id': 'o1'}):
        with pytest.raises(ValueError, match='Keputusan atau ID tidak valid'):
            fs.approve_transactions('tok', [], 'disetujui')
        with pytest.raises(ValueError, match='Keputusan atau ID tidak valid'):
            fs.approve_transactions('tok', ['tx1'], 'invalid_decision')


def test_dashboard_summary():
    cats = [
        {'id': 'c_sales', 'nama': 'Penjualan', 'tipe': 'pemasukan', 'bukan_operasional': False},
        {'id': 'c_modal', 'nama': 'Tambahan Modal', 'tipe': 'pemasukan', 'bukan_operasional': True},
        {'id': 'c_bahan', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran', 'bukan_operasional': False},
    ]
    txs = [
        {'id': 't1', 'nominal': '1000000', 'tipe': 'pemasukan', 'kategori_id': 'c_sales', 'status_persetujuan': 'disetujui'},
        {'id': 't2', 'nominal': '500000', 'tipe': 'pemasukan', 'kategori_id': 'c_modal', 'status_persetujuan': 'disetujui'},
        {'id': 't3', 'nominal': '300000', 'tipe': 'pengeluaran', 'kategori_id': 'c_bahan', 'status_persetujuan': 'disetujui'},
        {'id': 't4', 'nominal': '999999', 'tipe': 'pengeluaran', 'kategori_id': 'c_bahan', 'status_persetujuan': 'pending'},
    ]
    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.finance_service.categories', return_value=cats), \
         patch('services.finance_service.transactions', return_value=txs):
        summary = fs.dashboard_summary('tok')
        assert summary['pemasukan'] == Decimal('1500000')
        assert summary['pengeluaran'] == Decimal('300000')
        assert summary['tambahan_modal'] == Decimal('500000')
        assert summary['omzet'] == Decimal('1000000')
        assert summary['arus_kas_operasional'] == Decimal('700000')
        assert summary['arus_kas_total'] == Decimal('1200000')
        assert summary['jumlah_disetujui'] == 3
        assert summary['pengeluaran_per_kategori']['Bahan Baku'] == Decimal('300000')
