import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
import services.ai_service as ai_service
from ai.csv_cluster import cluster_rows


def test_prepare_categories():
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {'bidang_usaha': 'bakery'}
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.ai_service.privileged_client', return_value=client), \
         patch('services.ai_service.recommend', return_value={'pengeluaran': ['Bahan Baku'], 'pemasukan': ['Penjualan']}):
        res = ai_service.prepare_categories('tok')
        assert 'pengeluaran' in res
        assert 'pemasukan' in res


def test_save_initial_categories_already_exists():
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.ai_service.finance.list_categories', return_value=[{'id': 'c1', 'nama': 'Gaji'}]):
        with pytest.raises(ValueError, match='Kategori sudah ada'):
            ai_service.save_initial_categories('tok', {})


def test_save_initial_categories_validation_and_auto_adds():
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.ai_service.finance.list_categories', side_effect=[[], [{'id': 'c1'}]]), \
         patch('services.ai_service.finance.create_category') as mock_create:
        selection = {
            'pengeluaran': ['Gaji', 'Operasional', 'Pemasaran'],
            'pemasukan': ['Penjualan']
        }
        res = ai_service.save_initial_categories('tok', selection)
        created_names = [call.args[1] for call in mock_create.call_args_list]
        assert 'Others' in created_names
        assert 'Tambahan Modal' in created_names
        assert len(res) == 1


def test_save_initial_categories_exceed_cap():
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.ai_service.finance.list_categories', return_value=[]):
        selection = {
            'pengeluaran': [f'Cat{i}' for i in range(12)],
            'pemasukan': ['Penjualan']
        }
        with pytest.raises(ValueError, match='Kategori melewati batas'):
            ai_service.save_initial_categories('tok', selection)


def test_quick_preview():
    with patch('services.ai_service.extract_money', return_value=Decimal('250000')), \
         patch('services.ai_service.detect_type', return_value='pengeluaran'), \
         patch('services.ai_service.classify_preview', return_value={'kategori_id': 'c_telur', 'usulan': {}, 'message': 'OK'}):
        res = ai_service.quick_preview('tok', 'beli telur 250rb')
        assert res['nominal'] == '250000'
        assert res['tipe'] == 'pengeluaran'
        assert res['kategori_id'] == 'c_telur'


def test_commit_preview():
    with patch('services.ai_service.finance.add_transaction', return_value=[{'id': 'tx1'}]) as mock_add:
        row = {'tanggal': '2026-10-30', 'deskripsi': 'beli tepung', 'nominal': '150000', 'tipe': 'pengeluaran'}
        ai_service.commit_preview('tok', row, 'c_bahan')
        mock_add.assert_called_once_with('tok', '2026-10-30', 'beli tepung', '150000', 'pengeluaran', 'c_bahan')


def test_suggest_others():
    cats = [
        {'id': 'c_others', 'nama': 'Others', 'tipe': 'pengeluaran'},
        {'id': 'c_gaji', 'nama': 'Gaji', 'tipe': 'pengeluaran'}
    ]
    txs = [
        {'deskripsi': 'beli lakban', 'tipe': 'pengeluaran', 'kategori_id': 'c_others', 'status_persetujuan': 'disetujui'}
    ]
    with patch('services.ai_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.ai_service.finance.list_categories', return_value=cats), \
         patch('repositories.finance_repository.transactions', return_value=txs), \
         patch('ai.ollama_client.ask_json', return_value={'usulan': ['Peralatan Packing', 'Gaji']}):
        suggestions = ai_service.suggest_others('tok', 'pengeluaran')
        assert 'Peralatan Packing' in suggestions
        assert 'Gaji' not in suggestions


def test_cluster_rows_fallback():
    rows = [
        {'deskripsi': 'Beli tepung terigu', 'tipe': 'pengeluaran'},
        {'deskripsi': 'Beli mentega', 'tipe': 'pengeluaran'},
        {'deskripsi': 'Penjualan roti tawar', 'tipe': 'pemasukan'},
    ]
    with patch('ai.csv_cluster.ask_json', side_effect=Exception('offline')):
        clustered = cluster_rows(rows, level='Sedang')
        assert len(clustered) == 3
        assert clustered[0]['group'].startswith('pengeluaran-')
        assert clustered[2]['group'].startswith('pemasukan-')
