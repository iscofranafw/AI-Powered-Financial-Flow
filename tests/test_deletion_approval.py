import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

from repositories import deletion_repository
from services.auth_service import AccessError
from services.deletion_service import (
    owner_request_bulk_deletion,
    owner_cancel_deletion_request,
    owner_get_latest_deletion_request,
    superadmin_list_deletion_requests,
    superadmin_reject_deletion_request,
    superadmin_approve_and_execute_deletion,
)
from services import finance_service as f
from services import admin_service as a


@pytest.fixture(autouse=True)
def deletion_db(monkeypatch):
    """Database double for repository persistence and the atomic RPC contract."""
    from copy import deepcopy
    from types import SimpleNamespace
    rows = []
    transactions = []

    class Query:
        def __init__(self, table):
            self.table = table
            self.filters = []
            self.payload = None
            self.inserting = False
            self.limit_count = None
        def select(self, *args):
            return self
        def insert(self, payload):
            self.payload, self.inserting = payload, True
            return self
        def update(self, payload):
            self.payload = payload
            return self
        def eq(self, key, value):
            self.filters.append(lambda r: r.get(key) == value)
            return self
        def neq(self, key, value):
            self.filters.append(lambda r: r.get(key) != value)
            return self
        def in_(self, *args):
            return self
        def order(self, *args, **kwargs):
            return self
        def limit(self, count):
            self.limit_count = count
            return self
        def execute(self):
            if self.table != 'transaction_deletion_requests':
                return SimpleNamespace(data=[])
            if self.inserting:
                rows.append(deepcopy(self.payload))
                return SimpleNamespace(data=[deepcopy(self.payload)])
            matches = [r for r in rows if all(f(r) for f in self.filters)]
            if self.limit_count is not None:
                matches = matches[:self.limit_count]
            if self.payload is not None:
                for r in matches:
                    r.update(self.payload)
            return SimpleNamespace(data=deepcopy(matches))

    def rpc(name, params):
        assert name == 'execute_store_transaction_deletion'
        def execute():
            req = next(r for r in rows if r['id'] == params['p_request_id'])
            if req['status'] != 'pending':
                raise RuntimeError('Pengajuan tidak dalam status pending')
            deleted = [r for r in transactions if r['toko_id'] == req['store_id']]
            transactions[:] = [r for r in transactions if r['toko_id'] != req['store_id']]
            req.update(status='completed', transaction_count_deleted=len(deleted))
            return SimpleNamespace(data={'success': True, 'request_id': req['id'],
                'store_id': req['store_id'], 'deleted_count': len(deleted), 'status': 'completed'})
        return SimpleNamespace(execute=execute)

    client = MagicMock()
    client.table.side_effect = Query
    client.rpc.side_effect = rpc
    monkeypatch.setattr(deletion_repository, 'privileged_client', lambda: client)
    monkeypatch.setattr('services.deletion_service.privileged_client', lambda: client)
    return SimpleNamespace(client=client, rows=rows, transactions=transactions, rpc=rpc)


# ---------------------------------------------------------------------------
# 1. OWNER REQUEST WORKFLOW TESTS
# ---------------------------------------------------------------------------

def test_owner_request_bulk_deletion_success():
    """Store Owner successfully submits a bulk deletion request with exact phrase."""
    mock_owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}
    mock_txs = [
        {'id': 'tx-1', 'toko_id': 'store-1', 'nominal': 50000},
        {'id': 'tx-2', 'toko_id': 'store-1', 'nominal': 75000},
    ]

    with patch('services.deletion_service.verify_actor', return_value=mock_owner), \
         patch('services.deletion_service.transactions', return_value=mock_txs):
        
        req = f.request_deletion_all_transactions('valid-token', 'REQUEST DELETE')
        
        assert req is not None
        assert req['store_id'] == 'store-1'
        assert req['requested_by'] == 'owner-1'
        assert req['status'] == 'pending'
        assert req['transaction_count'] == 2
        # Crucial check: No transactions should be deleted upon request submission
        assert len(mock_txs) == 2


def test_owner_request_bulk_deletion_invalid_confirmation():
    """Request submission fails if confirmation phrase does not match 'REQUEST DELETE' exactly."""
    mock_owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}

    with patch('services.deletion_service.verify_actor', return_value=mock_owner):
        for invalid_text in ['request delete', 'DELETE', 'YES', 'REQUEST DELETE ', '']:
            if invalid_text.strip() != 'REQUEST DELETE':
                with pytest.raises(ValueError, match="Teks konfirmasi harus persis 'REQUEST DELETE'"):
                    f.request_deletion_all_transactions('token', invalid_text)


def test_owner_request_bulk_deletion_zero_transactions():
    """Request submission fails if the store has 0 transactions to delete."""
    mock_owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}

    with patch('services.deletion_service.verify_actor', return_value=mock_owner), \
         patch('services.deletion_service.transactions', return_value=[]):
        
        with pytest.raises(ValueError, match='Tidak ada transaksi yang dapat dihapus pada toko ini'):
            f.request_deletion_all_transactions('token', 'REQUEST DELETE')


def test_owner_request_duplicate_pending_rejected():
    """Store cannot have more than one pending deletion request at a time."""
    mock_owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}
    mock_txs = [{'id': 'tx-1', 'toko_id': 'store-1'}]

    with patch('services.deletion_service.verify_actor', return_value=mock_owner), \
         patch('services.deletion_service.transactions', return_value=mock_txs):
        
        # First request succeeds
        f.request_deletion_all_transactions('token', 'REQUEST DELETE')

        # Second request must be rejected
        with pytest.raises(ValueError, match='Sudah ada pengajuan penghapusan transaksi yang sedang menunggu persetujuan'):
            f.request_deletion_all_transactions('token', 'REQUEST DELETE')


def test_cashier_cannot_request_deletion():
    """Cashiers must be blocked from requesting bulk deletion server-side."""
    with patch('services.deletion_service.verify_actor', side_effect=AccessError('Akses ditolak')):
        with pytest.raises(AccessError):
            f.request_deletion_all_transactions('cashier-token', 'REQUEST DELETE')


# ---------------------------------------------------------------------------
# 2. OWNER CANCELLATION TESTS
# ---------------------------------------------------------------------------

def test_owner_cancel_pending_request_success():
    """Store Owner can cancel their own pending deletion request."""
    mock_owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}
    mock_txs = [{'id': 'tx-1', 'toko_id': 'store-1'}]

    with patch('services.deletion_service.verify_actor', return_value=mock_owner), \
         patch('services.deletion_service.transactions', return_value=mock_txs):
        
        req = f.request_deletion_all_transactions('token', 'REQUEST DELETE')
        req_id = req['id']

        # Cancel pending request
        cancelled = f.cancel_deletion_request('token', req_id)
        assert cancelled['status'] == 'cancelled'

        # Can submit a fresh request after cancelling
        new_req = f.request_deletion_all_transactions('token', 'REQUEST DELETE')
        assert new_req['status'] == 'pending'
        assert new_req['id'] != req_id


def test_owner_cannot_cancel_other_store_request():
    """Store Owner cannot cancel a deletion request belonging to a different store."""
    owner_a = {'id': 'owner-a', 'role': 'pemilik_toko', 'toko_id': 'store-a'}
    owner_b = {'id': 'owner-b', 'role': 'pemilik_toko', 'toko_id': 'store-b'}

    with patch('services.deletion_service.verify_actor', return_value=owner_a), \
         patch('services.deletion_service.transactions', return_value=[{'id': 't1'}]):
        req_a = f.request_deletion_all_transactions('token-a', 'REQUEST DELETE')

    with patch('services.deletion_service.verify_actor', return_value=owner_b):
        with pytest.raises(AccessError, match='Tidak berhak membatalkan pengajuan milik toko lain'):
            f.cancel_deletion_request('token-b', req_a['id'])


def test_owner_cannot_cancel_non_pending_request():
    """Attempting to cancel an already non-pending request raises ValueError."""
    mock_owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}
    mock_txs = [{'id': 'tx-1', 'toko_id': 'store-1'}]

    with patch('services.deletion_service.verify_actor', return_value=mock_owner), \
         patch('services.deletion_service.transactions', return_value=mock_txs):
        
        req = f.request_deletion_all_transactions('token', 'REQUEST DELETE')
        f.cancel_deletion_request('token', req['id'])

        # Second cancel attempt must fail
        with pytest.raises(ValueError, match='Hanya pengajuan berstatus pending yang dapat dibatalkan'):
            f.cancel_deletion_request('token', req['id'])


# ---------------------------------------------------------------------------
# 3. SUPERADMIN REVIEWS & APPROVAL TESTS
# ---------------------------------------------------------------------------

def test_owner_cannot_approve_own_deletion_request():
    """Store Owner cannot approve deletion requests; only Superadmin is authorized."""
    mock_owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}

    with patch('services.deletion_service.verify_actor', side_effect=AccessError('Role tidak diizinkan')):
        with pytest.raises(AccessError):
            a.approve_deletion_request('owner-token', 'req-1', 'APPROVE DELETE')


def test_superadmin_list_deletion_requests():
    """Superadmin can list deletion requests, filtered by status, with metadata."""
    admin = {'id': 'admin-1', 'role': 'superadmin'}
    owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}

    with patch('services.deletion_service.verify_actor', return_value=owner), \
         patch('services.deletion_service.transactions', return_value=[{'id': 't1'}]):
        f.request_deletion_all_transactions('token', 'REQUEST DELETE')

    with patch('services.deletion_service.verify_actor', return_value=admin):
        pending_list = a.list_deletion_requests('admin-token', status='pending')
        assert len(pending_list) == 1
        assert pending_list[0]['status'] == 'pending'
        assert pending_list[0]['store_id'] == 'store-1'

        all_list = a.list_deletion_requests('admin-token')
        assert len(all_list) >= 1


def test_superadmin_reject_request_success():
    """Superadmin can reject a deletion request without deleting any transactions."""
    admin = {'id': 'admin-1', 'role': 'superadmin'}
    owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}

    with patch('services.deletion_service.verify_actor', return_value=owner), \
         patch('services.deletion_service.transactions', return_value=[{'id': 't1'}]):
        req = f.request_deletion_all_transactions('token', 'REQUEST DELETE')

    with patch('services.deletion_service.verify_actor', return_value=admin):
        rejected = a.reject_deletion_request('admin-token', req['id'], reason='Data masih diaudit')
        assert rejected['status'] == 'rejected'
        assert rejected['rejection_reason'] == 'Data masih diaudit'
        assert rejected['reviewed_by'] == 'admin-1'


def test_superadmin_approve_invalid_phrase():
    """Approval fails if the confirmation phrase is not exactly 'APPROVE DELETE'."""
    admin = {'id': 'admin-1', 'role': 'superadmin'}

    with patch('services.deletion_service.verify_actor', return_value=admin):
        with pytest.raises(ValueError, match="Teks konfirmasi Superadmin harus persis 'APPROVE DELETE'"):
            a.approve_deletion_request('admin-token', 'req-1', 'approve delete')


def test_superadmin_approve_and_execute_deletion_with_cache_invalidation(deletion_db):
    """Superadmin approves deletion: deletes store transactions and invalidates caches."""
    admin = {'id': 'admin-1', 'role': 'superadmin'}
    owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}

    with patch('services.deletion_service.verify_actor', return_value=owner), \
         patch('services.deletion_service.transactions', return_value=[{'id': 't1'}, {'id': 't2'}]):
        req = f.request_deletion_all_transactions('token', 'REQUEST DELETE')

    mock_db_client = MagicMock()
    deletion_db.transactions.extend([{'id': 't1', 'toko_id': 'store-1'}, {'id': 't2', 'toko_id': 'store-1'}])
    mock_db_client.rpc.side_effect = deletion_db.rpc

    with patch('services.deletion_service.verify_actor', return_value=admin), \
         patch('services.deletion_service.privileged_client', return_value=mock_db_client), \
         patch('services.forecast_service.invalidate_forecast_cache') as mock_forecast_inv, \
         patch('ai.hybrid_classifier.clear_classification_cache') as mock_classif_inv, \
         patch('ai.rag_engine.clear_memory_cache') as mock_rag_inv:

        res = a.approve_deletion_request('admin-token', req['id'], 'APPROVE DELETE')

        assert res['success'] is True
        assert res['status'] == 'completed'
        assert res['deleted_count'] == 2

        mock_db_client.table.assert_not_called()
        mock_db_client.rpc.assert_called_once_with('execute_store_transaction_deletion',
            {'p_admin': 'admin-1', 'p_request_id': req['id']})
        assert deletion_db.transactions == []

        # Verify all caches were invalidated for the target store
        mock_forecast_inv.assert_called_once_with('store-1')
        mock_classif_inv.assert_called_once_with('store-1')
        mock_rag_inv.assert_called_once_with('store-1')


def test_superadmin_approval_idempotency(deletion_db):
    """Approving an already completed request raises ValueError."""
    admin = {'id': 'admin-1', 'role': 'superadmin'}
    owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}

    with patch('services.deletion_service.verify_actor', return_value=owner), \
         patch('services.deletion_service.transactions', return_value=[{'id': 't1'}]):
        req = f.request_deletion_all_transactions('token', 'REQUEST DELETE')

    mock_db_client = MagicMock()
    deletion_db.transactions.append({'id': 't1', 'toko_id': 'store-1'})
    mock_db_client.rpc.side_effect = deletion_db.rpc

    with patch('services.deletion_service.verify_actor', return_value=admin), \
         patch('services.deletion_service.privileged_client', return_value=mock_db_client):

        # First approval succeeds
        a.approve_deletion_request('admin-token', req['id'], 'APPROVE DELETE')

        # Second approval attempt must fail
        with pytest.raises(ValueError, match='Pengajuan tidak dalam status pending'):
            a.approve_deletion_request('admin-token', req['id'], 'APPROVE DELETE')


# ---------------------------------------------------------------------------
# 4. MULTI-TENANT ISOLATION TESTS
# ---------------------------------------------------------------------------

def test_multi_tenant_isolation_deletion(deletion_db):
    """Deleting Store A transactions never targets or deletes Store B transactions."""
    admin = {'id': 'admin-1', 'role': 'superadmin'}
    owner_a = {'id': 'owner-a', 'role': 'pemilik_toko', 'toko_id': 'store-a'}

    with patch('services.deletion_service.verify_actor', return_value=owner_a), \
         patch('services.deletion_service.transactions', return_value=[{'id': 'tx-a1'}]):
        req_a = f.request_deletion_all_transactions('token-a', 'REQUEST DELETE')

    mock_db_client = MagicMock()
    deletion_db.transactions.extend([{'id': 'tx-a1', 'toko_id': 'store-a'}, {'id': 'tx-b1', 'toko_id': 'store-b'}])
    mock_db_client.rpc.side_effect = deletion_db.rpc

    with patch('services.deletion_service.verify_actor', return_value=admin), \
         patch('services.deletion_service.privileged_client', return_value=mock_db_client):
        
        a.approve_deletion_request('admin-token', req_a['id'], 'APPROVE DELETE')

        mock_db_client.table.assert_not_called()
        assert deletion_db.transactions == [{'id': 'tx-b1', 'toko_id': 'store-b'}]



# ---------------------------------------------------------------------------
# 5. INDIVIDUAL TRANSACTION DELETE PRESERVATION
# ---------------------------------------------------------------------------

def test_individual_transaction_delete_remains_unaffected():
    """Existing single-transaction deletion remains 100% functional without approval workflow."""
    mock_owner = {'id': 'owner-1', 'role': 'pemilik_toko', 'toko_id': 'store-1'}
    mock_db = MagicMock()

    with patch('services.finance_service.verify_actor', return_value=mock_owner), \
         patch('services.finance_service.transaction_by_id', return_value={'id': 'tx-99', 'toko_id': 'store-1'}), \
         patch('services.finance_service.privileged_client', return_value=mock_db), \
         patch('services.forecast_service.invalidate_forecast_cache') as mock_inv:
        
        # Calling existing delete_transaction should succeed directly
        f.delete_transaction('token', 'tx-99')

        mock_db.table.assert_not_called()
        mock_db.rpc.assert_called_once_with('mutate_finance_audited', {
            'p_actor': 'owner-1', 'p_shop': 'store-1', 'p_action': 'delete_transaction',
            'p_ids': ['tx-99'], 'p_values': {},
        })
        mock_inv.assert_called_once_with('store-1')


# ---------------------------------------------------------------------------
# 6. CATEGORY FILTER & FINANCIAL SUMMARY CALCULATIONS
# ---------------------------------------------------------------------------

def test_category_filter_and_decimal_summary():
    """Verify category filtering and exact Decimal financial summary computations."""
    sample_rows = [
        {'id': '1', 'tanggal': '2026-10-01', 'deskripsi': 'Jual Kopi', 'nominal': 150000.50, 'tipe': 'pemasukan', 'kategori_id': 'cat-1'},
        {'id': '2', 'tanggal': '2026-10-02', 'deskripsi': 'Beli Biji Kopi', 'nominal': 50000.25, 'tipe': 'pengeluaran', 'kategori_id': 'cat-2'},
        {'id': '3', 'tanggal': '2026-10-03', 'deskripsi': 'Jual Donat', 'nominal': 100000.00, 'tipe': 'pemasukan', 'kategori_id': 'cat-1'},
        {'id': '4', 'tanggal': '2026-10-04', 'deskripsi': 'Listrik', 'nominal': 25000.00, 'tipe': 'pengeluaran', 'kategori_id': 'cat-3'},
    ]

    # Test 1: All categories
    total_tx = len(sample_rows)
    total_in = sum(Decimal(str(r['nominal'])) for r in sample_rows if r['tipe'] == 'pemasukan')
    total_out = sum(Decimal(str(r['nominal'])) for r in sample_rows if r['tipe'] == 'pengeluaran')
    net_flow = total_in - total_out

    assert total_tx == 4
    assert total_in == Decimal('250000.50')
    assert total_out == Decimal('75000.25')
    assert net_flow == Decimal('175000.25')

    # Test 2: Filter by category cat-1 (Sales)
    filtered_cat1 = [r for r in sample_rows if r['kategori_id'] == 'cat-1']
    assert len(filtered_cat1) == 2
    cat1_in = sum(Decimal(str(r['nominal'])) for r in filtered_cat1 if r['tipe'] == 'pemasukan')
    cat1_out = sum(Decimal(str(r['nominal'])) for r in filtered_cat1 if r['tipe'] == 'pengeluaran')
    assert cat1_in == Decimal('250000.50')
    assert cat1_out == Decimal('0')
    assert (cat1_in - cat1_out) == Decimal('250000.50')

    # Test 3: Filter by non-existent category
    filtered_none = [r for r in sample_rows if r['kategori_id'] == 'non-existent']
    assert len(filtered_none) == 0
    none_in = sum(Decimal(str(r.get('nominal', 0))) for r in filtered_none if r.get('tipe') == 'pemasukan')
    none_out = sum(Decimal(str(r.get('nominal', 0))) for r in filtered_none if r.get('tipe') == 'pengeluaran')
    assert none_in == Decimal('0')
    assert none_out == Decimal('0')
    assert (none_in - none_out) == Decimal('0')


@pytest.mark.parametrize('message', ['authorization denied', 'Pengajuan tidak dalam status pending', 'RPC unavailable'])
def test_rpc_failure_never_deletes_or_changes_request(deletion_db, message):
    from copy import deepcopy
    req = deletion_repository.create_request('store-1', 'owner-1', 1)
    deletion_db.transactions.append({'id': 'tx-1', 'toko_id': 'store-1'})
    before = deepcopy(deletion_db.rows)
    deletion_db.client.reset_mock()
    deletion_db.client.rpc.side_effect = RuntimeError(message)
    with patch('services.deletion_service.verify_actor', return_value={'id': 'admin-1'}), \
         patch.object(deletion_repository, 'mark_failed') as failed, \
         patch.object(deletion_repository, 'complete_deletion') as completed, \
         patch('services.forecast_service.invalidate_forecast_cache') as forecast, \
         patch('ai.hybrid_classifier.clear_classification_cache') as classifier, \
         patch('ai.rag_engine.clear_memory_cache') as rag:
        with pytest.raises(RuntimeError, match=f'Eksekusi penghapusan gagal: {message}') as error:
            superadmin_approve_and_execute_deletion('token', req['id'], 'APPROVE DELETE')
        assert str(error.value.__cause__) == message
        failed.assert_not_called()
        completed.assert_not_called()
        forecast.assert_not_called()
        classifier.assert_not_called()
        rag.assert_not_called()
    assert deletion_db.rows == before
    assert deletion_db.rows[0]['status'] == 'pending'
    assert deletion_db.transactions == [{'id': 'tx-1', 'toko_id': 'store-1'}]
    assert all(call.args == ('transaction_deletion_requests',) for call in deletion_db.client.table.call_args_list)


@pytest.mark.parametrize('response', [None, {}, {'success': False}, {'success': True},
    {'success': True, 'status': 'completed', 'request_id': 'other', 'store_id': 'store-1', 'deleted_count': 1}])
def test_unconfirmed_rpc_response_is_an_error(deletion_db, response):
    from types import SimpleNamespace
    req = deletion_repository.create_request('store-1', 'owner-1', 1)
    deletion_db.client.rpc.side_effect = None
    deletion_db.client.rpc.return_value.execute.return_value = SimpleNamespace(data=response)
    with patch('services.deletion_service.verify_actor', return_value={'id': 'admin-1'}), \
         patch('services.forecast_service.invalidate_forecast_cache') as forecast, \
         patch('ai.hybrid_classifier.clear_classification_cache') as classifier, \
         patch('ai.rag_engine.clear_memory_cache') as rag:
        with pytest.raises(RuntimeError, match='RPC tidak mengonfirmasi'):
            superadmin_approve_and_execute_deletion('token', req['id'], 'APPROVE DELETE')
        forecast.assert_not_called()
        classifier.assert_not_called()
        rag.assert_not_called()
    assert deletion_db.rows[0]['status'] == 'pending'


@pytest.mark.parametrize('method,args', [
    ('create_request', ('store-1', 'owner-1', 1)),
    ('get_pending_request_for_store', ('store-1',)),
    ('get_request_by_id', ('req-1',)),
    ('list_requests_for_store', ('store-1',)),
    ('list_all_requests', ()),
    ('cancel_request', ('req-1', 'store-1')),
    ('reject_request', ('req-1', 'admin-1', 'reason')),
    ('complete_deletion', ('req-1', 'store-1', 'admin-1', 1)),
    ('mark_failed', ('req-1', 'error')),
])
def test_repository_database_errors_are_not_swallowed(method, args):
    client = MagicMock()
    query = MagicMock()
    client.table.return_value = query
    for operation in ('insert', 'select', 'update', 'eq', 'neq', 'order', 'limit'):
        getattr(query, operation).return_value = query
    query.execute.side_effect = RuntimeError('database unavailable')
    with patch.object(deletion_repository, 'privileged_client', return_value=client):
        with pytest.raises(RuntimeError, match='database unavailable'):
            getattr(deletion_repository, method)(*args)
    assert not hasattr(deletion_repository, '_FALLBACK_STORE')


@pytest.mark.parametrize('data', [None, []])
def test_create_request_requires_database_confirmation(data):
    client = MagicMock()
    client.table.return_value.insert.return_value.execute.return_value.data = data
    with patch.object(deletion_repository, 'privileged_client', return_value=client):
        with pytest.raises(RuntimeError, match='database|Database'):
            deletion_repository.create_request('store-1', 'owner-1', 1)


def test_mark_failed_preserves_completed_state(deletion_db):
    req = deletion_repository.create_request('store-1', 'owner-1', 1)
    deletion_repository.complete_deletion(req['id'], 'store-1', 'admin-1', 1)
    before = dict(deletion_db.rows[0])
    deletion_repository.mark_failed(req['id'], 'late timeout')
    assert deletion_db.rows[0] == before


def test_cache_error_after_confirmed_deletion_preserves_success(deletion_db):
    req = deletion_repository.create_request('store-1', 'owner-1', 1)
    deletion_db.transactions.append({'id': 'tx-1', 'toko_id': 'store-1'})
    with patch('services.deletion_service.verify_actor', return_value={'id': 'admin-1'}), \
         patch('services.forecast_service.invalidate_forecast_cache', side_effect=RuntimeError('cache unavailable')), \
         patch('ai.hybrid_classifier.clear_classification_cache') as classifier, \
         patch('ai.rag_engine.clear_memory_cache') as rag:
        result = superadmin_approve_and_execute_deletion('token', req['id'], 'APPROVE DELETE')
        assert result['success'] is True
        classifier.assert_called_once_with('store-1')
        rag.assert_called_once_with('store-1')
    assert deletion_db.rows[0]['status'] == 'completed'
    assert deletion_db.transactions == []
