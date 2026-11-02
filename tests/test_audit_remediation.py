"""Comprehensive Post-Audit Remediation and Financial Data Integrity Test Suite.

Verifies fixes for P0, P1, and P2 audit findings:
1. Pending transactions do not inflate approved financial totals (P0-02)
2. Rejected transactions do not inflate reports (P0-02)
3. Importing the same file twice does not produce accidental duplicates (P0-03)
4. Legitimate repeated transactions remain supported (P0-03)
5. Double-clicking Import does not duplicate data / idempotency (P0-03, P1-07)
6. Partial import failures report actual committed counts (P1-01)
7. Cashier changes cannot bypass approval (P0-04)
8. Approved Owner workflows remain functional (P0-04)
9. Bulk deletion requires Superadmin authorization (P0-05)
10. Atomic deletion failures preserve consistent request state (P0-05)
11. Cross-store isolation enforced (P0-06, P1-06)
12. Forecast values match actual requested dates / explicit insufficient coverage (P0-07)
13. Monetary parsing standardizes Indonesian and international formats (P1-03)
14. RAG embedding dimension compatibility and degraded mode (P1-06)
15. Chat proposals cannot execute twice (P1-07)
16. Logout clears session and revokes tokens (P1-09)
17. Category limits match agreed rules (P2-02)
"""

import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
import io
import uuid
from datetime import datetime, date, timezone, timedelta

# 1 & 2: Pending and Rejected transactions do not inflate approved totals
def test_pending_and_rejected_transactions_do_not_inflate_approved_financial_totals():
    from services.finance_service import dashboard_summary
    from services.report_service import period_rows, metrics

    cats = [
        {'id': 'c_sales', 'nama': 'Penjualan', 'tipe': 'pemasukan', 'bukan_operasional': False},
        {'id': 'c_modal', 'nama': 'Tambahan Modal', 'tipe': 'pemasukan', 'bukan_operasional': True},
        {'id': 'c_bahan', 'nama': 'Bahan Baku', 'tipe': 'pengeluaran', 'bukan_operasional': False},
    ]
    txs = [
        {'id': 't1', 'tanggal': '2026-10-25', 'nominal': '1000000', 'tipe': 'pemasukan', 'kategori_id': 'c_sales', 'status_persetujuan': 'disetujui'},
        {'id': 't2', 'tanggal': '2026-10-26', 'nominal': '500000', 'tipe': 'pemasukan', 'kategori_id': 'c_modal', 'status_persetujuan': 'disetujui'},
        {'id': 't3', 'tanggal': '2026-10-27', 'nominal': '300000', 'tipe': 'pengeluaran', 'kategori_id': 'c_bahan', 'status_persetujuan': 'disetujui'},
        # Pending should NOT be included
        {'id': 't4', 'tanggal': '2026-10-28', 'nominal': '9999999', 'tipe': 'pengeluaran', 'kategori_id': 'c_bahan', 'status_persetujuan': 'pending'},
        # Rejected should NOT be included
        {'id': 't5', 'tanggal': '2026-10-29', 'nominal': '8888888', 'tipe': 'pemasukan', 'kategori_id': 'c_sales', 'status_persetujuan': 'ditolak'},
    ]

    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.finance_service.categories', return_value=cats), \
         patch('services.finance_service.transactions', return_value=txs):

        summary = dashboard_summary('fake_token')
        assert summary['pemasukan'] == Decimal('1500000')
        assert summary['pengeluaran'] == Decimal('300000')
        assert summary['arus_kas_operasional'] == Decimal('700000')
        assert summary['jumlah_disetujui'] == 3

    with patch('services.report_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.report_service.transactions', return_value=txs):

        rows = period_rows('fake_token')
        assert len(rows) == 3
        assert all(r['status_persetujuan'] == 'disetujui' for r in rows)
        rep_metrics = metrics(rows, cats)
        assert rep_metrics['pemasukan'] == Decimal('1500000')
        assert rep_metrics['pengeluaran'] == Decimal('300000')
        assert rep_metrics['omzet'] == Decimal('1000000')


# 3 & 4: Duplicate imports vs legitimate repeated transactions
def test_importing_same_file_twice_prevents_accidental_duplicates_while_preserving_legitimate_repeated():
    from services.smart_import_engine import detect_duplicates

    # File contains 2 legitimate repeated sales of "Pocari Sweat 7000" without source ID
    incoming_rows = [
        {'fingerprint': '2026-10-31|pocari sweat|7000.00|pemasukan', 'source_id': None, 'deskripsi': 'Pocari Sweat', 'nominal': '7000.00', 'tipe': 'pemasukan'},
        {'fingerprint': '2026-10-31|pocari sweat|7000.00|pemasukan', 'source_id': None, 'deskripsi': 'Pocari Sweat', 'nominal': '7000.00', 'tipe': 'pemasukan'},
    ]

    # First run: DB currently has 0 transactions
    new_rows, dups = detect_duplicates('shop-1', incoming_rows, existing_transactions=[])
    # Both sales must be preserved as legitimate distinct retail transactions
    assert len(new_rows) == 2
    assert len(dups) == 0

    # Second run / re-import: DB now has the 2 transactions recorded earlier
    db_after_first_import = [
        {'tanggal': '2026-10-31', 'deskripsi': 'Pocari Sweat', 'nominal': '7000.00', 'tipe': 'pemasukan', 'usulan': {}},
        {'tanggal': '2026-10-31', 'deskripsi': 'Pocari Sweat', 'nominal': '7000.00', 'tipe': 'pemasukan', 'usulan': {}},
    ]
    new_rows2, dups2 = detect_duplicates('shop-1', incoming_rows, existing_transactions=db_after_first_import)
    # Re-importing the same file must be 100% prevented from duplicating
    assert len(new_rows2) == 0
    assert len(dups2) == 2


# 5: Paginated transaction retrieval handles >1000 historical transactions
def test_duplicate_detection_pages_beyond_1000_transactions():
    from services.smart_import_engine import detect_duplicates

    # Mock 1,500 existing historical transactions
    mock_history = [
        {'id': f't_{i}', 'tanggal': '2026-01-01', 'deskripsi': f'Barang {i}', 'nominal': 1000, 'tipe': 'pengeluaran', 'usulan': {}}
        for i in range(1500)
    ]
    # Add item 1200 as an incoming row
    incoming = [
        {'fingerprint': '2026-01-01|barang 1200|1000.00|pengeluaran', 'source_id': None, 'deskripsi': 'Barang 1200', 'nominal': '1000.00', 'tipe': 'pengeluaran'},
        {'fingerprint': '2026-10-31|barang baru|50000.00|pengeluaran', 'source_id': None, 'deskripsi': 'Barang Baru', 'nominal': '50000.00', 'tipe': 'pengeluaran'},
    ]

    with patch('repositories.finance_repository.transactions', return_value=mock_history):
        new_rows, dups = detect_duplicates('shop-1', incoming, existing_transactions=None)
        assert len(new_rows) == 1
        assert new_rows[0]['deskripsi'] == 'Barang Baru'
        assert len(dups) == 1
        assert dups[0]['deskripsi'] == 'Barang 1200'


# 6: Partial import failure reporting
def test_partial_import_failure_reports_actual_committed_counts():
    from services.import_service import PartialImportError, execute_structured_import
    from services.smart_import_engine import execute_scalable_import

    # 1,200 rows -> batches of 500, 500, 200
    rows = [
        {'tanggal': '2026-10-30', 'deskripsi': f'Item {i}', 'nominal': '10000.00', 'tipe': 'pengeluaran', 'kategori_id': 'c1'}
        for i in range(1200)
    ]

    fake_client = MagicMock()
    # Batch 1 (500) succeeds, Batch 2 fails
    fake_client.rpc.return_value.execute.side_effect = [
        MagicMock(data=500),
        RuntimeError("DB constraint violation on row 650"),
    ]

    with patch('services.smart_import_engine.verify_actor', return_value={'id': 'u1', 'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.smart_import_engine.privileged_client', return_value=fake_client), \
         patch('repositories.finance_repository.categories', return_value=[{'id': 'c1', 'tipe': 'pengeluaran'}]):

        res = execute_scalable_import('token', rows, batch_size=500)
        assert res['total_imported'] == 500
        assert res['inserted_count'] == 500
        assert res['attempted_count'] == 1200
        assert res['failed_count'] == 700
        assert len(res['failed_rows']) == 700  # All uncommitted rows
        assert res['completion_status'] == 'partial_failure'
        assert len(res['failed_batches']) == 1


# 7 & 8: Cashier cannot edit approved transactions; Resubmitting rejected resets to pending; Audit trail
def test_transaction_editing_approval_state_and_audit_trail():
    from services.finance_service import update_transaction
    from services.auth_service import AccessError

    old_tx = {
        'id': 'tx-100',
        'toko_id': 'store-1',
        'tanggal': '2026-10-25',
        'deskripsi': 'Beli Bensin',
        'nominal': Decimal('50000.00'),
        'tipe': 'pengeluaran',
        'kategori_id': 'cat-1',
        'dicatat_oleh': 'cashier-1',
        'status_persetujuan': 'disetujui',
        'usulan': {'source': 'manual'},
    }

    # 1. Cashier cannot modify approved transaction
    with patch('services.finance_service.verify_actor', return_value={'id': 'cashier-1', 'role': 'kasir', 'toko_id': 'store-1'}), \
         patch('services.finance_service.transaction_by_id', return_value=old_tx):
        with pytest.raises(AccessError, match='Kasir tidak dapat mengubah transaksi yang sudah disetujui'):
            update_transaction('token', 'tx-100', '2026-10-25', 'Beli Bensin Edit', 50000, 'pengeluaran', 'cat-1')

    # 2. Cashier editing rejected transaction resets to pending and records audit trail
    rejected_tx = dict(old_tx)
    rejected_tx['status_persetujuan'] = 'ditolak'
    rejected_tx['disetujui_oleh'] = 'owner-1'

    fake_client = MagicMock()
    fake_client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = [{'id': 'tx-100'}]

    with patch('services.finance_service.verify_actor', return_value={'id': 'cashier-1', 'role': 'kasir', 'toko_id': 'store-1'}), \
         patch('services.finance_service.transaction_by_id', return_value=rejected_tx), \
         patch('services.finance_service.category_by_id', return_value={'id': 'cat-1', 'tipe': 'pengeluaran'}), \
         patch('services.finance_service.privileged_client', return_value=fake_client):

        update_transaction('token', 'tx-100', '2026-10-25', 'Beli Bensin Koreksi', 50000, 'pengeluaran', 'cat-1')

        params = fake_client.rpc.call_args.args[1]
        assert params['p_action'] == 'update_transaction'
        assert params['p_actor'] == 'cashier-1'
        assert params['p_shop'] == 'store-1'
        assert params['p_values']['deskripsi'] == 'Beli Bensin Koreksi'
        # Reapproval and audit_history are verified below against real PostgreSQL.
        fake_client.table.assert_not_called()


# 9 & 10: Bulk deletion requires Superadmin & cannot overwrite completed request
def test_bulk_deletion_authorization_and_consistent_failure_handling():
    from services.deletion_service import superadmin_approve_and_execute_deletion
    from services.auth_service import AccessError
    from repositories import deletion_repository

    # 1. Non-superadmin cannot approve bulk deletion
    with patch('services.deletion_service.verify_actor', side_effect=AccessError('Hanya Superadmin')):
        with pytest.raises(AccessError):
            superadmin_approve_and_execute_deletion('token', 'req-1', 'APPROVE DELETE')

    # 2. mark_failed cannot overwrite an already completed deletion request
    completed = {'id': 'req-1', 'status': 'completed'}
    client = MagicMock()
    query = client.table.return_value
    query.update.return_value = query
    query.eq.return_value = query
    query.neq.return_value = query
    query.execute.return_value.data = []  # completed row excluded by the predicate
    with patch('repositories.deletion_repository.privileged_client', return_value=client):
        deletion_repository.mark_failed(completed['id'], 'Late network timeout')
    query.neq.assert_called_once_with('status', 'completed')
    query.eq.assert_called_once_with('id', completed['id'])
    query.execute.assert_called_once()



# 11: Cross-store isolation
def test_cross_store_isolation_enforced():
    from services.chat_service import confirm_proposal
    from services.auth_service import AccessError

    # Store A actor attempting to confirm Store B proposal
    proposal = {
        'id': 'prop-1',
        'toko_id': 'store-B',
        'tanggal': '2026-10-31',
        'deskripsi': 'Beli Telur',
        'nominal': 20000,
        'tipe': 'pengeluaran',
        'kategori_id': 'cat-1',
    }
    with patch('services.chat_service.verify_actor', return_value={'id': 'u1', 'role': 'pemilik_toko', 'toko_id': 'store-A'}):
        with pytest.raises(AccessError, match='dibuat untuk toko lain'):
            confirm_proposal('token', proposal)


# 12: Forecasting date alignment and explicit insufficient coverage
def test_forecast_date_alignment_returns_explicit_insufficient_coverage():
    from services.forecast_service import get_forecast_for_period

    mock_fcst = {
        'success': True,
        'projections': [
            {'ds': '2026-11-01', 'yhat': 100.0, 'yhat_lower': 80.0, 'yhat_upper': 120.0},
            {'ds': '2026-11-02', 'yhat': 150.0, 'yhat_lower': 120.0, 'yhat_upper': 180.0},
        ]
    }

    # Query for 2026-12-01 which is beyond generated projections
    with patch('services.forecast_service._get_eligible_expense_rows', return_value=({'toko_id': 's1'}, [{'tanggal': '2026-10-30'}], [])), \
         patch('services.forecast_service.train_and_forecast_expenses', return_value=mock_fcst):

        res = get_forecast_for_period('token', start_date='2026-12-01', end_date='2026-12-05', period_label='Desember 2026')
        assert res['success'] is False
        assert res['error'] == 'INSUFFICIENT_COVERAGE'
        assert 'Tidak tersedia proyeksi' in res['message']
        assert res['total_predicted'] == 0.0


# 13: Standardized monetary parsing
def test_monetary_parsing_standardization():
    from services.validation_service import validate_amount, parse_monetary_value

    assert parse_monetary_value('Rp 2.500') == Decimal('2500')
    assert parse_monetary_value('2.500,50') == Decimal('2500.50')
    assert parse_monetary_value('2,500.50') == Decimal('2500.50')
    assert parse_monetary_value('Rp 250.000') == Decimal('250000')
    assert parse_monetary_value('IDR 1.250.000,00') == Decimal('1250000.00')
    assert parse_monetary_value(15000) == Decimal('15000')

    with pytest.raises(ValueError):
        parse_monetary_value('100.123')
    with pytest.raises(ValueError):
        parse_monetary_value('1.999')
    with pytest.raises(ValueError):
        parse_monetary_value('-5000')
    with pytest.raises(ValueError):
        parse_monetary_value('abc')


# 14: RAG embedding dimension check
def test_rag_embedding_dimension_rebuilds_on_mismatch(tmp_path):
    from ai import rag_engine

    # Simulate existing index metadata with mismatched dimension 512 vs expected 384
    mock_meta = {
        'store_id': 's1',
        'toko_id': 's1',
        'version': 1,
        'embedding_dimension': 512,  # Incompatible
        'entries': []
    }

    with patch('ai.rag_engine.get_remote_version', return_value=1), \
         patch('ai.rag_engine.os.path.exists', return_value=True), \
         patch('builtins.open', MagicMock()), \
         patch('json.load', return_value=mock_meta), \
         patch('ai.rag_engine.build_and_save_index') as mock_rebuild:

        rag_engine.get_or_sync_index('s1', 'pengeluaran')
        mock_rebuild.assert_called_once_with('s1', 'pengeluaran', version=1)


# 15: Chat proposal idempotency / cannot execute twice
def test_chat_proposal_cannot_execute_twice():
    from services.chat_service import confirm_proposal

    prop = {
        'id': 'prop-unique-123',
        'toko_id': 's1',
        'tanggal': '2026-10-31',
        'deskripsi': 'Beli Telur',
        'nominal': 25000,
        'tipe': 'pengeluaran',
        'kategori_id': 'cat-1',
    }

    with patch('services.chat_service.verify_actor', return_value={'id': 'u1', 'role': 'pemilik_toko', 'toko_id': 's1'}), \
         patch('services.finance_service.add_transaction', return_value=[{'id': 't1', 'status_persetujuan': 'disetujui'}]):

        res1 = confirm_proposal('token', prop)
        assert res1['success'] is True
        assert prop['status'] == 'confirmed'

        # Second click on the same proposal object
        res2 = confirm_proposal('token', prop)
        assert res2['success'] is True
        assert 'sudah dikonfirmasi sebelumnya' in res2['message']


# 16: Logout clears session and revokes tokens
def test_logout_session_management():
    from services.auth_service import logout

    mock_client = MagicMock()
    with patch('services.auth_service.privileged_client', return_value=mock_client), \
         patch('services.auth_service.public_client', return_value=mock_client):

        res = logout('fake-jwt-token')
        assert res is True
        mock_client.auth.admin.sign_out.assert_called_once_with(jwt='fake-jwt-token', scope='global')
        mock_client.auth.sign_out.assert_called_once()


# 17: Category limits: AI discovery <= 7, Owner unlimited
def test_category_limits_match_agreed_business_rules():
    from services.finance_service import check_category_quota

    with patch('services.finance_service.verify_actor', return_value={'toko_id': 's1', 'role': 'pemilik_toko'}), \
         patch('services.finance_service.categories', return_value=[{'id': f'c{i}', 'tipe': 'pengeluaran'} for i in range(15)]):

        quota = check_category_quota('token', 'pengeluaran')
        # Even with 15 categories, owner can still add unlimited categories
        assert quota['can_add'] is True
        assert quota['exceeded'] is False
        assert quota['ai_recommendation_limit'] == 7

# P0-03 write-layer regressions. Use a disposable PostgreSQL cluster, never the
# application database. The small schema below covers the RPC's table contract.
@pytest.fixture(scope='module')
def import_postgres(tmp_path_factory):
    import os
    import shutil
    import socket
    import subprocess
    from pathlib import Path

    psql = shutil.which('psql')
    if not psql:
        pytest.skip('PostgreSQL binaries required for write-layer integration tests')
    bindir = Path(psql).parent
    suffix = '.exe' if os.name == 'nt' else ''
    initdb = bindir / ('initdb' + suffix)
    pg_ctl = bindir / ('pg_ctl' + suffix)
    if not initdb.exists() or not pg_ctl.exists():
        pytest.skip('initdb and pg_ctl required for isolated PostgreSQL tests')
    root = tmp_path_factory.mktemp('import_postgres')
    cluster = root / 'cluster'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]

    def run(args, input=None):
        import tempfile
        # Server children on Windows can inherit capture_output pipes. A file
        # avoids waiting for those handles after pg_ctl itself has exited.
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as output:
            result = subprocess.run(args, input=input, text=True, encoding='utf-8',
                                    stdout=output, stderr=output, timeout=60)
            output.seek(0)
            text = output.read().strip()
        if result.returncode:
            raise AssertionError(text)
        return text

    run([str(initdb), '-D', str(cluster), '-U', 'postgres', '-A', 'trust', '--encoding=UTF8', '--no-locale'])
    run([str(pg_ctl), '-D', str(cluster), '-l', str(root / 'server.log'),
         '-o', f'-h 127.0.0.1 -p {port}', '-w', 'start'])
    def sql(statement):
        return run([psql, '-X', '-h', '127.0.0.1', '-p', str(port), '-U', 'postgres',
                    '-d', 'postgres', '-v', 'ON_ERROR_STOP=1', '-Atq'], statement)
    try:
        sql('''
            CREATE ROLE anon; CREATE ROLE authenticated; CREATE ROLE service_role;
            CREATE TABLE profiles(id uuid, toko_id uuid, role text, status text);
            CREATE TABLE kategori(id uuid PRIMARY KEY, toko_id uuid, tipe text);
            CREATE TABLE transaksi(
                toko_id uuid, tanggal date, deskripsi text, nominal numeric(18,2),
                tipe text, kategori_id uuid, dicatat_oleh uuid, status_persetujuan text,
                disetujui_oleh uuid, disetujui_pada timestamptz, status_kategori text,
                skor numeric, usulan jsonb, batch_id uuid);
        ''')
        migration = Path(__file__).resolve().parents[1] / 'database/migrations/007_import_idempotency.sql'
        sql(migration.read_text(encoding='utf-8'))
        yield sql
    finally:
        run([str(pg_ctl), '-D', str(cluster), '-m', 'immediate', '-w', 'stop'])


@pytest.fixture
def import_database(import_postgres, monkeypatch):
    import json
    from types import SimpleNamespace
    sql = import_postgres
    actor = {'id': str(uuid.uuid4()), 'toko_id': str(uuid.uuid4()), 'role': 'pemilik_toko'}
    cats = [{'id': str(uuid.uuid4()), 'tipe': tp} for tp in ('pemasukan', 'pengeluaran')]
    sql(f"INSERT INTO profiles VALUES ('{actor['id']}', '{actor['toko_id']}', 'pemilik_toko', 'aktif');")
    for cat in cats:
        sql(f"INSERT INTO kategori VALUES ('{cat['id']}', '{actor['toko_id']}', '{cat['tipe']}');")

    def rpc(name, params):
        assert name == 'import_transactions_atomic'
        payload = json.dumps(params['p_rows'], default=str, ensure_ascii=False).replace("'", "''")
        statement = (f"SELECT public.import_transactions_atomic('{params['p_actor']}', "
                     f"'{params['p_shop']}', '{payload}'::jsonb, '{params['p_batch']}');")
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=int(sql(statement))))

    client = MagicMock()
    client.rpc.side_effect = rpc
    for module in ('services.smart_import_engine', 'services.import_service'):
        monkeypatch.setattr(module + '.verify_actor', lambda *a: actor)
        monkeypatch.setattr(module + '.privileged_client', lambda: client)
    monkeypatch.setattr('repositories.finance_repository.categories', lambda *a: cats)
    # Deliberately stale preflight: only the real RPC can reject repeat writes.
    monkeypatch.setattr('repositories.finance_repository.transactions', lambda *a: [])
    return actor, cats, client, sql


@pytest.mark.parametrize('entrypoint', ['atomic', 'scalable'])
def test_sample_csv_twice_at_write_layer(import_database, entrypoint):
    from pathlib import Path
    from services.import_service import preview_csv, atomic_import
    from services.smart_import_engine import execute_scalable_import
    actor, cats, client, sql = import_database
    path = Path(__file__).resolve().parents[1] / 'data/Daftar_Transaksi_Terkoreksi_Semua.csv'
    rows = preview_csv(path.read_bytes())
    assert rows and all(not r['error'] for r in rows)
    ids = {c['tipe']: c['id'] for c in cats}
    rows = [{**r, 'kategori_id': ids[r['tipe']]} for r in rows]
    if entrypoint == 'atomic':
        assert atomic_import('token', rows) == len(rows)
        assert atomic_import('token', rows) == 0
    else:
        assert execute_scalable_import('token', rows, batch_size=137)['inserted_count'] == len(rows)
        second = execute_scalable_import('token', rows, batch_size=333)
        assert second['inserted_count'] == 0
        assert second['duplicate_count'] == len(rows)
        assert second['failed_count'] == 0
        assert second['completion_status'] == 'completed'
    assert int(sql(f"SELECT count(*) FROM transaksi WHERE toko_id='{actor['toko_id']}';")) == len(rows)


@pytest.mark.parametrize('entrypoint', ['atomic', 'scalable'])
def test_legitimate_source_ids_and_explicit_import_keys(import_database, entrypoint):
    from services.import_service import atomic_import, preview_csv
    from services.smart_import_engine import execute_scalable_import
    actor, cats, client, sql = import_database
    rows = preview_csv(b'no_ref,tanggal,deskripsi,nominal,tipe\nA,2026-10-30,Sale,100,pemasukan\nB,2026-10-30,Sale,100,pemasukan\n')
    rows = [{**r, 'kategori_id': cats[0]['id']} for r in rows]
    def invoke(rows, key=None):
        if entrypoint == 'atomic':
            return atomic_import('token', rows, import_id=key)
        return execute_scalable_import('token', rows, batch_size=1, import_id=key)['inserted_count']
    assert invoke(rows) == 2
    assert invoke(rows) == 0
    # No source IDs: retain multiplicity within one import and permit a genuinely
    # new import of identical values when its explicit key differs.
    plain = [{**r, 'source_id': '', 'usulan': {}} for r in rows]
    assert invoke(plain, 'delivery-one') == 2
    assert invoke(plain, 'delivery-one') == 0
    assert invoke(plain, 'delivery-two') == 2


def test_rpc_concurrent_retry_and_rollback(import_database):
    from concurrent.futures import ThreadPoolExecutor
    actor, cats, client, sql = import_database
    row = {'tanggal': '2026-10-30', 'deskripsi': 'Sale', 'nominal': '100.00',
           'tipe': 'pemasukan', 'kategori_id': cats[0]['id']}
    params = {'p_actor': actor['id'], 'p_shop': actor['toko_id'],
              'p_batch': str(uuid.uuid4()), 'p_rows': [row, row]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        counts = list(pool.map(lambda _: client.rpc('import_transactions_atomic', params).execute().data, range(2)))
    assert sorted(counts) == [0, 2]
    # A failed batch must roll back both its transaction and identity claim.
    params['p_batch'] = str(uuid.uuid4())
    params['p_rows'] = [row, {**row, 'kategori_id': str(uuid.uuid4())}]
    with pytest.raises(AssertionError, match='Kategori tidak cocok'):
        client.rpc('import_transactions_atomic', params).execute()
    params['p_rows'] = [row, row]
    assert client.rpc('import_transactions_atomic', params).execute().data == 2


def test_import_rows_returns_committed_count_on_partial_error():
    from services.import_service import import_rows, PartialImportError
    error = PartialImportError('batch two failed', {'inserted_count': 500, 'failed_count': 700})
    with patch('services.import_service.atomic_import', side_effect=error):
        assert import_rows('token', [{'tipe': 'pemasukan'}], {'pemasukan': 'cat'}) == (500, ['batch two failed'])


def test_preflight_matches_import_identity_without_global_fingerprint_constraint():
    from services.smart_import_engine import prepare_import_rows
    row = {'tanggal': '2026-10-30', 'deskripsi': 'Sale', 'nominal': '100', 'tipe': 'pemasukan'}
    with patch('repositories.finance_repository.transactions', return_value=[]):
        prepared, _, _ = prepare_import_rows('s1', [row, row], 'first')
    with patch('repositories.finance_repository.transactions', return_value=prepared):
        fresh, duplicates, _ = prepare_import_rows('s1', [row, row], 'first')
        assert not fresh and len(duplicates) == 2
        fresh, duplicates, _ = prepare_import_rows('s1', [row, row], 'second')
        assert len(fresh) == 2 and not duplicates


def test_partial_retry_keeps_occurrences_after_zero_insert_batch(import_database):
    from services.smart_import_engine import prepare_import_rows, execute_scalable_import
    from services.import_service import atomic_import
    actor, cats, client, sql = import_database
    row = {'tanggal': '2026-10-30', 'deskripsi': 'Repeated sale', 'nominal': '100',
           'tipe': 'pemasukan', 'kategori_id': cats[0]['id']}
    original = [dict(row) for _ in range(6)]
    prepared, _, key = prepare_import_rows(actor['toko_id'], original)
    assert atomic_import('token', prepared[:2]) == 2
    real_rpc = client.rpc.side_effect
    calls = 0
    def fail_second(name, params):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('temporary batch failure')
        return real_rpc(name, params)
    client.rpc.side_effect = fail_second
    result = execute_scalable_import('token', original, batch_size=2)
    assert result['inserted_count'] == 0
    assert result['duplicate_count'] == 2
    assert result['failed_count'] == 4
    assert [r['usulan']['import_occurrence'] for r in result['failed_rows']] == [3, 4, 5, 6]
    client.rpc.side_effect = real_rpc
    retry = execute_scalable_import('token', result['failed_rows'], batch_size=3, import_id=key)
    assert retry['inserted_count'] == 4
    assert retry['failed_count'] == 0
    assert execute_scalable_import('token', original)['inserted_count'] == 0


# 18. Future-Date Validation & Jakarta Timezone Handling (P1-04 & P2-04)
def test_future_date_rejected_for_actual_transactions():
    from services.validation_service import validate_date, validate_transaction_payload, get_jakarta_today

    today_wib = get_jakarta_today()
    tomorrow_wib = today_wib + timedelta(days=1)
    day_after_tomorrow = today_wib + timedelta(days=2)

    # Tomorrow rejected as date object
    with pytest.raises(ValueError, match='tidak boleh melebihi hari ini'):
        validate_date(tomorrow_wib)

    # Tomorrow rejected as ISO string
    with pytest.raises(ValueError, match='tidak boleh melebihi hari ini'):
        validate_date(tomorrow_wib.isoformat())

    # Day after tomorrow rejected in full payload validation
    with pytest.raises(ValueError, match='tidak boleh melebihi hari ini'):
        validate_transaction_payload(day_after_tomorrow.isoformat(), 'Pembelian stok', 50000, 'pengeluaran')


def test_today_and_past_dates_accepted_for_actual_transactions():
    from services.validation_service import validate_date, validate_transaction_payload, get_jakarta_today

    today_wib = get_jakarta_today()
    yesterday_wib = today_wib - timedelta(days=1)
    last_week_wib = today_wib - timedelta(days=7)

    # Today accepted as date object
    assert validate_date(today_wib) == today_wib.isoformat()

    # Today accepted as ISO string
    assert validate_date(today_wib.isoformat()) == today_wib.isoformat()

    # Yesterday accepted
    assert validate_date(yesterday_wib.isoformat()) == yesterday_wib.isoformat()

    # Past date accepted
    assert validate_date(last_week_wib.isoformat()) == last_week_wib.isoformat()

    # Full payload for today accepted
    payload = validate_transaction_payload(today_wib.isoformat(), 'Penjualan harian', 250000, 'pemasukan', 'cat-1')
    assert payload['tanggal'] == today_wib.isoformat()
    assert payload['tipe'] == 'pemasukan'


def test_future_date_allowed_with_warning_when_scheduled():
    from services.validation_service import validate_date, validate_transaction_payload, get_jakarta_today

    today_wib = get_jakarta_today()
    tomorrow_wib = today_wib + timedelta(days=1)
    future_date = today_wib + timedelta(days=14)

    # validate_date with is_scheduled=True issues UserWarning and succeeds
    with pytest.warns(UserWarning, match='transaksi terjadwal/rencana'):
        res = validate_date(tomorrow_wib, is_scheduled=True)
    assert res == tomorrow_wib.isoformat()

    # validate_transaction_payload with is_scheduled=True issues UserWarning and records flag
    with pytest.warns(UserWarning, match='transaksi terjadwal/rencana'):
        payload = validate_transaction_payload(
            future_date.isoformat(),
            'Rencana pembayaran sewa ruko',
            5000000,
            'pengeluaran',
            kategori_id='cat-rent',
            is_scheduled=True,
        )
    assert payload['tanggal'] == future_date.isoformat()
    assert payload['is_scheduled'] is True


def test_jakarta_timezone_boundary_evaluation():
    from services.validation_service import validate_date, get_jakarta_today

    today_wib = get_jakarta_today()
    assert isinstance(today_wib, date)

    # When Jakarta business date evaluates to 2026-11-02 (e.g., late evening UTC on previous day)
    with patch('services.validation_service.get_jakarta_today', return_value=date(2026, 11, 2)):
        # 2026-11-02 is today in Jakarta, so it must be accepted
        assert validate_date('2026-11-02') == '2026-11-02'
        # 2026-11-01 was yesterday in Jakarta, so it must be accepted
        assert validate_date('2026-11-01') == '2026-11-01'
        # 2026-11-03 is tomorrow in Jakarta, so it must be rejected
        with pytest.raises(ValueError, match='tidak boleh melebihi hari ini'):
            validate_date('2026-11-03')


@pytest.mark.parametrize('lookup', ['success', 'unavailable', 'missing'])
def test_proposal_unique_conflict_is_resolved_only_by_confirmed_database_row(lookup):
    from services.chat_service import confirm_proposal
    class UniqueViolation(Exception):
        code = '23505'
    prop = {'id': str(uuid.uuid4()), 'toko_id': 'shop-1', 'status': 'pending_confirmation',
            'tanggal': '2026-10-31', 'deskripsi': 'Sale', 'nominal': '100',
            'tipe': 'pemasukan', 'kategori_id': 'cat-1'}
    client = MagicMock()
    query = client.table.return_value.select.return_value.eq.return_value.contains.return_value.limit.return_value
    query.execute.return_value.data = [{'id': 'persisted-1', 'status_persetujuan': 'disetujui'}] if lookup == 'success' else []
    if lookup == 'unavailable':
        query.execute.side_effect = RuntimeError('lookup unavailable')
    with patch('services.chat_service.verify_actor', return_value={'id': 'owner-1', 'toko_id': 'shop-1'}), \
         patch('services.chat_service.privileged_client', return_value=client), \
         patch('services.finance_service.add_transaction', side_effect=UniqueViolation('duplicate proposal')) as insert:
        result = confirm_proposal('token', prop)
    insert.assert_called_once()
    assert result['success'] is (lookup == 'success')
    if lookup == 'success':
        assert prop['status'] == 'confirmed'
        assert result['already_persisted'] is True
        assert result['record']['id'] == 'persisted-1'
    else:
        assert prop['status'] == 'pending_confirmation'
        assert 'Gagal menyimpan' in result['message']
    client.table.return_value.select.return_value.eq.assert_called_once_with('toko_id', 'shop-1')


@pytest.mark.parametrize('failure', ['database-error', 'empty-response'])
def test_proposal_persistence_failure_never_confirms(failure):
    from services.chat_service import confirm_proposal
    prop = {'id': str(uuid.uuid4()), 'status': 'pending_confirmation',
            'tanggal': '2026-10-31', 'deskripsi': 'Sale', 'nominal': '100',
            'tipe': 'pemasukan', 'kategori_id': 'cat-1'}
    with patch('services.chat_service.verify_actor', return_value={'id': 'owner-1', 'toko_id': 'shop-1'}), \
         patch('services.chat_service.privileged_client') as client, \
         patch('services.finance_service.add_transaction') as insert:
        if failure == 'database-error':
            insert.side_effect = RuntimeError('database unavailable')
        else:
            insert.return_value = []
        result = confirm_proposal('token', prop)
    assert result['success'] is False
    assert prop['status'] == 'pending_confirmation'
    client.assert_not_called()
    insert.assert_called_once()


def test_proposal_constraint_serializes_concurrent_workers(import_postgres):
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    sql = import_postgres
    sql((Path(__file__).resolve().parents[1] / 'database/migrations/008_chat_proposal_idempotency.sql').read_text())
    shop, other_shop = str(uuid.uuid4()), str(uuid.uuid4())
    proposal = str(uuid.uuid4())
    def insert(target_shop, metadata):
        return sql(f"INSERT INTO public.transaksi(toko_id,usulan) VALUES ('{target_shop}','{metadata}'::jsonb);")
    def worker(_):
        try:
            insert(shop, '{"proposal_id":"' + proposal + '"}')
            return 'inserted'
        except AssertionError as exc:
            assert 'transaksi_shop_proposal_unique' in str(exc)
            return 'duplicate'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(worker, range(2))) == ['duplicate', 'inserted']
    assert sql(f"SELECT count(*) FROM public.transaksi WHERE toko_id='{shop}';") == '1'
    insert(other_shop, '{"proposal_id":"' + proposal + '"}')
    # Ordinary transactions have no proposal identity and remain unrestricted.
    insert(shop, '{}')
    insert(shop, '{}')
    assert sql(f"SELECT count(*) FROM public.transaksi WHERE toko_id='{shop}';") == '3'


@pytest.mark.parametrize('row_count', [2, 1002])
def test_structured_import_threads_explicit_operation_id_to_database(import_database, row_count):
    from services.import_service import execute_structured_import
    actor, cats, client, sql = import_database
    row = {'tanggal': '2026-10-30', 'deskripsi': 'Identical legitimate sale',
           'nominal': '100.00', 'tipe': 'pemasukan', 'kategori_id': cats[0]['id']}
    rows = [dict(row) for _ in range(row_count)]
    first_id, second_id = str(uuid.uuid4()), str(uuid.uuid4())
    for key, expected in [(first_id, row_count), (second_id, row_count), (first_id, 0), (second_id, 0)]:
        client.rpc.reset_mock()
        result = execute_structured_import('token', rows, import_id=key)
        assert result['import_id'] == key
        assert result['inserted_count'] == expected
        assert result['duplicate_count'] == row_count - expected
        assert result['failed_count'] == 0
        assert result['completion_status'] == 'completed'
        for call in client.rpc.call_args_list:
            params = call.args[1]
            assert {r['usulan']['import_id'] for r in params['p_rows']} == {key}
            if row_count <= 1000:
                assert params['p_batch'] == key
    assert int(sql(f"SELECT count(*) FROM transaksi WHERE toko_id='{actor['toko_id']}';")) == 2 * row_count


@pytest.mark.parametrize('row_count', [2, 1002])
def test_structured_import_generates_new_id_for_each_save(import_database, row_count):
    from services.import_service import execute_structured_import
    actor, cats, client, sql = import_database
    rows = [{'tanggal': '2026-10-30', 'deskripsi': 'Another sale', 'nominal': '100.00',
             'tipe': 'pemasukan', 'kategori_id': cats[0]['id']} for _ in range(row_count)]
    first = execute_structured_import('token', rows)
    second = execute_structured_import('token', rows)
    assert first['import_id'] != second['import_id']
    for result in (first, second):
        assert str(uuid.UUID(result['import_id'])) == result['import_id']
        assert result['inserted_count'] == row_count
        assert result['completion_status'] == 'completed'
    retry = execute_structured_import('token', rows, import_id=first['import_id'])
    assert retry['inserted_count'] == 0
    assert retry['duplicate_count'] == row_count


def test_structured_import_preserves_source_id_dedup_across_new_saves(import_database):
    from services.import_service import execute_structured_import
    actor, cats, client, sql = import_database
    rows = [{'tanggal': '2026-10-30', 'deskripsi': 'Sale with source ID', 'nominal': '100',
             'tipe': 'pemasukan', 'kategori_id': cats[0]['id'], 'source_id': sid}
            for sid in ('sale-A', 'sale-B')]
    first = execute_structured_import('token', rows)
    second = execute_structured_import('token', rows)
    assert first['import_id'] != second['import_id']
    assert first['inserted_count'] == 2
    assert second['inserted_count'] == 0
    assert second['duplicate_count'] == 2


def test_atomic_import_large_delegation_keeps_operation_id(import_database):
    from services.import_service import atomic_import
    actor, cats, client, sql = import_database
    rows = [{'tanggal': '2026-10-30', 'deskripsi': 'Sale', 'nominal': '100',
             'tipe': 'pemasukan', 'kategori_id': cats[0]['id']} for _ in range(1001)]
    key = str(uuid.uuid4())
    assert atomic_import('token', rows, import_id=key) == len(rows)
    for call in client.rpc.call_args_list:
        assert {r['usulan']['import_id'] for r in call.args[1]['p_rows']} == {key}
    assert atomic_import('token', rows, import_id=key) == 0


@pytest.fixture(scope='module')
def audit_postgres(import_postgres):
    from pathlib import Path
    sql = import_postgres
    root = Path(__file__).resolve().parents[1] / 'database/migrations'
    sql('ALTER TABLE transaksi ADD COLUMN id uuid DEFAULT gen_random_uuid() PRIMARY KEY, ADD COLUMN updated_at timestamptz DEFAULT now();')
    sql((root / '009_audit_log.sql').read_text())
    sql((root / '010_atomic_audit_mutations.sql').read_text())
    return sql


@pytest.fixture
def atomic_audit_context(audit_postgres, monkeypatch):
    import json
    from types import SimpleNamespace
    from services import finance_service as fs
    sql = audit_postgres
    owner, admin, cashier, shop, cat, cat2 = [str(uuid.uuid4()) for _ in range(6)]
    ids = sorted(str(uuid.uuid4()) for _ in range(2))
    sql(f"INSERT INTO profiles VALUES ('{owner}','{shop}','pemilik_toko','aktif'), ('{admin}',NULL,'superadmin','aktif'), ('{cashier}','{shop}','kasir','aktif'); INSERT INTO kategori VALUES ('{cat}','{shop}','pemasukan'), ('{cat2}','{shop}','pengeluaran');")
    for ident in ids:
        sql(f"INSERT INTO transaksi(id,toko_id,tanggal,deskripsi,nominal,tipe,kategori_id,dicatat_oleh,status_persetujuan,disetujui_oleh,disetujui_pada,status_kategori,usulan) VALUES ('{ident}','{shop}','2026-10-25','Sale',100,'pemasukan','{cat}','{cashier}','disetujui','{owner}',now(),'dikonfirmasi','{{\"source_id\":\"source\",\"audit_history\":[{{\"action\":\"previous\"}}]}}');")
    actor = {'id': owner, 'toko_id': shop, 'role': 'pemilik_toko'}
    def literal(value):
        return "'" + str(value).replace("'", "''") + "'"
    def rpc(name, args):
        values = []
        for key, value in args.items():
            if key == 'p_ids':
                expr = 'ARRAY[' + ','.join(literal(v) for v in value) + ']::uuid[]'
            elif key == 'p_values':
                expr = literal(json.dumps(value)) + '::jsonb'
            else:
                expr = literal(value)
            values.append(key + ' => ' + expr)
        def execute():
            return SimpleNamespace(data=json.loads(sql('SELECT public.' + name + '(' + ','.join(values) + ');')))
        return SimpleNamespace(execute=execute)
    client = MagicMock()
    client.rpc.side_effect = rpc
    def snapshot():
        return json.loads(sql(f"SELECT coalesce(jsonb_agg(to_jsonb(t) ORDER BY id),'[]') FROM transaksi t WHERE toko_id='{shop}';"))
    def logs():
        return json.loads(sql(f"SELECT coalesce(jsonb_agg(to_jsonb(a) ORDER BY target_id),'[]') FROM audit_log a WHERE toko_id='{shop}';"))
    monkeypatch.setattr(fs, 'verify_actor', lambda *a: actor)
    monkeypatch.setattr(fs, 'transaction_by_id', lambda store, ident: next((r for r in snapshot() if r['id']==ident and store==shop), None))
    monkeypatch.setattr(fs, '_validate_category', lambda *a: None)
    monkeypatch.setattr(fs, 'privileged_client', lambda: client)
    monkeypatch.setattr('services.admin_service.verify_actor', lambda *a: {'id': admin, 'role': 'superadmin'})
    monkeypatch.setattr('services.admin_service.privileged_client', lambda: client)
    client.table.return_value.select.return_value.eq.return_value.single.return_value.execute.side_effect = lambda: SimpleNamespace(data=json.loads(sql(f"SELECT to_jsonb(p) FROM profiles p WHERE id='{owner}';")))
    def invoke(action):
        if action == 'update_transaction':
            return fs.update_transaction('token', ids[0], '2026-10-26', 'Corrected', 120, 'pemasukan', cat)
        if action == 'delete_transaction':
            return fs.delete_transaction('token', ids[0])
        if action == 'approve_transactions':
            return fs.approve_transactions('token', ids, 'disetujui')
        from services.admin_service import change_owner_status
        return change_owner_status('token', owner, 'nonaktif')
    return SimpleNamespace(sql=sql, owner=owner, admin=admin, cashier=cashier, shop=shop, cat=cat,
        cat2=cat2, ids=ids, client=client, snapshot=snapshot, logs=logs, invoke=invoke, actor=actor)


@pytest.mark.parametrize('action', ['update_transaction','delete_transaction','approve_transactions','change_owner_status'])
def test_audit_failure_rolls_back_real_mutation(atomic_audit_context, action):
    c = atomic_audit_context
    if action == 'approve_transactions':
        c.sql(f"UPDATE transaksi SET status_persetujuan='pending' WHERE toko_id='{c.shop}';")
    before = c.snapshot()
    owner_before = c.sql(f"SELECT to_jsonb(p) FROM profiles p WHERE id='{c.owner}';")
    # Approval fails on its SECOND audit row: proves rollback includes the first
    # successful mutation and audit insert, not just the failing target.
    target = c.owner if action == 'change_owner_status' else c.ids[-1] if action == 'approve_transactions' else c.ids[0]
    c.sql(f"ALTER TABLE audit_log ADD CONSTRAINT injected_audit_failure CHECK(target_id <> '{target}'::uuid) NOT VALID;")
    try:
        with patch('services.forecast_service.invalidate_forecast_cache') as invalidate:
            with pytest.raises(AssertionError, match='injected_audit_failure'):
                c.invoke(action)
            invalidate.assert_not_called()
        assert c.snapshot() == before
        assert c.sql(f"SELECT to_jsonb(p) FROM profiles p WHERE id='{c.owner}';") == owner_before
        assert c.logs() == []
    finally:
        c.sql('ALTER TABLE audit_log DROP CONSTRAINT injected_audit_failure;')
    c.invoke(action)
    logs = c.logs()
    assert len(logs) == (2 if action == 'approve_transactions' else 1)
    assert all(log['action'] == action for log in logs)
    assert all(log['actor_id'] == (c.admin if action == 'change_owner_status' else c.owner) for log in logs)
    if action == 'delete_transaction':
        assert logs[0]['after_json'] is None and len(c.snapshot()) == 1
    elif action == 'change_owner_status':
        assert logs[0]['before_json']['status'] == 'aktif'
        assert logs[0]['after_json']['status'] == 'nonaktif'
    else:
        assert logs[0]['before_json'] == before[0]
        assert logs[0]['after_json'] == c.snapshot()[0]


@pytest.mark.parametrize('field,value', [('tanggal','2026-10-26'),('deskripsi','Corrected'),('nominal','120'),('kategori_id',None),('tipe','pengeluaran')])
def test_atomic_owner_edit_reapproval_and_secondary_history(atomic_audit_context, field, value):
    from services.finance_service import update_transaction
    c = atomic_audit_context
    if field == 'kategori_id':
        value = str(uuid.uuid4())
        c.sql(f"INSERT INTO kategori VALUES ('{value}','{c.shop}','pemasukan');")
    values = dict(tanggal='2026-10-25',deskripsi='Sale',nominal='100',tipe='pemasukan',kategori_id=c.cat)
    values[field] = value
    if field == 'tipe':
        values['kategori_id'] = c.cat2
    update_transaction('token', c.ids[0], **values)
    row = c.snapshot()[0]
    assert row['status_persetujuan'] == 'pending'
    assert row['disetujui_oleh'] is None and row['disetujui_pada'] is None
    assert row['usulan']['source_id'] == 'source'
    assert len(row['usulan']['audit_history']) == 2
    assert row['usulan']['audit_history'][0] == {'action': 'previous'}
    assert c.logs()[0]['after_json'] == row


@pytest.mark.parametrize('status,expected', [('disetujui','disetujui'),('ditolak','pending'),('pending','pending')])
def test_atomic_noop_and_rejected_edit(atomic_audit_context, status, expected):
    from services.finance_service import update_transaction
    c = atomic_audit_context
    c.sql(f"UPDATE transaksi SET status_persetujuan='{status}' WHERE id='{c.ids[0]}';")
    update_transaction('token', c.ids[0], '2026-10-25','Sale',Decimal('100.00'),'pemasukan',c.cat)
    assert c.snapshot()[0]['status_persetujuan'] == expected


def test_atomic_rpc_rechecks_role_and_store_under_lock(atomic_audit_context):
    c = atomic_audit_context
    for actor, shop, action in [(c.cashier,c.shop,'delete_transaction'),(c.owner,str(uuid.uuid4()),'delete_transaction'),(c.cashier,c.shop,'update_transaction')]:
        with pytest.raises(AssertionError):
            c.client.rpc('mutate_finance_audited', {'p_actor':actor,'p_shop':shop,'p_action':action,'p_ids':[c.ids[0]],'p_values':{}}).execute()
    assert len(c.snapshot()) == 2 and c.logs() == []


def test_audit_permissions_unchanged(audit_postgres):
    sql = audit_postgres
    for role in ('anon','authenticated','service_role'):
        assert sql(f"SELECT has_table_privilege('{role}','public.audit_log','UPDATE') OR has_table_privilege('{role}','public.audit_log','DELETE');") == 'f'
    assert sql("SELECT relrowsecurity FROM pg_class WHERE oid='public.audit_log'::regclass;") == 't'
    assert sql("SELECT has_function_privilege('authenticated','public.mutate_finance_audited(uuid,uuid,text,uuid[],jsonb)','EXECUTE');") == 'f'


def test_import_optional_failures_log_warnings_and_preserve_committed_rows(caplog):
    import logging
    from services.smart_import_engine import execute_scalable_import
    row = {'tanggal': '2026-10-30', 'deskripsi': 'Sale', 'nominal': '100',
           'tipe': 'pemasukan', 'kategori_id': 'cat-1'}
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.execute.side_effect = RuntimeError('count unavailable')
    client.rpc.return_value.execute.return_value.data = 1
    with patch('services.smart_import_engine.verify_actor', return_value={'id': 'owner', 'toko_id': 'shop', 'role': 'pemilik_toko'}), \
         patch('services.smart_import_engine.privileged_client', return_value=client), \
         patch('repositories.finance_repository.transactions', side_effect=RuntimeError('history unavailable')), \
         patch('repositories.finance_repository.categories', side_effect=RuntimeError('categories unavailable')), \
         patch('services.smart_import_engine._validate_category') as validate, \
         patch('services.forecast_service.invalidate_forecast_cache', side_effect=RuntimeError('cache unavailable')), \
         caplog.at_level(logging.WARNING, logger='services.smart_import_engine'):
        result = execute_scalable_import('token', [row], import_id=str(uuid.uuid4()))
    assert result['inserted_count'] == 1
    assert result['failed_count'] == 0
    assert result['completion_status'] == 'completed'
    assert result['count_before'] == 0 and result['count_after'] == 1
    assert result['warnings'] == ['Forecast cache invalidation failed; imported rows remain committed.']
    validate.assert_called_once_with('shop', 'cat-1', 'pemasukan')
    for phrase in ('Duplicate preflight unavailable', 'Pre-import count unavailable',
                   'Category prefetch failed', 'Forecast cache invalidation failed',
                   'Post-import count unavailable'):
        assert phrase in caplog.text
    assert all(r.levelno >= logging.WARNING for r in caplog.records if r.name == 'services.smart_import_engine')


def test_finance_cache_failure_logs_and_returns_committed_result(caplog):
    import logging
    from services.finance_service import add_transaction
    client = MagicMock()
    committed = [{'id': 'persisted', 'status_persetujuan': 'disetujui'}]
    client.table.return_value.insert.return_value.execute.return_value.data = committed
    with patch('services.finance_service.verify_actor', return_value={'id': 'owner', 'toko_id': 'shop', 'role': 'pemilik_toko'}), \
         patch('services.finance_service._validate_category'), \
         patch('services.finance_service.privileged_client', return_value=client), \
         patch('services.forecast_service.invalidate_forecast_cache', side_effect=RuntimeError('cache offline')), \
         caplog.at_level(logging.WARNING, logger='services.finance_service'):
        result = add_transaction('token', '2026-10-30', 'Sale', '100', 'pemasukan', 'cat')
    assert result == committed
    assert 'Cache invalidation failed after add_transaction' in caplog.text
    assert 'returning committed result' in caplog.text


def test_chat_summary_failure_warns_and_returns_deterministic_numbers(caplog):
    import logging
    from services.chat_service import handle_chat_message
    actor = {'id': 'owner', 'toko_id': 'shop', 'role': 'pemilik_toko'}
    rows = [{'id': 'tx', 'tanggal': '2026-10-10', 'nominal': '200000', 'tipe': 'pemasukan',
             'kategori_id': 'cat', 'status_persetujuan': 'disetujui'}]
    with patch('services.chat_service.verify_actor', return_value=actor), \
         patch('services.chat_service.finance.list_categories', return_value=[{'id': 'cat', 'nama': 'Penjualan', 'tipe': 'pemasukan'}]), \
         patch('services.chat_service.period_rows', return_value=rows), \
         patch('services.chat_service.ask_text', side_effect=RuntimeError('LLM offline')), \
         caplog.at_level(logging.WARNING, logger='services.chat_service'):
        result = handle_chat_message('token', 'Berapa total pemasukan bulan ini?')
    assert 'Rp 200.000' in result['message']
    assert 'Chat summary generation failed; returning deterministic financial summary' in caplog.text


def test_hot_path_handlers_never_silently_pass():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for filename in ('smart_import_engine.py', 'finance_service.py', 'chat_service.py'):
        tree = ast.parse((root / 'services' / filename).read_text(encoding='utf-8-sig'))
        silent = [node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.ExceptHandler) and len(node.body) == 1 and isinstance(node.body[0], ast.Pass)]
        assert silent == [], (filename, silent)
