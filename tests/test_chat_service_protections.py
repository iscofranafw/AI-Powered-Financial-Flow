"""Regression coverage for protections lost between the v5 and v6 chat service."""
import logging
from copy import deepcopy
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from ai.financial_chat import Intent
from services import chat_service as chat


@pytest.fixture(autouse=True)
def isolated_confirmation_state():
    for name in ('_CONFIRMING_PROPOSALS', '_CONFIRMED_PROPOSALS'):
        getattr(chat, name, {}).clear()
    yield
    for name in ('_CONFIRMING_PROPOSALS', '_CONFIRMED_PROPOSALS'):
        getattr(chat, name, {}).clear()


@pytest.fixture
def actor(monkeypatch):
    value = {'id': 'owner-restore', 'role': 'pemilik_toko', 'toko_id': 'shop-restore'}
    monkeypatch.setattr(chat, 'verify_actor', lambda *args: value)
    return value


@pytest.fixture
def proposal(actor):
    return {'id': str(uuid4()), 'toko_id': actor['toko_id'], 'status': 'pending_confirmation',
            'tanggal': '2026-10-31', 'deskripsi': 'Penjualan', 'nominal': '100000',
            'tipe': 'pemasukan', 'kategori_id': 'category-sale',
            'usulan': {'source_id': 'existing-source'}}


@pytest.fixture
def persisted():
    return {'id': 'persisted-restore', 'status_persetujuan': 'disetujui'}


def test_restore_confirm_injects_proposal_id_without_losing_metadata(proposal, persisted):
    original_meta = deepcopy(proposal['usulan'])
    with patch.object(chat.finance, 'add_transaction', return_value=[persisted]) as insert:
        result = chat.confirm_proposal('token', proposal)
    assert result['success'] is True
    assert insert.call_args.kwargs['usulan'] == {**original_meta, 'proposal_id': proposal['id']}
    assert proposal['usulan'] == original_meta


def test_restore_same_worker_reuses_result_for_an_independent_proposal_copy(proposal, persisted):
    second_proposal = deepcopy(proposal)
    with patch.object(chat.finance, 'add_transaction', return_value=[persisted]) as insert:
        first = chat.confirm_proposal('token', proposal)
        second = chat.confirm_proposal('token', second_proposal)
    insert.assert_called_once()
    assert first['record'] == second['record'] == persisted
    assert second['already_confirmed'] is True
    assert second_proposal['status'] == 'confirmed'


def test_restore_other_worker_unique_conflict_returns_persisted_row(proposal, persisted, actor):
    class UniqueViolation(Exception):
        code = '23505'

    other_worker_proposal = deepcopy(proposal)
    client = MagicMock()
    query = client.table.return_value.select.return_value.eq.return_value.contains.return_value.limit.return_value
    query.execute.return_value.data = [persisted]
    with patch.object(chat.finance, 'add_transaction', side_effect=[[persisted], UniqueViolation('duplicate proposal')]) as insert, \
         patch.object(chat, 'privileged_client', return_value=client):
        first = chat.confirm_proposal('token', proposal)
        getattr(chat, '_CONFIRMED_PROPOSALS', {}).clear()  # Independent worker has no local confirmed cache.
        second = chat.confirm_proposal('token', other_worker_proposal)
    assert first['success'] is True
    assert second['success'] is True
    assert second['already_persisted'] is True
    assert second['record'] == first['record'] == persisted
    assert other_worker_proposal['status'] == 'confirmed'
    assert insert.call_count == 2  # First insert succeeds; the second is rejected by the database.
    client.table.assert_called_once_with('transaksi')
    client.table.return_value.select.return_value.eq.assert_called_once_with('toko_id', actor['toko_id'])
    client.table.return_value.select.return_value.eq.return_value.contains.assert_called_once_with(
        'usulan', {'proposal_id': proposal['id']})


def test_restore_inflight_guard_prevents_reentrant_insert(proposal, persisted, actor):
    nested = []
    def insert_transaction(*args, **kwargs):
        if not nested:
            nested.append(None)
            nested[0] = chat.confirm_proposal('token', deepcopy(proposal))
        return [persisted]
    with patch.object(chat.finance, 'add_transaction', side_effect=insert_transaction) as insert:
        result = chat.confirm_proposal('token', proposal)
    assert result['success'] is True
    assert nested[0]['success'] is False
    assert 'sedang diproses' in nested[0]['message']
    insert.assert_called_once()
    assert (actor['toko_id'], proposal['id']) not in chat._CONFIRMING_PROPOSALS


def test_restore_missing_proposal_id_rejected_before_insert(proposal, persisted):
    proposal.pop('id')
    with patch.object(chat.finance, 'add_transaction', return_value=[persisted]) as insert:
        result = chat.confirm_proposal('token', proposal)
    assert result['success'] is False
    insert.assert_not_called()
    assert proposal['status'] == 'pending_confirmation'


def test_restore_unconfirmed_database_record_does_not_confirm(proposal):
    with patch.object(chat.finance, 'add_transaction', return_value=[{'status_persetujuan': 'disetujui'}]):
        result = chat.confirm_proposal('token', proposal)
    assert result['success'] is False
    assert proposal['status'] == 'pending_confirmation'


@pytest.mark.parametrize('lookup', ['missing', 'unavailable'])
def test_restore_unique_conflict_lookup_failure_stays_pending(proposal, actor, lookup, caplog):
    class UniqueViolation(Exception):
        code = '23505'
    client = MagicMock()
    query = client.table.return_value.select.return_value.eq.return_value.contains.return_value.limit.return_value
    query.execute.return_value.data = []
    if lookup == 'unavailable':
        query.execute.side_effect = RuntimeError('lookup offline')
    with patch.object(chat.finance, 'add_transaction', side_effect=UniqueViolation('duplicate')), \
         patch.object(chat, 'privileged_client', return_value=client), caplog.at_level(logging.ERROR):
        result = chat.confirm_proposal('token', proposal)
    assert result['success'] is False
    assert proposal['status'] == 'pending_confirmation'
    query.execute.assert_called_once()
    assert (actor['toko_id'], proposal['id']) not in chat._CONFIRMING_PROPOSALS
    assert caplog.records


@pytest.mark.parametrize('number', ['42', '14', '30'])
def test_restore_rejects_ungrounded_two_digit_number(number):
    assert not chat._is_numeric_consistent('Total Rp 200.000 dari 5 transaksi.',
                                           f'Total Rp 200.000 dari {number} transaksi.')


@pytest.mark.parametrize('fact,candidate', [
    ('Ada 5 transaksi.', 'Pertumbuhan mencapai 5%.'),
    ('Margin 15% dari 5 transaksi.', 'Margin 5% dari 5 transaksi.'),
])
def test_restore_rejects_ungrounded_percentage(fact, candidate):
    assert not chat._is_numeric_consistent(fact, candidate)


def test_restore_accepts_grounded_numbers_and_percentage():
    assert chat._is_numeric_consistent('Total 42 transaksi dengan margin 15%.',
                                      'Margin tercatat 15 % dari 42 transaksi.')


def test_probe_a_rejects_invented_rp_amount_from_count():
    """(a) Probe A: bare count in facts ('500 transaksi') invented as Rp amount ('Rp 500.') in candidate must return False."""
    fact = 'Total pemasukan Rp 200.000 dari 500 transaksi.'
    candidate = 'Total pemasukan Rp 500.'
    assert chat._is_numeric_consistent(fact, candidate) is False


def test_probe_b_accepts_grounded_rp_amount_and_count():
    """(b) Probe B: grounded fact with valid Rp and count must return True."""
    fact = 'Total pemasukan Rp 200.000 dari 5 transaksi.'
    candidate = 'Total pemasukan Rp 200.000.'
    assert chat._is_numeric_consistent(fact, candidate) is True


def test_angka_rp_parsing():
    """(c) _angka_rp parsing: 'Rp 1.500.000' -> {1500000}, 'Rp200.000' -> {200000}, '500 transaksi' -> empty set."""
    assert chat._angka_rp('Rp 1.500.000') == {1500000}
    assert chat._angka_rp('Rp200.000') == {200000}
    assert chat._angka_rp('500 transaksi') == set()


def test_candidate_without_rp_amounts_passes_when_grounded():
    """(d) Candidate with no Rp amounts at all still passes when its other numbers are grounded."""
    fact = 'Total pemasukan Rp 200.000 dari 42 transaksi.'
    candidate = 'Tercatat ada 42 transaksi dalam periode ini.'
    assert chat._is_numeric_consistent(fact, candidate) is True


def test_restore_cashier_lookup_failure_is_logged(actor, caplog):
    rows = [{'id': 'pending-row', 'tanggal': '2026-10-31', 'nominal': '100', 'tipe': 'pemasukan',
             'kategori_id': 'category-sale', 'dicatat_oleh': 'cashier', 'status_persetujuan': 'pending'}]
    with patch.object(chat.finance, 'list_transactions', return_value=rows), \
         patch.object(chat.finance, 'list_categories', return_value=[]), \
         patch.object(chat.cashier_service, 'list_cashiers', side_effect=RuntimeError('cashier lookup offline')), \
         caplog.at_level(logging.WARNING):
        result = chat._handle_pending_approvals('token', 'transaksi pending', actor)
    assert 'Kasir' in result['message']
    assert 'cashier lookup offline' in caplog.text


def test_restore_summary_failure_is_logged(actor, caplog):
    rows = [{'tanggal': '2026-10-31', 'nominal': '200000', 'tipe': 'pemasukan',
             'kategori_id': 'category-sale', 'status_persetujuan': 'disetujui'}]
    with patch.object(chat.finance, 'list_categories', return_value=[]), \
         patch.object(chat, 'period_rows', return_value=rows), \
         patch.object(chat, 'ask_text', side_effect=RuntimeError('summary offline')), caplog.at_level(logging.WARNING):
        result = chat.handle_chat_message('token', 'Berapa total pemasukan bulan ini?')
    assert '200.000' in result['message']
    assert 'summary offline' in caplog.text


def test_restore_inventory_clarification_does_not_create_proposal(actor):
    with patch.object(chat, 'build_transaction_proposal') as build:
        result = chat.handle_chat_message('token', 'ambil 5 bungkus indomie')
    assert result['intent'] == Intent.CLARIFICATION_NEEDED
    assert result['type'] == 'message'
    assert 'prive' in result['message'] and 'rusak/hilang' in result['message']
    build.assert_not_called()


def test_restore_page_context_is_explained_before_planning(actor):
    with patch.object(chat, 'plan_intent', side_effect=AssertionError('Page inquiry should be answered directly')):
        result = chat.handle_chat_message('token', 'Jelaskan halaman ini', page_context='Kelola Kategori')
    assert result['intent'] == Intent.EXPLAIN_APP_FEATURE
    assert 'Kelola Kategori' in result['message']
    assert 'pos-pos anggaran' in result['message']


def test_restore_store_name_in_greeting(actor):
    with patch('ai.store_context.get_store_business_context', return_value={'nama': 'Toko Regresi'}):
        result = chat.handle_chat_message('token', 'Halo, kamu bisa bantu apa saja?')
    assert 'Toko Regresi' in result['message']


def test_restore_category_counts_do_not_claim_fixed_quotas(actor):
    categories = [{'id': 'income', 'nama': 'Penjualan', 'tipe': 'pemasukan'},
                  {'id': 'expense', 'nama': 'Operasional', 'tipe': 'pengeluaran'}]
    with patch.object(chat.finance, 'list_categories', return_value=categories):
        result = chat._handle_category_listing('token', Intent.LIST_ALL_CATEGORIES, actor)
    assert 'Kategori Pemasukan (1)' in result['message']
    assert 'Kategori Pengeluaran (1)' in result['message']
