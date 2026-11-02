import logging
from typing import Any, Dict, List, Optional
from database.client import privileged_client
from repositories import deletion_repository
from repositories.finance_repository import transactions
from services.auth_service import AccessError, verify_actor

logger = logging.getLogger(__name__)


def owner_request_bulk_deletion(token: str, confirmation_text: str) -> Dict[str, Any]:
    """Submit a store-scoped request to delete all transactions, requiring Superadmin approval."""
    actor = verify_actor(token, {'pemilik_toko'})
    shop_id = actor['toko_id']

    if confirmation_text.strip() != 'REQUEST DELETE':
        raise ValueError("Teks konfirmasi harus persis 'REQUEST DELETE'")

    # Check for existing pending request for this store
    existing_pending = deletion_repository.get_pending_request_for_store(shop_id)
    if existing_pending:
        raise ValueError('Sudah ada pengajuan penghapusan transaksi yang sedang menunggu persetujuan Superadmin')

    # Count current transactions
    current_txs = transactions(shop_id)
    tx_count = len(current_txs)
    if tx_count == 0:
        raise ValueError('Tidak ada transaksi yang dapat dihapus pada toko ini')

    # Create pending request without deleting any transactions
    req = deletion_repository.create_request(
        store_id=shop_id,
        requested_by=actor['id'],
        count=tx_count,
    )
    return req


def owner_cancel_deletion_request(token: str, request_id: str) -> Dict[str, Any]:
    """Store Owner cancels their own pending deletion request."""
    actor = verify_actor(token, {'pemilik_toko'})
    shop_id = actor['toko_id']

    req = deletion_repository.get_request_by_id(request_id)
    if not req:
        raise ValueError('Pengajuan tidak ditemukan')

    if req['store_id'] != shop_id:
        raise AccessError('Tidak berhak membatalkan pengajuan milik toko lain')

    if req['status'] != 'pending':
        raise ValueError('Hanya pengajuan berstatus pending yang dapat dibatalkan')

    return deletion_repository.cancel_request(request_id, shop_id)


def owner_get_latest_deletion_request(token: str) -> Optional[Dict[str, Any]]:
    """Get the active pending or most recent deletion request for the owner's store."""
    actor = verify_actor(token, {'pemilik_toko'})
    shop_id = actor['toko_id']

    pending = deletion_repository.get_pending_request_for_store(shop_id)
    if pending:
        return pending

    all_reqs = deletion_repository.list_requests_for_store(shop_id)
    return all_reqs[0] if all_reqs else None


def superadmin_list_deletion_requests(token: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Superadmin reviews all store deletion requests with store and owner metadata."""
    verify_actor(token, {'superadmin'})
    raw_requests = deletion_repository.list_all_requests(status=status)
    if not raw_requests:
        return []

    client = privileged_client()

    # Preload store names and requester profiles
    store_ids = list({r['store_id'] for r in raw_requests if r.get('store_id')})
    requester_ids = list({r['requested_by'] for r in raw_requests if r.get('requested_by')})

    store_map = {}
    if store_ids:
        try:
            s_rows = client.table('toko').select('id, nama').in_('id', store_ids).execute().data or []
            store_map = {s['id']: s['nama'] for s in s_rows}
        except Exception as exc:
            # Safe fallback: non-critical display enrichment falls back to short ID
            logger.debug("Failed preloading store names for deletion requests: %s", exc)

    user_map = {}
    if requester_ids:
        try:
            u_rows = client.table('profiles').select('id, nama, email').in_('id', requester_ids).execute().data or []
            user_map = {u['id']: f"{u.get('nama', '')} ({u.get('email', '')})" for u in u_rows}
        except Exception as exc:
            # Safe fallback: non-critical display enrichment falls back to short requester ID
            logger.debug("Failed preloading requester info for deletion requests: %s", exc)

    enriched = []
    for r in raw_requests:
        item = dict(r)
        item['store_name'] = store_map.get(r['store_id'], f"Toko {str(r['store_id'])[:8]}")
        item['requester_info'] = user_map.get(r['requested_by'], f"User {str(r['requested_by'])[:8]}")
        enriched.append(item)

    return enriched


def superadmin_reject_deletion_request(token: str, request_id: str, reason: str = '') -> Dict[str, Any]:
    """Superadmin rejects a deletion request without deleting any transactions."""
    admin = verify_actor(token, {'superadmin'})

    req = deletion_repository.get_request_by_id(request_id)
    if not req:
        raise ValueError('Pengajuan tidak ditemukan')

    if req['status'] != 'pending':
        raise ValueError(f"Pengajuan tidak dalam status pending (status saat ini: {req['status']})")

    clean_reason = reason.strip() if reason and reason.strip() else 'Ditolak oleh Superadmin'
    return deletion_repository.reject_request(
        request_id=request_id,
        reviewed_by=admin['id'],
        reason=clean_reason,
    )


def superadmin_approve_and_execute_deletion(
    token: str,
    request_id: str,
    confirmation_text: str,
) -> Dict[str, Any]:
    """Superadmin approves deletion and securely executes permanent removal of all store transactions."""
    admin = verify_actor(token, {'superadmin'})

    if confirmation_text.strip() != 'APPROVE DELETE':
        raise ValueError("Teks konfirmasi Superadmin harus persis 'APPROVE DELETE'")

    req = deletion_repository.get_request_by_id(request_id)
    if not req:
        raise ValueError('Pengajuan tidak ditemukan')

    if req['status'] != 'pending':
        raise ValueError(f"Pengajuan tidak dalam status pending (status saat ini: {req['status']})")

    target_store_id = req['store_id']
    client = privileged_client()

    try:
        rpc_res = client.rpc('execute_store_transaction_deletion', {
            'p_admin': str(admin['id']),
            'p_request_id': str(request_id),
        }).execute()
        result = rpc_res.data
        if (not isinstance(result, dict)
                or result.get('success') is not True
                or result.get('status') != 'completed'
                or str(result.get('request_id')) != str(request_id)
                or str(result.get('store_id')) != str(target_store_id)
                or type(result.get('deleted_count')) is not int
                or result['deleted_count'] < 0):
            raise ValueError('RPC tidak mengonfirmasi keberhasilan penghapusan')
        deleted_count = result['deleted_count']
    except Exception as exc:
        # The RPC owns both deletion and status transition atomically. A rejected
        # or ambiguous response must never trigger another write from the client.
        raise RuntimeError(f'Eksekusi penghapusan gagal: {exc}') from exc

    # Invalidate affected store financial and AI caches
    try:
        from services.forecast_service import invalidate_forecast_cache
        invalidate_forecast_cache(target_store_id)
    except Exception as exc:
        # Safe post-commit cleanup: forecast cache invalidation failure does not revert deletion
        logger.debug("Post-deletion forecast cache invalidation skipped: %s", exc)

    try:
        from ai.hybrid_classifier import clear_classification_cache
        clear_classification_cache(target_store_id)
    except Exception as exc:
        # Safe post-commit cleanup: classification cache invalidation failure does not revert deletion
        logger.debug("Post-deletion classification cache invalidation skipped: %s", exc)

    try:
        from ai.rag_engine import clear_memory_cache
        clear_memory_cache(target_store_id)
    except Exception as exc:
        # Safe post-commit cleanup: RAG memory cache invalidation failure does not revert deletion
        logger.debug("Post-deletion RAG cache invalidation skipped: %s", exc)

    return {
        'success': True,
        'request_id': request_id,
        'store_id': target_store_id,
        'deleted_count': deleted_count,
        'status': 'completed',
    }

