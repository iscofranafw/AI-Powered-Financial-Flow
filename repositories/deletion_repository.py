import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from database.client import privileged_client


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_request(r: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not r:
        return None
    out = dict(r)
    if 'transaction_count' not in out and 'transaction_count_requested' in out:
        out['transaction_count'] = out['transaction_count_requested']
    if 'transaction_count_requested' not in out and 'transaction_count' in out:
        out['transaction_count_requested'] = out['transaction_count']
    return out


def create_request(store_id: str, requested_by: str, count: int) -> Dict[str, Any]:
    """Create a new deletion request in pending status."""
    client = privileged_client()
    req_id = str(uuid.uuid4())
    payload = {
        'id': req_id,
        'store_id': str(store_id),
        'requested_by': str(requested_by),
        'requested_at': _now_iso(),
        'status': 'pending',
        'transaction_count_requested': count,
        'transaction_count_deleted': 0,
        'created_at': _now_iso(),
        'updated_at': _now_iso(),
    }

    res = client.table('transaction_deletion_requests').insert(payload).execute()
    if res.data is None:
        raise RuntimeError('Database tidak mengembalikan hasil pengajuan penghapusan')
    if res.data:
        return _normalize_request(res.data[0])

    raise RuntimeError('Pengajuan penghapusan tidak tersimpan: database tidak mengembalikan hasil')


def get_pending_request_for_store(store_id: str) -> Optional[Dict[str, Any]]:
    """Return the currently pending deletion request for a store, if any."""
    client = privileged_client()
    res = (
        client.table('transaction_deletion_requests')
        .select('*')
        .eq('store_id', str(store_id))
        .eq('status', 'pending')
        .order('requested_at', desc=True)
        .limit(1)
        .execute()
    )
    if res.data is None:
        raise RuntimeError('Database tidak mengembalikan hasil pengajuan penghapusan')
    if res.data:
        return _normalize_request(res.data[0])

    return None


def get_request_by_id(request_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve deletion request by UUID."""
    client = privileged_client()
    res = (
        client.table('transaction_deletion_requests')
        .select('*')
        .eq('id', str(request_id))
        .limit(1)
        .execute()
    )
    if res.data is None:
        raise RuntimeError('Database tidak mengembalikan hasil pengajuan penghapusan')
    if res.data:
        return _normalize_request(res.data[0])

    return None


def list_requests_for_store(store_id: str) -> List[Dict[str, Any]]:
    """List all deletion requests belonging to a specific store."""
    client = privileged_client()
    res = (
        client.table('transaction_deletion_requests')
        .select('*')
        .eq('store_id', str(store_id))
        .order('requested_at', desc=True)
        .execute()
    )
    if res.data is not None:
        return [_normalize_request(r) for r in res.data]

    raise RuntimeError('Database tidak mengembalikan daftar pengajuan penghapusan')


def list_all_requests(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List deletion requests across all stores with optional status filter."""
    client = privileged_client()
    query = client.table('transaction_deletion_requests').select('*')
    if status and status != 'Semua':
        query = query.eq('status', status.lower())
    res = query.order('requested_at', desc=True).execute()
    if res.data is not None:
        return [_normalize_request(r) for r in res.data]

    raise RuntimeError('Database tidak mengembalikan daftar pengajuan penghapusan')


def cancel_request(request_id: str, store_id: str) -> Dict[str, Any]:
    """Mark pending request as cancelled."""
    client = privileged_client()
    now_str = _now_iso()
    res = (
        client.table('transaction_deletion_requests')
        .update({'status': 'cancelled', 'updated_at': now_str})
        .eq('id', str(request_id))
        .eq('store_id', str(store_id))
        .eq('status', 'pending')
        .execute()
    )
    if res.data is None:
        raise RuntimeError('Database tidak mengembalikan hasil pengajuan penghapusan')
    if res.data:
        return _normalize_request(res.data[0])

    raise ValueError('Pengajuan tidak dapat dibatalkan atau status bukan pending')


def reject_request(request_id: str, reviewed_by: str, reason: str) -> Dict[str, Any]:
    """Mark pending request as rejected."""
    client = privileged_client()
    now_str = _now_iso()
    clean_reason = reason.strip() if reason and reason.strip() else 'Ditolak oleh Superadmin'
    res = (
        client.table('transaction_deletion_requests')
        .update({
            'status': 'rejected',
            'reviewed_by': str(reviewed_by),
            'reviewed_at': now_str,
            'rejection_reason': clean_reason,
            'updated_at': now_str,
        })
        .eq('id', str(request_id))
        .eq('status', 'pending')
        .execute()
    )
    if res.data is None:
        raise RuntimeError('Database tidak mengembalikan hasil pengajuan penghapusan')
    if res.data:
        return _normalize_request(res.data[0])

    raise ValueError('Pengajuan tidak dapat ditolak atau status bukan pending')


def complete_deletion(
    request_id: str,
    store_id: str,
    reviewed_by: str,
    deleted_count: int,
) -> Dict[str, Any]:
    """Mark request as completed with execution metrics."""
    client = privileged_client()
    now_str = _now_iso()
    res = (
        client.table('transaction_deletion_requests')
        .update({
            'status': 'completed',
            'reviewed_by': str(reviewed_by),
            'reviewed_at': now_str,
            'executed_at': now_str,
            'transaction_count_deleted': deleted_count,
            'updated_at': now_str,
        })
        .eq('id', str(request_id))
        .eq('store_id', str(store_id))
        .eq('status', 'pending')
        .execute()
    )
    if res.data is None:
        raise RuntimeError('Database tidak mengembalikan hasil pengajuan penghapusan')
    if res.data:
        return _normalize_request(res.data[0])

    raise ValueError('Pengajuan tidak dapat diselesaikan atau status bukan pending')


def mark_failed(request_id: str, error_message: str) -> None:
    """Record execution failure state (only if request was not already completed)."""
    client = privileged_client()
    now_str = _now_iso()
    res = client.table('transaction_deletion_requests').update({
        'status': 'failed',
        'error_message': str(error_message),
        'updated_at': now_str,
    }).eq('id', str(request_id)).neq('status', 'completed').execute()
    if res.data is None:
        raise RuntimeError('Database tidak mengonfirmasi pembaruan status pengajuan')
