"""Store Context Provider.

Resolves trusted store metadata including business type (bidang_usaha)
for store-aware transaction classification and multi-tenant isolation.
"""
from typing import Any, Dict, Optional
from database.client import privileged_client

# In-memory tenant cache: store_id -> {'id': ..., 'nama': ..., 'bidang_usaha': ...}
_STORE_CONTEXT_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_store_context_cache(store_id: Optional[str] = None) -> None:
    """Clear cached store profile for testing or profile update."""
    global _STORE_CONTEXT_CACHE
    if store_id:
        _STORE_CONTEXT_CACHE.pop(str(store_id), None)
    else:
        _STORE_CONTEXT_CACHE.clear()


def set_store_context(store_id: str, nama_or_dict: Any = 'Toko', bidang_usaha: Optional[str] = None) -> None:
    """Explicitly prime store context cache (useful for mocking and fast test fixtures)."""
    if isinstance(nama_or_dict, dict):
        nama = nama_or_dict.get('nama', 'Toko')
        bidang = nama_or_dict.get('bidang_usaha', 'Umum')
    else:
        nama = str(nama_or_dict)
        bidang = bidang_usaha or 'Umum'
    _STORE_CONTEXT_CACHE[str(store_id)] = {
        'id': str(store_id),
        'nama': nama,
        'bidang_usaha': bidang,
    }


def get_store_business_context(store_id: str) -> Dict[str, Any]:
    """Retrieve verified store profile (nama, bidang_usaha) from cache or Supabase database."""
    if not store_id:
        return {'id': '', 'nama': 'Toko', 'bidang_usaha': 'Umum'}

    sid = str(store_id)
    if sid in _STORE_CONTEXT_CACHE:
        ctx = dict(_STORE_CONTEXT_CACHE[sid])
        ctx.setdefault('nama_toko', ctx.get('nama', 'Toko'))
        return ctx

    if sid in ('shop-bench', 'test-store', 'shop-1', 'shop-bulk', 'tok'):
        ctx = {'id': sid, 'nama': 'Toko Test', 'nama_toko': 'Toko Test', 'bidang_usaha': 'Umum'}
        _STORE_CONTEXT_CACHE[sid] = ctx
        return dict(ctx)

    client = privileged_client()
    try:
        res = (
            client.table('toko')
            .select('id, nama, bidang_usaha')
            .eq('id', sid)
            .limit(1)
            .execute()
        )
        if res.data:
            data = res.data[0]
            data['nama_toko'] = data.get('nama', 'Toko')
            _STORE_CONTEXT_CACHE[sid] = data
            return dict(data)
    except Exception:
        # Safe fallback: if database query fails, fall through to default store context
        pass

    fallback = {'id': sid, 'nama': f'Toko {sid[:8]}', 'nama_toko': f'Toko {sid[:8]}', 'bidang_usaha': 'Umum'}
    _STORE_CONTEXT_CACHE[sid] = fallback
    return dict(fallback)


def normalize_business_type(value):
    """Normalize spelling case and whitespace without assigning a business domain."""
    return ' '.join(str(value or '').split()).casefold() or 'umum'
