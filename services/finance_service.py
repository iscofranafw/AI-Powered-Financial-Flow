import logging
import json
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from database.client import privileged_client
from services.auth_service import verify_actor, AccessError
from repositories.finance_repository import categories, transactions, category_by_id, transaction_by_id
from services.validation_service import (
    validate_amount,
    validate_date,
    validate_type,
    validate_description,
)

logger = logging.getLogger(__name__)

EXPENSE_DEFAULTS=['Gaji','Operasional','Pemasaran','Others']
INCOME_DEFAULTS=['Penjualan','Tambahan Modal','Pendapatan Lain','Others']
# Batas rekomendasi AI per operasi penemuan kategori (Pemilik Toko memiliki kuota kategori tanpa batas)
AI_RECOMMENDATION_LIMITS = {'pengeluaran': 7, 'pemasukan': 7}
LIMITS = AI_RECOMMENDATION_LIMITS

def check_category_quota(token, tipe=None):
    """Inspect category count. Store owners have unlimited categories; AI recommendations are capped at 7 per discovery."""
    a = _member(token)
    shop = a['toko_id']
    cats = categories(shop)
    types = [tipe] if tipe else ['pengeluaran', 'pemasukan']
    res = {}
    for tp in types:
        count = len([c for c in cats if c['tipe'] == tp])
        limit = LIMITS.get(tp, 7)
        res[tp] = {
            'count': count,
            'limit': limit,
            'ai_recommendation_limit': limit,
            'exceeded': False,
            'at_limit': False,
            'can_add': True,  # Owners have unlimited category creation capability
            'warning': None,
        }
    return res if not tipe else res[tipe]

def _owner(token): return verify_actor(token,{'pemilik_toko'})
def _member(token): return verify_actor(token,{'pemilik_toko','kasir'})
def _money(value): return validate_amount(value)
def _date(value): return validate_date(value)
def _type(value): return validate_type(value)

def list_categories(token):
 a=_member(token); return categories(a['toko_id'])
def initialize_categories(token):
 a=_owner(token); shop=a['toko_id']; existing=categories(shop)
 if existing: return existing
 admin=privileged_client()
 for tipe,names in [('pengeluaran',EXPENSE_DEFAULTS),('pemasukan',INCOME_DEFAULTS)]:
  for name in names:
   admin.table('kategori').upsert({'toko_id':shop,'nama':name,'tipe':tipe,'sistem':name=='Others','bukan_operasional':name=='Tambahan Modal'},on_conflict='toko_id,tipe,nama').execute()
 return categories(shop)
def _invalidate_classification_state(shop, tipe=None, rebuild_index=False):
    """Clear classification/RAG caches and optionally rebuild FAISS index for shop."""
    try:
        from ai.hybrid_classifier import clear_classification_cache
        clear_classification_cache(shop)
        from ai.rag_engine import clear_memory_cache, rebuild_store_index
        clear_memory_cache(shop)
        if rebuild_index:
            rebuild_store_index(shop, tipe)
    except Exception as exc:
        logger.warning("Classification/RAG cache refresh failed for shop %s: %s", shop, exc)


def create_category(token,name,tipe,description='',system_seed=False,keywords=None):
 a=_owner(token);shop=a['toko_id'];tipe=_type(tipe);name=name.strip()
 if not name or len(name)>80: raise ValueError('Nama kategori wajib diisi (maksimal 80 karakter)')
 existing=categories(shop)
 if any(c['nama'].casefold()==name.casefold() and c['tipe']==tipe for c in existing): raise ValueError('Kategori sudah ada')
 if name.casefold()=='others' and not system_seed: raise ValueError('Others adalah kategori sistem')
 if name.casefold()=='tambahan modal' and tipe!='pemasukan': raise ValueError('Tambahan Modal harus pemasukan')
 sanitized_kw = [str(k).strip()[:80] for k in (keywords or []) if str(k).strip()][:20]
 res = privileged_client().table('kategori').insert({'toko_id':shop,'nama':name,'tipe':tipe,'deskripsi':description.strip()[:500],'sistem':name.casefold()=='others','bukan_operasional':name.casefold()=='tambahan modal','kata_kunci':sanitized_kw}).execute().data
 is_real_row = isinstance(res, list) and bool(res) and isinstance(res[0], dict) and bool(res[0].get('toko_id'))
 _invalidate_classification_state(shop, tipe, rebuild_index=is_real_row)
 return res

def rename_category(token,category_id,name,description='',keywords=None):
 a=_owner(token);shop=a['toko_id'];old=category_by_id(shop,category_id)
 if not old: raise ValueError('Kategori tidak ditemukan')
 if old['sistem'] or old['bukan_operasional']: raise ValueError('Kategori sistem/modal tidak dapat diubah namanya')
 name=name.strip()
 if not name or len(name)>80: raise ValueError('Nama kategori tidak valid')
 updates = {'nama':name,'deskripsi':description.strip()[:500]}
 if keywords is not None:
     sanitized_kw = [str(k).strip()[:80] for k in keywords if str(k).strip()]
     if len(sanitized_kw) > 20:
         raise ValueError('Maksimal 20 kata kunci')
     updates['kata_kunci'] = sanitized_kw
 res = privileged_client().table('kategori').update(updates).eq('id',category_id).eq('toko_id',shop).execute().data
 is_real_row = isinstance(res, list) and bool(res) and isinstance(res[0], dict) and bool(res[0].get('toko_id'))
 _invalidate_classification_state(shop, old.get('tipe'), rebuild_index=is_real_row)
 return res

def set_category_keywords(token,category_id,keywords):
 actor=_owner(token);shop=actor['toko_id']
 cat=category_by_id(shop,category_id)
 if not cat:raise ValueError('Kategori tidak ditemukan')
 if cat['sistem']:raise ValueError('Kategori Others tidak diindeks')
 sanitized=[str(k).strip()[:80] for k in keywords if str(k).strip()]
 if len(sanitized)>20:raise ValueError('Maksimal 20 kata kunci')
 res = privileged_client().table('kategori').update({'kata_kunci':sanitized}).eq('toko_id',shop).eq('id',category_id).execute().data
 is_real_row = isinstance(res, list) and bool(res) and isinstance(res[0], dict) and bool(res[0].get('toko_id'))
 _invalidate_classification_state(shop, cat.get('tipe'), rebuild_index=is_real_row)
 return res


def plan_category_keywords(store_categories, history, business_type='Umum', csv_rows=None):
    """Read-only plan; the same operation is used by UI and measurement tools."""
    from ai.category_evidence import compile_keyword_plan, resolve_sample_owner
    evidence = list(history)
    prepared = compile_keyword_plan(store_categories, history, business_type)['categories']
    # Explicit CSV category input outranks inferred examples. Remaining upload
    # descriptions are proposed against the owner's allowed categories only.
    indexed = {(r.get('tipe'), str(r.get('deskripsi', '')).strip().casefold()): r
               for r in evidence}
    seen_upload = set()
    for row in csv_rows or []:
        if row.get('error') or not row.get('deskripsi'):
            continue
        key = (row.get('tipe'), str(row['deskripsi']).strip().casefold())
        declared_name = str(row.get('csv_category') or row.get('kategori') or '').strip()
        upload_key = (*key, str(row.get('kategori_id') or declared_name))
        if upload_key in seen_upload:
            continue
        seen_upload.add(upload_key)
        cat = next((c for c in store_categories if c.get('tipe') == row.get('tipe')
                    and (str(c['id']) == str(row.get('kategori_id') or '')
                         or (declared_name and str(c['nama']).casefold() == declared_name.casefold()))), None)
        if cat:
            mapped = {**row, 'kategori_id': cat['id'], '_source_priority': 2, '_sample_priority': 2}
            evidence.append(mapped)
            indexed[key] = mapped
            continue
        if key in indexed:
            evidence.append({**row, 'kategori_id': indexed[key]['kategori_id'], '_sample_priority': 2})
            continue
        # Same user item across action/quantity variants retains its confirmed
        # category. No product vocabulary is embedded in this equivalence rule.
        from ai.category_evidence import phrase_key
        matching_ids = {r.get('kategori_id') for r in evidence
                        if r.get('tipe') == row.get('tipe')
                        and phrase_key(r.get('deskripsi', '')) == phrase_key(row['deskripsi'])}
        if len(matching_ids) == 1:
            assignment = next(iter(matching_ids))
        else:
            assignment = resolve_sample_owner(row['tipe'], row['deskripsi'], prepared, business_type)['kategori_id']
        if assignment:
            mapped = {**row, 'kategori_id': assignment, '_sample_priority': 2}
            evidence.append(mapped)
            indexed[key] = mapped
    plan = compile_keyword_plan(store_categories, evidence, business_type)
    plan['evidence'] = evidence
    return plan


def auto_fill_category_keywords(token, overwrite=False, csv_rows=None):
    """Fill missing keywords from tenant history/upload, preserving filled fields."""
    from ai.store_context import get_store_business_context
    actor = _owner(token)
    shop = actor['toko_id']
    b_type = get_store_business_context(shop).get('bidang_usaha', 'Umum')
    store_categories = categories(shop)
    # Existing all-type 1000-row contract is retained. Corpus building queries
    # 1000 per type independently, so income history is also indexed there.
    history = (privileged_client().table('transaksi').select('kategori_id,deskripsi,tipe')
               .eq('toko_id', shop).eq('status_persetujuan', 'disetujui')
               .eq('status_kategori', 'dikonfirmasi')
               .order('tanggal', desc=True).limit(1000).execute().data or [])
    plan = plan_category_keywords(store_categories, history, b_type, csv_rows)
    originals = {c['id']: c for c in store_categories}
    updated = []
    for cat in plan['categories']:
        old = originals[cat['id']]
        if cat.get('sistem') or cat.get('bukan_operasional') or str(cat.get('nama', '')).casefold() == 'others':
            continue
        if old.get('kata_kunci') and not overwrite:
            continue
        if cat.get('kata_kunci'):
            set_category_keywords(token, cat['id'], cat['kata_kunci'])
            updated.append({'id': cat['id'], 'nama': cat['nama'], 'keywords': cat['kata_kunci']})
    for warning in plan['warnings']:
        logger.warning(warning)
    return {'updated_count': len(updated), 'updated': updated,
            'warnings': plan['warnings'], 'keyword_audit': plan['audit']}

def delete_category(token,category_id):
 a=_owner(token);shop=a['toko_id'];old=category_by_id(shop,category_id)
 if not old: raise ValueError('Kategori tidak ditemukan')
 if old['sistem'] or old['bukan_operasional']: raise ValueError('Kategori terlindungi')
 if privileged_client().table('transaksi').select('id').eq('toko_id',shop).eq('kategori_id',category_id).limit(1).execute().data: raise ValueError('Kategori sudah dipakai; gunakan gabung kategori')
 privileged_client().table('kategori').delete().eq('id',category_id).eq('toko_id',shop).execute()
 _invalidate_classification_state(shop, old.get('tipe'), rebuild_index=True)

def merge_category(token,source_id,target_id):
 a=_owner(token);shop=a['toko_id'];src=category_by_id(shop,source_id);dst=category_by_id(shop,target_id)
 if not src or not dst or source_id==target_id or src['tipe']!=dst['tipe']: raise ValueError('Kategori sumber/tujuan tidak sesuai')
 if src['sistem'] or src['bukan_operasional'] or dst['bukan_operasional']: raise ValueError('Kategori terlindungi tidak dapat digabung')
 admin=privileged_client()
 res = admin.rpc('merge_category_atomic',{'p_actor':a['id'],'p_shop':shop,'p_source':source_id,'p_target':target_id}).execute().data
 try:
     from services.forecast_service import invalidate_forecast_cache
     invalidate_forecast_cache(shop)
     from ai.hybrid_classifier import clear_classification_cache
     clear_classification_cache(shop)
     from ai.rag_engine import clear_memory_cache
     clear_memory_cache(shop)
 except Exception as exc:
     logger.warning("Cache invalidation failed after merge_category for shop %s; returning committed result: %s", shop, exc)
     return res
 return res

def list_transactions(token):
 a=_member(token);data=transactions(a['toko_id'])
 return data if a['role']=='pemilik_toko' else [t for t in data if t['dicatat_oleh']==a['id']]

def _validate_category(shop,category_id,tipe):
 cat=category_by_id(shop,category_id)
 if not cat or cat['tipe']!=tipe: raise ValueError('Kategori tidak sesuai jenis transaksi/toko')
 return cat

def add_transaction(token,tanggal,deskripsi,nominal,tipe,kategori_id,status_kategori='dikonfirmasi',skor=None,usulan=None):
 a=_member(token);shop=a['toko_id'];tipe=_type(tipe)
 ai_pending = (usulan or {}).get('ai_pending') is True
 if ai_pending:
     kategori_id = None
     status_kategori = 'perlu_tinjauan'
 else:
     _validate_category(shop,kategori_id,tipe)
 desc=validate_description(deskripsi)
 owner=a['role']=='pemilik_toko' and not ai_pending
 res = privileged_client().table('transaksi').insert({'toko_id':shop,'tanggal':_date(tanggal),'deskripsi':desc,'nominal':_money(nominal),'tipe':tipe,'kategori_id':kategori_id,'dicatat_oleh':a['id'],'status_persetujuan':'disetujui' if owner else 'pending','disetujui_oleh':a['id'] if owner else None,'disetujui_pada':datetime.now(timezone.utc).isoformat() if owner else None,'status_kategori':status_kategori,'skor':skor,'usulan':usulan or {}}).execute().data
 if owner:
     try:
         from services.forecast_service import invalidate_forecast_cache
         invalidate_forecast_cache(shop)
         from ai.hybrid_classifier import clear_classification_cache
         clear_classification_cache(shop)
         from ai.rag_engine import clear_memory_cache
         clear_memory_cache(shop)
     except Exception as exc:
         logger.warning("Cache invalidation failed after add_transaction for shop %s; returning committed result: %s", shop, exc)
         return res
 return res

def _audited_mutation(client, actor_id, shop_id, action, ids, values=None):
 return client.rpc('mutate_finance_audited', {
     'p_actor': actor_id, 'p_shop': shop_id, 'p_action': action,
     'p_ids': ids, 'p_values': json.loads(json.dumps(values or {}, default=str)),
 }).execute().data


def update_transaction(token,transaction_id,tanggal,deskripsi,nominal,tipe,kategori_id):
 a=_member(token);shop=a['toko_id'];old=transaction_by_id(shop,transaction_id)
 if not old: raise ValueError('Transaksi tidak ditemukan')
 if a['role']=='kasir':
     if old['dicatat_oleh'] != a['id']:
         raise AccessError('Kasir hanya dapat mengubah entri yang dicatat sendiri')
     if old.get('status_persetujuan') not in ('pending', 'ditolak'):
         raise AccessError('Kasir tidak dapat mengubah transaksi yang sudah disetujui')
 tipe=_type(tipe)
 if not kategori_id and (old.get('usulan') or {}).get('ai_pending') is True:
     kategori_id = None
 else:
     _validate_category(shop,kategori_id,tipe)
 try:
     desc=validate_description(deskripsi)
 except ValueError:
     raise ValueError('Keterangan tidak valid')

 payload = {'tanggal': _date(tanggal), 'deskripsi': desc, 'nominal': _money(nominal),
            'tipe': tipe, 'kategori_id': kategori_id}
 # Authorization, current snapshots, reapproval and secondary history are
 # evaluated again under database row locks by the audited mutation RPC.
 res = _audited_mutation(privileged_client(), a['id'], shop, 'update_transaction', [transaction_id], payload)
 if not res:
     raise RuntimeError('Perubahan transaksi tidak tersimpan')
 try:
     from ai.hybrid_classifier import clear_classification_cache
     clear_classification_cache(shop)
     from ai.rag_engine import clear_memory_cache
     clear_memory_cache(shop)
     from services.forecast_service import invalidate_forecast_cache
     invalidate_forecast_cache(shop)
 except Exception as exc:
     logger.warning("Cache invalidation failed after update_transaction for shop %s; returning committed result: %s", shop, exc)
     return res
 return res

def delete_transaction(token,transaction_id):
 a=_owner(token);shop=a['toko_id']
 old = transaction_by_id(shop,transaction_id)
 if not old: raise ValueError('Transaksi tidak ditemukan')
 client = privileged_client()
 deleted = _audited_mutation(client, a['id'], shop, 'delete_transaction', [transaction_id])
 if not deleted:
     raise RuntimeError('Penghapusan transaksi tidak dikonfirmasi')
 try:
     from services.forecast_service import invalidate_forecast_cache
     invalidate_forecast_cache(shop)
 except Exception as exc:
     logger.warning("Cache invalidation failed after delete_transaction for shop %s; returning committed result: %s", shop, exc)
     return None

def approve_transactions(token,ids,decision):
 a=_owner(token);shop=a['toko_id']
 if decision not in ('disetujui','ditolak') or not ids:raise ValueError('Keputusan atau ID tidak valid')
 client = privileged_client()
 unique_ids = list(dict.fromkeys(ids))
 res = client.rpc('approve_transactions_atomic',{
  'p_actor':a['id'],'p_shop':shop,'p_ids':unique_ids,'p_decision':decision}).execute().data
 try:
     from services.forecast_service import invalidate_forecast_cache
     invalidate_forecast_cache(shop)
     _invalidate_classification_state(shop)
 except Exception as exc:
     logger.warning("Cache invalidation failed after approve_transactions for shop %s; returning committed result: %s", shop, exc)
     return res
 return res


def dashboard_summary(token):
 a=_owner(token);shop=a['toko_id'];approved=[t for t in transactions(shop) if t['status_persetujuan']=='disetujui'];cats={c['id']:c for c in categories(shop)}
 income=sum(Decimal(str(t['nominal'])) for t in approved if t['tipe']=='pemasukan')
 expense=sum(Decimal(str(t['nominal'])) for t in approved if t['tipe']=='pengeluaran')
 capital=sum(Decimal(str(t['nominal'])) for t in approved if t['tipe']=='pemasukan' and cats.get(t['kategori_id'],{}).get('bukan_operasional'))
 by_category={}
 for t in approved:
  if t['tipe']=='pengeluaran':
   name=cats.get(t['kategori_id'],{}).get('nama','?');by_category[name]=by_category.get(name,Decimal(0))+Decimal(str(t['nominal']))
 return {'pemasukan':income,'pengeluaran':expense,'tambahan_modal':capital,'omzet':income-capital,'arus_kas_operasional':income-capital-expense,'arus_kas_total':income-expense,'pengeluaran_per_kategori':by_category,'jumlah_disetujui':len(approved)}


def request_deletion_all_transactions(token, confirmation_text):
    from services.deletion_service import owner_request_bulk_deletion
    return owner_request_bulk_deletion(token, confirmation_text)


def cancel_deletion_request(token, request_id):
    from services.deletion_service import owner_cancel_deletion_request
    return owner_cancel_deletion_request(token, request_id)


def get_deletion_status(token):
    from services.deletion_service import owner_get_latest_deletion_request
    return owner_get_latest_deletion_request(token)
