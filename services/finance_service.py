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

EXPENSE_DEFAULTS=['Gaji','Operasional','Pemasaran','Others']
INCOME_DEFAULTS=['Penjualan','Tambahan Modal','Pendapatan Lain','Others']
LIMITS={'pengeluaran':10,'pemasukan':5}

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
def create_category(token,name,tipe,description='',system_seed=False,keywords=None):
 a=_owner(token);shop=a['toko_id'];tipe=_type(tipe);name=name.strip()
 if not name or len(name)>80: raise ValueError('Nama kategori wajib diisi (maksimal 80 karakter)')
 existing=categories(shop)
 if len([c for c in existing if c['tipe']==tipe])>=LIMITS[tipe]: raise ValueError('Batas kategori tercapai; gabungkan kategori terlebih dahulu')
 if any(c['nama'].casefold()==name.casefold() and c['tipe']==tipe for c in existing): raise ValueError('Kategori sudah ada')
 if name.casefold()=='others' and not system_seed: raise ValueError('Others adalah kategori sistem')
 if name.casefold()=='tambahan modal' and tipe!='pemasukan': raise ValueError('Tambahan Modal harus pemasukan')
 return privileged_client().table('kategori').insert({'toko_id':shop,'nama':name,'tipe':tipe,'deskripsi':description.strip()[:500],'sistem':name.casefold()=='others','bukan_operasional':name.casefold()=='tambahan modal','kata_kunci':keywords or []}).execute().data

def rename_category(token,category_id,name,description=''):
 a=_owner(token);shop=a['toko_id'];old=category_by_id(shop,category_id)
 if not old: raise ValueError('Kategori tidak ditemukan')
 if old['sistem'] or old['bukan_operasional']: raise ValueError('Kategori sistem/modal tidak dapat diubah namanya')
 name=name.strip()
 if not name or len(name)>80: raise ValueError('Nama kategori tidak valid')
 return privileged_client().table('kategori').update({'nama':name,'deskripsi':description.strip()[:500]}).eq('id',category_id).eq('toko_id',shop).execute().data

def set_category_keywords(token,category_id,keywords):
 actor=_owner(token);shop=actor['toko_id']
 cat=category_by_id(shop,category_id)
 if not cat:raise ValueError('Kategori tidak ditemukan')
 if cat['sistem']:raise ValueError('Kategori Others tidak diindeks')
 sanitized=[str(k).strip()[:80] for k in keywords if str(k).strip()]
 if len(sanitized)>20:raise ValueError('Maksimal 20 kata kunci')
 return privileged_client().table('kategori').update({'kata_kunci':sanitized}).eq('toko_id',shop).eq('id',category_id).execute().data

def delete_category(token,category_id):
 a=_owner(token);shop=a['toko_id'];old=category_by_id(shop,category_id)
 if not old: raise ValueError('Kategori tidak ditemukan')
 if old['sistem'] or old['bukan_operasional']: raise ValueError('Kategori terlindungi')
 if privileged_client().table('transaksi').select('id').eq('toko_id',shop).eq('kategori_id',category_id).limit(1).execute().data: raise ValueError('Kategori sudah dipakai; gunakan gabung kategori')
 privileged_client().table('kategori').delete().eq('id',category_id).eq('toko_id',shop).execute()

def merge_category(token,source_id,target_id):
 a=_owner(token);shop=a['toko_id'];src=category_by_id(shop,source_id);dst=category_by_id(shop,target_id)
 if not src or not dst or source_id==target_id or src['tipe']!=dst['tipe']: raise ValueError('Kategori sumber/tujuan tidak sesuai')
 if src['sistem'] or src['bukan_operasional'] or dst['bukan_operasional']: raise ValueError('Kategori terlindungi tidak dapat digabung')
 admin=privileged_client()
 return admin.rpc('merge_category_atomic',{'p_actor':a['id'],'p_shop':shop,'p_source':source_id,'p_target':target_id}).execute().data

def list_transactions(token):
 a=_member(token);data=transactions(a['toko_id'])
 return data if a['role']=='pemilik_toko' else [t for t in data if t['dicatat_oleh']==a['id']]

def _validate_category(shop,category_id,tipe):
 cat=category_by_id(shop,category_id)
 if not cat or cat['tipe']!=tipe: raise ValueError('Kategori tidak sesuai jenis transaksi/toko')
 return cat

def add_transaction(token,tanggal,deskripsi,nominal,tipe,kategori_id,status_kategori='dikonfirmasi',skor=None,usulan=None):
 a=_member(token);shop=a['toko_id'];tipe=_type(tipe);_validate_category(shop,kategori_id,tipe)
 desc=validate_description(deskripsi)
 owner=a['role']=='pemilik_toko'
 return privileged_client().table('transaksi').insert({'toko_id':shop,'tanggal':_date(tanggal),'deskripsi':desc,'nominal':_money(nominal),'tipe':tipe,'kategori_id':kategori_id,'dicatat_oleh':a['id'],'status_persetujuan':'disetujui' if owner else 'pending','disetujui_oleh':a['id'] if owner else None,'disetujui_pada':datetime.now(timezone.utc).isoformat() if owner else None,'status_kategori':status_kategori,'skor':skor,'usulan':usulan or {}}).execute().data

def update_transaction(token,transaction_id,tanggal,deskripsi,nominal,tipe,kategori_id):
 a=_member(token);shop=a['toko_id'];old=transaction_by_id(shop,transaction_id)
 if not old: raise ValueError('Transaksi tidak ditemukan')
 if a['role']=='kasir' and (old['dicatat_oleh']!=a['id'] or old['status_persetujuan']!='pending'): raise AccessError('Kasir hanya dapat mengubah entri sendiri yang masih pending')
 tipe=_type(tipe);_validate_category(shop,kategori_id,tipe)
 try:
     desc=validate_description(deskripsi)
 except ValueError:
     raise ValueError('Keterangan tidak valid')
 payload={'tanggal':_date(tanggal),'deskripsi':desc,'nominal':_money(nominal),'tipe':tipe,'kategori_id':kategori_id,'updated_at':datetime.now(timezone.utc).isoformat(),'status_kategori':'dikonfirmasi','usulan':{}}
 return privileged_client().table('transaksi').update(payload).eq('toko_id',shop).eq('id',transaction_id).execute().data

def delete_transaction(token,transaction_id):
 a=_owner(token);shop=a['toko_id']
 if not transaction_by_id(shop,transaction_id): raise ValueError('Transaksi tidak ditemukan')
 privileged_client().table('transaksi').delete().eq('toko_id',shop).eq('id',transaction_id).execute()

def approve_transactions(token,ids,decision):
 a=_owner(token);shop=a['toko_id']
 if decision not in ('disetujui','ditolak') or not ids:raise ValueError('Keputusan atau ID tidak valid')
 return privileged_client().rpc('approve_transactions_atomic',{
  'p_actor':a['id'],'p_shop':shop,'p_ids':list(set(ids)),'p_decision':decision}).execute().data


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
