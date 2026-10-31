import csv,io,uuid
from services.auth_service import verify_actor
from services.finance_service import _money,_date,_type,_validate_category
from database.client import privileged_client
from ai.text_normalizer import clean_text_for_storage


def preview_csv(content:bytes):
    if len(content)>5_000_000:raise ValueError('CSV maksimal 5 MB')
    try:raw=content.decode('utf-8-sig')
    except UnicodeDecodeError:raise ValueError('CSV harus UTF-8')
    reader=csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames or not {'tanggal','deskripsi','nominal'}.issubset(set(reader.fieldnames)):
        raise ValueError('Kolom wajib: tanggal, deskripsi, nominal; opsional: tipe')
    result=[]
    for i,row in enumerate(reader,2):
        if i>1001:raise ValueError('Maksimal 1000 baris per impor')
        try:
            if None in row:raise ValueError('Jumlah kolom tidak sesuai')
            desc=clean_text_for_storage(row.get('deskripsi') or '')
            if not desc or len(desc)>500:raise ValueError('Deskripsi tidak valid')
            tp=(row.get('tipe') or '').strip().lower()
            if tp:_type(tp)
            result.append({'baris':i,'tanggal':_date((row.get('tanggal') or '').strip()),
                           'deskripsi':desc,'nominal':_money((row.get('nominal') or '').strip()),
                           'tipe':tp,'error':''})
        except Exception as exc:
            result.append({'baris':i,'tanggal':row.get('tanggal',''),
                           'deskripsi':row.get('deskripsi',''),'nominal':row.get('nominal',''),
                           'tipe':row.get('tipe',''),'error':str(exc)})
    if not result:raise ValueError('CSV kosong')
    return result

def enrich_rows(token,rows,progress=None):
    from services.ai_service import classify_preview
    from ai.classifier import detect_type
    output=[]
    for i,r in enumerate(rows):
        if r['error']:raise ValueError('Perbaiki kesalahan CSV terlebih dahulu')
        tp=r['tipe'] or detect_type(r['deskripsi'])
        decision=classify_preview(token,tp,r['deskripsi'])
        output.append({**r,'tipe':tp,**decision})
        if progress:progress((i+1)/len(rows))
    return output

def atomic_import(token,rows):
    actor=verify_actor(token,{'pemilik_toko','kasir'})
    if not rows or len(rows)>1000:raise ValueError('Jumlah baris harus 1–1000')
    batch=[]
    for i,r in enumerate(rows,1):
        tp=_type(r.get('tipe'));_validate_category(actor['toko_id'],r.get('kategori_id'),tp)
        desc=clean_text_for_storage(r.get('deskripsi', ''))
        if not desc or len(desc)>500:raise ValueError(f'Deskripsi baris {i} tidak valid')
        batch.append({'tanggal':_date(r['tanggal']),'deskripsi':desc,'nominal':_money(r['nominal']),
                      'tipe':tp,'kategori_id':r['kategori_id'],
                      'status_kategori':r.get('status_kategori','dikonfirmasi'),
                      'skor':r.get('skor'),'usulan':r.get('usulan') or {}})
    return privileged_client().rpc('import_transactions_atomic',{
        'p_actor':actor['id'],'p_shop':actor['toko_id'],'p_rows':batch,
        'p_batch':str(uuid.uuid4())}).execute().data

def import_rows(token,rows,category_map):
    try:
        mapped=[{**r,'kategori_id':category_map[r['tipe']]} for r in rows]
        return atomic_import(token,mapped),[]
    except Exception as exc:return 0,[str(exc)]
