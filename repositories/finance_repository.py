from database.client import privileged_client


def _all(table, shop, order):
    client=privileged_client();items=[];page=0
    while True:
        chunk=(client.table(table).select('*').eq('toko_id',shop)
               .order(order,desc=order=='tanggal')
               .range(page*1000,(page+1)*1000-1).execute().data or [])
        items.extend(chunk)
        if len(chunk)<1000:break
        page+=1
    return items

def categories(shop):return _all('kategori',shop,'tipe')

def transactions(shop):return _all('transaksi',shop,'tanggal')

def category_by_id(shop,category_id):
    if not category_id:return None
    rows=privileged_client().table('kategori').select('*').eq('toko_id',shop).eq('id',category_id).limit(1).execute().data
    return rows[0] if rows else None

def transaction_by_id(shop,transaction_id):
    rows=privileged_client().table('transaksi').select('*').eq('toko_id',shop).eq('id',transaction_id).limit(1).execute().data
    return rows[0] if rows else None
