def own_profile(client, uid):
    rows = client.table("profiles").select("id,email,nama,role,toko_id,status").eq("id",uid).limit(1).execute().data
    return rows[0] if rows else None

def cashier_profiles(admin, shop_id):
    return admin.table("profiles").select("id,email,nama,status").eq("toko_id",shop_id).eq("role","kasir").execute().data

def pending_owners(admin):
    return admin.table("profiles").select("id,email,nama,status,toko_id,created_at").eq("role","pemilik_toko").order("created_at",desc=True).execute().data
