from database.client import privileged_client
from services.auth_service import verify_actor, AccessError
from repositories.profile_repository import pending_owners

VALID_STATES = {"aktif", "ditolak", "nonaktif"}

def list_owners(token):
    verify_actor(token,{"superadmin"})
    return pending_owners(privileged_client())

def change_owner_status(token, owner_id, new_status):
    actor = verify_actor(token,{"superadmin"})
    if new_status not in VALID_STATES:
        raise ValueError("Status tidak valid")
    if owner_id == actor["id"]:
        raise AccessError("Tidak boleh mengubah akun sendiri")
    admin = privileged_client()
    row = admin.table("profiles").select("role,status").eq("id",owner_id).single().execute().data
    if row["role"] != "pemilik_toko":
        raise AccessError("Target bukan pemilik toko")
    if new_status == "aktif":
        admin.auth.admin.update_user_by_id(owner_id,{"email_confirm":True})
    admin.table("profiles").update({"status":new_status}).eq("id",owner_id).execute()

def shop_summary(token):
    verify_actor(token,{"superadmin"})
    return privileged_client().rpc("admin_shop_summary").execute().data

def list_all_users(token):
    verify_actor(token, {'superadmin'})
    admin=privileged_client();result=[];offset=0
    while True:
        batch=(admin.table('profiles').select('id,email,nama,role,toko_id,status,created_at')
               .order('created_at',desc=True).range(offset,offset+999).execute().data or [])
        result.extend(batch)
        if len(batch)<1000:break
        offset+=1000
    return result
