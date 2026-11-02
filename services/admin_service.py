from database.client import privileged_client
from services.auth_service import verify_actor, AccessError
from repositories.profile_repository import pending_owners
from services.finance_service import _audited_mutation

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
    row = admin.table("profiles").select("id,role,status,toko_id").eq("id",owner_id).single().execute().data
    if not row:
        raise ValueError("Pemilik toko tidak ditemukan")
    if row["role"] != "pemilik_toko":
        raise AccessError("Target bukan pemilik toko")
    if new_status == "aktif":
        admin.auth.admin.update_user_by_id(owner_id,{"email_confirm":True})
    updated = _audited_mutation(admin, actor['id'], row.get('toko_id'),
                                'change_owner_status', [owner_id], {'status': new_status})
    if not updated:
        raise RuntimeError("Perubahan status pemilik tidak tersimpan")

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


def list_deletion_requests(token, status=None):
    from services.deletion_service import superadmin_list_deletion_requests
    return superadmin_list_deletion_requests(token, status=status)


def approve_deletion_request(token, request_id, confirmation_text):
    from services.deletion_service import superadmin_approve_and_execute_deletion
    return superadmin_approve_and_execute_deletion(token, request_id, confirmation_text)


def reject_deletion_request(token, request_id, reason=''):
    from services.deletion_service import superadmin_reject_deletion_request
    return superadmin_reject_deletion_request(token, request_id, reason=reason)
