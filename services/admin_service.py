from database.client import privileged_client
from services.auth_service import verify_actor, AccessError
from repositories.profile_repository import pending_owners
from services.finance_service import _audited_mutation
from services.shop_data_cleanup import clear_shop_local_data

VALID_STATES = {"aktif", "ditolak", "nonaktif"}

def list_owners(token):
    verify_actor(token,{"superadmin"})
    admin = privileged_client()
    owners = pending_owners(admin) or []
    jobs = admin.table("shop_deletion_jobs").select("*").execute().data or []
    for owner in owners:
        owner["nama_toko"] = (owner.get("toko") or {}).get("nama", "")
    owners.extend({
        "id": job["owner_id"], "email": job["email"], "nama": job["owner_name"],
        "status": "nonaktif", "toko_id": job["shop_id"],
        "nama_toko": job["shop_name"], "deletion_pending": True,
    } for job in jobs)
    return owners


class ShopAuthCleanupError(RuntimeError):
    """Data toko sudah dihapus; job durable menunggu indeks lokal atau akun auth."""


def delete_inactive_shop(token, owner_id, shop_id, owner_email):
    actor = verify_actor(token, {"superadmin"})
    if owner_id == actor["id"]:
        raise AccessError("Tidak boleh menghapus akun sendiri")
    if not shop_id or not owner_email:
        raise ValueError("Identitas toko dan pemilik harus lengkap")
    admin = privileged_client()
    job = admin.rpc("delete_inactive_shop_atomic", {
        "p_actor": actor["id"], "p_owner": owner_id,
        "p_shop": shop_id, "p_email": owner_email,
    }).execute().data
    if not job or job.get("owner_id") != owner_id or job.get("shop_id") != shop_id:
        raise RuntimeError("Penghapusan toko belum terkonfirmasi")
    remaining = list(job["auth_user_ids"])
    try:
        clear_shop_local_data(shop_id)
        for user_id in tuple(remaining):
            try:
                admin.auth.admin.delete_user(user_id)
            except Exception as exc:
                # A previous request may have deleted auth before its job update failed.
                not_found = getattr(exc, "code", None) == "user_not_found"
                legacy_not_found = (
                    getattr(exc, "status", None) in (404, "404")
                    and str(exc).strip().casefold() == "user not found"
                )
                if not (not_found or legacy_not_found):
                    raise
            remaining.remove(user_id)
            admin.table("shop_deletion_jobs").update({"auth_user_ids": remaining}).eq("owner_id", owner_id).execute()
        admin.table("shop_deletion_jobs").delete().eq("owner_id", owner_id).execute()
    except Exception as exc:
        raise ShopAuthCleanupError(
            "Data toko dan profil di database sudah dihapus, tetapi pembersihan indeks lokal "
            "atau akun login belum selesai. Tekan Hapus lagi untuk melanjutkan pembersihan."
        ) from exc
    return {"shop_name": job["shop_name"], "email": job["email"]}

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
