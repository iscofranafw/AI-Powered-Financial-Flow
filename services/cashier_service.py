import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from database.client import privileged_client
from services.auth_service import verify_actor
from repositories.profile_repository import cashier_profiles


def create_cashier(token, nama, email, password):
    owner = verify_actor(token, {'pemilik_toko'})
    nama, email = nama.strip(), email.strip().lower()
    if not nama or not email or '@' not in email or len(password) < 8:
        raise ValueError('Nama/email wajib; password minimal 8 karakter')
    admin = privileged_client()
    ticket = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(ticket.encode()).hexdigest()
    admin.table('cashier_provisioning').insert({
        'token_hash': token_hash, 'email': email, 'nama': nama,
        'toko_id': owner['toko_id'], 'created_by': owner['id'],
        'expires_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    }).execute()
    try:
        created = admin.auth.admin.create_user({
            'email': email, 'password': password, 'email_confirm': True,
            'user_metadata': {'nama': nama, 'bizflow_cashier_ticket': ticket},
        })
        if not created.user:
            raise RuntimeError('Supabase tidak mengembalikan pengguna baru')
        admin.auth.admin.update_user_by_id(created.user.id, {
            'user_metadata': {'nama': nama},
            'app_metadata': {'bizflow_role': 'kasir', 'bizflow_toko_id': owner['toko_id']},
        })
        return created
    finally:
        admin.table('cashier_provisioning').delete().eq('token_hash', token_hash).execute()


def list_cashiers(token):
    owner = verify_actor(token, {'pemilik_toko'})
    return cashier_profiles(privileged_client(), owner['toko_id'])
