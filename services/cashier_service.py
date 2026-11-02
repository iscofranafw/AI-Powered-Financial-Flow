import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from database.client import privileged_client
from services.auth_service import verify_actor
from repositories.profile_repository import cashier_profiles


def create_cashier(token, nama, email, password):
    """Securely provision a new store cashier account using an expiring single-use server ticket."""
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
        # Immediately strip ticket from user-visible metadata and seal role inside server app_metadata
        admin.auth.admin.update_user_by_id(created.user.id, {
            'user_metadata': {'nama': nama},
            'app_metadata': {'bizflow_role': 'kasir', 'bizflow_toko_id': owner['toko_id']},
        })
        return created
    except Exception as exc:
        raise RuntimeError(f"Gagal mendaftarkan akun kasir: {exc}") from None
    finally:
        # Guarantee single-use cleanup of server-side provisioning ticket
        try:
            admin.table('cashier_provisioning').delete().eq('token_hash', token_hash).execute()
        except Exception:
            # Safe fallback: ticket has 5-minute TTL; cleanup failure does not abort user creation
            pass


def list_cashiers(token):
    owner = verify_actor(token, {'pemilik_toko'})
    return cashier_profiles(privileged_client(), owner['toko_id'])
