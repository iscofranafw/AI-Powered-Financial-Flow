from database.client import public_client, privileged_client
from repositories.profile_repository import own_profile

class AccessError(Exception):
    pass

def register_owner(nama, email, password, toko, bidang):
    if not all(x.strip() for x in (nama,email,password,toko,bidang)):
        raise ValueError("Semua kolom wajib diisi")
    if len(password) < 8:
        raise ValueError("Kata sandi minimal 8 karakter")
    return public_client().auth.sign_up({"email":email.strip(),"password":password,"options":{"data":{"nama":nama.strip(),"nama_toko":toko.strip(),"bidang_usaha":bidang.strip()}}})

def login(email,password):
    result = public_client().auth.sign_in_with_password({"email":email,"password":password})
    if not result.session:
        raise AccessError("Sesi login tidak tersedia")
    return result.session.access_token

def verify_actor(token, allowed_roles=None, active=True):
    if not token:
        raise AccessError("Silakan login")
    admin = privileged_client()
    user_response = admin.auth.get_user(token)
    user = user_response.user
    if not user:
        raise AccessError("Token tidak valid")
    profile = own_profile(admin,user.id)
    if not profile:
        raise AccessError("Profil tidak ditemukan")
    if active and profile["status"] != "aktif":
        raise AccessError("Akun belum aktif atau telah dinonaktifkan")
    if allowed_roles and profile["role"] not in allowed_roles:
        raise AccessError("Anda tidak memiliki izin")
    return profile

def logout(token):
    return None
