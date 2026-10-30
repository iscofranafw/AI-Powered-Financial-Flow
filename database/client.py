from supabase import create_client
from config.settings import get_settings

def public_client():
    s = get_settings()
    return create_client(s["SUPABASE_URL"], s["SUPABASE_ANON_KEY"])

def privileged_client():
    s = get_settings()
    return create_client(s["SUPABASE_URL"], s["SUPABASE_SERVICE_ROLE_KEY"])
