from supabase import create_client
from config.settings import get_settings

def public_client():
    s = get_settings()
    return create_client(s["SUPABASE_URL"], s["SUPABASE_ANON_KEY"])

def privileged_client():
    s = get_settings()
    return create_client(s["SUPABASE_URL"], s["SUPABASE_SERVICE_ROLE_KEY"])

def authenticated_client(token: str):
    """Create a scoped client executing under the authenticated user's session and RLS policies."""
    s = get_settings()
    client = create_client(s["SUPABASE_URL"], s["SUPABASE_ANON_KEY"])
    if token:
        try:
            client.postgrest.auth(token)
        except Exception:
            # Safe fallback: if auth header cannot be attached, return base anon client
            pass
    return client
