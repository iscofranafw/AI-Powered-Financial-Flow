import streamlit as st
from config.settings import get_settings
from services.auth_service import verify_actor, AccessError
from pages import login, owner, cashier, superadmin
from utils.ui_components import apply_custom_styles, user_profile_sidebar

st.set_page_config(
    page_title="AI-Powered Financial Recording & Cash Flow Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles()

try:
    get_settings()
except RuntimeError as exc:
    st.error(f"Konfigurasi sistem belum lengkap: {exc}")
    st.stop()

token = st.session_state.get("token")
if not token:
    login.render()
    st.stop()

try:
    profile = verify_actor(token)
except AccessError as exc:
    st.warning(str(exc))
    if st.button("Kembali ke Halaman Masuk"):
        st.session_state.pop("token", None)
        st.rerun()
    st.stop()
except Exception as exc:
    st.error(f"Gagal memverifikasi sesi aktif: {exc}")
    st.stop()

user_profile_sidebar(profile["nama"], profile["role"])

if profile["role"] == "superadmin":
    superadmin.render(token)
elif profile["role"] == "pemilik_toko":
    owner.render(token, profile)
elif profile["role"] == "kasir":
    cashier.render(token, profile)
else:
    st.error("Peran akun tidak dikenali dalam sistem.")

st.sidebar.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
if st.sidebar.button("Keluar", use_container_width=True):
    st.session_state.pop("token", None)
    st.rerun()
