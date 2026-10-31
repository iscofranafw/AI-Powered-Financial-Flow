import streamlit as st
from services.auth_service import login, register_owner

def render():
    st.markdown("""
    <style>
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarHeader"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem; margin-top: 2rem;'>
            <div style='font-size: 1.5rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;'>AI-Powered Financial Recording & Cash Flow Assistant</div>
            <div style='font-size: 0.9375rem; color: #64748B; margin-top: 0.25rem;'>Sistem Pencatatan & Analisis Arus Kas UMKM</div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Masuk ke Akun", "Daftar Pemilik Toko"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Alamat Email", key="login_email")
                password = st.text_input("Kata Sandi", type="password", key="login_password")
                submitted = st.form_submit_button("Masuk", use_container_width=True)

            if submitted:
                if not email.strip() or not password:
                    st.error("Alamat email dan kata sandi wajib diisi.")
                else:
                    try:
                        st.session_state["token"] = login(email, password)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Gagal masuk: {exc}")

        with tab_register:
            with st.form("register_form"):
                nama = st.text_input("Nama Lengkap Pemilik")
                reg_email = st.text_input("Alamat Email", key="reg_email")
                reg_password = st.text_input("Kata Sandi (Minimal 8 karakter)", type="password", key="reg_password")
                toko = st.text_input("Nama Toko / Usaha")
                bidang = st.text_input("Bidang Usaha (Contoh: Bakery, Bengkel Motor, Grosir, Warung Makan)")
                reg_submitted = st.form_submit_button("Kirim Pendaftaran", use_container_width=True)

            if reg_submitted:
                try:
                    register_owner(nama, reg_email, reg_password, toko, bidang)
                    st.success("Pendaftaran berhasil dikirim. Akun Anda sedang menunggu verifikasi superadmin.")
                except Exception as exc:
                    st.error(f"Pendaftaran gagal: {exc}")
