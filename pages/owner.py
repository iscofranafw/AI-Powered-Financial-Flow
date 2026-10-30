import streamlit as st
import pandas as pd
from services.cashier_service import create_cashier, list_cashiers
from services import finance_service as f
from pages import ai_tools, import_view, report_view
from pages.finance import transactions_ui, categories_ui, approvals_ui
from utils.ui_components import page_header, section_header


def cashier_management_ui(token):
    page_header("Pengelolaan Akun Kasir", "Pendaftaran akun kasir untuk pencatatan transaksi harian toko.")

    with st.form("create_cashier"):
        section_header("Tambah Akun Kasir Baru")
        nama = st.text_input("Nama Lengkap Kasir")
        email = st.text_input("Alamat Email Kasir")
        password = st.text_input("Kata Sandi Kasir (Minimal 8 karakter)", type="password")
        submit = st.form_submit_button("Simpan Akun Kasir")

    if submit:
        try:
            create_cashier(token, nama, email, password)
            st.success("Akun kasir berhasil dibuat.")
            st.rerun()
        except Exception as exc:
            st.error(f"Gagal membuat akun kasir: {exc}")

    st.divider()
    section_header("Daftar Kasir Toko Terdaftar")
    try:
        data = list_cashiers(token)
        if data:
            display_cashiers = [{
                "Nama Kasir": c["nama"],
                "Alamat Email": c["email"],
                "Status": "Aktif" if c.get("status") == "aktif" else c.get("status", "-")
            } for c in data]
            st.dataframe(pd.DataFrame(display_cashiers), use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada akun kasir terdaftar untuk toko ini.")
    except Exception as exc:
        st.error(str(exc))


def render(token, profile):
    menu_options = [
        "Ringkasan Bisnis",
        "Transaksi",
        "Pencatatan Cepat",
        "Persetujuan Kasir",
        "Kategori Pembukuan",
        "Impor Data CSV",
        "Akun Kasir",
        "Status Sistem"
    ]

    st.sidebar.markdown(
        "<div style='font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Navigasi Menu</div>",
        unsafe_allow_html=True
    )
    pilihan = st.sidebar.radio("Pilih Menu:", menu_options, key="owner_menu", label_visibility="collapsed")

    if pilihan == "Ringkasan Bisnis":
        report_view.render(token)
    elif pilihan == "Transaksi":
        transactions_ui(token, owner=True)
    elif pilihan == "Pencatatan Cepat":
        ai_tools.quick_entry(token, owner=True)
    elif pilihan == "Persetujuan Kasir":
        approvals_ui(token)
    elif pilihan == "Kategori Pembukuan":
        if not f.list_categories(token):
            ai_tools.initial_categories(token)
        categories_ui(token)
    elif pilihan == "Impor Data CSV":
        import_view.render(token)
    elif pilihan == "Akun Kasir":
        cashier_management_ui(token)
    elif pilihan == "Status Sistem":
        ai_tools.diagnostics()
