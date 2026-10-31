import streamlit as st
from pages.finance import transactions_ui
from pages import ai_tools, chat_view


def render(token, profile):
    st.sidebar.markdown(
        "<div style='font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Navigasi Kasir</div>",
        unsafe_allow_html=True
    )
    pilihan = st.sidebar.radio("Pilih Menu:", ["Transaksi", "Pencatatan Cepat", "AI Financial Chat", "Status Sistem"], key="cashier_menu", label_visibility="collapsed")

    if pilihan in ["Transaksi", "Pencatatan Cepat", "AI Financial Chat", "Asisten Keuangan AI"]:
        st.markdown("""
        <div style='border: 1px solid #E2E8F0; background-color: #F8FAFC; border-left: 3px solid #64748B; border-radius: 4px; padding: 0.75rem 1rem; color: #475569; font-size: 0.8125rem; margin-bottom: 1.5rem;'>
            Catatan Operasional: Transaksi yang Anda catat berstatus <strong>Menunggu Persetujuan</strong> dan akan ditinjau oleh Pemilik Toko sebelum masuk ke laporan keuangan resmi.
        </div>
        """, unsafe_allow_html=True)

    if pilihan == "Transaksi":
        transactions_ui(token, owner=False)
    elif pilihan == "Pencatatan Cepat":
        ai_tools.quick_entry(token, owner=False)
    elif pilihan in ("AI Financial Chat", "Asisten Keuangan AI"):
        chat_view.render(token, profile)
    elif pilihan == "Status Sistem":
        ai_tools.diagnostics()
