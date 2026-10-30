import streamlit as st
import pandas as pd
from services.admin_service import list_owners, change_owner_status, shop_summary, list_all_users
from utils.ui_components import page_header, section_header

def render(token):
    page_header("Manajemen Platform", "Persetujuan akun pemilik toko baru dan rekapitulasi data sistem.")

    try:
        owners = list_owners(token)
        section_header("Pendaftaran Pemilik Toko Menunggu Verifikasi")
        if owners:
            for owner in owners:
                status_labels = {
                    'pending': 'Menunggu Verifikasi',
                    'aktif': 'Aktif',
                    'ditolak': 'Ditolak',
                    'nonaktif': 'Nonaktif'
                }
                status_text = status_labels.get(owner['status'], owner['status'])

                with st.container():
                    col_info, col_action, col_btn = st.columns([3, 2, 1])
                    col_info.markdown(f"**{owner['nama']}** &bull; `{owner['email']}`  \nStatus: {status_text}")
                    options = ["aktif", "ditolak", "nonaktif"]
                    current_idx = options.index(owner["status"]) if owner["status"] in options else 0
                    choice = col_action.selectbox("Ubah Status:", options, index=current_idx, key=f"state_{owner['id']}")
                    if col_btn.button("Simpan", key=f"save_{owner['id']}", use_container_width=True):
                        change_owner_status(token, owner["id"], choice)
                        st.success(f"Status {owner['nama']} berhasil diubah menjadi {choice}.")
                        st.rerun()
                    st.markdown("<hr style='margin: 0.75rem 0;'>", unsafe_allow_html=True)
        else:
            st.info("Tidak ada akun pemilik toko yang perlu ditinjau.")

        st.divider()
        section_header("Direktori Pengguna Sistem")
        users = list_all_users(token)
        if users:
            display_users = [{
                'Email': u['email'],
                'Nama': u['nama'],
                'Peran': u['role'].replace('_', ' ').title(),
                'Status': u['status'].capitalize(),
                'ID Toko': u.get('toko_id') or '-',
                'Terdaftar Pada': u.get('created_at', '')[:10]
            } for u in users]
            st.dataframe(pd.DataFrame(display_users), use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada pengguna terdaftar.")

        st.divider()
        section_header("Rekapitulasi Usaha Terdaftar", "Informasi agregat kuantitatif tanpa mengekspos isi transaksi toko.")
        summary = shop_summary(token)
        if summary:
            display_summary = [{
                'Nama Toko': s['nama_toko'],
                'Bidang Usaha': s.get('bidang_usaha') or '-',
                'Pengguna': s['jumlah_pengguna'],
                'Jumlah Transaksi': s['jumlah_transaksi'],
                'Jumlah Kategori': s['jumlah_kategori'],
                'Transaksi Terakhir': s.get('transaksi_terakhir') or '-'
            } for s in summary]
            st.dataframe(pd.DataFrame(display_summary), use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada data toko yang tersimpan.")

    except Exception as exc:
        st.error(f"Gagal memuat data administrasi: {exc}")
