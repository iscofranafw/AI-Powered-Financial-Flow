import streamlit as st
import pandas as pd
from services.admin_service import (
    list_owners,
    change_owner_status,
    shop_summary,
    list_all_users,
    list_deletion_requests,
    approve_deletion_request,
    reject_deletion_request,
    delete_inactive_shop,
    ShopAuthCleanupError,
)
from utils.ui_components import page_header, section_header


def _dismiss_shop_deletion():
    st.session_state.pop("shop_deletion_target", None)


@st.dialog("Konfirmasi Hapus Toko", on_dismiss=_dismiss_shop_deletion)
def _confirm_shop_deletion(token, owner):
    st.write(f"Toko: {owner['nama_toko']}")
    st.write(f"Email pemilik: {owner['email']}")
    st.error("Toko, seluruh data operasional, profil, dan akun login pengguna toko akan dihapus permanen. Tindakan ini tidak bisa dibatalkan.")
    st.caption("Jejak audit administratif tetap disimpan.")
    if owner.get("deletion_pending"):
        st.warning("Data toko sudah dihapus. Lanjutkan pembersihan indeks lokal dan akun login yang belum selesai.")
    yes, cancel = st.columns(2)
    if cancel.button("Batal", key="cancel_shop_delete", use_container_width=True):
        _dismiss_shop_deletion()
        st.rerun()
    with yes.container(key="owner_delete_confirm_style"):
        confirmed = st.button("Ya, Hapus", key="owner_delete_confirm", use_container_width=True)
    if confirmed:
        try:
            result = delete_inactive_shop(token, owner["id"], owner["toko_id"], owner["email"])
        except ShopAuthCleanupError as exc:
            _dismiss_shop_deletion()
            st.session_state["shop_deletion_warning"] = str(exc)
            st.rerun()
        except Exception as exc:
            st.error(f"Gagal menghapus toko: {exc}")
        else:
            _dismiss_shop_deletion()
            st.session_state["shop_deletion_success"] = f"Toko {result['shop_name']} berhasil dihapus."
            st.rerun()


def render(token):
    page_header("Manajemen Platform", "Persetujuan akun pemilik toko baru dan rekapitulasi data sistem.")
    st.markdown("""<style>
    [class*="st-key-owner_delete_"] button {background:#b91c1c;color:#fff;border-color:#b91c1c;}
    [class*="st-key-owner_delete_"] button:hover {background:#991b1b;color:#fff;border-color:#991b1b;}
    </style>""", unsafe_allow_html=True)
    for key, display in (("shop_deletion_success", st.success), ("shop_deletion_warning", st.warning)):
        if key in st.session_state:
            display(st.session_state.pop(key))

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
                    col_info, col_action, col_btn, col_delete = st.columns([3, 2, 1, 1])
                    col_info.markdown(f"**{owner['nama']}** &bull; `{owner['email']}`  \nStatus: {status_text}")
                    options = ["aktif", "ditolak", "nonaktif"]
                    current_idx = options.index(owner["status"]) if owner["status"] in options else 0
                    pending_cleanup = owner.get("deletion_pending", False)
                    choice = col_action.selectbox("Ubah Status:", options, index=current_idx, key=f"state_{owner['id']}", disabled=pending_cleanup)
                    if col_btn.button("Simpan", key=f"save_{owner['id']}", use_container_width=True, disabled=pending_cleanup):
                        change_owner_status(token, owner["id"], choice)
                        st.success(f"Status {owner['nama']} berhasil diubah menjadi {choice}.")
                        st.rerun()
                    if owner["status"] == "nonaktif":
                        with col_delete.container(key=f"owner_delete_style_{owner['id']}"):
                            if st.button("Hapus", key=f"owner_delete_{owner['id']}", use_container_width=True):
                                st.session_state["shop_deletion_target"] = dict(owner)
                    if pending_cleanup:
                        col_info.warning("Pembersihan belum selesai; tekan Hapus untuk melanjutkan.")
                    st.markdown("<hr style='margin: 0.75rem 0;'>", unsafe_allow_html=True)
        else:
            st.info("Tidak ada akun pemilik toko yang perlu ditinjau.")

        if "shop_deletion_target" in st.session_state:
            _confirm_shop_deletion(token, st.session_state["shop_deletion_target"])

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

        st.divider()
        section_header("Pengajuan Penghapusan Transaksi Toko", "Tinjau dan eksekusi permohonan penghapusan massal data transaksi toko.")
        
        del_status_filter = st.selectbox(
            "Filter Status Pengajuan Penghapusan:",
            ["Semua", "pending", "completed", "rejected", "cancelled", "failed"],
            index=1,
            key="superadmin_del_req_filter"
        )
        
        filter_arg = None if del_status_filter == "Semua" else del_status_filter
        del_requests = list_deletion_requests(token, status=filter_arg)
        
        if del_requests:
            for req in del_requests:
                status_raw = req.get('status', 'pending')
                status_badge_map = {
                    'pending': 'Menunggu Persetujuan',
                    'completed': 'Selesai Dihapus',
                    'rejected': 'Ditolak',
                    'cancelled': 'Dibatalkan Pemilik',
                    'failed': 'Gagal Eksekusi'
                }
                status_display = status_badge_map.get(status_raw, status_raw.capitalize())
                req_tx_count = req.get('transaction_count', req.get('transaction_count_requested', 0))
                store_display = req.get('store_name', f"Toko {str(req.get('store_id'))[:8]}")
                requester_display = req.get('requester_info', f"User {str(req.get('requested_by'))[:8]}")
                req_date = req.get('requested_at') or req.get('created_at') or '-'

                with st.container():
                    col_info, col_status = st.columns([3, 1])
                    col_info.markdown(
                        f"**Toko:** {store_display}  \n"
                        f"**Diajukan Oleh:** {requester_display}  \n"
                        f"**Waktu Pengajuan:** {req_date}  \n"
                        f"**Jumlah Transaksi yang Diminta Dihapus:** `{req_tx_count}` transaksi"
                    )
                    col_status.markdown(f"**Status:**  \n{status_display}")

                    if status_raw == 'pending':
                        col_app, col_rej = st.columns(2)
                        with col_app.expander("Setujui & Hapus Transaksi", expanded=False):
                            st.error(
                                f"**TINDAKAN KRITIS:** Anda akan menghapus secara permanen **{req_tx_count} transaksi** "
                                f"milik toko **{store_display}**. Tindakan ini **TIDAK DAPAT DIBATALKAN**."
                            )
                            st.markdown("Ketik **`APPROVE DELETE`** untuk mengonfirmasi:")
                            with st.form(key=f"form_approve_{req['id']}"):
                                approve_confirm = st.text_input("Konfirmasi Teks", placeholder="APPROVE DELETE", key=f"input_app_{req['id']}")
                                btn_approve = st.form_submit_button("Konfirmasi Persetujuan & Eksekusi Hapus", type="primary")

                            if btn_approve:
                                if approve_confirm.strip() != "APPROVE DELETE":
                                    st.error("Teks konfirmasi tidak cocok. Anda harus mengetik 'APPROVE DELETE' persis.")
                                else:
                                    try:
                                        res = approve_deletion_request(token, req['id'], approve_confirm.strip())
                                        deleted_cnt = res.get('transaction_count_deleted', req_tx_count)
                                        st.success(f"Berhasil menyetujui dan menghapus {deleted_cnt} transaksi toko {store_display}.")
                                        st.rerun()
                                    except Exception as e_app:
                                        st.error(f"Gagal memproses persetujuan: {e_app}")

                        with col_rej.expander("Tolak Pengajuan", expanded=False):
                            st.write("Masukkan alasan penolakan permohonan penghapusan transaksi:")
                            with st.form(key=f"form_reject_{req['id']}"):
                                rej_reason = st.text_input("Alasan Penolakan", placeholder="Contoh: Data masih diperlukan untuk audit tahunan", key=f"input_rej_{req['id']}")
                                btn_reject = st.form_submit_button("Tolak Permohonan", type="secondary")

                            if btn_reject:
                                try:
                                    reject_deletion_request(token, req['id'], reason=rej_reason)
                                    st.success(f"Permohonan penghapusan transaksi toko {store_display} berhasil ditolak.")
                                    st.rerun()
                                except Exception as e_rej:
                                    st.error(f"Gagal menolak permohonan: {e_rej}")
                    elif status_raw == 'completed':
                        del_cnt = req.get('transaction_count_deleted', req.get('deleted_count', req_tx_count))
                        exec_date = req.get('executed_at', '-')
                        st.caption(f"Telah dihapus permanen sebanyak {del_cnt} transaksi pada {exec_date}.")
                    elif status_raw == 'rejected':
                        rej_msg = req.get('rejection_reason', '-')
                        st.caption(f"Alasan penolakan: {rej_msg}")
                    elif status_raw == 'cancelled':
                        canc_date = req.get('cancelled_at') or req.get('updated_at') or '-'
                        st.caption(f"Permohonan dibatalkan oleh pemilik toko pada {canc_date}.")

                    st.markdown("<hr style='margin: 0.75rem 0;'>", unsafe_allow_html=True)
        else:
            st.info("Tidak ada pengajuan penghapusan transaksi yang sesuai dengan filter.")

    except Exception as exc:
        st.error(f"Gagal memuat data administrasi: {exc}")
