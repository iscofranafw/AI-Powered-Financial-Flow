import streamlit as st
import pandas as pd
from datetime import date
from decimal import Decimal
from services import finance_service as f
from services.validation_service import get_jakarta_today
from utils.ui_components import page_header, section_header, format_rupiah

def _cats(token, tipe=None):
    data = f.list_categories(token)
    return [c for c in data if c['tipe'] == tipe] if tipe else data

def _choices(cats):
    return {f"{c['nama']} ({c['tipe']})": c['id'] for c in cats}

def render(token, profile):
    if profile.get('role') == 'pemilik_toko':
        from pages import owner
        owner.render(token, profile)
    else:
        from pages import cashier
        cashier.render(token, profile)

def render_ai_pending_queue(token, *, auto_retry=True):
    """One bounded retry on page load or button; never a background loop."""
    import time
    from services.ai_service import pending_ai_transactions, retry_pending_ai_classifications
    from ai.ollama_client import diagnosis
    try:
        pending = pending_ai_transactions(token)
        if not pending:
            return
        st.warning(f"Klasifikasi tertunda - {len(pending)} transaksi menunggu AI. Transaksi sudah tersimpan.")
        requested = st.button(f"Klasifikasi ulang ({len(pending)} menunggu AI)", key='retry_pending_ai')
        key = 'last_ai_retry_' + str(pending[0].get('toko_id',''))
        if requested or (auto_retry and time.monotonic() - st.session_state.get(key, 0) > 30):
            st.session_state[key] = time.monotonic()
            if diagnosis()['ok']:
                result = retry_pending_ai_classifications(token, limit=20 if requested else 3)
                st.info(f"AI: {result['processed']} diproses; {result['remaining']} masih menunggu.")
                if result['errors']:
                    st.warning(result['errors'][0])
            elif requested:
                st.warning('Jalankan Ollama llama3:8b; antrean tetap tersimpan.')
    except Exception as exc:
        st.warning(f'Antrean AI belum dapat diproses: {exc}')


def _render_others_reclassification_panel(token, cats, key_prefix='cat'):
    others_ids = {
        str(c['id']) for c in cats
        if c.get('sistem') or str(c.get('nama', '')).strip().casefold() == 'others'
    }
    if not others_ids:
        return
    all_tx = f.list_transactions(token)
    others_txs = [
        r for r in all_tx
        if r.get('kategori_id') and str(r['kategori_id']) in others_ids
        and r.get('status_persetujuan') != 'ditolak'
    ]
    st.caption(f"Transaksi saat ini pada kategori Others: **{len(others_txs):,} transaksi**.")
    if not others_txs:
        st.session_state.pop('others_reclass_preview', None)
        return

    if st.button('Klasifikasi ulang transaksi Others', type='primary', key=f'{key_prefix}_btn_reclassify_others'):
        with st.spinner('Menganalisis ulang transaksi Others dengan Similarity Gate & AI...'):
            try:
                from services.ai_service import preview_others_reclassification
                st.session_state['others_reclass_preview'] = preview_others_reclassification(token)
            except Exception as exc:
                st.error(f'Gagal memproses pratinjau klasifikasi ulang Others: {exc}')

    preview = st.session_state.get('others_reclass_preview')
    if preview and preview.get('total_others', 0) > 0:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric('Total Transaksi Others', f"{preview['total_others']:,}")
        mc2.metric('Siap Dipindahkan', f"{preview['reclassified_count']:,}")
        mc3.metric('Tetap Perlu Tinjauan', f"{preview['unresolved_count']:,}")

        table_rows = [
            {
                'Deskripsi': g['deskripsi'],
                'Jenis': str(g['tipe']).capitalize(),
                'Jumlah': g['count'],
                'Total Nominal': format_rupiah(g['total_nominal']),
                'Usulan Kategori': g['suggested_kategori_nama'],
                'Status': 'Terpetakan' if g.get('suggested_kategori_id') else 'Perlu Tinjauan',
            }
            for g in preview.get('groups', [])
        ]
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        c_ok, c_cancel = st.columns([2, 1])
        updates = preview.get('updates', [])
        if updates and c_ok.button(
            f"Konfirmasi & Terapkan ke {len(updates):,} Transaksi",
            type='primary',
            key=f'{key_prefix}_btn_confirm_others',
        ):
            with st.spinner('Menyimpan hasil klasifikasi ulang & memperbarui indeks FAISS...'):
                try:
                    from services.ai_service import confirm_others_reclassification
                    res = confirm_others_reclassification(token, updates)
                    st.session_state.pop('others_reclass_preview', None)
                    st.success(
                        f"Berhasil memindahkan {res['updated_count']:,} transaksi dari Others "
                        f"({res['reapproved_count']:,} disetujui ulang dengan audit log)."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f'Gagal menerapkan klasifikasi ulang: {exc}')
        elif not updates:
            st.info('Tidak ada transaksi Others yang mencapai keyakinan cukup; lengkapi kata kunci kategori terlebih dahulu.')
        if c_cancel.button('Tutup Pratinjau', key=f'{key_prefix}_btn_cancel_others'):
            st.session_state.pop('others_reclass_preview', None)
            st.rerun()


def transactions_ui(token, owner):
    page_header('Pencatatan Transaksi', 'Pencatatan mutasi kas pemasukan dan pengeluaran operasional toko.')
    render_ai_pending_queue(token)
    cats = _cats(token)
    if owner and cats:
        with st.expander('Klasifikasi Ulang Transaksi Others', expanded=False):
            _render_others_reclassification_panel(token, cats, key_prefix='tx')
    if not cats:
        st.warning('Belum ada kategori terdaftar. Pemilik toko perlu menyiapkan kategori di menu Kelola Kategori terlebih dahulu.')
        return

    tipe = st.selectbox('Jenis Transaksi', ['pengeluaran', 'pemasukan'], key='tx_type_select')
    valid_cats = [c for c in cats if c['tipe'] == tipe]
    opts = _choices(valid_cats)
    if not opts:
        st.warning(f"Belum ada kategori untuk jenis '{tipe}'. Silakan tambah kategori baru terlebih dahulu.")
        return

    try:
        quota_data = f.check_category_quota(token, tipe)
        if quota_data.get('warning'):
            st.warning(quota_data['warning'])
    except Exception:
        # Safe fallback: UI quota check warning is advisory and should not block the transaction form
        pass

    with st.form('add_finance'):
        tanggal = st.date_input('Tanggal Transaksi', value=get_jakarta_today())
        desc = st.text_input('Keterangan Transaksi')
        nominal = st.number_input('Nominal (Rp)', min_value=0.01, step=1000.0, format='%.2f')
        selected = st.selectbox('Kategori', list(opts))
        submit = st.form_submit_button('Simpan Transaksi')

    if submit:
        try:
            f.add_transaction(token, tanggal, desc, nominal, tipe, opts[selected])
            st.success('Transaksi berhasil disimpan.')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    section_header('Riwayat Transaksi Terdaftar')
    rows = f.list_transactions(token)

    cmap = {c['id']: c['nama'] for c in cats}
    status_label_map = {
        'disetujui': 'Disetujui',
        'pending': 'Menunggu Persetujuan',
        'ditolak': 'Ditolak'
    }

    if not rows:
        st.info('Belum ada transaksi tercatat pada toko ini.')
    else:
        # Filter controls: Category, Type, Status, Search
        f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1, 1.1, 1.2])

        unique_cat_names = sorted(list({c['nama'] for c in cats}))
        cat_filter_options = ["Semua Kategori"] + unique_cat_names
        selected_cat_name = f_col1.selectbox("Filter Kategori", cat_filter_options, key="filter_cat_name")

        type_filter_options = ["Semua Jenis", "Pemasukan", "Pengeluaran"]
        selected_type = f_col2.selectbox("Filter Jenis", type_filter_options, key="filter_type")

        status_filter_options = ["Semua Status", "Disetujui (Sah)", "Menunggu Persetujuan", "Ditolak"]
        selected_status = f_col3.selectbox("Filter Status", status_filter_options, key="filter_status")

        search_query = f_col4.text_input("Cari Keterangan", placeholder="Ketik kata kunci...", key="filter_search")

        # Apply filtering
        filtered_rows = rows
        if selected_cat_name != "Semua Kategori":
            target_cat_ids = {c['id'] for c in cats if c['nama'] == selected_cat_name}
            filtered_rows = [r for r in filtered_rows if r.get('kategori_id') in target_cat_ids]

        if selected_type == "Pemasukan":
            filtered_rows = [r for r in filtered_rows if str(r.get('tipe', '')).lower() == 'pemasukan']
        elif selected_type == "Pengeluaran":
            filtered_rows = [r for r in filtered_rows if str(r.get('tipe', '')).lower() == 'pengeluaran']

        if selected_status == "Disetujui (Sah)":
            filtered_rows = [r for r in filtered_rows if r.get('status_persetujuan') == 'disetujui']
        elif selected_status == "Menunggu Persetujuan":
            filtered_rows = [r for r in filtered_rows if r.get('status_persetujuan') == 'pending']
        elif selected_status == "Ditolak":
            filtered_rows = [r for r in filtered_rows if r.get('status_persetujuan') == 'ditolak']

        if search_query.strip():
            sq = search_query.strip().lower()
            filtered_rows = [r for r in filtered_rows if sq in str(r.get('deskripsi', '')).lower()]

        # Official Financial Metrics: ONLY count financially effective (disetujui) transactions
        effective_rows = [r for r in filtered_rows if r.get('status_persetujuan') == 'disetujui']
        pending_rows = [r for r in filtered_rows if r.get('status_persetujuan') == 'pending']
        rejected_rows = [r for r in filtered_rows if r.get('status_persetujuan') == 'ditolak']

        total_tx = len(effective_rows)
        total_in = sum(Decimal(str(r.get('nominal', 0))) for r in effective_rows if str(r.get('tipe', '')).lower() == 'pemasukan')
        total_out = sum(Decimal(str(r.get('nominal', 0))) for r in effective_rows if str(r.get('tipe', '')).lower() == 'pengeluaran')
        net_cash = total_in - total_out

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Total Transaksi Sah", f"{total_tx} Transaksi")
        s_col2.metric("Total Pemasukan Sah", format_rupiah(total_in))
        s_col3.metric("Total Pengeluaran Sah", format_rupiah(total_out))
        s_col4.metric("Arus Kas Bersih Sah", format_rupiah(net_cash))

        if pending_rows or rejected_rows:
            pending_total = sum(Decimal(str(r.get('nominal', 0))) for r in pending_rows)
            st.caption(f"ℹ️ Terdapat **{len(pending_rows)} transaksi pending** ({format_rupiah(pending_total)}) dan **{len(rejected_rows)} ditolak** dalam daftar yang tidak mempengaruhi total kas sah.")

        if not filtered_rows:
            st.info("Tidak ada transaksi yang sesuai dengan filter yang dipilih.")
        else:
            display_data = []
            for idx, r in enumerate(filtered_rows, 1):
                display_data.append({
                    'No.': idx,
                    'Tanggal': r['tanggal'],
                    'Keterangan': r['deskripsi'],
                    'Nominal': format_rupiah(r['nominal']),
                    'Jenis': str(r.get('tipe', '')).capitalize(),
                    'Kategori': cmap.get(r.get('kategori_id'), '-'),
                    'Status': status_label_map.get(r.get('status_persetujuan'), str(r.get('status_persetujuan', '-')))
                })
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

    # Individual edit / delete (100% preserved)
    editable = [r for r in rows if owner or r['status_persetujuan'] == 'pending'] if rows else []
    if editable:
        st.divider()
        section_header('Koreksi atau Hapus Transaksi')
        ids = {f"{r['tanggal']} | {r['deskripsi']} | {format_rupiah(r['nominal'])} | ID: {r['id'][:8]}": r for r in editable}
        picked = st.selectbox('Pilih transaksi yang ingin dikoreksi', list(ids), key='edit_pick')
        r = ids[picked]

        tp_edit = st.selectbox('Jenis Transaksi Koreksi', ['pengeluaran', 'pemasukan'], index=['pengeluaran', 'pemasukan'].index(r['tipe']), key='edit_type_select')
        opts_edit = _choices([c for c in cats if c['tipe'] == tp_edit])
        default_idx = next((i for i, v in enumerate(opts_edit.values()) if v == r['kategori_id']), 0)

        with st.form('edit_finance'):
            d = st.date_input('Tanggal Koreksi', value=date.fromisoformat(r['tanggal']))
            desc = st.text_input('Keterangan Koreksi', value=r['deskripsi'])
            amt = st.number_input('Nominal Koreksi', min_value=0.01, value=float(r['nominal']), step=1000.0)
            selected = st.selectbox('Kategori Koreksi', list(opts_edit), index=default_idx)
            save = st.form_submit_button('Simpan Koreksi')

        if save:
            try:
                f.update_transaction(token, r['id'], d, desc, amt, tp_edit, opts_edit[selected])
                st.success('Transaksi berhasil diperbarui.')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if owner and st.button('Hapus Transaksi', type='secondary'):
            try:
                f.delete_transaction(token, r['id'])
                st.success('Transaksi berhasil dihapus.')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    # Zona Bahaya: Pengajuan Hapus Seluruh Transaksi Toko (HANYA PEMILIK TOKO)
    if owner:
        st.divider()
        section_header('Zona Bahaya (Danger Zone)', 'Fitur berisiko tinggi yang memerlukan persetujuan otorisasi Superadmin.')
        try:
            active_req = f.get_deletion_status(token)
        except Exception:
            active_req = None

        if active_req and active_req.get('status') == 'pending':
            req_time = active_req.get('requested_at') or active_req.get('created_at') or '-'
            tx_count_val = active_req.get('transaction_count', active_req.get('transaction_count_requested', 0))
            st.warning(
                f"**Pengajuan Penghapusan Seluruh Transaksi Sedang Menunggu Persetujuan Superadmin**\n\n"
                f"- **Waktu Pengajuan:** {req_time}\n"
                f"- **Jumlah Transaksi yang Diajukan:** {tx_count_val} transaksi\n"
                f"- **Status:** Menunggu Persetujuan (Pending)\n\n"
                f"Data transaksi Anda belum dihapus dan operasional toko tetap berjalan normal sampai Superadmin menyetujui."
            )
            if st.button('Batalkan Pengajuan Penghapusan', type='secondary', key='btn_cancel_bulk_delete'):
                try:
                    f.cancel_deletion_request(token, active_req['id'])
                    st.success('Pengajuan penghapusan transaksi berhasil dibatalkan.')
                    st.rerun()
                except Exception as exc:
                    st.error(f"Gagal membatalkan pengajuan: {exc}")
        else:
            total_tx_all = len(rows) if rows else 0
            with st.expander('Permohonan Hapus Seluruh Transaksi Toko', expanded=False):
                st.error(
                    '**PERINGATAN:** Tindakan ini akan mengajukan permohonan ke Superadmin untuk '
                    'menghapus **SELURUH transaksi toko Anda secara permanen**. '
                    'Data transaksi yang dihapus tidak dapat dipulihkan kembali.'
                )
                st.write(f"- **Jumlah transaksi toko saat ini:** {total_tx_all} transaksi")
                st.markdown(
                    "Untuk mengonfirmasi bahwa Anda benar-benar bermaksud mengajukan permohonan ini, "
                    "silakan ketik **`REQUEST DELETE`** pada kolom di bawah:"
                )
                with st.form('form_owner_request_delete'):
                    confirm_text = st.text_input('Konfirmasi Teks', placeholder='REQUEST DELETE', key='owner_delete_confirm_input')
                    btn_submit = st.form_submit_button('Kirim Pengajuan ke Superadmin', type='primary')

                if btn_submit:
                    if confirm_text.strip() != 'REQUEST DELETE':
                        st.error("Teks konfirmasi salah. Anda wajib mengetik 'REQUEST DELETE' persis.")
                    elif total_tx_all == 0:
                        st.warning("Tidak ada transaksi untuk dihapus pada toko ini.")
                    else:
                        try:
                            f.request_deletion_all_transactions(token, confirm_text.strip())
                            st.success('Pengajuan penghapusan seluruh transaksi berhasil dikirim ke Superadmin.')
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Gagal mengirim pengajuan: {exc}")

def categories_ui(token):
    page_header('Kelola Kategori', 'Kelola struktur taksonomi pemasukan dan pengeluaran toko.')
    if st.button('Siapkan Kategori Dasar Otomatis (Jika Kosong)'):
        try:
            f.initialize_categories(token)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    cats = _cats(token)
    try:
        quota_summary = f.check_category_quota(token)
        for tp_k, q in quota_summary.items():
            if q.get('warning'):
                st.warning(q['warning'])
    except Exception:
        # Safe fallback: UI quota summary warning is advisory and should not block category display
        pass

    if cats:
        formatted_cats = []
        for c in cats:
            tipe_tag = 'Kustom'
            if c.get('sistem'):
                tipe_tag = 'Sistem (Others)'
            elif c.get('bukan_operasional'):
                tipe_tag = 'Non-Operasional (Modal)'
            formatted_cats.append({
                'Nama Kategori': c['nama'],
                'Jenis': c['tipe'].capitalize(),
                'Deskripsi': c.get('deskripsi') or '-',
                'Kata Kunci': ', '.join(c.get('kata_kunci') or []) or '-',
                'Tipe Akun': tipe_tag
            })
        st.dataframe(pd.DataFrame(formatted_cats), use_container_width=True, hide_index=True)

        missing_kw_count = sum(
            1 for c in cats
            if not c.get('sistem') and not c.get('bukan_operasional')
            and str(c.get('nama', '')).strip().casefold() != 'others'
            and not [k for k in (c.get('kata_kunci') or []) if str(k).strip()]
        )
        if missing_kw_count > 0:
            st.warning(f"Terdapat {missing_kw_count} kategori aktif yang belum memiliki kata kunci pencocokan (Similarity Gate berpotensi rendah).")
        if st.button('Lengkapi Kata Kunci Default Kategori (Sesuai Bidang Usaha)', key='btn_autofill_keywords'):
            try:
                res_kw = f.auto_fill_category_keywords(token, overwrite=False)
                for warning in res_kw.get('warnings', []):
                    st.warning(warning)
                if res_kw['updated_count']:
                    st.success(f"Kata kunci berhasil diisi pada {res_kw['updated_count']} kategori dan indeks FAISS diperbarui.")
                    if not res_kw.get('warnings'):
                        st.rerun()
                else:
                    st.info('Tidak ada kata kunci baru yang dapat diisi. Isi kata kunci kategori kosong secara manual lewat Kelola Kategori.')
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info('Belum ada kategori yang terdaftar pada toko ini.')

    st.divider()
    with st.form('new_category'):
        section_header('Tambah Kategori Baru')
        name = st.text_input('Nama Kategori')
        tp = st.selectbox('Jenis Kategori', ['pengeluaran', 'pemasukan'])
        desc = st.text_input('Deskripsi Singkat (Opsional)')
        kw_new = st.text_input('Kata Kunci Pencocokan (Opsional, pisahkan koma, maks 20)', placeholder='Misal: tepung terigu, gula pasir, ragi instan')
        auto_kw = st.checkbox('Gunakan kata kunci default usaha bila kolom kata kunci dikosongkan', value=True)
        create = st.form_submit_button('Simpan Kategori Baru')

    if create:
        try:
            parsed_kw = [k.strip()[:80] for k in kw_new.split(',') if k.strip()][:20]
            if not parsed_kw and auto_kw:
                from ai.category_discovery import suggest_default_keywords
                from ai.store_context import get_store_business_context
                shop_ctx = f._owner(token)
                b_type = get_store_business_context(shop_ctx['toko_id']).get('bidang_usaha', 'Umum')
                parsed_kw = suggest_default_keywords(name, tp, business_type=b_type)
            f.create_category(token, name, tp, desc, keywords=parsed_kw)
            st.success('Kategori baru berhasil ditambahkan.')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if cats:
        st.divider()
        section_header('Ubah atau Hapus Kategori')
        opts = _choices(cats)
        selected = st.selectbox('Pilih Kategori:', list(opts))
        cid = opts[selected]
        current = next(c for c in cats if c['id'] == cid)

        with st.form('edit_category'):
            name = st.text_input('Nama Baru', value=current['nama'])
            desc = st.text_input('Deskripsi Baru', value=current.get('deskripsi', ''))
            kw_edit = st.text_input(
                'Kata Kunci Pencocokan (pisahkan koma, maks 20)',
                value=', '.join(current.get('kata_kunci') or []),
                disabled=bool(current.get('sistem') or current.get('bukan_operasional')),
            )
            save = st.form_submit_button('Perbarui Kategori')

        if save:
            try:
                f.rename_category(token, cid, name, desc)
                if not current.get('sistem') and not current.get('bukan_operasional'):
                    parsed_kw = [k.strip()[:80] for k in kw_edit.split(',') if k.strip()]
                    f.set_category_keywords(token, cid, parsed_kw)
                st.success('Kategori berhasil diubah dan indeks diperbarui.')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if not current.get('sistem') and not current.get('bukan_operasional'):
            if st.button('Hapus Kategori Ini (Jika Belum Ada Transaksi)'):
                try:
                    f.delete_category(token, cid)
                    st.success('Kategori berhasil dihapus.')
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.divider()
        section_header('Gabungkan Kategori', 'Pindahkan seluruh transaksi dari kategori sumber ke kategori tujuan dengan jenis yang sama.')
        mergeable_sources = [c for c in cats if not c.get('sistem') and not c.get('bukan_operasional')]
        if mergeable_sources:
            src_opts = _choices(mergeable_sources)
            src_selected = st.selectbox('Kategori Sumber (Akan Dihapus):', list(src_opts), key='merge_src')
            src_cid = src_opts[src_selected]
            src_cat = next(c for c in mergeable_sources if c['id'] == src_cid)

            dst_candidates = [c for c in cats if c['tipe'] == src_cat['tipe'] and c['id'] != src_cid and not c.get('bukan_operasional')]
            if dst_candidates:
                dst_opts = _choices(dst_candidates)
                dst_selected = st.selectbox('Kategori Tujuan (Menerima Seluruh Transaksi):', list(dst_opts), key='merge_dst')
                if st.button('Proses Penggabungan Kategori'):
                    try:
                        f.merge_category(token, src_cid, dst_opts[dst_selected])
                        st.success('Kategori berhasil digabungkan.')
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            else:
                st.info('Tidak ada kategori tujuan lain yang cocok untuk digabungkan.')
        else:
            st.info('Tidak ada kategori kustom yang dapat digabungkan.')

    st.divider()
    section_header('Analisis Kategori Tidak Terpetakan (Others)', 'Klasifikasi ulang transaksi Others ke kategori aktif atau temukan peluang kategori baru.')
    _render_others_reclassification_panel(token, cats, key_prefix='cat')
    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
    typ = st.selectbox('Jenis Transaksi untuk Usulan Kategori Baru:', ['pengeluaran', 'pemasukan'])
    if st.button('Analisis Pola Transaksi'):
        try:
            from services.ai_service import suggest_others
            st.session_state['other_suggestions'] = suggest_others(token, typ)
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get('other_suggestions'):
        st.write('Usulan Kategori Baru Berdasarkan Pola Deskripsi:', st.session_state['other_suggestions'])
        st.caption('Anda dapat menambahkan salah satu usulan di atas melalui formulir Tambah Kategori.')

def approvals_ui(token):
    page_header('Persetujuan Transaksi Kasir', 'Verifikasi transaksi kasir sebelum dibukukan secara permanen ke laporan keuangan.')
    pending = [r for r in f.list_transactions(token) if r['status_persetujuan'] == 'pending']
    if not pending:
        st.info('Tidak ada transaksi kasir yang sedang menunggu persetujuan.')
        return

    cats = _cats(token)
    cmap = {c['id']: c['nama'] for c in cats}

    from statistics import median
    recent = f.list_transactions(token)
    amounts = [float(t['nominal']) for t in recent if t['status_persetujuan'] == 'disetujui']
    typical = median(amounts) if amounts else None

    from services.cashier_service import list_cashiers
    cashier_map = {}
    try:
        cashiers = list_cashiers(token)
        cashier_map = {c['id']: c['nama'] for c in cashiers}
    except Exception:
        # Safe fallback: if cashier name lookup fails, cashier_map remains empty and fallback 'Kasir' is used
        pass

    display_pending = []
    for idx, r in enumerate(pending, 1):
        is_large = bool(typical and float(r['nominal']) > 3 * typical)
        display_pending.append({
            'No.': idx,
            'Tanggal': r['tanggal'],
            'Keterangan': r['deskripsi'],
            'Nominal': format_rupiah(r['nominal']),
            'Jenis': r['tipe'].capitalize(),
            'Kategori': cmap.get(r['kategori_id'], '-'),
            'Pencatat': cashier_map.get(r['dicatat_oleh'], 'Kasir'),
            'Status Validasi': 'Nominal di atas 3x rata-rata' if is_large else 'Normal'
        })
    st.dataframe(pd.DataFrame(display_pending), hide_index=True, use_container_width=True)

    choices = {f"{r['tanggal']} | {r['deskripsi']} | {format_rupiah(r['nominal'])} ({cmap.get(r['kategori_id'], '-')}) | ID: {r['id'][:8]}": r['id'] for r in pending}

    col_sel1, col_sel2 = st.columns([1, 4])
    select_all = col_sel1.checkbox('Pilih Semua Transaksi')
    default_selected = list(choices.keys()) if select_all else []
    selected = st.multiselect('Pilih transaksi untuk verifikasi massal:', list(choices), default=default_selected)

    col_btn_a, col_btn_b = st.columns(2)
    if col_btn_a.button('Setujui Terpilih', disabled=not selected, use_container_width=True):
        try:
            f.approve_transactions(token, [choices[x] for x in selected], 'disetujui')
            st.success(f'{len(selected)} transaksi berhasil disetujui.')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if col_btn_b.button('Tolak Terpilih', disabled=not selected, use_container_width=True):
        try:
            f.approve_transactions(token, [choices[x] for x in selected], 'ditolak')
            st.success(f'{len(selected)} transaksi berhasil ditolak.')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
