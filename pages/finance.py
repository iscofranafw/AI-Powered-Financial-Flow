import streamlit as st
import pandas as pd
from datetime import date
from services import finance_service as f
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

def transactions_ui(token, owner):
    page_header('Pencatatan Transaksi', 'Pencatatan mutasi kas pemasukan dan pengeluaran operasional toko.')
    cats = _cats(token)
    if not cats:
        st.warning('Belum ada kategori terdaftar. Pemilik toko perlu menyiapkan kategori di menu Kategori Pembukuan terlebih dahulu.')
        return

    tipe = st.selectbox('Jenis Transaksi', ['pengeluaran', 'pemasukan'], key='tx_type_select')
    valid_cats = [c for c in cats if c['tipe'] == tipe]
    opts = _choices(valid_cats)
    if not opts:
        st.warning(f"Belum ada kategori untuk jenis '{tipe}'. Silakan tambah kategori baru terlebih dahulu.")
        return

    with st.form('add_finance'):
        tanggal = st.date_input('Tanggal Transaksi', value=date.today())
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
    if not rows:
        st.info('Belum ada transaksi tercatat pada toko ini.')
        return

    cmap = {c['id']: c['nama'] for c in cats}
    status_label_map = {
        'disetujui': 'Disetujui',
        'pending': 'Menunggu Persetujuan',
        'ditolak': 'Ditolak'
    }

    display_data = []
    for idx, r in enumerate(rows, 1):
        display_data.append({
            'No.': idx,
            'Tanggal': r['tanggal'],
            'Keterangan': r['deskripsi'],
            'Nominal': format_rupiah(r['nominal']),
            'Jenis': r['tipe'].capitalize(),
            'Kategori': cmap.get(r['kategori_id'], '-'),
            'Status': status_label_map.get(r['status_persetujuan'], r['status_persetujuan'])
        })
    st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

    editable = [r for r in rows if owner or r['status_persetujuan'] == 'pending']
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

def categories_ui(token):
    page_header('Kategori Pembukuan', 'Kelola struktur taksonomi pemasukan dan pengeluaran toko.')
    if st.button('Siapkan Kategori Dasar Otomatis (Jika Kosong)'):
        try:
            f.initialize_categories(token)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    cats = _cats(token)
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
                'Tipe Akun': tipe_tag
            })
        st.dataframe(pd.DataFrame(formatted_cats), use_container_width=True, hide_index=True)
    else:
        st.info('Belum ada kategori yang terdaftar pada toko ini.')

    st.divider()
    with st.form('new_category'):
        section_header('Tambah Kategori Baru')
        name = st.text_input('Nama Kategori')
        tp = st.selectbox('Jenis Kategori', ['pengeluaran', 'pemasukan'])
        desc = st.text_input('Deskripsi Singkat (Opsional)')
        create = st.form_submit_button('Simpan Kategori Baru')

    if create:
        try:
            f.create_category(token, name, tp, desc)
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
            save = st.form_submit_button('Perbarui Kategori')

        if save:
            try:
                f.rename_category(token, cid, name, desc)
                st.success('Kategori berhasil diubah.')
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
    section_header('Analisis Kategori Tidak Terpetakan (Others)', 'Menganalisis transaksi yang masuk ke kategori Others untuk menemukan peluang kategori baru.')
    typ = st.selectbox('Jenis Transaksi untuk Dianalisis:', ['pengeluaran', 'pemasukan'])
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
