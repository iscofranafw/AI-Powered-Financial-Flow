import hashlib
import streamlit as st
import pandas as pd
from services import finance_service as finance
from services.import_service import preview_csv, enrich_rows, atomic_import
from ai.csv_cluster import cluster_rows
from ai.classifier import detect_type
from utils.ui_components import page_header, section_header


def _edit_view(rows, cols):
    return st.data_editor(
        pd.DataFrame(rows)[cols], num_rows='fixed', hide_index=True,
        use_container_width=True, key='csv_review_editor'
    )


def render(token):
    page_header('Impor Data Transaksi (CSV)', 'Unggah berkas riwayat transaksi dalam format CSV untuk dibukukan secara massal.')

    with st.expander('Format Berkas & Data Sampel'):
        sample_template = "tanggal,deskripsi,nominal,tipe\n2026-10-30,Beli bahan baku,250000,pengeluaran\n2026-10-30,Penjualan harian kasir,750000,pemasukan\n"
        st.download_button(
            'Unduh Template CSV Kosong',
            sample_template.encode('utf-8-sig'),
            'template_transaksi.csv',
            'text/csv'
        )
        st.caption('Contoh data riil berdasarkan sektor bisnis UMKM:')
        c1, c2, c3, c4 = st.columns(4)
        sample_files = [
            (c1, 'Bakery', 'data/samples/bakery.csv'),
            (c2, 'Grosir Sembako', 'data/samples/grosir.csv'),
            (c3, 'Bengkel Motor', 'data/samples/bengkel_motor.csv'),
            (c4, 'Warung Makan', 'data/samples/warung_makan.csv')
        ]
        for col, label, path in sample_files:
            try:
                with open(path, 'rb') as f_sample:
                    col.download_button(
                        f'Sampel {label}', f_sample.read(),
                        f'sampel_{label.lower().replace(" ", "_")}.csv',
                        'text/csv', use_container_width=True
                    )
            except Exception:
                pass

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    section_header('Tahap 1: Unggah Berkas')
    file = st.file_uploader('Pilih berkas CSV (UTF-8, maksimal 1.000 baris, 5 MB):', type=['csv'], key='upload_csv')
    if file is None:
        return

    content = file.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    try:
        rows = preview_csv(content)
    except Exception as exc:
        st.error(str(exc))
        return

    st.divider()
    section_header('Tahap 2: Tinjau & Verifikasi Baris')
    if any(not row['tipe'] for row in rows):
        st.warning('Jenis transaksi kosong pada beberapa baris. Tentukan jenis pemasukan/pengeluaran secara manual pada tabel di bawah:')
        start = pd.DataFrame([{**row, 'tipe': row['tipe'] or 'pengeluaran'} for row in rows])
        edited = st.data_editor(
            start, hide_index=True, use_container_width=True, key='csv_type_editor',
            column_config={'tipe': st.column_config.SelectboxColumn('Jenis', options=['pengeluaran', 'pemasukan'], required=True)},
            disabled=['baris', 'tanggal', 'deskripsi', 'nominal', 'error']
        )
        rows = edited.to_dict('records')
    else:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if any(row['error'] for row in rows):
        st.error('Terdapat baris data yang belum valid. Harap perbaiki berkas CSV Anda terlebih dahulu.')
        return

    digest = hashlib.sha256((digest + str([r['tipe'] for r in rows])).encode()).hexdigest()
    if st.session_state.get('csv_digest') != digest:
        st.session_state['csv_digest'] = digest
        st.session_state.pop('csv_proposals', None)
        st.session_state.pop('csv_clusters', None)

    existing = finance.list_categories(token)
    owner_categories = bool(existing)

    if not owner_categories:
        st.info('Kategori toko belum tersedia. Kelompokkan data transaksi terlebih dahulu untuk membentuk kategori awal.')
        detail = st.selectbox('Tingkat Keringkasan Kelompok:', ['Rinci', 'Sedang', 'Ringkas'])
        if st.button('Kelompokkan Pola Transaksi'):
            try:
                fixed = []
                for r in rows:
                    tp = r['tipe'] or detect_type(r['deskripsi'])
                    fixed.append({**r, 'tipe': tp})
                st.session_state['csv_clusters'] = {'fixed': fixed, 'groups': cluster_rows(fixed, detail)}
            except Exception as exc:
                st.error(f'Pengelompokan gagal: {exc}')
        prepared = st.session_state.get('csv_clusters')
        if not prepared:
            return
        named = {}
        with st.form('clusters_confirm'):
            for group, seed in {v['group']: v['suggested_name'] for v in prepared['groups'].values()}.items():
                named[group] = st.text_input('Nama ' + group, value=seed, key='group_' + group)
            confirmed = st.form_submit_button('Buat Kategori dan Impor Transaksi')
        if confirmed:
            try:
                for tp in ('pengeluaran', 'pemasukan'):
                    names = list(dict.fromkeys([name.strip() for gid, name in named.items() if gid.startswith(tp + '-')]))
                    for name in names:
                        finance.create_category(token, name, tp)
                    finance.create_category(token, 'Others', tp, system_seed=True)
                    if tp == 'pemasukan' and 'Tambahan Modal' not in names:
                        finance.create_category(token, 'Tambahan Modal', tp, system_seed=True)
                shop_cats = {(c['tipe'], c['nama'].casefold()): c['id'] for c in finance.list_categories(token)}
                items = []
                for ix, r in enumerate(prepared['fixed']):
                    name = named[prepared['groups'][ix]['group']].strip()
                    items.append({**r, 'kategori_id': shop_cats[(r['tipe'], name.casefold())]})
                count = atomic_import(token, items)
                st.success(f'{count} transaksi berhasil diimpor.')
                st.session_state.pop('csv_clusters', None)
                st.rerun()
            except Exception as exc:
                st.error(f'Impor ditolak: {exc}')
        return

    st.divider()
    section_header('Tahap 3: Pemetaan Kategori & Konfirmasi Final')
    if st.button('Klasifikasikan Kategori Otomatis'):
        bar = st.progress(0)
        try:
            st.session_state['csv_proposals'] = enrich_rows(token, rows, progress=bar.progress)
        except Exception as exc:
            st.error(f'Klasifikasi gagal: {exc}')

    suggested = st.session_state.get('csv_proposals')
    if not suggested:
        return

    names = {c['id']: c['nama'] for c in existing}
    edited = [{**r, 'kategori': names.get(r.get('kategori_id'), 'Others')} for r in suggested]
    st.caption('Periksa kategori per baris. Anda dapat menyunting nama kategori sebelum melakukan impor permanen.')
    data = _edit_view(edited, ['baris', 'tanggal', 'deskripsi', 'nominal', 'tipe', 'kategori', 'status_kategori'])

    if st.button('Konfirmasi dan Impor Seluruh Transaksi'):
        try:
            cmap = {(c['tipe'], c['nama'].casefold()): c['id'] for c in finance.list_categories(token)}
            items = []
            for i, r in data.iterrows():
                row = r.to_dict()
                tp = str(row['tipe']).strip()
                name = str(row['kategori']).strip().casefold()
                cid = cmap.get((tp, name))
                if not cid:
                    raise ValueError(f"Baris {row['baris']}: Kategori '{name}' belum tersedia untuk jenis {tp}.")
                source = suggested[i]
                items.append({**source, **row, 'kategori_id': cid, 'status_kategori': 'dikonfirmasi'})
            count = atomic_import(token, items)
            st.success(f'{count} transaksi berhasil diimpor secara atomik.')
            st.session_state.pop('csv_proposals', None)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
