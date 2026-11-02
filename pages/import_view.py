import hashlib
import io
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from ai.classifier import detect_type
from ai.text_normalizer import clean_text_for_storage, normalize
from services import finance_service as finance
from services.ai_service import discover_categories_for_import
from services.import_service import (
    analyze_csv_upload,
    enrich_rows,
)
from services.smart_import_engine import (
    detect_duplicates,
    execute_scalable_import,
    parse_and_validate_dataframe,
)
from utils.ui_components import page_header, section_header


def _edit_view(rows: List[Dict[str, Any]], cols: List[str]):
    return st.data_editor(
        pd.DataFrame(rows)[cols],
        num_rows='fixed',
        hide_index=True,
        use_container_width=True,
        key='csv_review_editor',
    )


def render(token: str):
    page_header(
        'Impor Data Transaksi (CSV)',
        'Unggah berkas CSV riwayat transaksi untuk penemuan kategori otomatis, persetujuan pemilik toko, dan klasifikasi massal.',
    )

    from pages.finance import render_ai_pending_queue
    render_ai_pending_queue(token)

    with st.expander('Format Berkas & Data Sampel'):
        sample_template = "tanggal,deskripsi,nominal,tipe\n2026-10-30,Beli bahan baku,250000,pengeluaran\n2026-10-30,Penjualan harian kasir,750000,pemasukan\n"
        st.download_button(
            'Unduh Template CSV Kosong',
            sample_template.encode('utf-8-sig'),
            'template_transaksi.csv',
            'text/csv',
        )
        st.caption('Contoh data riil berdasarkan sektor bisnis UMKM:')
        c1, c2, c3, c4 = st.columns(4)
        sample_files = [
            (c1, 'Bakery', 'data/samples/bakery.csv'),
            (c2, 'Grosir Sembako', 'data/samples/grosir.csv'),
            (c3, 'Bengkel Motor', 'data/samples/bengkel_motor.csv'),
            (c4, 'Warung Makan', 'data/samples/warung_makan.csv'),
        ]
        for col, label, path in sample_files:
            try:
                with open(path, 'rb') as f_sample:
                    col.download_button(
                        f'Sampel {label}',
                        f_sample.read(),
                        f'sampel_{label.lower().replace(" ", "_")}.csv',
                        'text/csv',
                        use_container_width=True,
                    )
            except Exception:
                pass

        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        if st.button('Periksa Integritas Dataset Sampel Referensi'):
            from services.dataset_service import audit_reference_datasets
            report = audit_reference_datasets()
            if report['status'] == 'PASSED':
                st.success(
                    f"Seluruh {report['total_files_audited']} berkas referensi "
                    f"({report['total_rows_audited']} baris data) terverifikasi valid dan siap digunakan."
                )
            else:
                st.error('Ditemukan ketidaksesuaian pada berkas referensi.')

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAHAP 1 — Unggah Berkas & Deteksi Format
    # ----------------------------------------------------
    section_header('Tahap 1: Unggah Berkas Transaksi (CSV)')
    file = st.file_uploader(
        'Pilih berkas CSV transaksi (Mendukung UTF-8, Windows CSV, hingga 25 MB):',
        type=['csv'],
        key='upload_csv',
    )
    if file is None:
        # Clear upload-related session state when file is removed
        for key in list(st.session_state.keys()):
            if key.startswith('csv_'):
                st.session_state.pop(key, None)
        return

    content = file.getvalue()
    content_hash = hashlib.sha256(content).hexdigest()

    # Reset cache when a different file is uploaded
    if st.session_state.get('csv_content_hash') != content_hash:
        st.session_state['csv_content_hash'] = content_hash
        st.session_state.pop('csv_analysis', None)
        st.session_state.pop('csv_discovery_data', None)
        st.session_state.pop('review_expense_cats', None)
        st.session_state.pop('review_income_cats', None)
        st.session_state.pop('csv_category_approved', None)
        st.session_state.pop('csv_approved_categories', None)
        st.session_state.pop('csv_proposals', None)
        st.session_state.pop('csv_import_result', None)

    if 'csv_analysis' not in st.session_state:
        try:
            st.session_state['csv_analysis'] = analyze_csv_upload(content)
        except Exception as exc:
            st.error(f'Gagal membaca format berkas CSV: {exc}')
            return

    analysis = st.session_state['csv_analysis']
    schema_info = analysis['schema_analysis']
    headers = analysis['sample_df'].columns.tolist()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Encoding Berkas', analysis['encoding'].upper())
    delim_label = 'Koma (,)' if analysis['delimiter'] == ',' else ('Titik Koma (;)' if analysis['delimiter'] == ';' else 'Tab')
    m2.metric('Pemisah Kolom', delim_label)
    m3.metric('Estimasi Baris', f"{analysis['estimated_rows']:,}")
    m4.metric('Total Kolom Sumber', len(headers))

    if schema_info.get('warnings'):
        for warn in schema_info['warnings']:
            st.info(warn)

    # Pemetaan Kolom Otomatis
    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    with st.expander('Sesuaikan Pemetaan Kolom (Opsional)', expanded=False):
        auto_map = schema_info['mappings']
        c_map1, c_map2 = st.columns(2)
        header_options_with_none = ['(Tidak ada)'] + headers

        def _get_index(col_val, opts):
            return opts.index(col_val) if col_val in opts else 0

        with c_map1:
            sel_tanggal = st.selectbox(
                'Kolom Tanggal (Wajib):',
                headers,
                index=_get_index(auto_map.get('tanggal'), headers),
                key='map_tanggal',
            )
            sel_deskripsi = st.selectbox(
                'Kolom Keterangan / Barang (Wajib):',
                headers,
                index=_get_index(auto_map.get('deskripsi'), headers),
                key='map_deskripsi',
            )
            sel_nominal = st.selectbox(
                'Kolom Nominal (Wajib):',
                headers,
                index=_get_index(auto_map.get('nominal'), headers),
                key='map_nominal',
            )

        with c_map2:
            type_mode_options = [
                'Otomatis dari awalan ID (I- = Pemasukan, E- = Pengeluaran)',
                'Gunakan kolom tipe transaksi',
                'Semua transaksi adalah Pengeluaran',
                'Semua transaksi adalah Pemasukan',
            ]
            default_mode_idx = 0 if schema_info['type_inference_mode'] == 'id_prefix' else (1 if auto_map.get('tipe') else 2)
            sel_type_mode = st.selectbox(
                'Metode Penentuan Jenis Transaksi:',
                type_mode_options,
                index=default_mode_idx,
                key='map_type_mode',
            )

            sel_tipe_col = None
            if sel_type_mode == 'Gunakan kolom tipe transaksi':
                sel_tipe_col = st.selectbox(
                    'Pilih Kolom Jenis Transaksi:',
                    header_options_with_none,
                    index=_get_index(auto_map.get('tipe'), header_options_with_none),
                    key='map_tipe_col',
                )

            sel_source_id = st.selectbox(
                'Kolom ID Transaksi Sumber (Opsional):',
                header_options_with_none,
                index=_get_index(auto_map.get('source_id'), header_options_with_none),
                key='map_source_id',
            )
            sel_notes = st.selectbox(
                'Kolom Catatan Tambahan (Opsional):',
                header_options_with_none,
                index=_get_index(auto_map.get('notes'), header_options_with_none),
                key='map_notes',
            )

    active_mappings = {
        'tanggal': sel_tanggal if 'sel_tanggal' in locals() else auto_map.get('tanggal'),
        'deskripsi': sel_deskripsi if 'sel_deskripsi' in locals() else auto_map.get('deskripsi'),
        'nominal': sel_nominal if 'sel_nominal' in locals() else auto_map.get('nominal'),
        'tipe': sel_tipe_col if ('sel_tipe_col' in locals() and sel_type_mode == 'Gunakan kolom tipe transaksi' and sel_tipe_col != '(Tidak ada)') else auto_map.get('tipe'),
        'source_id': sel_source_id if ('sel_source_id' in locals() and sel_source_id != '(Tidak ada)') else auto_map.get('source_id'),
        'notes': sel_notes if ('sel_notes' in locals() and sel_notes != '(Tidak ada)') else auto_map.get('notes'),
        'staff': auto_map.get('staff'),
        'time': auto_map.get('time'),
    }

    resolved_inference_mode = (
        'id_prefix' if ('sel_type_mode' in locals() and sel_type_mode.startswith('Otomatis dari awalan ID'))
        else ('column' if ('sel_type_mode' in locals() and sel_type_mode == 'Gunakan kolom tipe transaksi')
        else ('constant_income' if ('sel_type_mode' in locals() and 'Pemasukan' in sel_type_mode) else 'constant_expense'))
    )
    resolved_default_type = 'pemasukan' if resolved_inference_mode == 'constant_income' else 'pengeluaran'

    # Normalisasi & Validasi Dataframe
    try:
        raw_df = pd.read_csv(
            io.StringIO(analysis['raw_text']),
            sep=analysis['delimiter'],
            dtype=str,
            keep_default_na=False,
        )
        valid_rows, rejected_rows = parse_and_validate_dataframe(
            df=raw_df,
            mappings=active_mappings,
            type_inference_mode=resolved_inference_mode,
            default_type=resolved_default_type,
            append_notes=(active_mappings.get('notes') is not None),
            start_row_index=2,
        )
    except Exception as exc:
        st.error(f'Kesalahan dalam memproses data CSV: {exc}')
        return

    # Deteksi Duplikat
    actor = finance._member(token)
    shop_id = actor['toko_id']
    new_rows, duplicate_rows = detect_duplicates(shop_id=shop_id, valid_rows=valid_rows)

    v1, v2, v3 = st.columns(3)
    v1.metric('Baris Baru Siap Proses', f"{len(new_rows):,}")
    v2.metric('Potensi Duplikat Terdeteksi', f"{len(duplicate_rows):,}")
    v3.metric('Baris Tidak Valid / Ditolak', f"{len(rejected_rows):,}")

    skip_duplicates = True
    if duplicate_rows:
        dup_choice = st.radio(
            'Tindakan untuk Potensi Transaksi Duplikat:',
            [
                'Lewati transaksi duplikat (Rekomendasi - hanya impor transaksi baru)',
                'Tetap impor semua transaksi termasuk potensi duplikat',
            ],
            key='dup_handling_choice',
        )
        skip_duplicates = dup_choice.startswith('Lewati')

    rows_to_import = new_rows if skip_duplicates else valid_rows
    if not rows_to_import:
        st.warning('Tidak ada data transaksi baru yang dapat diproses.')
        return

    # ----------------------------------------------------
    # TAHAP 2 — Penemuan Kategori Otomatis oleh AI (Category Discovery)
    # ----------------------------------------------------
    st.divider()
    section_header('Tahap 2: Penemuan Kategori Otomatis oleh AI (Category Discovery)')

    if 'csv_discovery_data' not in st.session_state:
        with st.spinner('Menganalisis deskripsi transaksi dan menemukan pos kategori yang relevan...'):
            t_disc_start = time.perf_counter()
            discovery = discover_categories_for_import(token, rows_to_import)
            discovery['elapsed_time'] = time.perf_counter() - t_disc_start
            st.session_state['csv_discovery_data'] = discovery

            # Initialize review lists
            st.session_state['review_expense_cats'] = [
                {
                    'id': f'exp_rec_{i}',
                    'nama': r['nama'],
                    'tipe': 'pengeluaran',
                    'deskripsi': r.get('deskripsi', ''),
                    'contoh': r.get('contoh', []),
                    'is_new': True,
                    'source': r.get('source', 'ai_discovery'),
                }
                for i, r in enumerate(discovery.get('pengeluaran', []))
            ]

            st.session_state['review_income_cats'] = [
                {
                    'id': f'inc_rec_{i}',
                    'nama': r['nama'],
                    'tipe': 'pemasukan',
                    'deskripsi': r.get('deskripsi', ''),
                    'contoh': r.get('contoh', []),
                    'is_new': True,
                    'source': r.get('source', 'ai_discovery'),
                }
                for i, r in enumerate(discovery.get('pemasukan', []))
            ]

    discovery = st.session_state['csv_discovery_data']
    if discovery.get('ai_pending_count'):
        st.warning(f"Ollama tidak tersedia - {discovery['ai_pending_count']} transaksi sudah tersimpan dan menunggu AI. Gunakan klasifikasi ulang pada halaman Transaksi.")
        return
    coverage_info = discovery.get('coverage_stats', {})
    existing_store_cats = finance.list_categories(token)

    exp_cov = coverage_info.get('pengeluaran', {})
    inc_cov = coverage_info.get('pemasukan', {})

    cd1, cd2, cd3, cd4 = st.columns(4)
    cd1.metric('Kategori Toko Saat Ini', len(existing_store_cats))
    cd2.metric('Cakupan Pengeluaran', f"{exp_cov.get('coverage_pct', 0.0)}%")
    cd3.metric('Rekomendasi Pengeluaran', f"{len(st.session_state.get('review_expense_cats', []))} (Maks 7 AI)")
    cd4.metric('Rekomendasi Pemasukan', f"{len(st.session_state.get('review_income_cats', []))} (Maks 7 AI)")

    st.caption(
        'AI telah menganalisis pola deskripsi transaksi pada berkas CSV Anda. '
        'Batas rekomendasi otomatis AI maksimal 7 kategori per jenis. '
        'Sebagai Pemilik Toko, Anda memiliki kendali penuh untuk meninjau, menyunting, menghapus, atau '
        'menambahkan kategori baru tanpa batasan jumlah.'
    )

    # ----------------------------------------------------
    # TAHAP 3 — Tinjau & Sesuaikan Kategori (Review Categories)
    # ----------------------------------------------------
    st.divider()
    section_header('Tahap 3: Tinjau & Sesuaikan Kategori Toko')

    tab_exp, tab_inc, tab_add = st.tabs(['Pos Pengeluaran Toko', 'Pos Pemasukan Toko', 'Tambah Kategori Manual (Tanpa Batas)'])

    # --- TAB PENGELUARAN ---
    with tab_exp:
        existing_exp = [c for c in existing_store_cats if c.get('tipe') == 'pengeluaran']
        if existing_exp:
            st.markdown("<div style='font-size: 0.875rem; font-weight: 600; color: #334155; margin-bottom: 0.5rem;'>Kategori Pengeluaran Toko yang Sudah Aktif:</div>", unsafe_allow_html=True)
            tags_html = " ".join([
                f"<span style='display:inline-block; padding: 0.25rem 0.6rem; margin: 0.2rem; background-color: #F1F5F9; border: 1px solid #CBD5E1; border-radius: 4px; font-size: 0.8125rem; color: #1E293B;'>{c['nama']} "
                f"<small style='color:#64748B;'>({'Sistem' if c.get('sistem') else 'Aktif'})</small></span>"
                for c in existing_exp
            ])
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        exp_recs = st.session_state.get('review_expense_cats', [])
        st.markdown(f"<div style='font-size: 0.875rem; font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Rekomendasi Pos Pengeluaran Baru ({len(exp_recs)} kategori):</div>", unsafe_allow_html=True)

        if not exp_recs:
            st.info('Tidak ada rekomendasi kategori pengeluaran baru. Seluruh transaksi pengeluaran sudah terpetakan ke kategori yang ada, atau Anda dapat menambahkannya secara manual.')

        to_remove_exp = None
        for idx, item in enumerate(exp_recs):
            with st.container():
                c_name, c_desc, c_del = st.columns([3, 4, 1])
                with c_name:
                    new_val = st.text_input(
                        f"Nama Pengeluaran #{idx+1}:",
                        value=item['nama'],
                        key=f"exp_name_in_{item['id']}",
                    )
                    item['nama'] = new_val.strip()
                with c_desc:
                    new_desc = st.text_input(
                        f"Deskripsi Pos #{idx+1}:",
                        value=item.get('deskripsi', ''),
                        key=f"exp_desc_in_{item['id']}",
                    )
                    item['deskripsi'] = new_desc.strip()
                with c_del:
                    st.markdown("<div style='margin-top: 1.7rem;'></div>", unsafe_allow_html=True)
                    if st.button('Hapus', key=f"btn_del_exp_{item['id']}"):
                        to_remove_exp = idx

                if item.get('contoh'):
                    st.caption(f"Contoh transaksi terkait: {', '.join(item['contoh'][:3])}")
                st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px dashed #E2E8F0;'>", unsafe_allow_html=True)

        if to_remove_exp is not None:
            st.session_state['review_expense_cats'].pop(to_remove_exp)
            st.rerun()

    # --- TAB PEMASUKAN ---
    with tab_inc:
        existing_inc = [c for c in existing_store_cats if c.get('tipe') == 'pemasukan']
        if existing_inc:
            st.markdown("<div style='font-size: 0.875rem; font-weight: 600; color: #334155; margin-bottom: 0.5rem;'>Kategori Pemasukan Toko yang Sudah Aktif:</div>", unsafe_allow_html=True)
            tags_html = " ".join([
                f"<span style='display:inline-block; padding: 0.25rem 0.6rem; margin: 0.2rem; background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 4px; font-size: 0.8125rem; color: #166534;'>{c['nama']} "
                f"<small style='color:#64748B;'>({'Sistem' if c.get('sistem') else 'Aktif'})</small></span>"
                for c in existing_inc
            ])
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        inc_recs = st.session_state.get('review_income_cats', [])
        st.markdown(f"<div style='font-size: 0.875rem; font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Rekomendasi Pos Pemasukan Baru ({len(inc_recs)} kategori):</div>", unsafe_allow_html=True)

        if not inc_recs:
            st.info('Tidak ada rekomendasi kategori pemasukan baru. Seluruh transaksi pemasukan sudah terpetakan ke kategori yang ada, atau Anda dapat menambahkannya secara manual.')

        to_remove_inc = None
        for idx, item in enumerate(inc_recs):
            with st.container():
                c_name, c_desc, c_del = st.columns([3, 4, 1])
                with c_name:
                    new_val = st.text_input(
                        f"Nama Pemasukan #{idx+1}:",
                        value=item['nama'],
                        key=f"inc_name_in_{item['id']}",
                    )
                    item['nama'] = new_val.strip()
                with c_desc:
                    new_desc = st.text_input(
                        f"Deskripsi Pos #{idx+1}:",
                        value=item.get('deskripsi', ''),
                        key=f"inc_desc_in_{item['id']}",
                    )
                    item['deskripsi'] = new_desc.strip()
                with c_del:
                    st.markdown("<div style='margin-top: 1.7rem;'></div>", unsafe_allow_html=True)
                    if st.button('Hapus', key=f"btn_del_inc_{item['id']}"):
                        to_remove_inc = idx

                if item.get('contoh'):
                    st.caption(f"Contoh transaksi terkait: {', '.join(item['contoh'][:3])}")
                st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px dashed #E2E8F0;'>", unsafe_allow_html=True)

        if to_remove_inc is not None:
            st.session_state['review_income_cats'].pop(to_remove_inc)
            st.rerun()

    # --- TAB TAMBAH MANUAL (UNLIMITED) ---
    with tab_add:
        st.markdown("<div style='font-size: 0.875rem; font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Tambah Kategori Kustom Tambahan (Tanpa Batas Kuota):</div>", unsafe_allow_html=True)
        st.caption('Pemilik toko bebas menambahkan kategori baru sebanyak apa pun yang dibutuhkan untuk pembukuan.')

        with st.form('form_add_manual_category'):
            m_col1, m_col2 = st.columns([2, 1])
            with m_col1:
                man_name = st.text_input('Nama Kategori Baru (Maksimal 80 karakter):')
            with m_col2:
                man_type = st.selectbox('Jenis Kategori:', ['pengeluaran', 'pemasukan'], format_func=lambda x: 'Pengeluaran' if x == 'pengeluaran' else 'Pemasukan')
            man_desc = st.text_input('Deskripsi Kategori (Opsional):')
            man_submit = st.form_submit_button('Tambahkan ke Daftar Kategori')

        if man_submit:
            man_name_clean = clean_text_for_storage(man_name).strip()
            if not man_name_clean:
                st.error('Nama kategori tidak boleh kosong.')
            elif len(man_name_clean) > 80:
                st.error('Nama kategori maksimal 80 karakter.')
            else:
                target_list_key = 'review_expense_cats' if man_type == 'pengeluaran' else 'review_income_cats'
                existing_in_list = [c['nama'].casefold() for c in st.session_state[target_list_key]]
                existing_in_db = [c['nama'].casefold() for c in existing_store_cats if c.get('tipe') == man_type]

                if man_name_clean.casefold() in existing_in_list or man_name_clean.casefold() in existing_in_db:
                    st.error(f"Kategori '{man_name_clean}' sudah ada dalam daftar {man_type}.")
                else:
                    new_item = {
                        'id': f"manual_{man_type}_{int(time.time()*1000)}",
                        'nama': man_name_clean,
                        'tipe': man_type,
                        'deskripsi': man_desc.strip(),
                        'contoh': [],
                        'is_new': True,
                        'source': 'manual_owner',
                    }
                    st.session_state[target_list_key].append(new_item)
                    st.success(f"Kategori '{man_name_clean}' berhasil ditambahkan ke daftar {man_type}.")
                    st.rerun()

    # ----------------------------------------------------
    # TAHAP 4 — Persetujuan Pemilik Toko (Owner Approval)
    # ----------------------------------------------------
    st.divider()
    section_header('Tahap 4: Persetujuan Konfigurasi Kategori')

    all_reviewed_exp = st.session_state.get('review_expense_cats', [])
    all_reviewed_inc = st.session_state.get('review_income_cats', [])
    total_new_cats = len(all_reviewed_exp) + len(all_reviewed_inc)

    st.markdown(
        f"<div style='border: 1px solid #CBD5E1; background-color: #F8FAFC; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1rem;'>"
        f"<strong>Ringkasan Persetujuan:</strong> Terdapat <strong>{total_new_cats} kategori baru</strong> yang akan disetujui dan didaftarkan ke toko "
        f"({len(all_reviewed_exp)} pengeluaran, {len(all_reviewed_inc)} pemasukan). "
        f"Kategori toko aktif saat ini: <strong>{len(existing_store_cats)} kategori</strong>. "
        f"Setelah Anda menyetujui, sistem akan memulai klasifikasi massal menggunakan pos kategori yang telah disetujui."
        f"</div>",
        unsafe_allow_html=True,
    )

    is_approved = st.session_state.get('csv_category_approved', False)

    btn_approve = st.button(
        'Setujui Kategori & Lanjutkan',
        type='primary',
        use_container_width=True,
        key='btn_approve_and_continue',
    )

    if btn_approve:
        # Validate category names
        validation_errors = []
        for tp_label, cat_list in [('Pengeluaran', all_reviewed_exp), ('Pemasukan', all_reviewed_inc)]:
            names_seen = set()
            for c in cat_list:
                c_name = c['nama'].strip()
                if not c_name:
                    validation_errors.append(f"Terdapat nama kategori {tp_label} yang masih kosong.")
                elif len(c_name) > 80:
                    validation_errors.append(f"Kategori '{c_name}' melebihi batas 80 karakter.")
                elif c_name.casefold() in names_seen:
                    validation_errors.append(f"Kategori ganda ditemukan pada {tp_label}: '{c_name}'.")
                names_seen.add(c_name.casefold())

        if validation_errors:
            for err in validation_errors:
                st.error(err)
            return

        with st.spinner('Menyimpan kategori yang disetujui ke database toko...'):
            try:
                # 1. Ensure system seed categories exist if store has zero categories
                existing_now = finance.list_categories(token)
                existing_map = {(c['tipe'], c['nama'].casefold()): c for c in existing_now}

                # Ensure system categories
                for tp in ('pengeluaran', 'pemasukan'):
                    if (tp, 'others') not in existing_map:
                        try:
                            finance.create_category(token, 'Others', tp, system_seed=True)
                        except Exception:
                            pass
                if ('pemasukan', 'tambahan modal') not in existing_map:
                    try:
                        finance.create_category(token, 'Tambahan Modal', 'pemasukan', system_seed=True)
                    except Exception:
                        pass

                # Refresh existing
                existing_now = finance.list_categories(token)
                existing_map = {(c['tipe'], c['nama'].casefold()): c for c in existing_now}

                # 2. Insert new approved categories
                created_count = 0
                for c in all_reviewed_exp + all_reviewed_inc:
                    c_name = c['nama'].strip()
                    tp = c['tipe']
                    if (tp, c_name.casefold()) not in existing_map:
                        finance.create_category(token, c_name, tp, description=c.get('deskripsi', ''))
                        created_count += 1

                # 3. Fetch final approved categories with valid UUIDs
                final_approved_cats = finance.list_categories(token)
                st.session_state['csv_approved_categories'] = final_approved_cats
                st.session_state['csv_new_categories_created'] = created_count
                st.session_state['csv_category_approved'] = True
                st.session_state.pop('csv_proposals', None)  # Trigger fresh classification
                st.rerun()

            except Exception as exc:
                st.error(f'Gagal mendaftarkan kategori yang disetujui: {exc}')
                return

    # If categories have not been approved yet, stop here.
    # REQUIREMENT: Bulk classification must NOT start until owner approves!
    if not is_approved or 'csv_approved_categories' not in st.session_state:
        st.info('Silakan tinjau dan klik **Setujui Kategori & Lanjutkan** di atas untuk melanjutkan ke klasifikasi transaksi.')
        return

    # ----------------------------------------------------
    # TAHAP 5 — Klasifikasi Transaksi Otomatis (Automatic Classification)
    # ----------------------------------------------------
    st.divider()
    section_header('Tahap 5: Klasifikasi Transaksi Otomatis')

    approved_categories = st.session_state['csv_approved_categories']

    if 'csv_proposals' not in st.session_state:
        progress_bar = st.progress(0, text='Mengklasifikasikan seluruh transaksi...')
        status_info = st.empty()
        t_classify_start = time.perf_counter()

        def _on_classify_progress(pct):
            progress_bar.progress(pct, text=f"Mengklasifikasikan transaksi ({int(pct*100)}%)...")
            status_info.caption(f"Memproses {len(rows_to_import):,} transaksi dengan pos kategori yang telah disetujui...")

        try:
            classified_rows = enrich_rows(
                token=token,
                rows=rows_to_import,
                progress=_on_classify_progress,
                categories=approved_categories,
            )
            elapsed_classify = time.perf_counter() - t_classify_start
            st.session_state['csv_classify_elapsed'] = elapsed_classify
            st.session_state['csv_proposals'] = classified_rows
            progress_bar.empty()
            status_info.empty()
            st.rerun()
        except Exception as exc:
            st.error(f'Proses klasifikasi mengalami kendala: {exc}')
            return

    suggested = st.session_state['csv_proposals']
    saved_pending = sum(r.get('_saved_ai_pending',False) for r in suggested)
    if saved_pending:
        st.warning(f'Ollama tidak tersedia - {saved_pending} transaksi sudah tersimpan dan menunggu AI.')

    # Transparency Metrics
    auto_cnt = sum(1 for r in suggested if r.get('status_kategori') == 'otomatis' and not r.get('is_unresolved'))
    review_cnt = sum(1 for r in suggested if r.get('is_unresolved') or r.get('status_kategori') == 'perlu_tinjauan')
    confirmed_cnt = sum(1 for r in suggested if r.get('status_kategori') == 'dikonfirmasi' and not r.get('is_unresolved'))

    cat_name_map = {c['id']: c['nama'] for c in approved_categories}
    unique_cats_used = len(set(r.get('kategori_id') for r in suggested if r.get('kategori_id')))

    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric('Total Transaksi', f"{len(suggested):,}")
    cm2.metric('Otomatis Terpetakan', f"{auto_cnt + confirmed_cnt:,}")
    cm3.metric('Perlu Tinjauan (Unresolved)', f"{review_cnt:,}")
    cm4.metric('Kategori Toko Dipakai', unique_cats_used)

    # UNRESOLVED TRANSACTIONS GROUPING & BULK RESOLUTION
    unresolved_rows = [r for r in suggested if r.get('is_unresolved') or r.get('status_kategori') == 'perlu_tinjauan']

    if unresolved_rows:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.warning(
            f"Terdapat {len(unresolved_rows)} transaksi yang belum dapat dipetakan secara meyakinkan ke kategori yang disetujui. "
            "Gunakan fitur resolusi massal di bawah untuk memetakan transaksi serupa sekaligus dalam satu klik."
        )

        with st.expander('Resolusi Massal Transaksi Unresolved Berdasarkan Pola Deskripsi', expanded=True):
            # Group unresolved by normalized description
            groups: Dict[Tuple[str, str], List[int]] = {}  # (tipe, norm_desc) -> list of indices in suggested
            for idx, r in enumerate(suggested):
                if r.get('is_unresolved') or r.get('status_kategori') == 'perlu_tinjauan':
                    tp = r.get('tipe', 'pengeluaran')
                    norm_desc = normalize(clean_text_for_storage(r.get('deskripsi', '')))
                    key = (tp, norm_desc)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(idx)

            # Sort groups by count descending
            sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

            for (g_type, g_desc), row_indices in sorted_groups[:15]:
                sample_desc = suggested[row_indices[0]]['deskripsi']
                count_tx = len(row_indices)
                total_amt = sum(float(suggested[i].get('nominal', 0) or 0) for i in row_indices)

                allowed_for_type = [c for c in approved_categories if c.get('tipe') == g_type]
                type_opts = {c['nama']: c['id'] for c in allowed_for_type}

                with st.container():
                    c_info, c_sel, c_act = st.columns([4, 3, 2])
                    with c_info:
                        st.markdown(
                            f"<strong>\"{sample_desc}\"</strong><br>"
                            f"<small style='color: #64748B;'>{count_tx} transaksi ({g_type.capitalize()}) | Total: Rp {total_amt:,.0f}</small>",
                            unsafe_allow_html=True,
                        )
                    with c_sel:
                        sel_cat_name = st.selectbox(
                            'Pilih Kategori:',
                            options=list(type_opts.keys()),
                            key=f"sel_res_{g_type}_{abs(hash(g_desc))}",
                            label_visibility='collapsed',
                        )
                    with c_act:
                        if st.button('Terapkan Semua', key=f"btn_res_{g_type}_{abs(hash(g_desc))}"):
                            target_cid = type_opts[sel_cat_name]
                            for row_idx in row_indices:
                                suggested[row_idx]['kategori_id'] = target_cid
                                suggested[row_idx]['status_kategori'] = 'dikonfirmasi'
                                suggested[row_idx]['is_unresolved'] = False
                                suggested[row_idx]['message'] = f"Diselesaikan manual oleh pemilik: {sel_cat_name}"
                            st.session_state['csv_proposals'] = suggested
                            st.success(f"Berhasil menerapkan ke {count_tx} transaksi.")
                            st.rerun()
                    st.markdown("<hr style='margin: 0.4rem 0; border: none; border-top: 1px dotted #CBD5E1;'>", unsafe_allow_html=True)

    # Preview classified transactions
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.caption('Pratinjau transaksi hasil klasifikasi:')
    preview_data = [
        {
            'Baris': r.get('baris', i+1),
            'Tanggal': r.get('tanggal', ''),
            'Deskripsi': r.get('deskripsi', ''),
            'Nominal': f"Rp {float(r.get('nominal', 0) or 0):,.0f}",
            'Jenis': r.get('tipe', '').capitalize(),
            'Kategori': cat_name_map.get(r.get('kategori_id'), 'Others'),
            'Status': 'Perlu Tinjauan' if (r.get('is_unresolved') or r.get('status_kategori') == 'perlu_tinjauan') else 'Terpetakan',
        }
        for i, r in enumerate(suggested[:50])
    ]
    st.dataframe(pd.DataFrame(preview_data), hide_index=True, use_container_width=True)

    # ----------------------------------------------------
    # TAHAP 6 — Simpan Transaksi ke Database (Save Transactions)
    # ----------------------------------------------------
    st.divider()
    section_header('Tahap 6: Simpan Transaksi ke Database Toko')

    btn_confirm_save = st.button(
        f"Simpan {len(rows_to_import):,} Transaksi ke Database Toko",
        type='primary',
        use_container_width=True,
        key='btn_confirm_save_transactions',
    )

    if btn_confirm_save:
        t_save_start = time.perf_counter()
        progress_bar = st.progress(0, text='Mempersiapkan batch penyimpanan database...')
        status_text = st.empty()
        total_items = len(suggested)
        total_batches = max(1, (total_items + 499) // 500)

        # Fallback dictionary for others category
        other_cats_by_type = {
            c['tipe']: c['id']
            for c in approved_categories
            if c.get('nama', '').casefold() == 'others'
        }

        # Prepare payload with strict category ID enforcement
        items_to_save = []
        for r in suggested:
            if r.get('_saved_ai_pending'):
                continue
            tp = str(r['tipe']).strip().lower()
            cat_id = r.get('kategori_id')
            if not cat_id or cat_id not in cat_name_map:
                cat_id = other_cats_by_type.get(tp)

            items_to_save.append({
                **r,
                'kategori_id': cat_id,
                'status_kategori': r.get('status_kategori', 'dikonfirmasi'),
            })

        def _on_save_progress(pct, current_batch=None):
            elapsed = time.perf_counter() - t_save_start
            b_info = f"Batch {current_batch}/{total_batches}" if current_batch else ""
            progress_bar.progress(pct, text=f"Menyimpan ke database ({int(pct*100)}%) {b_info} [{elapsed:.1f}s]")
            status_text.caption(
                f"Transaksi: {min(total_items, int(pct * total_items)):,}/{total_items:,} | "
                f"Batch: {current_batch or 1}/{total_batches} | Waktu berjalan: {elapsed:.1f}s"
            )

        try:
            result = execute_scalable_import(
                token=token,
                rows=items_to_save,
                batch_size=500,
                progress_callback=_on_save_progress,
            )
            progress_bar.empty()
            status_text.empty()

            elapsed = time.perf_counter() - t_save_start
            result['elapsed_time'] = elapsed
            result['total_file_rows'] = len(raw_df)
            result['valid_rows_count'] = len(valid_rows)
            result['rejected_rows_count'] = len(rejected_rows)
            result['duplicate_rows_count'] = len(duplicate_rows)
            result['unresolved_count'] = sum(1 for r in items_to_save if r.get('status_kategori') == 'perlu_tinjauan')
            result['categories_used_count'] = len(set(r['kategori_id'] for r in items_to_save))
            result['new_categories_created_count'] = st.session_state.get('csv_new_categories_created', 0)

            st.session_state['csv_import_result'] = result
            st.session_state.pop('csv_proposals', None)
            st.rerun()

        except Exception as exc:
            st.error(f'Penyimpanan database gagal: {exc}')

    # ----------------------------------------------------
    # TAHAP 7 — Ringkasan Penyelesaian (Completion Summary)
    # ----------------------------------------------------
    res = st.session_state.get('csv_import_result')
    if res:
        st.divider()
        section_header('Tahap 7: Ringkasan Penyelesaian Impor')

        if res.get('failed_batches'):
            st.warning(
                f"Impor selesai sebagian: {res['total_imported']:,} dari {res['total_submitted']:,} transaksi berhasil disimpan. "
                f"Batch {res['failed_batches'][0]['batch_index']} mengalami kendala: {res['failed_batches'][0]['error']}"
            )
        else:
            st.success(
                f"Impor Berhasil: {res['total_imported']:,} transaksi telah berhasil disimpan ke database toko "
                f"dalam {res['batches_completed']} batch aman."
            )

        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
        row1_c1.metric('Total Berkas', f"{res.get('total_file_rows', 0):,} baris")
        row1_c2.metric('Transaksi Valid', f"{res.get('valid_rows_count', 0):,}")
        row1_c3.metric('Baris Ditolak', f"{res.get('rejected_rows_count', 0):,}")
        row1_c4.metric('Duplikat Dilewati', f"{res.get('duplicate_rows_count', 0):,}")

        row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
        row2_c1.metric('Berhasil Disimpan', f"{res['total_imported']:,}")
        row2_c2.metric('Transaksi Unresolved', f"{res.get('unresolved_count', 0):,}")
        row2_c3.metric('Kategori Baru Dibuat', f"{res.get('new_categories_created_count', 0):,}")
        row2_c4.metric('Waktu Pemrosesan', f"{res.get('elapsed_time', 0.0):.2f} detik")

        if 'count_before' in res and 'count_after' in res:
            st.caption(
                f"Audit Database Toko: Jumlah transaksi sebelum impor: {res['count_before']:,} | "
                f"Setelah impor: {res['count_after']:,}."
            )

        st.caption('Model peramalan keuangan (Prophet & statistik) serta indeks semantik telah diperbarui otomatis dengan data transaksi terbaru.')
