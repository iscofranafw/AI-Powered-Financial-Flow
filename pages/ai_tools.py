import streamlit as st
from services import ai_service
from services import finance_service as finance
from ai.ollama_client import diagnosis
from datetime import date
from utils.ui_components import page_header, section_header, format_rupiah


def quick_entry(token, owner=True):
    page_header('Pencatatan Cepat Cerdas', 'Input transaksi dalam bahasa sehari-hari. Sistem mengekstrak nominal dan merekomendasikan kategori secara otomatis.')
    st.caption('Contoh penulisan: "beli telur 3 tray 250rb", "bayar listrik oven 480rb", atau "omzet kasir hari ini 1.5jt".')

    all_cats = finance.list_categories(token)
    if not all_cats:
        st.warning('Toko Anda belum memiliki kategori. Silakan siapkan kategori dasar pembukuan terlebih dahulu.')
        if owner and st.button('Inisialisasi Kategori Dasar Sekarang'):
            try:
                finance.initialize_categories(token)
                st.success('Kategori dasar berhasil dibuat.')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with st.form('quick_entry'):
        desc = st.text_input('Deskripsi Transaksi')
        type_choice = st.selectbox('Jenis Transaksi', ['Deteksi otomatis', 'pengeluaran', 'pemasukan'])
        submit = st.form_submit_button('Proses Analisis')

    if submit:
        if not desc.strip():
            st.error('Deskripsi transaksi wajib diisi.')
        else:
            try:
                st.session_state['quick_preview'] = ai_service.quick_preview(token, desc, None if type_choice == 'Deteksi otomatis' else type_choice)
            except Exception as exc:
                st.error(str(exc))

    data = st.session_state.get('quick_preview')
    if not data:
        return

    if data.get('_saved_ai_pending'):
        st.warning('Ollama tidak tersedia - transaksi sudah tersimpan tanpa kategori dan menunggu AI. Gunakan klasifikasi ulang pada halaman Transaksi.')
        return

    st.divider()
    section_header('Hasil Analisis Transaksi')
    badge = ''
    if data.get('skor') is not None:
        status_lbl = 'Otomatis' if data.get('status_kategori') == 'otomatis' else 'Perlu Tinjauan'
        badge = f" &nbsp; <span style='display:inline-block; padding: 0.15rem 0.5rem; background-color: #F1F5F9; border-radius: 4px; font-size: 0.75rem; color: #475569;'>Tingkat Keyakinan: {data['skor']:.2f} ({status_lbl})</span>"
    st.markdown(f"<div style='border: 1px solid #E2E8F0; background-color: #FFFFFF; border-radius: 6px; padding: 0.875rem 1rem; margin-bottom: 1.25rem; font-size: 0.875rem; color: #334155;'><strong>Rekomendasi Sistem:</strong> {data.get('message', '-')}{badge}</div>", unsafe_allow_html=True)

    cats = [x for x in all_cats if x['tipe'] == data['tipe']]
    if not cats:
        st.warning(f"Belum ada kategori untuk jenis '{data['tipe']}'. Silakan buat kategori baru terlebih dahulu.")
        return

    options = {x['nama']: x['id'] for x in cats}
    other_cat = next((c for c in cats if c['nama'].casefold() == 'others'), None)

    can_create_new = owner  # Owners have unlimited category creation (see finance_service.check_category_quota)
    suggested_new = (data.get('usulan') or {}).get('nama_baru', '').strip()

    cat_modes = []
    if owner and can_create_new and suggested_new:
        cat_modes = ['Buat kategori baru (sesuai rekomendasi)', 'Pilih kategori yang sudah ada', 'Masukkan ke Others']
    elif owner and can_create_new:
        cat_modes = ['Pilih kategori yang sudah ada', 'Buat kategori baru', 'Masukkan ke Others']
    else:
        cat_modes = ['Pilih kategori yang sudah ada', 'Masukkan ke Others']

    if suggested_new:
        st.info(f"Rekomendasi nama kategori baru: {suggested_new}")

    cat_choice = st.radio('Tindakan Kategori:', cat_modes, key='quick_cat_mode')

    with st.form('commit_quick'):
        when = st.date_input('Tanggal Transaksi', value=date.fromisoformat(data['tanggal']))
        nominal_float = float(data['nominal']) if data.get('nominal') else 0.0
        amount = st.number_input('Nominal Terverifikasi (Rp)', min_value=0.01, value=nominal_float, step=1000.0, format='%.2f')
        description = st.text_input('Keterangan Transaksi Final', value=data['deskripsi'])

        new_cat_name = ''
        selected_cat_id = None

        if cat_choice.startswith('Buat kategori baru'):
            default_name = suggested_new if suggested_new else ''
            new_cat_name = st.text_input('Nama Kategori Baru:', value=default_name)
            st.caption(f"Kategori baru akan ditambahkan untuk jenis: {data['tipe']}")
        elif cat_choice == 'Masukkan ke Others':
            if other_cat:
                st.caption(f"Transaksi akan dimasukkan ke kategori sistem: {other_cat['nama']}")
                selected_cat_id = other_cat['id']
            else:
                st.error("Kategori 'Others' tidak ditemukan.")
        else:
            default = next((i for i, c in enumerate(cats) if c['id'] == data.get('kategori_id')), 0)
            chosen = st.selectbox('Pilih Kategori:', list(options), index=default)
            selected_cat_id = options[chosen]

        saved = st.form_submit_button('Konfirmasi dan Simpan Transaksi')
    if saved:
        try:
            if cat_choice.startswith('Buat kategori baru'):
                if not new_cat_name.strip():
                    raise ValueError('Nama kategori baru wajib diisi.')
                created = finance.create_category(token, new_cat_name.strip(), data['tipe'])
                if not created:
                    raise ValueError('Gagal membuat kategori baru.')
                selected_cat_id = created[0]['id']
            elif not selected_cat_id:
                raise ValueError('Kategori transaksi belum dipilih.')

            nom_str = str(int(amount)) if amount.is_integer() else f"{amount:.2f}"
            row = {**data, 'tanggal': when.isoformat(), 'nominal': nom_str, 'deskripsi': description}
            ai_service.commit_preview(token, row, selected_cat_id)
            st.session_state.pop('quick_preview', None)
            st.success('Transaksi berhasil tercatat.')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def initial_categories(token):
    page_header('Rekomendasi Kategori Usaha', 'Saran kategori standar berdasarkan bidang usaha toko Anda.')
    st.caption('Kategori yang diusulkan hanya akan disimpan ke pembukuan setelah Anda konfirmasi.')
    if st.button('Muat Rekomendasi Kategori'):
        try:
            st.session_state['initial_suggested'] = ai_service.prepare_categories(token)
        except Exception as exc:
            st.error(str(exc))
    suggested = st.session_state.get('initial_suggested')
    if not suggested:
        return
    with st.form('save_initial_categories'):
        exp = st.multiselect('Kategori Pengeluaran (Maksimal 7):', suggested['pengeluaran'], default=suggested['pengeluaran'])
        inc = st.multiselect('Kategori Pemasukan (Maksimal 7):', suggested['pemasukan'], default=suggested['pemasukan'])
        save = st.form_submit_button('Simpan Kategori Terpilih')
    if save:
        try:
            ai_service.save_initial_categories(token, {'pengeluaran': exp, 'pemasukan': inc})
            st.session_state.pop('initial_suggested', None)
            st.success('Kategori berhasil disimpan.')
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def diagnostics():
    page_header('Status Sistem & Layanan AI', 'Diagnosa koneksi layanan inferensi lokal Ollama dan model.')
    if st.button('Periksa Status Koneksi'):
        result = diagnosis()
        if result['ok']:
            st.success(f"Status Layanan: {result['message']}")
        else:
            st.error(f"Status Layanan: {result['message']}")
        if result['models']:
            st.markdown(f"<div style='font-size: 0.8125rem; color: #64748B; margin-top: 0.5rem;'>Model terdeteksi: {', '.join(result['models'])}</div>", unsafe_allow_html=True)
