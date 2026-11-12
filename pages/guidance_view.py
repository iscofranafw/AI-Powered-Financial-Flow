import streamlit as st

from ai.knowledge_rag import (
    BUSINESS_GROUP_LABELS,
    DEFAULT_KNOWLEDGE_DIR,
    answer_guidance_for_actor,
    get_knowledge_index_status,
    map_business_type,
    parse_markdown_sections,
)
from ai.store_context import get_store_business_context
from services.auth_service import verify_actor
from utils.ui_components import page_header, section_header

_SAMPLE_QUESTIONS = {
    'bakery': [
        'Bagaimana cara mengelola stok bahan baku dan roti agar tidak banyak yang sisa?',
        'Bagaimana cara menghitung HPP produk bakery per batch termasuk energi?',
        'Apa saja perizinan dasar (NIB, PIRT, Halal) untuk usaha toko roti?',
    ],
    'warung_makan': [
        'Bagaimana menjaga food cost agar warung makan tetap menguntungkan?',
        'Bagaimana memisahkan pengelolaan stok bahan basah dan bahan kering?',
        'Apa perbedaan harga menu dine-in dan aplikasi pesan antar online?',
    ],
    'retail': [
        'Apa perbedaan manajemen margin usaha retail eceran dan grosir sembako?',
        'Bagaimana menghindari overstock saat supplier menawarkan diskon grosir besar?',
        'Bagaimana aturan stock opname dan metode ABC di toko kelontong?',
    ],
    'bengkel_motor': [
        'Bagaimana memisahkan pembukuan pendapatan jasa servis dan penjualan sparepart?',
        'Apa pengelompokan stok sparepart fast, medium, dan slow moving di bengkel?',
        'Bagaimana aturan NIB KBLI dan pengelolaan limbah oli bekas bengkel motor?',
    ],
    'lainnya': [
        'Bagaimana cara memisahkan keuangan pribadi pemilik dan kas operasional usaha?',
        'Bagaimana langkah mengurus NIB gratis melalui portal OSS-RBA untuk UMKM?',
        'Apa SOP sederhana saat membuka dan menutup usaha harian?',
    ],
}


def render(token: str):
    page_header(
        'Panduan Operasional Usaha',
        'Referensi manajemen stok, HPP, pencatatan kas, SOP harian, dan legalitas sesuai bidang usaha toko.',
    )

    try:
        actor = verify_actor(token, {'pemilik_toko', 'superadmin'})
    except Exception as exc:
        st.error(f'Gagal memverifikasi akses pengguna: {exc}')
        return

    store_id = actor.get('toko_id')
    ctx = get_store_business_context(store_id) if store_id else {}
    raw_bidang = (ctx.get('bidang_usaha') or '').strip() if ctx else ''
    store_name = (ctx.get('nama_toko') or ctx.get('nama') or 'Toko Anda').strip() if ctx else 'Toko Anda'
    mapped_group = map_business_type(raw_bidang)
    group_label = BUSINESS_GROUP_LABELS.get(mapped_group, mapped_group)

    idx_status = get_knowledge_index_status().get(mapped_group, {})

    c1, c2, c3 = st.columns(3)
    c1.metric('Toko Aktif', store_name)
    c2.metric('Bidang Usaha Terpetakan', f'{group_label} ({mapped_group})')
    c3.metric(
        'Indeks Pengetahuan',
        f"{idx_status.get('chunk_count', 0)} bagian" if idx_status.get('ready') else 'Belum siap',
    )

    st.info(
        f"Pencarian panduan ini **terisolasi penuh** hanya pada korpus pengetahuan kelompok **{mapped_group}** "
        f"(asal bidang usaha toko: `{raw_bidang or '-'}`). Data transaksi toko Anda tidak pernah "
        "dicampur ke indeks panduan ini."
    )

    section_header('Tanya Panduan Operasional Toko')
    samples = _SAMPLE_QUESTIONS.get(mapped_group, _SAMPLE_QUESTIONS['lainnya'])

    if 'guidance_query_input' not in st.session_state:
        st.session_state['guidance_query_input'] = ''

    st.caption('Contoh topik cepat:')
    cols = st.columns(len(samples))
    for i, sample in enumerate(samples):
        if cols[i].button(sample[:52] + ('...' if len(sample) > 52 else ''), key=f'guidance_ex_{i}', use_container_width=True):
            st.session_state['guidance_query_input'] = sample

    query = st.text_area(
        'Pertanyaan seputar operasional, HPP, stok, SOP, atau legalitas usaha:',
        key='guidance_query_input',
        height=90,
        placeholder='Contoh: Bagaimana cara mengelola stok FIFO dan menghindari dead stock?',
    )

    if st.button('Cari Jawaban Berbasis Panduan', type='primary'):
        cleaned = (query or '').strip()
        if not cleaned:
            st.warning('Silakan tuliskan pertanyaan Anda terlebih dahulu.')
        else:
            with st.spinner('Mencari kutipan panduan terverifikasi dan menyusun jawaban...'):
                res = answer_guidance_for_actor(token, query=cleaned)
            st.session_state['guidance_last_result'] = res

    res = st.session_state.get('guidance_last_result')
    if res:
        st.divider()
        section_header('Hasil Jawaban Panduan')
        if not res.get('found'):
            st.warning(res.get('answer', 'belum ada referensi untuk pertanyaan itu'))
        else:
            st.markdown(res.get('answer', ''))
            citations = res.get('citations') or []
            if citations:
                with st.expander('Sumber & Kutipan yang Dirujuk', expanded=False):
                    for item in citations:
                        st.markdown(
                            f"- **{item['judul_bagian']}** (`{item['nama_file']}`, skor relevansi: `{item['score']:.2f}`)  \n"
                            f"  Sumber: {item['sumber_url']}"
                        )

    st.divider()
    with st.expander(f'Daftar Isi Lengkap Panduan {group_label}', expanded=False):
        guide_dir = DEFAULT_KNOWLEDGE_DIR / mapped_group
        md_files = sorted(guide_dir.glob('*.md')) if guide_dir.exists() else []
        if not md_files:
            st.info('Belum ada berkas panduan untuk kelompok usaha ini.')
        else:
            for md_path in md_files:
                sections = parse_markdown_sections(md_path.read_text(encoding='utf-8'))
                st.markdown(f'#### `{md_path.name}`')
                for sec in sections:
                    st.markdown(f"**{sec['judul_bagian']}**")
                    st.caption(f"Sumber: {sec['sumber_url']}")
                    st.markdown(sec['body'])
                    st.markdown('---')
