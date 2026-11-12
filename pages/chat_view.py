from ai.guidance_prompts import get_guidance_starters
from ai.store_context import get_store_business_context
import streamlit as st
from utils.ui_components import page_header, format_rupiah
from services.chat_service import handle_chat_message, confirm_proposal, cancel_proposal


def render(token, profile):
    is_owner = profile.get('role') == 'pemilik_toko'
    subtitle = (
        "Asisten dan Copilot Bisnis Toko: pencatatan transaksi, analisis pos pembukuan, persetujuan kasir, dan ringkasan keuangan."
        if is_owner else
        "Pencatatan transaksi kasir percakapan dengan tinjauan langsung sebelum disimpan."
    )
    page_header("AI Financial Chat", subtitle)

    user_id = profile.get('id', 'default_user')
    history_key = f"chat_messages_{user_id}"

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Reset chat button
    col_info, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Reset Obrolan", use_container_width=True):
            st.session_state[history_key] = []
            st.rerun()

    messages = st.session_state[history_key]

    guidance_input = None
    if not messages:
        business = get_store_business_context(profile.get('toko_id', '')).get('bidang_usaha', 'Umum')
        starters = get_guidance_starters(business)
        if starters:
            st.caption('Panduan untuk usaha Anda:')
            for index, starter in enumerate(starters):
                if st.button(starter['question'], key=f'guide_chat_{user_id}_{index}',
                             help=f"{starter['filename']} — {starter['section']}", use_container_width=True):
                    guidance_input = starter['question']

    # Show role-aware starter tips if conversation is empty
    if not messages:
        if is_owner:
            st.markdown("""
            <div style='border: 1px solid #E2E8F0; background-color: #F8FAFC; border-radius: 6px; padding: 1.25rem; margin-bottom: 1.5rem; font-size: 0.875rem; color: #334155;'>
                <div style='font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Contoh Pertanyaan & Perintah Copilot Pemilik Toko:</div>
                <ul style='margin-bottom: 0; padding-left: 1.25rem; line-height: 1.7;'>
                    <li><em>"Ada kategori pemasukan dan pengeluaran apa saja di aplikasi ini?"</em></li>
                    <li><em>"Ada persetujuan pengeluaran/pemasukan yang dilakukan kasir hari ini yang belum dikonfirmasi?"</em></li>
                    <li><em>"Berapa total pemasukan bulan ini?"</em> &rarr; tindak lanjuti dengan <em>"Bandingkan dengan bulan lalu."</em></li>
                    <li><em>"Berapa saldo kas toko saya saat ini?"</em></li>
                    <li><em>"Siapa saja kasir yang terdaftar di toko saya?"</em></li>
                    <li><em>"Catat pemasukan penjualan 10 produk dengan total Rp500.000."</em></li>
                    <li><em>"Bagaimana cara kerja persetujuan transaksi kasir?"</em></li>
                    <li><em>"Bagaimana mekanisme peramalan pengeluaran Prophet?"</em></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='border: 1px solid #E2E8F0; background-color: #F8FAFC; border-radius: 6px; padding: 1.25rem; margin-bottom: 1.5rem; font-size: 0.875rem; color: #334155;'>
                <div style='font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Contoh Perintah Kasir:</div>
                <ul style='margin-bottom: 0; padding-left: 1.25rem; line-height: 1.7;'>
                    <li><em>"Catat pemasukan penjualan roti Rp150.000 hari ini."</em></li>
                    <li><em>"Saya membeli kantong kresek Rp25.000 untuk kasir."</em></li>
                    <li><em>"Tampilkan transaksi yang saya catat hari ini."</em></li>
                    <li><em>"Ada kategori pemasukan apa saja di toko ini?"</em></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Render conversation history
    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

            # Render structured transaction proposal preview if present
            if msg.get("proposal"):
                prop = msg["proposal"]
                status = prop.get("status", "pending_confirmation")
                is_income = prop.get("tipe") == "pemasukan"
                badge_bg = "#ECFDF5" if is_income else "#FEF2F2"
                badge_color = "#047857" if is_income else "#B91C1C"
                tp_text = "Pemasukan" if is_income else "Pengeluaran"

                st.markdown(f"""
                <div style='border: 1px solid #CBD5E1; background-color: #FFFFFF; border-radius: 6px; padding: 1rem 1.25rem; margin: 0.75rem 0;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                        <span style='background-color: {badge_bg}; color: {badge_color}; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px; text-transform: uppercase;'>
                            {tp_text}
                        </span>
                        <span style='font-size: 0.8125rem; color: #64748B;'>Tanggal: {prop.get('tanggal')}</span>
                    </div>
                    <div style='font-size: 1.25rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;'>
                        {format_rupiah(prop.get('nominal', 0))}
                    </div>
                    <div style='font-size: 0.875rem; color: #334155; margin-bottom: 0.25rem;'>
                        <strong>Keterangan:</strong> {prop.get('deskripsi', '-')}
                    </div>
                    <div style='font-size: 0.875rem; color: #334155; margin-bottom: 0.5rem;'>
                        <strong>Kategori:</strong> {prop.get('kategori_nama', 'Others')}
                    </div>
                    <div style='font-size: 0.8125rem; color: #64748B; border-top: 1px dashed #E2E8F0; padding-top: 0.5rem; margin-top: 0.5rem;'>
                        {msg.get('approval_note', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show interactive confirmation actions or locked result
                if status == "pending_confirmation":
                    col_confirm, col_cancel, _ = st.columns([2, 2, 4])
                    prop_id = prop.get("id")

                    if col_confirm.button("Konfirmasi & Simpan", key=f"confirm_{prop_id}", type="primary"):
                        result = confirm_proposal(token, prop)
                        if result.get("success"):
                            st.success(result.get("message"))
                        else:
                            st.error(result.get("message"))
                        st.rerun()

                    if col_cancel.button("Batalkan", key=f"cancel_{prop_id}"):
                        result = cancel_proposal(prop)
                        st.info(result.get("message"))
                        st.rerun()

                elif status == "confirmed":
                    res_msg = prop.get("result", {}).get("message", "Transaksi telah dikonfirmasi dan disimpan.")
                    st.success(f"{res_msg}")

                elif status == "cancelled":
                    st.info("Transaksi ini telah dibatalkan oleh pengguna.")

    # Chat input box
    user_input = st.chat_input("Ketik instruksi transaksi atau pertanyaan bisnis/keuangan Anda di sini...") or guidance_input
    if user_input:
        # Append user message
        st.session_state[history_key].append({
            "role": "user",
            "content": user_input
        })

        # Process message via chat service passing conversation history for context awareness
        with st.spinner("Memproses permintaan Anda..."):
            response = handle_chat_message(token, user_input, history=messages)

        assistant_msg = {
            "role": "assistant",
            "content": response.get("message", ""),
            "metadata": response.get("metadata", {})
        }
        if response.get("type") == "proposal":
            assistant_msg["proposal"] = response["proposal"]
            assistant_msg["approval_note"] = response.get("approval_note", "")

        st.session_state[history_key].append(assistant_msg)
        st.rerun()
