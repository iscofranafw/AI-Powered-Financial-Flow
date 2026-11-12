"""Global Floating Financial AI Agent Component.

Provides a modern, universally accessible floating chat assistant positioned at the
bottom-right corner across all authenticated pages. Supports multi-session conversations,
page-context awareness, streaming responses, and safe financial transaction confirmations.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional
import uuid

import streamlit as st

from ai.guidance_prompts import get_guidance_starters
from ai.store_context import get_store_business_context
from services.chat_service import cancel_proposal, confirm_proposal, handle_chat_message
from utils.ui_components import format_rupiah


# ---------------------------------------------------------------------------
# Multi-Session Conversation State Management
# ---------------------------------------------------------------------------

def _get_store_key(user_id: str, store_id: str) -> str:
    """Generate session state storage key isolated by user and store."""
    return f"floating_chat_store_{store_id}_{user_id}"


def get_or_create_sessions(user_id: str, store_id: str) -> Dict[str, Any]:
    """Retrieve or initialize multi-session store for authenticated user."""
    key = _get_store_key(user_id, store_id)
    if key not in st.session_state:
        initial_session_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        st.session_state[key] = {
            "active_session_id": initial_session_id,
            "sessions": {
                initial_session_id: {
                    "id": initial_session_id,
                    "title": "Obrolan Baru",
                    "messages": [],
                    "visible_count": 20,
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            },
        }
    return st.session_state[key]


def get_active_session(user_id: str, store_id: str) -> Dict[str, Any]:
    """Get the currently active conversation session."""
    store = get_or_create_sessions(user_id, store_id)
    active_id = store.get("active_session_id")
    sessions = store.get("sessions", {})
    if not active_id or active_id not in sessions:
        new_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        sessions[new_id] = {
            "id": new_id,
            "title": "Obrolan Baru",
            "messages": [],
            "visible_count": 20,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        store["active_session_id"] = new_id
        store["sessions"] = sessions
    return store["sessions"][store["active_session_id"]]


def create_new_session(user_id: str, store_id: str, title: str = "Obrolan Baru") -> str:
    """Create and activate a fresh chat session."""
    store = get_or_create_sessions(user_id, store_id)
    new_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    store["sessions"][new_id] = {
        "id": new_id,
        "title": title,
        "messages": [],
        "visible_count": 20,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    store["active_session_id"] = new_id
    return new_id


def switch_session(user_id: str, store_id: str, session_id: str) -> None:
    """Switch active session to an existing session."""
    store = get_or_create_sessions(user_id, store_id)
    if session_id in store["sessions"]:
        store["active_session_id"] = session_id


def rename_session(user_id: str, store_id: str, session_id: str, new_title: str) -> None:
    """Rename an existing conversation session."""
    clean_title = new_title.strip()[:60]
    if not clean_title:
        return
    store = get_or_create_sessions(user_id, store_id)
    if session_id in store["sessions"]:
        store["sessions"][session_id]["title"] = clean_title
        store["sessions"][session_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


def delete_session(user_id: str, store_id: str, session_id: str) -> None:
    """Delete a session, activating another session or creating a fresh one."""
    store = get_or_create_sessions(user_id, store_id)
    sessions = store["sessions"]
    was_active = store.get("active_session_id") == session_id
    if session_id in sessions:
        sessions.pop(session_id, None)

    if not sessions:
        create_new_session(user_id, store_id)
    elif was_active:
        store["active_session_id"] = list(sessions.keys())[-1]


def _auto_generate_session_title(first_prompt: str) -> str:
    """Derive a clean, concise 3-6 word session title from the first prompt."""
    tokens = first_prompt.strip().split()
    if not tokens:
        return "Obrolan Keuangan"
    summary = " ".join(tokens[:5]).capitalize()
    if len(summary) > 40:
        summary = summary[:37] + "..."
    return summary


def _stream_response_generator(text: str):
    """Yield word-by-word streaming chunks for smooth typewriter feel."""
    words = text.split(" ")
    for idx, word in enumerate(words):
        chunk = word if idx == len(words) - 1 else word + " "
        yield chunk
        time.sleep(0.015)


# ---------------------------------------------------------------------------
# Context Detection
# ---------------------------------------------------------------------------

def detect_current_page_context(profile: Dict[str, Any]) -> str:
    """Detect current active application menu context from session state."""
    role = profile.get("role")
    if role == "pemilik_toko":
        return st.session_state.get("owner_menu", "Ringkasan Bisnis")
    elif role == "kasir":
        return st.session_state.get("cashier_menu", "Transaksi")
    return "Umum"


def get_contextual_starter_prompts(page_context: str, is_owner: bool) -> List[str]:
    """Produce context-sensitive starter questions based on the active screen."""
    if not is_owner:
        return [
            "Catat pemasukan kasir Rp150.000",
            "Beli kantong plastik kasir Rp20.000",
            "Tampilkan transaksi yang saya catat hari ini",
        ]

    context_map = {
        "Ringkasan Bisnis": [
            "Berapa total saldo kas toko saat ini?",
            "Bandingkan pemasukan bulan ini dengan bulan lalu",
            "Apa kategori pengeluaran terbesar toko?",
        ],
        "Transaksi": [
            "Tampilkan 5 transaksi terakhir toko",
            "Ada transaksi pengeluaran terbesar minggu ini?",
            "Catat pengeluaran beli bensin operasional Rp25.000",
        ],
        "Pencatatan Cepat": [
            "Catat penjualan tunai kasir Rp350.000",
            "Bayar sales kopi Rp120.000 untuk stok",
            "Ada pos kategori apa saja untuk belanja stok?",
        ],
        "Persetujuan Kasir": [
            "Ada berapa transaksi kasir yang belum disetujui?",
            "Berapa total nominal transaksi kasir yang pending?",
            "Jelaskan aturan persetujuan transaksi kasir",
        ],
        "Kelola Kategori": [
            "Ada pos pemasukan dan pengeluaran apa saja di toko ini?",
            "Jelaskan fungsi kategori sistem Others",
            "Bagaimana cara menggabungkan kategori pembukuan?",
        ],
        "Impor Data CSV": [
            "Bagaimana format CSV yang didukung aplikasi?",
            "Berapa batas maksimal rekomendasi kategori otomatis AI?",
            "Apakah transaksi CSV langsung disimpan sebelum disetujui?",
        ],
        "Akun Kasir": [
            "Siapa saja kasir yang terdaftar di toko saya?",
            "Berapa jumlah kasir aktif saat ini?",
            "Bagaimana cara kerja akun kasir?",
        ],
    }
    return context_map.get(
        page_context,
        [
            "Berapa saldo kas toko saya saat ini?",
            "Tampilkan ringkasan keuangan bulan ini",
            "Ada kategori pengeluaran apa saja?",
        ],
    )


# ---------------------------------------------------------------------------
# Global Floating Agent Rendering
# ---------------------------------------------------------------------------

def _visible_messages(messages, visible_count):
    """Return the newest visible_count messages (all of them if fewer)."""
    if visible_count is None or visible_count >= len(messages):
        return list(messages)
    if visible_count <= 0:
        return []
    return list(messages[-visible_count:])


@st.fragment
def render_floating_agent(token: str, profile: Dict[str, Any]) -> None:
    """Render the global floating financial AI agent launcher and conversation panel.

    Visible across all authenticated pages for Owners and Cashiers.
    Hidden for Superadmin (who manages platform accounts rather than shop books).
    Isolated using @st.fragment to prevent rerunning the parent application page.
    """
    role = profile.get("role")
    if role not in ("pemilik_toko", "kasir"):
        return

    is_owner = role == "pemilik_toko"
    user_id = str(profile.get("id", "default_user"))
    store_id = str(profile.get("toko_id", "default_store"))

    # Active page context
    active_page = detect_current_page_context(profile)

    # Resolve store business name
    store_info = get_store_business_context(store_id)
    store_name = store_info.get("nama_toko") or store_info.get("nama") or "Toko Anda"

    # Panel open/close toggle in session state
    if "floating_agent_open" not in st.session_state:
        st.session_state["floating_agent_open"] = False

    is_open = st.session_state["floating_agent_open"]

    # 1. FLOATING LAUNCHER BUTTON (Bottom-Right Fixed)
    with st.container(key="floating_agent_launcher_container"):
        launcher_label = "Tutup Copilot" if is_open else "Asisten Keuangan AI"
        launcher_type = "primary" if is_open else "secondary"
        if st.button(
            launcher_label,
            key="btn_floating_agent_toggle",
            type=launcher_type,
            help="Buka/Tutup Asisten Keuangan AI global",
        ):
            st.session_state["floating_agent_open"] = not is_open
            st.rerun(scope="fragment")

    # If panel is closed, stop here (launcher remains visible)
    if not is_open:
        return

    # 2. FLOATING CONVERSATION PANEL (Bottom-Right Overlay)
    with st.container(key="floating_agent_panel_container"):
        # Retrieve active session
        session_store = get_or_create_sessions(user_id, store_id)
        active_session = get_active_session(user_id, store_id)
        session_id = active_session["id"]
        messages = active_session["messages"]

        # --- PANEL HEADER ---
        h_col_title, h_col_actions = st.columns([7, 3])
        with h_col_title:
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; gap: 0.5rem;'>
                    <span style='display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #10B981;'></span>
                    <span style='font-size: 0.9375rem; font-weight: 700; color: #0F172A;'>Financial AI Copilot</span>
                </div>
                <div style='font-size: 0.75rem; color: #64748B; margin-top: 2px;'>
                    <strong>{store_name}</strong> &bull; <span style='background-color: #F1F5F9; padding: 1px 6px; border-radius: 4px; font-size: 0.7rem; color: #0284C7; font-weight: 600;'>{active_page}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with h_col_actions:
            act_col1, act_col2 = st.columns(2)
            with act_col1:
                if st.button("Baru", key="btn_float_new_chat", help="Mulai obrolan baru"):
                    create_new_session(user_id, store_id)
                    st.rerun(scope="fragment")
            with act_col2:
                if st.button("Tutup", key="btn_float_close", help="Tutup panel asisten"):
                    st.session_state["floating_agent_open"] = False
                    st.rerun(scope="fragment")

        # --- MULTI-SESSION SELECTOR BAR ---
        all_sessions = session_store.get("sessions", {})
        if len(all_sessions) > 1:
            session_opts = {sid: s["title"] for sid, s in all_sessions.items()}
            current_idx = list(session_opts.keys()).index(session_id) if session_id in session_opts else 0

            s_col1, s_col2 = st.columns([5, 1])
            with s_col1:
                sel_sid = st.selectbox(
                    "Pilih Obrolan:",
                    options=list(session_opts.keys()),
                    format_func=lambda sid: session_opts[sid],
                    index=current_idx,
                    key="sel_floating_session",
                    label_visibility="collapsed",
                )
                if sel_sid != session_id:
                    switch_session(user_id, store_id, sel_sid)
                    st.rerun(scope="fragment")
            with s_col2:
                if st.button("Hapus", key="btn_float_del_session", help="Hapus obrolan ini"):
                    delete_session(user_id, store_id, session_id)
                    st.rerun(scope="fragment")

        st.markdown("<hr style='margin: 0.4rem 0 0.6rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

        # --- SCROLLABLE MESSAGES CONTAINER ---
        with st.container(height=390):
            if not messages:
                role_label = "Pemilik Toko" if is_owner else "Kasir"
                st.markdown(
                    f"""
                    <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.875rem; margin-bottom: 0.75rem; font-size: 0.8125rem; color: #334155;'>
                        <div style='font-weight: 600; color: #0F172A; margin-bottom: 0.35rem;'>Halo! Asisten Keuangan Siap Membantu ({role_label})</div>
                        <div>Saya memahami konteks halaman <strong>{active_page}</strong> dan data keuangan <strong>{store_name}</strong>. Anda dapat bertanya atau memberikan instruksi pencatatan.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                starters = [item['question'] for item in get_guidance_starters(store_info.get('bidang_usaha', 'Umum'))] + get_contextual_starter_prompts(active_page, is_owner)
                st.caption("Panduan Usaha & Pertanyaan Cepat:")
                for s_idx, starter in enumerate(starters):
                    if st.button(starter, key=f"quick_prompt_{session_id}_{s_idx}", use_container_width=True):
                        # Treat quick button click as user submission
                        messages.append({"role": "user", "content": starter})
                        if active_session["title"] == "Obrolan Baru":
                            rename_session(user_id, store_id, session_id, _auto_generate_session_title(starter))

                        with st.chat_message("assistant"):
                            ack_slot = st.empty()
                            ack_slot.markdown("Baik, saya proses dulu...")

                        response = handle_chat_message(
                            token=token,
                            message_text=starter,
                            history=messages[:-1],
                            page_context=active_page,
                        )
                        asst_msg = {
                            "role": "assistant",
                            "content": response.get("message", ""),
                            "metadata": response.get("metadata", {}),
                        }
                        if response.get("type") == "proposal":
                            asst_msg["proposal"] = response["proposal"]
                            asst_msg["approval_note"] = response.get("approval_note", "")

                        messages.append(asst_msg)
                        active_session["updated_at"] = datetime.now(timezone.utc).isoformat()
                        st.rerun(scope="fragment")

            # Render existing messages
            visible_count = active_session.get("visible_count", 20)
            if len(messages) > visible_count:
                if st.button(
                    f"Lihat {len(messages) - visible_count} pesan lebih lama",
                    key=f"btn_float_see_more_{session_id}",
                    use_container_width=True,
                ):
                    active_session["visible_count"] = min(len(messages), visible_count + 20)
                    st.rerun(scope="fragment")

            visible = _visible_messages(messages, visible_count)
            offset = len(messages) - len(visible)
            for msg_idx, msg in enumerate(visible):
                global_idx = msg_idx + offset
                role_type = msg.get("role", "assistant")
                with st.chat_message(role_type):
                    st.markdown(msg.get("content", ""))

                    # Render transaction confirmation card if present
                    if msg.get("proposal"):
                        prop = msg["proposal"]
                        status = prop.get("status", "pending_confirmation")
                        is_income = prop.get("tipe") == "pemasukan"
                        badge_bg = "#ECFDF5" if is_income else "#FEF2F2"
                        badge_color = "#047857" if is_income else "#B91C1C"
                        tp_text = "Pemasukan" if is_income else "Pengeluaran"

                        st.markdown(
                            f"""
                            <div style='border: 1px solid #CBD5E1; background-color: #FFFFFF; border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 0;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;'>
                                    <span style='background-color: {badge_bg}; color: {badge_color}; font-weight: 700; font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 4px; text-transform: uppercase;'>
                                        {tp_text}
                                    </span>
                                    <span style='font-size: 0.75rem; color: #64748B;'>Tanggal: {prop.get('tanggal')}</span>
                                </div>
                                <div style='font-size: 1.125rem; font-weight: 700; color: #0F172A; margin-bottom: 0.35rem;'>
                                    {format_rupiah(prop.get('nominal', 0))}
                                </div>
                                <div style='font-size: 0.8125rem; color: #334155; margin-bottom: 0.2rem;'>
                                    <strong>Keterangan:</strong> {prop.get('deskripsi', '-')}
                                </div>
                                <div style='font-size: 0.8125rem; color: #334155; margin-bottom: 0.35rem;'>
                                    <strong>Kategori:</strong> {prop.get('kategori_nama', 'Others')}
                                </div>
                                <div style='font-size: 0.75rem; color: #64748B; border-top: 1px dashed #E2E8F0; padding-top: 0.35rem;'>
                                    {msg.get('approval_note', '')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        prop_id = prop.get("id")
                        if status == "pending_confirmation":
                            col_conf, col_canc = st.columns(2)
                            with col_conf:
                                if st.button("Konfirmasi & Simpan", key=f"float_conf_{session_id}_{prop_id}_{global_idx}", type="primary", use_container_width=True):
                                    res = confirm_proposal(token, prop)
                                    if res.get("success"):
                                        st.success(res.get("message"))
                                    else:
                                        st.error(res.get("message"))
                                    st.rerun()
                            with col_canc:
                                if st.button("Batalkan", key=f"float_canc_{session_id}_{prop_id}_{global_idx}", use_container_width=True):
                                    res = cancel_proposal(prop)
                                    st.info(res.get("message"))
                                    st.rerun(scope="fragment")

                        elif status == "confirmed":
                            res_text = prop.get("result", {}).get("message", f"Transaksi **{store_name}** telah dikonfirmasi dan disimpan.")
                            st.success(res_text)
                        elif status == "cancelled":
                            st.info("Draf transaksi ini telah dibatalkan.")

        # --- CHAT INPUT & STREAMING EXECUTION ---
        user_input = st.chat_input(
            "Tanya keuangan atau catat transaksi...",
            key=f"floating_chat_input_{session_id}",
        )

        if user_input:
            clean_input = user_input.strip()
            if clean_input:
                # Append user prompt
                messages.append({"role": "user", "content": clean_input})

                # Auto-name session if first message
                if active_session["title"] == "Obrolan Baru":
                    rename_session(user_id, store_id, session_id, _auto_generate_session_title(clean_input))

                # Process message via chat service with active page context
                with st.chat_message("assistant"):
                    ack_slot = st.empty()
                    ack_slot.markdown("Baik, saya proses dulu...")

                response = handle_chat_message(
                    token=token,
                    message_text=clean_input,
                    history=messages[:-1],
                    page_context=active_page,
                )

                response_text = response.get("message", "")
                asst_msg = {
                    "role": "assistant",
                    "content": response_text,
                    "metadata": response.get("metadata", {}),
                }
                if response.get("type") == "proposal":
                    asst_msg["proposal"] = response["proposal"]
                    asst_msg["approval_note"] = response.get("approval_note", "")

                messages.append(asst_msg)
                active_session["updated_at"] = datetime.now(timezone.utc).isoformat()
                st.rerun(scope="fragment")
