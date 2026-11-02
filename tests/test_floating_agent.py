"""Automated Regression and Integration Tests for Global Floating Financial AI Agent.

Tests:
1. Removal of dedicated AI Financial Chat from Owner and Cashier sidebar menus.
2. Global floating launcher rendering and visibility across pages.
3. Toggle open/close interaction without losing conversation state.
4. Multi-session lifecycle (create, switch, rename, delete, auto-titling).
5. Conversation persistence across page navigation.
6. Context-awareness (menu detection and tailored starter prompts).
7. Role-based scoping (Owner full tools, Cashier scoped, Superadmin hidden).
8. Financial action safety (proposal confirmation and cancellation).
9. Resolution of inventory withdrawal inquiry ("ambil 5 bungkus indomie") as clarification needed.
"""

from unittest.mock import MagicMock, patch
import pytest
import streamlit as st

from ai.financial_chat import Intent, detect_chat_intent
from services.chat_service import handle_chat_message
from utils.floating_agent import (
    _auto_generate_session_title,
    _visible_messages,
    create_new_session,
    delete_session,
    detect_current_page_context,
    get_active_session,
    get_contextual_starter_prompts,
    get_or_create_sessions,
    rename_session,
    render_floating_agent,
    switch_session,
)


@pytest.fixture(autouse=True)
def clean_streamlit_session_state():
    """Ensure clean session state for each test case."""
    keys_to_clear = [k for k in list(st.session_state.keys()) if "floating" in k or "chat" in k or "menu" in k]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    yield
    for k in keys_to_clear:
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# 1. Sidebar Navigation Removal Verification
# ---------------------------------------------------------------------------

def test_owner_sidebar_menu_excludes_ai_financial_chat():
    """Verify 'AI Financial Chat' is removed from the owner sidebar menu (chat is provided by the floating Asisten Keuangan AI)."""
    import inspect
    from pages import owner

    source = inspect.getsource(owner.render)
    menu_def = source.split("menu_options = [")[1].split("]")[0]
    assert "AI Financial Chat" not in menu_def
    assert "Ringkasan Bisnis" in menu_def
    assert "Transaksi" in menu_def
    assert "Kelola Kategori" in menu_def
    assert "Impor Data CSV" in menu_def


def test_cashier_sidebar_menu_excludes_ai_financial_chat():
    """Verify 'AI Financial Chat' is removed from the visible cashier sidebar menu."""
    import inspect
    from pages import cashier

    source = inspect.getsource(cashier.render)
    menu_def = source.split('pilihan = st.sidebar.radio("Pilih Menu:", [')[1].split("]")[0]
    assert "AI Financial Chat" not in menu_def
    assert "Transaksi" in menu_def
    assert "Pencatatan Cepat" in menu_def
    assert "Status Sistem" in menu_def


# ---------------------------------------------------------------------------
# 2. Multi-Session Conversation Management Tests
# ---------------------------------------------------------------------------

def test_initial_session_creation():
    """Verify initial session is automatically provisioned with default title."""
    user_id = "usr-test-1"
    store_id = "store-test-1"

    session = get_active_session(user_id, store_id)
    assert session is not None
    assert session["title"] == "Obrolan Baru"
    assert session["messages"] == []
    assert len(session["id"]) > 10


def test_create_and_switch_multi_sessions():
    """Verify creating multiple chat sessions and switching between them."""
    user_id = "usr-test-2"
    store_id = "store-test-2"

    # Initial session
    s1 = get_active_session(user_id, store_id)
    s1_id = s1["id"]

    # Create new session
    s2_id = create_new_session(user_id, store_id, title="Analisis Penjualan")
    assert s2_id != s1_id

    # Active session should now be s2
    active_now = get_active_session(user_id, store_id)
    assert active_now["id"] == s2_id
    assert active_now["title"] == "Analisis Penjualan"

    # Switch back to s1
    switch_session(user_id, store_id, s1_id)
    active_switched = get_active_session(user_id, store_id)
    assert active_switched["id"] == s1_id
    assert active_switched["title"] == "Obrolan Baru"


def test_rename_and_delete_session():
    """Verify renaming a session and deleting with automatic fallback."""
    user_id = "usr-test-3"
    store_id = "store-test-3"

    s1_id = create_new_session(user_id, store_id, title="Sesi Pertama")
    s2_id = create_new_session(user_id, store_id, title="Sesi Kedua")

    rename_session(user_id, store_id, s1_id, "Nama Baru Sesi 1")
    store = get_or_create_sessions(user_id, store_id)
    assert store["sessions"][s1_id]["title"] == "Nama Baru Sesi 1"

    # Delete s2 (active)
    delete_session(user_id, store_id, s2_id)
    store = get_or_create_sessions(user_id, store_id)
    assert s2_id not in store["sessions"]
    # Fallback activates s1
    assert store["active_session_id"] == s1_id


def test_auto_generate_session_title():
    """Verify clean titles are generated from user first prompt."""
    t1 = _auto_generate_session_title("Berapa total pemasukan bulan Oktober 2026?")
    assert t1 == "Berapa total pemasukan bulan oktober"

    t2 = _auto_generate_session_title("Catat pengeluaran beli bensin operasional toko sebesar 25000 rupiah")
    assert len(t2) <= 40
    assert "Catat" in t2


# ---------------------------------------------------------------------------
# 3. Conversation State Persistence Across Navigation
# ---------------------------------------------------------------------------

def test_conversation_persists_across_page_navigation():
    """Verify chat messages survive switching between Owner pages."""
    user_id = "usr-nav"
    store_id = "store-nav"

    # 1. User starts on 'Ringkasan Bisnis'
    st.session_state["owner_menu"] = "Ringkasan Bisnis"
    session = get_active_session(user_id, store_id)
    session["messages"].append({"role": "user", "content": "Tampilkan omzet bulan ini"})
    session["messages"].append({"role": "assistant", "content": "Total omzet bulan ini Rp15.000.000"})

    # 2. User navigates to 'Transaksi' page
    st.session_state["owner_menu"] = "Transaksi"
    page_now = detect_current_page_context({"role": "pemilik_toko"})
    assert page_now == "Transaksi"

    # Verify session messages are completely preserved
    session_after_nav = get_active_session(user_id, store_id)
    assert len(session_after_nav["messages"]) == 2
    assert session_after_nav["messages"][0]["content"] == "Tampilkan omzet bulan ini"

    # 3. User navigates to 'Kelola Kategori'
    st.session_state["owner_menu"] = "Kelola Kategori"
    session_after_nav2 = get_active_session(user_id, store_id)
    assert len(session_after_nav2["messages"]) == 2


# ---------------------------------------------------------------------------
# 4. Context Awareness Detection Tests
# ---------------------------------------------------------------------------

def test_detect_current_page_context():
    """Verify current page context resolution for Owner and Cashier."""
    st.session_state["owner_menu"] = "Impor Data CSV"
    ctx_owner = detect_current_page_context({"role": "pemilik_toko"})
    assert ctx_owner == "Impor Data CSV"

    st.session_state["cashier_menu"] = "Pencatatan Cepat"
    ctx_cashier = detect_current_page_context({"role": "kasir"})
    assert ctx_cashier == "Pencatatan Cepat"


def test_contextual_starter_prompts():
    """Verify starter prompts are tailored based on current screen."""
    p_report = get_contextual_starter_prompts("Ringkasan Bisnis", is_owner=True)
    assert any("saldo" in p.lower() or "pemasukan" in p.lower() for p in p_report)

    p_import = get_contextual_starter_prompts("Impor Data CSV", is_owner=True)
    assert any("csv" in p.lower() or "kategori" in p.lower() for p in p_import)

    p_cashier = get_contextual_starter_prompts("Transaksi", is_owner=False)
    assert any("kasir" in p.lower() for p in p_cashier)


# ---------------------------------------------------------------------------
# 5. Role-Based Permissions & Safety Scoping
# ---------------------------------------------------------------------------

def test_superadmin_does_not_render_floating_agent():
    """Verify Superadmin cannot access store-scoped floating agent."""
    with patch("streamlit.button") as mock_btn, patch("streamlit.container") as mock_ctr:
        profile_super = {"id": "sa-1", "role": "superadmin"}
        render_floating_agent("token-sa", profile_super)
        # Should return immediately with zero widgets rendered
        assert mock_btn.call_count == 0
        assert mock_ctr.call_count == 0


def test_cashier_cannot_access_owner_financial_summary_in_floating_agent():
    """Verify Cashier using floating agent is safely blocked from owner financials."""
    token = "token-cashier"
    mock_cashier = {"id": "ksr-1", "email": "kasir@toko.com", "role": "kasir", "toko_id": "tok-1"}

    with patch("services.chat_service.verify_actor", return_value=mock_cashier):
        res = handle_chat_message(token, "Berapa total pemasukan toko bulan ini?")
        assert "hanya dapat diakses oleh Pemilik Toko" in res["message"]


# ---------------------------------------------------------------------------
# 6. Intent Recognition: Inventory Withdrawal Inquiry Bug Fix
# ---------------------------------------------------------------------------

def test_ambil_indomie_detected_as_clarification_needed():
    """Verify 'ambil 5 bungkus indomie' is recognized as inventory withdrawal, NOT cash income."""
    token = "token-owner"
    mock_owner = {"id": "own-1", "email": "bos@toko.com", "role": "pemilik_toko", "toko_id": "tok-1"}

    with patch("services.chat_service.verify_actor", return_value=mock_owner):
        # 1. Direct intent detection
        intent = detect_chat_intent("ambil 5 bungkus indomie")
        assert intent == Intent.CLARIFICATION_NEEDED

        # 2. Chat service dispatch
        res = handle_chat_message(token, "ambil 5 bungkus indomie")
        assert res["intent"] == Intent.CLARIFICATION_NEEDED
        assert "Pengambilan barang dagangan" in res["message"]
        assert "bukan mutasi kas langsung" in res["message"]
        # Ensure it does NOT create a proposal or cash income
        assert res.get("type") != "proposal"


def test_ambil_beras_and_mie_instan_clarification():
    """Verify other inventory taking expressions also require clarification."""
    token = "token-owner"
    mock_owner = {"id": "own-1", "email": "bos@toko.com", "role": "pemilik_toko", "toko_id": "tok-1"}

    with patch("services.chat_service.verify_actor", return_value=mock_owner):
        res1 = handle_chat_message(token, "ambil beras 5kg untuk stok etalase")
        assert res1["intent"] == Intent.CLARIFICATION_NEEDED

        res2 = handle_chat_message(token, "ambil rokok 2 bungkus")
        assert res2["intent"] == Intent.CLARIFICATION_NEEDED


# ---------------------------------------------------------------------------
# 7. Contextual Page Inquiry Handling
# ---------------------------------------------------------------------------

def test_page_context_inquiry_handling():
    """Verify agent provides page-specific guidance when user asks 'jelaskan halaman ini'."""
    token = "token-owner"
    mock_owner = {"id": "own-1", "email": "bos@toko.com", "role": "pemilik_toko", "toko_id": "tok-1"}

    with patch("services.chat_service.verify_actor", return_value=mock_owner):
        res = handle_chat_message(
            token=token,
            message_text="Bisa jelaskan fungsi halaman ini?",
            page_context="Kelola Kategori"
        )
        assert "Kelola Kategori" in res["message"]
        assert "pos-pos anggaran" in res["message"]



# ---------------------------------------------------------------------------
# 8. Chat History Pagination Helper Tests
# ---------------------------------------------------------------------------

def test_visible_messages_pagination():
    """Verify _visible_messages returns newest slice or empty list as expected."""
    dummy_messages = [{"role": "user", "content": f"msg_{i}"} for i in range(25)]

    # visible_count=20 -> 20 pesan terakhir
    v20 = _visible_messages(dummy_messages, 20)
    assert len(v20) == 20
    assert v20 == dummy_messages[-20:]

    # visible_count=50 -> semua 25
    v50 = _visible_messages(dummy_messages, 50)
    assert len(v50) == 25
    assert v50 == dummy_messages

    # visible_count=0 -> daftar kosong
    v0 = _visible_messages(dummy_messages, 0)
    assert v0 == []
