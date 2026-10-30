import streamlit as st


def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .stMarkdown, .stText, p, label, input, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarHeader"],
    button[aria-label*="sidebar" i],
    button[aria-label*="Sidebar" i],
    button[data-testid="stBaseButton-header"],
    button[data-testid="stBaseButton-headerNoPadding"],
    header[data-testid="stHeader"] button,
    .stAppHeader button {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
    }

    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    button[aria-label*="password" i] span,
    button[aria-label*="Password" i] span,
    [data-testid="stTextInput"] button span {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        font-feature-settings: 'liga' 1 !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        -webkit-font-smoothing: antialiased !important;
    }

    .tabular-nums, [data-testid="stMetricValue"], [data-testid="stDataFrame"] {
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem !important;
    }

    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.5rem 0;
    }

    .stButton > button {
        border-radius: 6px !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }
    .stButton > button:active {
        background-color: #F1F5F9 !important;
    }

    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-color: #0F172A !important;
    }
    .stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1E293B !important;
        border-color: #1E293B !important;
        color: #FFFFFF !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
        margin-bottom: 0.25rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em;
    }

    [data-testid="stForm"] {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        background-color: #FFFFFF;
        padding: 1.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 6px !important;
        border: 1px solid #CBD5E1 !important;
        font-size: 0.875rem !important;
    }

    [data-testid="stAlert"] {
        border-radius: 6px !important;
        font-size: 0.875rem !important;
        padding: 0.75rem 1rem !important;
    }

    .status-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        line-height: 1.2;
    }
    .status-approved {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .status-pending {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .status-rejected {
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .status-neutral {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
    }

    .narrative-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #0F172A;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.9375rem;
        color: #334155;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


def format_rupiah(val):
    try:
        n = float(val)
        if n.is_integer():
            return f"Rp {int(n):,}".replace(',', '.')
        return f"Rp {n:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except Exception:
        return f"Rp {val}"


def page_header(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(
            f"<p style='color: #64748B; font-size: 0.875rem; margin-top: -0.35rem; margin-bottom: 1.25rem;'>{subtitle}</p>",
            unsafe_allow_html=True
        )


def section_header(title, description=None):
    st.markdown(f"#### {title}")
    if description:
        st.markdown(
            f"<p style='color: #64748B; font-size: 0.8125rem; margin-top: -0.35rem; margin-bottom: 1rem;'>{description}</p>",
            unsafe_allow_html=True
        )


def user_profile_sidebar(nama, role, shop_name=None):
    role_labels = {
        'pemilik_toko': 'Pemilik Toko',
        'kasir': 'Kasir',
        'superadmin': 'Superadmin'
    }
    role_name = role_labels.get(role, role)
    shop_line = f"<div style='font-size: 0.8125rem; color: #64748B; margin-top: 2px;'>{shop_name}</div>" if shop_name else ""

    st.sidebar.markdown(f"""
    <div style='border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.875rem 1rem; background-color: #FFFFFF; margin-bottom: 1.25rem;'>
        <div style='font-weight: 600; font-size: 0.9375rem; color: #0F172A;'>{nama}</div>
        <div style='font-size: 0.75rem; color: #475569; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;'>{role_name}</div>
        {shop_line}
    </div>
    """, unsafe_allow_html=True)
