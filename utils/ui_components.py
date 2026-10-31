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

    /* Top header and decorations cleanup */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 99999 !important;
    }
    [data-testid="stDeployButton"],
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }

    /* Sidebar Header container */
    [data-testid="stSidebarHeader"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        min-height: 2.75rem !important;
        padding: 0.5rem 0.75rem 0.25rem 0.75rem !important;
        align-items: center !important;
        justify-content: flex-end !important;
        background-color: transparent !important;
    }

    /* Sidebar Collapse Button (Hide side) */
    [data-testid="stSidebarCollapseButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        align-items: center !important;
        justify-content: center !important;
        width: 2rem !important;
        height: 2rem !important;
        min-width: 2rem !important;
        min-height: 2rem !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 6px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        cursor: pointer !important;
        position: relative !important;
        color: transparent !important;
        overflow: hidden !important;
        transition: all 0.15s ease-in-out !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
    }

    /* Suppress unrendered raw font ligature text inside collapse button */
    [data-testid="stSidebarCollapseButton"] button span,
    [data-testid="stSidebarCollapseButton"] button p,
    [data-testid="stSidebarCollapseButton"] button svg {
        display: none !important;
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
        visibility: hidden !important;
    }

    /* Render crisp SVG chevron-left for collapse button */
    [data-testid="stSidebarCollapseButton"] button::after {
        content: "" !important;
        display: block !important;
        width: 16px !important;
        height: 16px !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: contain !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23475569' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='15 18 9 12 15 6'%3E%3C/polyline%3E%3C/svg%3E") !important;
    }

    /* Sidebar Expand Button (Open side when collapsed) */
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] button,
    button[aria-label="Expand sidebar" i] {
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        align-items: center !important;
        justify-content: center !important;
        width: 2.25rem !important;
        height: 2.25rem !important;
        min-width: 2.25rem !important;
        min-height: 2.25rem !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 6px !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08) !important;
        cursor: pointer !important;
        position: relative !important;
        color: transparent !important;
        overflow: hidden !important;
        transition: all 0.15s ease-in-out !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="collapsedControl"] button:hover,
    button[aria-label="Expand sidebar" i]:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
    }

    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 0.75rem !important;
        left: 0.75rem !important;
        z-index: 999999 !important;
    }

    /* Suppress unrendered raw font ligature text inside expand button */
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stExpandSidebarButton"] p,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="collapsedControl"] span,
    [data-testid="collapsedControl"] p,
    [data-testid="collapsedControl"] svg,
    button[aria-label="Expand sidebar" i] span,
    button[aria-label="Expand sidebar" i] p,
    button[aria-label="Expand sidebar" i] svg {
        display: none !important;
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
        visibility: hidden !important;
    }

    /* Render crisp SVG chevron-right for expand button */
    [data-testid="stExpandSidebarButton"]::after,
    [data-testid="collapsedControl"] button::after,
    button[aria-label="Expand sidebar" i]::after {
        content: "" !important;
        display: block !important;
        width: 16px !important;
        height: 16px !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: contain !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23475569' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='9 18 15 12 9 6'%3E%3C/polyline%3E%3C/svg%3E") !important;
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

    #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
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
