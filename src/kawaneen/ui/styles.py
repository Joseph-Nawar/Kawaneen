"""Static visual system for the Kawaneen workspace."""

from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.html(
        """
        <style>
        :root {
          --kw-ink:#15263d;
          --kw-teal:#0f766e;
          --kw-gold:#b8893a;
          --kw-canvas:#f5f1eb;
          --kw-surface:#ffffff;
          --kw-line:#dfe5e3;
          --kw-muted:#627183;
          --kw-warning-ink:#76551d;
          --kw-warning-line:#d8b675;
          --kw-warning-surface:#fbf6ea;
          --kw-danger:#8d3f3f;
        }
        html, body, [class*="css"] { font-family:-apple-system, BlinkMacSystemFont,
          "Segoe UI", "Noto Sans Arabic", Arial, sans-serif; }
        .stApp { background:var(--kw-canvas); color:var(--kw-ink); }
        .block-container { max-width:1440px; padding:1.25rem clamp(1.5rem, 4vw, 3.5rem) 4rem; }
        [data-testid="stHeader"] { background:var(--kw-canvas); box-shadow:none; }
        [data-testid="stToolbar"] { visibility:hidden; height:0; }
        [data-testid="stNavigation"] { border-bottom:1px solid var(--kw-line); }
        [data-testid="stTopNavLinkContainer"], [data-testid="stTopNavLink"] {
          visibility:visible !important;
        }
        [data-testid="stPageLink-NavLink"] { color:var(--kw-ink); }
        [data-testid="stPageLink-NavLink"][aria-current="page"] {
          border-bottom:2px solid var(--kw-teal); color:var(--kw-teal); font-weight:700;
        }
        .block-container h1 { color:var(--kw-ink); font-size:2.35rem !important;
          letter-spacing:-.025em; line-height:1.15; margin:.65rem 0 .25rem !important; }
        .block-container h3 { color:var(--kw-ink); letter-spacing:-.012em; margin-top:1.25rem; }
        [data-testid="stForm"] { border:1px solid #cbd5d3; border-radius:8px;
          padding:1rem 1rem .85rem !important; }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
          border-color:var(--kw-line); border-radius:6px;
        }
        .stButton > button, [data-testid="stFormSubmitButton"] button,
        [data-testid="stDownloadButton"] button { border:1px solid #b9c7c5; border-radius:6px;
          color:var(--kw-ink); background:var(--kw-surface); min-height:2.35rem; }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {
          border-color:var(--kw-teal); color:var(--kw-teal);
        }
        .stButton > button:focus-visible, [data-testid="stFormSubmitButton"] button:focus-visible,
        [data-testid="stDownloadButton"] button:focus-visible,
        input:focus-visible, textarea:focus-visible { outline:2px solid var(--kw-teal);
          outline-offset:2px; }
        [data-testid="stMetric"] { background:var(--kw-surface); border:1px solid var(--kw-line);
          border-radius:6px; padding:.65rem .8rem; }
        .kw-product-header { align-items:flex-start; border-bottom:1px solid var(--kw-line);
          display:flex; justify-content:space-between; margin:0 0 .8rem; padding:36px 0 12px; }
        .kw-brand { color:var(--kw-ink); font-size:1.25rem; font-weight:800; letter-spacing:.02em; }
        .kw-subtitle { color:var(--kw-muted); font-size:.82rem; margin-top:.2rem; }
        .kw-status { border:1px solid var(--kw-line); border-radius:4px; color:#46616a;
          display:inline-block; font-size:.74rem; padding:.25rem .55rem; }
        .kw-status.demo { background:var(--kw-warning-surface); border-color:var(--kw-warning-line);
          color:var(--kw-warning-ink); }
        .kw-status.degraded { background:#f8eeee; border-color:#d9abab; color:var(--kw-danger); }
        .kw-public-banner { align-items:baseline; background:var(--kw-warning-surface);
          border:1px solid var(--kw-warning-line); border-left:3px solid var(--kw-gold);
          border-radius:6px; color:var(--kw-warning-ink); display:flex; flex-wrap:wrap;
          gap:.35rem .8rem; line-height:1.5; margin:.25rem 0 1rem; padding:.6rem .8rem; }
        .kw-public-label { font-size:.73rem; font-weight:800; letter-spacing:.08em; }
        .kw-public-facts { font-size:.79rem; }
        .kw-mode-note { border-bottom:1px solid var(--kw-line); color:var(--kw-muted);
          font-size:.78rem; margin:-.35rem 0 .85rem; padding:0 0 .65rem; }
        .kw-eyebrow { color:var(--kw-teal); font-size:.7rem; font-weight:800;
          letter-spacing:.14em; text-transform:uppercase; }
        .kw-page-intro { margin:0 0 1rem; }
        .kw-page-intro h1 { color:var(--kw-ink); font-size:2.35rem; letter-spacing:-.025em;
          line-height:1.15; margin:.35rem 0 .4rem; }
        .kw-page-description { color:var(--kw-muted); font-size:.9rem; margin:0; }
        .kw-surface { background:var(--kw-surface); border:1px solid var(--kw-line);
          border-radius:7px; padding:1rem 1.1rem; }
        .kw-answer-pane { min-height:7rem; }
        .kw-evidence-card { background:var(--kw-surface); border:1px solid var(--kw-line);
          border-radius:7px; margin:.65rem 0; padding:1rem 1.1rem; }
        .kw-evidence-heading {
          align-items:baseline; display:flex; gap:.65rem; justify-content:space-between;
        }
        .kw-rank { color:var(--kw-teal); font-variant-numeric:tabular-nums; }
        .kw-evidence-text { line-height:2; margin-top:.7rem; }
        .kw-evidence-footer { border-top:1px solid #edf0ef; margin-top:.8rem; padding-top:.55rem; }
        .kw-quote { background:#fcfaf6; border-left:3px solid var(--kw-gold);
          border-radius:0 6px 6px 0; line-height:1.9; padding:.85rem 1rem; }
        .kw-citation-index { color:var(--kw-teal); font-size:.75rem; font-weight:800; }
        .kw-abstention {
          background:var(--kw-warning-surface); border:1px solid var(--kw-warning-line);
          border-left:3px solid var(--kw-gold); border-radius:6px; padding:1rem 1.1rem; }
        .kw-findings-summary {
          border-bottom:1px solid var(--kw-line); border-top:1px solid var(--kw-line);
          display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); margin:.7rem 0 1rem; }
        .kw-finding-count {
          align-items:baseline; border-left:1px solid var(--kw-line); display:flex;
          gap:.35rem; justify-content:space-between; padding:.6rem .7rem;
        }
        .kw-finding-count:first-child { border-left:0; }
        .kw-finding-label { color:var(--kw-muted); font-size:.74rem; }
        .kw-finding-number {
          color:var(--kw-ink); font-size:1.2rem; font-variant-numeric:tabular-nums;
          font-weight:700; }
        .kw-finding-record {
          border-left:3px solid var(--kw-gold); border-bottom:1px solid var(--kw-line);
          margin:.65rem 0; padding:.25rem 0 .8rem .9rem; }
        .kw-finding-type {
          color:var(--kw-teal); font-size:.72rem; font-weight:800; letter-spacing:.1em;
          text-transform:uppercase; }
        .kw-finding-value { font-size:1rem; line-height:1.8; margin-top:.25rem; }
        .kw-status-strip {
          border-bottom:1px solid var(--kw-line); border-top:1px solid var(--kw-line);
          display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); margin:.7rem 0 1.4rem; }
        .kw-status-item { border-left:1px solid var(--kw-line); padding:.7rem .75rem; }
        .kw-status-item:first-child { border-left:0; }
        .kw-status-key { color:var(--kw-muted); font-size:.72rem; }
        .kw-status-value {
          color:var(--kw-ink); font-size:.9rem; font-weight:700; margin-top:.18rem; }
        .kw-rtl { direction:rtl; text-align:right; }
        .kw-ltr { direction:ltr; text-align:left; }
        .kw-meta { color:var(--kw-muted); font-size:.78rem; }
        .query-hit { background:#f5e7bb; border-radius:2px; color:inherit; padding:0 .1em; }
        @media (max-width: 800px) {
          .kw-findings-summary, .kw-status-strip {
            grid-template-columns:repeat(3, minmax(0, 1fr));
          }
          .kw-finding-count:nth-child(4), .kw-status-item:nth-child(4) { border-left:0; }
          .kw-product-header { gap:1rem; }
        }
        </style>
        """
    )
