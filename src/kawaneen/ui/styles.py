"""Static visual system for the Kawaneen workspace."""

from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.html(
        """
        <style>
        :root { --navy:#15263d; --teal:#0f766e; --gold:#b8893a; --canvas:#f5f1eb; --line:#dfe5e3; }
        .stApp { background: var(--canvas); color: var(--navy); }
        .block-container { max-width: 1440px; padding: 1.5rem 3rem 4rem; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line);
          border-radius:10px; padding:.8rem 1rem; }
        .kw-surface { background:#fff; border:1px solid var(--line);
          border-radius:10px; padding:1.1rem 1.25rem; }
        .kw-eyebrow { color:var(--teal); font-size:.72rem; font-weight:700;
          letter-spacing:.12em; text-transform:uppercase; }
        .kw-brand { color:var(--navy); font-size:1.35rem; font-weight:800; letter-spacing:.02em; }
        .kw-subtitle { color:#5e6b78; font-size:.85rem; margin-top:.18rem; }
        .kw-status { border:1px solid var(--line); border-radius:999px; color:#46616a;
          display:inline-block; font-size:.75rem; padding:.25rem .65rem; }
        .kw-status.demo { color:#8a6221; border-color:#ddc28c; background:#fffaf0; }
        .kw-status.degraded { color:#9a5d13; border-color:#e4bf87; background:#fff7e8; }
        .kw-quote { border-left:3px solid var(--gold); background:#fcfaf6;
          border-radius:0 8px 8px 0; padding:.8rem 1rem; line-height:1.8; }
        [data-testid="stAlert"] { margin:.65rem 0; padding:.65rem .9rem; }
        .kw-rtl { direction:rtl; text-align:right; }
        .kw-ltr { direction:ltr; text-align:left; }
        .kw-meta { color:#65727c; font-size:.78rem; }
        .query-hit { background:#f5e7bb; border-radius:3px; color:inherit; padding:0 .1em; }
        </style>
        """
    )
