"""Streamlit application shell; page behavior lives in ``kawaneen.ui.pages``."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Kawaneen | قوانين", page_icon="⚖", layout="wide")
    pages = [
        st.Page("pages/search.py", title="Search", icon="🔎", url_path="search"),
        st.Page("pages/ask.py", title="Ask", icon="💬", url_path="ask"),
        st.Page("pages/extract.py", title="Extract", icon="📄", url_path="extract"),
        st.Page("pages/evaluation.py", title="Evaluation", icon="📊", url_path="evaluation"),
    ]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    main()
