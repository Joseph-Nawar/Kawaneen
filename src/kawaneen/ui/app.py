"""Streamlit application shell; page behavior lives in ``kawaneen.ui.pages``."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Kawaneen | قوانين", layout="wide")
    pages = [
        st.Page("pages/search.py", title="Search", url_path="search"),
        st.Page("pages/ask.py", title="Ask", url_path="ask"),
        st.Page("pages/extract.py", title="Extract", url_path="extract"),
        st.Page("pages/evaluation.py", title="Evaluation", url_path="evaluation"),
    ]
    navigation = st.navigation(pages, position="hidden")
    navigation.run()


if __name__ == "__main__":
    main()
