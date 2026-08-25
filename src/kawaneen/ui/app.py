"""Streamlit application shell; page behavior lives in ``kawaneen.ui.pages``."""

from __future__ import annotations

import streamlit as st


def _search_page() -> None:
    from kawaneen.ui.pages.search import render

    render()


def _ask_page() -> None:
    from kawaneen.ui.pages.ask import render

    render()


def _extract_page() -> None:
    from kawaneen.ui.pages.extract import render

    render()


def _evaluation_page() -> None:
    from kawaneen.ui.pages.evaluation import render

    render()


def main() -> None:
    st.set_page_config(page_title="Kawaneen | قوانين", page_icon="⚖", layout="wide")
    pages = [
        st.Page(_search_page, title="Search", icon="⌕", url_path="search"),
        st.Page(_ask_page, title="Ask", icon="↗", url_path="ask"),
        st.Page(_extract_page, title="Extract", icon="⊞", url_path="extract"),
        st.Page(_evaluation_page, title="Evaluation", icon="◒", url_path="evaluation"),
    ]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    main()
