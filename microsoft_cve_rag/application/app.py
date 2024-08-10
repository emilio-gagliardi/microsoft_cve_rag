# Purpose: Entry point for the Streamlit application
# Inputs: None
# Outputs: Streamlit web application
# Dependencies: Streamlit pages and components

import streamlit as st
from application.frontend.streamlit.pages import home, chat, data_explorer
from application.frontend.streamlit.components import sidebar


def main():
    st.set_page_config(page_title="AI Knowledge Graph", layout="wide")
    sidebar.render()

    page = st.sidebar.selectbox("Select a page", ["Home", "Chat", "Data Explorer"])

    if page == "Home":
        home.render()
    elif page == "Chat":
        chat.render()
    elif page == "Data Explorer":
        data_explorer.render()


if __name__ == "__main__":
    main()
