# Purpose: Render the sidebar in Streamlit
# Inputs: None
# Outputs: Streamlit sidebar elements
# Dependencies: None

import streamlit as st


def render():
    st.sidebar.title("Navigation")
    st.sidebar.page_link("Home", icon="🏠")
    st.sidebar.page_link("Chat", icon="💬")
    st.sidebar.page_link("Data Explorer", icon="🔍")
