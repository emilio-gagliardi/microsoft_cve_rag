# Purpose: Render the chat interface in Streamlit
# Inputs: User messages
# Outputs: AI responses
# Dependencies: ChatServiceimport streamlit as st

import streamlit as st
from application.services.chat_service import ChatService


def render():
    st.title("AI Chat")

    chat_service = ChatService()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        response = chat_service.get_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)
