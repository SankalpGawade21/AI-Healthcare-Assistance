import streamlit as st

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! After entering your details, please describe your symptoms."}
        ]
    if 'conversation_stage' not in st.session_state:
        st.session_state.conversation_stage = "start"
    if 'specialization' not in st.session_state:
        st.session_state.specialization = ""
