import streamlit as st
import pandas as pd
from chatbot.session_manager import initialize_session_state
from chatbot.llm_handler import analyze_symptoms_and_extract_specialist
from chatbot.translator import translate_text
from chatbot.doctor_finder import load_doctor_data, find_doctors

# --- Streamlit Setup ---
st.set_page_config(page_title="HealthCare ChatBot", layout="wide")
st.title("HealthCare ChatBot 🧑🏽‍⚕")
st.subheader("Please provide your details in the sidebar and describe your symptoms below.")

# --- Sidebar ---
st.sidebar.header("Your Information")
st.session_state.user_age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1, key='user_age')
st.session_state.user_gender = st.sidebar.selectbox("Gender", ["Not specified", "Male", "Female", "Other"], key='user_gender')

# --- Load doctor data ---
doctors_df = load_doctor_data()

# --- Initialize session ---
initialize_session_state()

# --- Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if prompt := st.chat_input("Tell me your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.conversation_stage == "start":
            with st.spinner("Analyzing your symptoms..."):
                user_info = {"age": st.session_state.user_age, "gender": st.session_state.user_gender}
                analysis = analyze_symptoms_and_extract_specialist(prompt, user_info, doctors_df)
            
            specialization = analysis.get("specialization", "General Medicine")
            st.session_state.specialization = specialization
            response = (f"{analysis['full_response']}\n\n"
                        f"To find a {specialization} near you, could you please tell me your location (e.g., Deccan, Kothrud, Aundh)?")

            st.session_state.conversation_stage = "awaiting_location"
            st.markdown(response)

        elif st.session_state.conversation_stage == "awaiting_location":
            location = prompt
            specialization = st.session_state.specialization
            
            with st.spinner(f"Searching for {specialization}s in {location}..."):
                recommended_doctors = find_doctors(specialization, location, doctors_df)

            if not recommended_doctors.empty:
                response = f"Here are some {specialization}s in {location}:\n\n"
                for _, row in recommended_doctors.iterrows():
                    response += (
                        f"- {row['Name']}\n"
                        f"  - Hospital/Clinic: {row['Hospital/Clinic']}\n"
                        f"  - Qualifications: {row['Qualifications']}\n"
                        f"  - Availability: {row['Availability']}\n"
                        f"  - Contact: {row['Contact_no']}\n\n"
                    )
                response += "\n*Disclaimer: This is not a medical diagnosis. Please consult a professional for medical advice.*"
                st.session_state.conversation_stage = "start"
                st.session_state.specialization = ""
            else:
                response = f"Sorry, I couldn't find any {specialization}s in {location}. Try another area?"

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
