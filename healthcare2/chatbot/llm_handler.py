import re
import streamlit as st
from langchain_ollama.llms import OllamaLLM as Ollama

@st.cache_resource
def get_llm():
    return Ollama(model="phi3:mini", base_url="http://localhost:11434")

def analyze_symptoms_and_extract_specialist(symptoms, user_info, doctors_df):
    llm = get_llm()
    available_specializations = doctors_df['Specialization'].unique().tolist() if doctors_df is not None else []

    user_context = f"A {user_info['age']}-year-old {user_info['gender']} user"
    if user_info['age'] == 0 or user_info['gender'] == 'Not specified':
        user_context = "A user"

    prompt = (
        f"{user_context} has the following symptoms: '{symptoms}'.\n\n"
        f"Your response MUST start with the single most relevant medical specialization from this list: {available_specializations}.\n\n"
        f"Example: ENT\n"
        f"Follow with short bullet points about precautions (<100 words)."
    )

    try:
        conversational_response = llm.invoke(prompt)
        found_specialization = "General Medicine"
        for spec in available_specializations:
            if re.search(r'\b' + re.escape(spec) + r'\b', conversational_response, re.IGNORECASE):
                found_specialization = spec
                break

        return {"full_response": conversational_response, "specialization": found_specialization}
    except Exception as e:
        st.error(f"LLM error: {e}")
        return {"full_response": "Error analyzing your symptoms.", "specialization": "General Medicine"}
