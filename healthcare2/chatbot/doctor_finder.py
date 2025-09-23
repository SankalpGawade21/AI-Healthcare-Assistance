import pandas as pd
import streamlit as st

@st.cache_data
def load_doctor_data():
    try:
        df = pd.read_csv("data/doctors2_data.csv", encoding="ISO-8859-1")
        df['Location'] = df['Location'].str.strip().str.title()
        return df
    except FileNotFoundError:
        st.error("doctors2_data.csv not found in /data folder")
        return None

def find_doctors(specialization, location, df):
    if df is not None:
        location = location.strip().title()
        return df[(df['Specialization'].str.lower() == specialization.lower()) & (df['Location'] == location)]
    return pd.DataFrame()
