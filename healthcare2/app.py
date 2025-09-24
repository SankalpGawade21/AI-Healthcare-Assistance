import streamlit as st
import pandas as pd
from langchain_ollama.llms import OllamaLLM as Ollama
import os
import json
import re
import speech_recognition as sr
import io
import warnings
from deep_translator import GoogleTranslator

# Suppress a specific warning from the audio library
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

# --- 1. SETUP AND INITIALIZATION ---

st.set_page_config(page_title="HealthCare ChatBot", layout="wide")

# --- PRE-TRANSLATED STATIC TEXTS ---
translations = {
    "en": {
        "title": "AI-Powered Medical Assistant", "write": "Before we begin, please provide some information about yourself in the sidebar.",
        "sidebar_your_info": "Your Information", "age": "Age", "gender": "Gender",
        "gender_options": ["Not specified", "Male", "Female", "Other"], "sidebar_filters": "Filter Options",
        "insurance_checkbox": "Show only doctors who accept insurance", "sidebar_find_doctor": "Find a Doctor",
        "location_select": "Select Location(s)", "find_doctors_button": "Find Doctors",
        "sidebar_voice_input": "Voice Input", "voice_language_select": "Select Language for Voice",
        "health_corner": "Health Corner", "health_tip": "💡 Health Tip:", "next_tip_button": "Next Tip",
        "emergency_button": "🚑 Emergency Contacts", "emergency_info": "Nearest Emergency Numbers:\n\n- 🚨 Ambulance: 108 \n- ☎ Police: 100",
        "download_chat_button": "💾 Download Chat History", "init_spinner": "Initializing the chatbot...",
        "init_toast": "Chatbot is ready!", "initial_message": "Hello! After entering your details, please describe your symptoms.",
        "symptoms_placeholder": "Tell me your symptoms...", "follow_up_placeholder": "Ask a follow-up question or enter new symptoms...",
        "analysis_spinner": "Analyzing your symptoms...", "thinking_spinner": "Thinking...",
        "follow_up_response": "You can now ask a follow-up question, or select a location from the sidebar and click *Find Doctors*.",
        "new_symptoms_info": "It looks like you're describing new symptoms. I will start a new analysis.",
        "new_symptoms_response": "You can now ask a follow-up question, or find a doctor using the sidebar.",
        "listening_info": "Listening in LANG... Speak now!", "transcribing_spinner": "Transcribing your voice...",
        "listening_timeout_warning": "Listening timed out. Please try again.", "whisper_error_toast": "Whisper could not understand the audio.",
        "voice_error": "An error occurred during transcription: ERROR_MESSAGE", "toast_symptoms_first": "Please describe your symptoms in the chat first.",
        "toast_location_first": "Please select at least one location from the sidebar.",
        "search_spinner": "Searching for SPECIALIZATIONs in LOCATIONS...",
        "search_results_ok": "Okay, here are some *SPECIALIZATIONs* in *LOCATIONS*:",
        "search_results_fail": "I'm sorry, I couldn't find any *SPECIALIZATIONs* in the selected locations matching your criteria. Please try different areas or filters.",
        "disclaimer": "Disclaimer: This is not a medical diagnosis.",
        "tip_1": "💧 Stay hydrated and drink at least 2L of water daily.", "tip_2": "😴 Get 7–8 hours of sleep for a healthy immune system.",
        "tip_3": "🏃‍♀ Exercise at least 30 minutes, 5 days a week.", "tip_4": "🧼 Wash hands frequently to prevent infections.",
        "tip_5": "🥗 Eat more fruits and vegetables to boost immunity.", "tip_6": "🚭 Avoid smoking and limit alcohol consumption.",
        "tip_7": "🧘‍♀ Practice stress management techniques like meditation.", "tip_8": "☀ Get adequate sunlight for vitamin D synthesis.",
        "tip_9": "🏥 Have regular health check-ups and screenings.", "tip_10": "💊 Take medications as prescribed by your doctor.",
        "doc_hospital": "Hospital/Clinic", "doc_qualifications": "Qualifications", "doc_availability": "Availability", "doc_contact": "Contact"
    },
    "hi": {
        "title": "हेल्थकेयर चैटबॉट 🧑🏽‍⚕", "subheader": "१. विवरण दर्ज करें। २. लक्षण बताएं। ३. प्रश्न पूछें या डॉक्टर खोजें।",
        "sidebar_your_info": "आपकी जानकारी", "age": "आयु", "gender": "लिंग", "gender_options": ["निर्दिष्ट नहीं", "पुरुष", "महिला", "अन्य"],
        "sidebar_filters": "फ़िल्टर विकल्प", "insurance_checkbox": "केवल बीमा स्वीकार करने वाले डॉक्टर दिखाएं",
        "sidebar_find_doctor": "डॉक्टर खोजें", "location_select": "स्थान चुनें", "find_doctors_button": "डॉक्टर खोजें",
        "sidebar_voice_input": "आवाज इनपुट", "voice_language_select": "आवाज के लिए भाषा चुनें", "health_corner": "स्वास्थ्य कॉर्नर",
        "health_tip": "💡 स्वास्थ्य सुझाव:", "next_tip_button": "अगला सुझाव", "emergency_button": "🚑 आपातकालीन संपर्क",
        "emergency_info": "निकटतम आपातकालीन नंबर:\n\n- 🚨 एम्बुलेंस: 108 \n- ☎ पुलिस: 100",
        "download_chat_button": "💾 चैट इतिहास डाउनलोड करें", "init_spinner": "चैटबॉट शुरू हो रहा है...", "init_toast": "चैटबॉट तैयार है!",
        "initial_message": "नमस्ते! अपनी जानकारी दर्ज करने के बाद, कृपया अपने लक्षण बताएं।", "symptoms_placeholder": "मुझे अपने लक्षण बताएं...",
        "follow_up_placeholder": "एक और प्रश्न पूछें या नए लक्षण दर्ज करें...", "analysis_spinner": "आपके लक्षणों का विश्लेषण हो रहा है...",
        "thinking_spinner": "सोच रहा हूँ...", "follow_up_response": "अब आप एक और प्रश्न पूछ सकते हैं, या साइडबार से एक स्थान चुनकर *डॉक्टर खोजें* पर क्लिक कर सकते हैं।",
        "new_symptoms_info": "लगता है आप नए लक्षण बता रहे हैं। मैं एक नया विश्लेषण शुरू करूंगा।", "new_symptoms_response": "अब आप एक और प्रश्न पूछ सकते हैं, या साइडबार का उपयोग करके डॉक्टर खोज सकते हैं।",
        "listening_info": "LANG में सुन रहा हूँ... अब बोलें!", "transcribing_spinner": "आपकी आवाज को ट्रांसक्राइब किया जा रहा है...",
        "listening_timeout_warning": "सुनने का समय समाप्त हो गया। कृपया फिर प्रयास करें।", "whisper_error_toast": "व्हिस्पर ऑडियो को समझ नहीं सका।",
        "voice_error": "ट्रांसक्रिप्शन के दौरान एक त्रुटि हुई: ERROR_MESSAGE", "toast_symptoms_first": "कृपया पहले चैट में अपने लक्षण बताएं।",
        "toast_location_first": "कृपया साइडबार से कम से कम एक स्थान चुनें।", "search_spinner": "LOCATIONS में SPECIALIZATIONs खोज रहा हूँ...",
        "search_results_ok": "ठीक है, यहाँ LOCATIONS में कुछ *SPECIALIZATIONs* हैं:", "search_results_fail": "माफ़ करें, मुझे आपके मानदंडों से मेल खाने वाले चयनित स्थानों में कोई *SPECIALIZATIONs* नहीं मिला। कृपया दूसरे क्षेत्र या फ़िल्टर आज़माएँ।",
        "disclaimer": "अस्वीकरण: यह एक चिकित्सा निदान नहीं है।", "tip_1": "💧 हाइड्रेटेड रहें और रोजाना कम से कम 2 लीटर पानी पिएं।",
        "tip_2": "😴 स्वस्थ प्रतिरक्षा प्रणाली के लिए 7-8 घंटे की नींद लें।", "tip_3": "🏃‍♀ सप्ताह में 5 दिन कम से कम 30 मिनट व्यायाम करें।",
        "tip_4": "🧼 संक्रमण से बचने के लिए बार-बार हाथ धोएं।", "tip_5": "🥗 रोग प्रतिरोधक क्षमता बढ़ाने के लिए अधिक फल और सब्जियां खाएं।",
        "tip_6": "🚭 धूम्रपान से बचें और शराब का सेवन सीमित करें।", "tip_7": "🧘‍♀ ध्यान जैसी तनाव प्रबंधन तकनीकों का अभ्यास करें।",
        "tip_8": "☀ विटामिन डी संश्लेषण के लिए पर्याप्त धूप लें।", "tip_9": "🏥 नियमित स्वास्थ्य जांच और स्क्रीनिंग कराएं।",
        "tip_10": "💊 अपने डॉक्टर द्वारा बताई गई दवाएं लें।", "doc_hospital": "अस्पताल/क्लिनिक", "doc_qualifications": "योग्यता",
        "doc_availability": "उपलब्धता", "doc_contact": "संपर्क"
    },
    "mr": {
        "title": "आरोग्य चॅटबॉट 🧑🏽‍⚕", "subheader": "१. तपशील प्रविष्ट करा. २. लक्षण सांगा. ३. प्रश्न विचारा किंवा डॉक्टर शोधा.",
        "sidebar_your_info": "तुमची माहिती", "age": "वय", "gender": "लिंग", "gender_options": ["निर्दिष्ट नाही", "पुरुष", "स्त्री", "इतर"],
        "sidebar_filters": "फिल्टर पर्याय", "insurance_checkbox": "फक्त विमा स्वीकारणारे डॉक्टर दाखवा", "sidebar_find_doctor": "डॉक्टर शोधा",
        "location_select": "स्थान निवडा", "find_doctors_button": "डॉक्टर शोधा", "sidebar_voice_input": "व्हॉइस इनपुट",
        "voice_language_select": "आवाजासाठी भाषा निवडा", "health_corner": "आरोग्य कोपरा", "health_tip": "💡 आरोग्य टीप:",
        "next_tip_button": "पुढील टीप", "emergency_button": "🚑 आपत्कालीन संपर्क",
        "emergency_info": "जवळचे आपत्कालीन क्रमांक:\n\n- 🚨 रुग्णवाहिका: १०८ \n- ☎ पोलीस: १००",
        "download_chat_button": "💾 चॅट इतिहास डाउनलोड करा", "init_spinner": "चॅटबॉट सुरू होत आहे...", "init_toast": "चॅटबॉट तयार आहे!",
        "initial_message": "नमस्कार! तुमचा तपशील प्रविष्ट केल्यानंतर, कृपया तुमची लक्षणे सांगा.", "symptoms_placeholder": "मला तुमची लक्षणे सांगा...",
        "follow_up_placeholder": "पुढील प्रश्न विचारा किंवा नवीन लक्षणे प्रविष्ट करा...", "analysis_spinner": "तुमच्या लक्षणांचे विश्लेषण करत आहे...",
        "thinking_spinner": "विचार करत आहे...", "follow_up_response": "तुम्ही आता पुढील प्रश्न विचारू शकता, किंवा साइडबारमधून स्थान निवडून *डॉक्टर शोधा* वर क्लिक करू शकता.",
        "new_symptoms_info": "असे दिसते की तुम्ही नवीन लक्षणे सांगत आहात. मी नवीन विश्लेषण सुरू करेन.",
        "new_symptoms_response": "तुम्ही आता पुढील प्रश्न विचारू शकता, किंवा साइडबार वापरून डॉक्टर शोधू शकता.",
        "listening_info": "LANG मध्ये ऐकत आहे... आता बोला!", "transcribing_spinner": "तुमचा आवाज लिहित आहे...",
        "listening_timeout_warning": "ऐकण्याची वेळ संपली. कृपया पुन्हा प्रयत्न करा.", "whisper_error_toast": "व्हिस्परला ऑडिओ समजू शकला नाही.",
        "voice_error": "लिप्यंतरण दरम्यान एक त्रुटी आली: ERROR_MESSAGE", "toast_symptoms_first": "कृपया प्रथम चॅटमध्ये तुमची लक्षणे सांगा.",
        "toast_location_first": "कृपया साइडबारमधून किमान एक स्थान निवडा.", "search_spinner": "LOCATIONS मध्ये SPECIALIZATIONs शोधत आहे...",
        "search_results_ok": "ठीक आहे, येथे LOCATIONS मधील काही *SPECIALIZATIONs* आहेत:",
        "search_results_fail": "माफ करा, मला तुमच्या निकषांशी जुळणारे निवडलेल्या ठिकाणी कोणतेही *SPECIALIZATIONs* सापडले नाहीत. कृपया भिन्न क्षेत्रे किंवा फिल्टर वापरून पहा.",
        "disclaimer": "अस्वीकरण: हे वैद्यकीय निदान नाही.", "tip_1": "💧 हायड्रेटेड रहा आणि दररोज किमान 2 लिटर पाणी प्या.",
        "tip_2": "😴 निरोगी रोगप्रतिकारशक्तीसाठी 7-8 तास झोप घ्या.", "tip_3": "🏃‍♀ आठवड्यातून 5 दिवस किमान 30 मिनिटे व्यायाम करा.",
        "tip_4": "🧼 संसर्ग टाळण्यासाठी वारंवार हात धुवा.", "tip_5": "🥗 रोगप्रतिकारशक्ती वाढवण्यासाठी अधिक फळे आणि भाज्या खा.",
        "tip_6": "🚭 धूम्रपान टाळा आणि मद्यपान मर्यादित करा.", "tip_7": "🧘‍♀ ध्यानासारख्या तणाव व्यवस्थापन तंत्रांचा सराव करा.",
        "tip_8": "☀ व्हिटॅमिन डी संश्लेषणासाठी पुरेशी सूर्यप्रकाश घ्या.", "tip_9": "🏥 नियमित आरोग्य तपासणी आणि स्क्रीनिंग करा.",
        "tip_10": "💊 तुमच्या डॉक्टरांनी सांगितल्याप्रमाणे औषधे घ्या.", "doc_hospital": "रुग्णालय/क्लिनिक", "doc_qualifications": "पात्रता",
        "doc_availability": "उपलब्धता", "doc_contact": "संपर्क"
    }
}


# --- SESSION STATE INITIALIZATION ---
def initialize_session_state():
    """Initializes session state for the conversation."""
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": translations[st.session_state.lang]['initial_message']}]
    if 'conversation_stage' not in st.session_state:
        st.session_state.conversation_stage = "awaiting_symptoms"
    if 'specialization' not in st.session_state:
        st.session_state.specialization = ""
    if 'tip_index' not in st.session_state:
        st.session_state.tip_index = 0

initialize_session_state()
t = translations[st.session_state.lang]

# --- UI Language Selector ---
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title(t['title'])
with col2:
    lang_map = {'en': 'English', 'hi': 'हिंदी (Hindi)', 'mr': 'मराठी (Marathi)'}
    current_lang_name = lang_map.get(st.session_state.lang, 'English')
    selected_lang_name = st.selectbox("Language", options=list(lang_map.values()), index=list(lang_map.values()).index(current_lang_name), label_visibility="collapsed")
    selected_lang_code = [code for code, name in lang_map.items() if name == selected_lang_name][0]

    if st.session_state.lang != selected_lang_code:
        st.session_state.lang = selected_lang_code
        st.session_state.messages = [] # Clear messages on lang change
        st.rerun()

st.subheader(t['write'])

# --- SIDEBAR FOR USER INFO & FILTERS ---
with st.sidebar:
    st.header(t['sidebar_your_info'])
    age = st.number_input(t['age'], min_value=0, max_value=120, step=1, key='user_age')
    gender = st.selectbox(t['gender'], t['gender_options'], key='user_gender')
    
    st.header(t['sidebar_filters'])
    accepts_insurance = st.checkbox(t['insurance_checkbox'], key='accepts_insurance')
    
    st.header(t['sidebar_find_doctor'])
    location_selector = st.empty()
    find_doctors_button = st.button(t['find_doctors_button'], key='find_doctors_button')
    
    st.header(t['sidebar_voice_input'])
    language = st.selectbox(t['voice_language_select'], ["english", "hindi", "marathi", "spanish", "french", "german"], key='voice_language')
    
    st.divider()

    st.header(t['health_corner'])
    health_tips = [t[f'tip_{i}'] for i in range(1, 11)]
    current_tip = health_tips[st.session_state.tip_index % len(health_tips)]
    st.info(f"{t['health_tip']} {current_tip}")
    if st.button(t['next_tip_button']):
        st.session_state.tip_index += 1
        st.rerun()
    
    if st.button(t['emergency_button']):
        st.sidebar.warning(t['emergency_info'])

    st.divider()
    chat_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
    st.download_button(t['download_chat_button'], chat_history, file_name="chat_history.txt")

# --- RESOURCE LOADING ---
@st.cache_resource
def get_llm():
    return Ollama(model="phi3:mini", base_url="http://localhost:11434")

@st.cache_data
def load_doctor_data():
    try:
        df = pd.read_csv('C:\\Users\\rguja\\OneDrive\\Documents\\LY\\healthChatBot\\doctors2_data.csv', encoding="ISO-8859-1")
        df['Location'] = df['Location'].str.strip().str.title()
        return df
    except FileNotFoundError:
        st.error("doctors2_data.csv file not found.")
        return None

if 'resources_loaded' not in st.session_state:
    with st.spinner(t['init_spinner']):
        st.session_state.llm = get_llm()
        st.session_state.doctors_df = load_doctor_data()
        st.session_state.resources_loaded = True
        st.toast(t['init_toast'])

llm = st.session_state.llm
doctors_df = st.session_state.doctors_df

if doctors_df is not None:
    locations = sorted(doctors_df['Location'].unique().tolist())
    location_selector.multiselect(t['location_select'], options=locations, key='selected_locations')

# --- CORE LOGIC ---
def analyze_symptoms_and_extract_specialist(symptoms, user_info):
    if doctors_df is not None and not doctors_df.empty:
        available_specializations = doctors_df['Specialization'].unique().tolist()
        user_context = f"A {user_info['age']}-year-old {user_info['gender']} user"
        if user_info['age'] == 0 or user_info['gender'] == 'Not specified':
            user_context = "A user"
        prompt = (
            f"{user_context} has the following symptoms: '{symptoms}'.\n\n"
            f"Your response MUST start with the single most relevant medical specialization from this list: {available_specializations}.\n\n"
            "After stating the specialization, provide a brief (under 120 words) explanation covering:\n"
            "- *Immediate Self-Care:* Bulleted list of actions.\n"
            "- *When to See a Doctor:* Bulleted list of red flags.\n"
            "- *General Advice:* Broader health advice."
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
            st.error(f"Could not connect to the language model: {e}")
            return {"full_response": "I encountered an error.", "specialization": "General Medicine"}
    return {"full_response": "Doctor data not loaded.", "specialization": "General Medicine"}

def find_doctors(specialization, locations, df, accepts_insurance=False):
    if df is not None and locations:
        title_case_locations = [loc.strip().title() for loc in locations]
        matches = df[(df['Specialization'].str.lower() == specialization.lower()) & (df['Location'].isin(title_case_locations))]
        if accepts_insurance:
            matches = matches[matches['Insurance'] == 1]
        return matches
    return pd.DataFrame()

def listen_and_transcribe(lang="english"):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            status_placeholder = st.empty()
            listening_text = t['listening_info'].replace("LANG", lang)
            status_placeholder.info(listening_text)
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            status_placeholder.empty()

        with st.spinner(t['transcribing_spinner']):
            text = recognizer.recognize_whisper(audio_data, language=lang, model="base")
        return text
    except sr.WaitTimeoutError:
        status_placeholder.warning(t['listening_timeout_warning'])
        return None
    except sr.UnknownValueError:
        st.toast(t['whisper_error_toast'], icon="🤔")
        return None
    except Exception as e:
        error_text = t['voice_error'].replace("ERROR_MESSAGE", str(e))
        st.error(error_text)
        return None

# --- UI AND CONVERSATION FLOW ---
def handle_prompt(prompt_text):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    prompt_for_model = prompt_text
    if st.session_state.lang != 'en':
        prompt_for_model = GoogleTranslator(source=st.session_state.lang, target='en').translate(prompt_text)

    if st.session_state.conversation_stage == "awaiting_symptoms":
        with st.spinner(t['analysis_spinner']):
            user_info = {"age": st.session_state.user_age, "gender": st.session_state.user_gender}
            analysis = analyze_symptoms_and_extract_specialist(prompt_for_model, user_info)
            full_response_text = analysis.get("full_response", "Based on your symptoms...")
            specialization = analysis.get("specialization", "General Medicine")
        st.session_state.specialization = specialization
        
        response = full_response_text
        if st.session_state.lang != 'en':
            response = GoogleTranslator(source='en', target=st.session_state.lang).translate(full_response_text)

        response = f"{response}\n\n{t['follow_up_response']}"
        st.session_state.conversation_stage = "awaiting_doctor_search"
    
    elif st.session_state.conversation_stage == "awaiting_doctor_search":
        with st.spinner(t['thinking_spinner']):
            if len(prompt_for_model.split()) > 5:
                 st.info(t['new_symptoms_info'])
                 user_info = {"age": st.session_state.user_age, "gender": st.session_state.user_gender}
                 analysis = analyze_symptoms_and_extract_specialist(prompt_for_model, user_info)
                 full_response_text = analysis.get("full_response", "Based on your symptoms...")
                 specialization = analysis.get("specialization", "General Medicine")
                 st.session_state.specialization = specialization
                 
                 response = full_response_text
                 if st.session_state.lang != 'en':
                     response = GoogleTranslator(source='en', target=st.session_state.lang).translate(full_response_text)

                 response = f"{response}\n\n{t['new_symptoms_response']}"
            else:
                follow_up_prompt = f"The user has a follow-up question: '{prompt_for_model}'. Provide a concise and helpful answer in under 80 words."
                response = llm.invoke(follow_up_prompt)
                if st.session_state.lang != 'en':
                    response = GoogleTranslator(source='en', target=st.session_state.lang).translate(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Display chat messages from history
if not st.session_state.messages:
     st.session_state.messages = [{"role": "assistant", "content": t['initial_message']}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_input_placeholder = t['symptoms_placeholder']
if st.session_state.conversation_stage == "awaiting_doctor_search":
    chat_input_placeholder = t['follow_up_placeholder']

col1, col2 = st.columns([8, 1])
with col1:
    if prompt := st.chat_input(chat_input_placeholder, key="chat_box"):
        handle_prompt(prompt)
with col2:
    st.write("")
    st.write("")
    if st.button("🎤", key="voice_button"):
        transcribed_text = listen_and_transcribe(lang=st.session_state.voice_language)
        if transcribed_text:
            handle_prompt(transcribed_text)

if find_doctors_button:
    specialization = st.session_state.get('specialization')
    locations = st.session_state.get('selected_locations')
    insurance_filter = st.session_state.get('accepts_insurance', False)

    if not specialization:
        st.toast(t['toast_symptoms_first'], icon="ℹ")
    elif not locations:
        st.toast(t['toast_location_first'], icon="📍")
    else:
        display_spec = specialization
        display_locs = ', '.join(locations)
        translator = GoogleTranslator(source='en', target=st.session_state.lang)
        if st.session_state.lang != 'en':
            display_spec = translator.translate(specialization)
            display_locs = translator.translate(', '.join(locations))

        spinner_text = t['search_spinner'].replace("SPECIALIZATION", display_spec).replace("LOCATIONS", display_locs)
        with st.spinner(spinner_text):
            recommended_doctors = find_doctors(specialization, locations, doctors_df, accepts_insurance=insurance_filter)
        
        if not recommended_doctors.empty:
            contact_col = 'Contact_no' if 'Contact_no' in recommended_doctors.columns else 'Contact'
            response = t['search_results_ok'].replace("SPECIALIZATION", display_spec).replace("LOCATIONS", display_locs) + "\n\n"
            
            for index, row in recommended_doctors.iterrows():
                name = row['Name']
                hospital = row['Hospital/Clinic']
                quals = row['Qualifications']
                if st.session_state.lang != 'en':
                    translated_details = [translator.translate(item) for item in [name, hospital, quals] if pd.notna(item)]
                    name, hospital, quals = translated_details[0], translated_details[1], translated_details[2]

                response += (
                    f"- *{name}*\n"
                    f"  - *{t['doc_hospital']}:* {hospital},{row['Location']} \n"
                    f"  - *{t['doc_qualifications']}:* {quals}\n"
                    f"  - *{t['doc_availability']}:* {row['Availability']}\n"
                    f"  - *{t['doc_contact']}:* {row[contact_col]}\n\n"
                )
            response += f"\n{t['disclaimer']}"
        else:
            response = t['search_results_fail'].replace("SPECIALIZATION", display_spec)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.conversation_stage = "awaiting_symptoms"
        st.session_state.specialization = ""
        st.rerun()
