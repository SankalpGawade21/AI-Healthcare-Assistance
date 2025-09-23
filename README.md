# AI-Healthcare-Assistance
An AI-powered healthcare assistant built with Streamlit, LangChain, and Ollama that analyzes symptoms, recommends specialists, and helps patients find nearby doctors.


This project is an AI-driven healthcare assistant designed to make symptom checking and doctor discovery easier for patients.
It combines Large Language Models (LLMs) with Streamlit’s interactive UI to provide:

🩺 Symptom Analysis → User inputs symptoms (text or voice), which are analyzed by an LLM (phi3:mini).

🌍 Multilingual Support → Inputs in any language are translated into English for analysis, and responses are translated back to the user’s language.

🧠 AI Recommendations → The chatbot suggests the most relevant medical specialist based on symptoms.

👨‍⚕️ Doctor Finder → Using a CSV database, it filters doctors by specialization, location, and insurance preference.

🎤 Voice-to-Text Support → Users can speak their symptoms via Whisper AI.

💬 Interactive Chat → Patients can ask follow-up questions in natural language.

This project demonstrates end-to-end AI integration with healthcare applications, from speech processing → LLM-based reasoning → real doctor recommendations.
