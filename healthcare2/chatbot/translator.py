from deep_translator import GoogleTranslator

def translate_text(text, source='auto', target='en'):
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        return text  # fallback if translation fails
