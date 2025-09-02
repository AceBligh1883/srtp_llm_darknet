# src/processing/translator.py
import translators as ts

def translate_to_english(query_text: str) -> str:
    try:
        query_text.encode('ascii')
        return query_text
    except UnicodeEncodeError:
        pass

    try:
        translated_text = ts.translate_text(query_text, translator='google', to_language='en')
        return translated_text
    except Exception as e:
        return query_text

