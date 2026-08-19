from transformers import pipeline
import torch
import sys
import io
from src.config import TRANSLATION_MODEL_HI_EN, TRANSLATION_MODEL_EN_HI

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load translation pipeline
# Using Helsinki-NLP models for English-Hindi translation
print(f"Loading Hindi to English translation model: {TRANSLATION_MODEL_HI_EN}")
hi_to_en_pipeline = pipeline(
    "translation_hi_to_en",
    model=TRANSLATION_MODEL_HI_EN,
    device=device
)

print(f"Loading English to Hindi translation model: {TRANSLATION_MODEL_EN_HI}")
en_to_hi_pipeline = pipeline(
    "translation_en_to_hi",
    model=TRANSLATION_MODEL_EN_HI,
    device=device
)


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.
    Returns 'hi' for Hindi, 'en' for English, or 'unknown'
    """
    if not text or len(text) == 0:
        return 'en'  # Default to English for empty text
    
    # Check for Hindi characters (Unicode range for Devanagari)
    hindi_chars = 0
    for char in text:
        if '\u0900' <= char <= '\u097F':  # Devanagari Unicode range
            hindi_chars += 1
    
    # If more than 30% of characters are Hindi, classify as Hindi
    if hindi_chars / len(text) > 0.3:
        return 'hi'
    elif any(char.isalpha() for char in text):
        return 'en'
    else:
        return 'unknown'


def translate_to_english(text: str) -> str:
    """
    Translate Hindi text to English.
    If text is already in English, return as-is.
    """
    if not text or not text.strip():
        return text
    
    lang = detect_language(text)
    if lang == 'en':
        print("Text is already in English, skipping translation")
        return text
    
    print("Translating from Hindi to English...")
    try:
        result = hi_to_en_pipeline(text, max_length=512)
        translated_text = result[0]['translation_text']
        print("Translation completed")
        return translated_text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original if translation fails


def translate_to_hindi(text: str) -> str:
    """
    Translate English text to Hindi.
    If text is already in Hindi, return as-is.
    """
    if not text or not text.strip():
        return text
    
    lang = detect_language(text)
    if lang == 'hi':
        print("Text is already in Hindi, skipping translation")
        return text
    
    print(f"Translating from English to Hindi...")
    try:
        # Handle long texts by chunking
        if len(text) > 1000:
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            translated_chunks = []
            for chunk in chunks:
                result = en_to_hi_pipeline(chunk, max_length=512)
                translated_chunks.append(result[0]['translation_text'])
            translated_text = ' '.join(translated_chunks)
        else:
            result = en_to_hi_pipeline(text, max_length=512)
            translated_text = result[0]['translation_text']
        
        print("Translation completed")
        return translated_text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original if translation fails
