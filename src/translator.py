from transformers import MarianMTModel, MarianTokenizer
import torch
import sys
import io
from src.config import TRANSLATION_MODEL_HI_EN, TRANSLATION_MODEL_EN_HI

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace'
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding='utf-8',
        errors='replace'
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# Load Hindi → English model
# =========================

print(f"Loading Hindi to English translation model: {TRANSLATION_MODEL_HI_EN}")

hi_en_tokenizer = MarianTokenizer.from_pretrained(
    TRANSLATION_MODEL_HI_EN
)

hi_en_model = MarianMTModel.from_pretrained(
    TRANSLATION_MODEL_HI_EN
).to(device)


# =========================
# Load English → Hindi model
# =========================

print(f"Loading English to Hindi translation model: {TRANSLATION_MODEL_EN_HI}")

en_hi_tokenizer = MarianTokenizer.from_pretrained(
    TRANSLATION_MODEL_EN_HI
)

en_hi_model = MarianMTModel.from_pretrained(
    TRANSLATION_MODEL_EN_HI
).to(device)


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.
    Returns 'hi' for Hindi, 'en' for English, or 'unknown'
    """

    if not text or len(text) == 0:
        return 'en'

    # Check for Hindi characters
    hindi_chars = 0

    for char in text:
        if '\u0900' <= char <= '\u097F':
            hindi_chars += 1

    # If more than 30% of characters are Hindi
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

        inputs = hi_en_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():

            translated = hi_en_model.generate(
                **inputs,
                max_length=512
            )

        translated_text = hi_en_tokenizer.decode(
            translated[0],
            skip_special_tokens=True
        )

        print("Translation completed")

        return translated_text

    except Exception as e:

        print(f"Translation error: {str(e)}")

        return text


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

    print("Translating from English to Hindi...")

    try:

        # Handle long texts by chunking
        if len(text) > 1000:

            chunks = [
                text[i:i + 1000]
                for i in range(0, len(text), 1000)
            ]

            translated_chunks = []

            for chunk in chunks:

                inputs = en_hi_tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)

                with torch.no_grad():

                    translated = en_hi_model.generate(
                        **inputs,
                        max_length=512
                    )

                translated_chunk = en_hi_tokenizer.decode(
                    translated[0],
                    skip_special_tokens=True
                )

                translated_chunks.append(translated_chunk)

            translated_text = ' '.join(translated_chunks)

        else:

            inputs = en_hi_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():

                translated = en_hi_model.generate(
                    **inputs,
                    max_length=512
                )

            translated_text = en_hi_tokenizer.decode(
                translated[0],
                skip_special_tokens=True
            )

        print("Translation completed")

        return translated_text

    except Exception as e:

        print(f"Translation error: {str(e)}")

        return text