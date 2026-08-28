from transformers import MarianMTModel, MarianTokenizer
import torch
import sys
import io
import re

from src.config import (
    TRANSLATION_MODEL_HI_EN,
    TRANSLATION_MODEL_EN_HI
)


# ============================================================
# UTF-8 encoding for Windows console
# ============================================================

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace"
    )


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ============================================================
# Global translation models
# ============================================================

hi_en_tokenizer = None
hi_en_model = None

en_hi_tokenizer = None
en_hi_model = None


# ============================================================
# Load translation models
# ============================================================

def load_translation_models():
    """Load Hindi-English translation models on first use."""

    global hi_en_tokenizer
    global hi_en_model
    global en_hi_tokenizer
    global en_hi_model

    # Hindi → English
    if hi_en_tokenizer is None or hi_en_model is None:

        print(
            f"Loading Hindi to English translation model: "
            f"{TRANSLATION_MODEL_HI_EN}"
        )

        hi_en_tokenizer = MarianTokenizer.from_pretrained(
            TRANSLATION_MODEL_HI_EN
        )

        hi_en_model = MarianMTModel.from_pretrained(
            TRANSLATION_MODEL_HI_EN
        ).to(device)

        hi_en_model.eval()

        print("Hindi to English model loaded")

    # English → Hindi
    if en_hi_tokenizer is None or en_hi_model is None:

        print(
            f"Loading English to Hindi translation model: "
            f"{TRANSLATION_MODEL_EN_HI}"
        )

        en_hi_tokenizer = MarianTokenizer.from_pretrained(
            TRANSLATION_MODEL_EN_HI
        )

        en_hi_model = MarianMTModel.from_pretrained(
            TRANSLATION_MODEL_EN_HI
        ).to(device)

        en_hi_model.eval()

        print("English to Hindi model loaded")

    return (
        hi_en_tokenizer,
        hi_en_model,
        en_hi_tokenizer,
        en_hi_model
    )


# ============================================================
# Language detection
# ============================================================

def detect_language(text: str) -> str:
    """
    Detect whether text is primarily Hindi or English.

    Returns:
        'hi'      -> Hindi
        'en'      -> English
        'unknown' -> Unable to determine
    """

    if not text or not text.strip():
        return "en"

    text = text.strip()

    hindi_chars = 0
    alphabetic_chars = 0

    for char in text:

        # Devanagari Unicode range
        if "\u0900" <= char <= "\u097F":
            hindi_chars += 1

        if char.isalpha():
            alphabetic_chars += 1

    if alphabetic_chars == 0:
        return "unknown"

    hindi_ratio = hindi_chars / alphabetic_chars

    # If most alphabetic characters are Devanagari,
    # classify as Hindi.
    if hindi_ratio > 0.3:
        return "hi"

    return "en"


# ============================================================
# Translation generation helper
# ============================================================

def _generate_translation(
    tokenizer,
    model,
    text: str,
    max_new_tokens: int = 256
) -> str:
    """
    Generate a translation using safer decoding settings.

    no_repeat_ngram_size helps prevent repetitive output such as:
    'Caka Caka Caka Caka...'
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():

        translated = model.generate(
            **inputs,

            # Maximum output length
            max_new_tokens=max_new_tokens,

            # Beam search improves translation quality
            num_beams=4,

            # Prevent repetitive loops
            no_repeat_ngram_size=3,

            # Stop when the translation is complete
            early_stopping=True
        )

    result = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return result.strip()


# ============================================================
# Text chunking helper
# ============================================================

def _split_text_into_chunks(
    text: str,
    max_chars: int = 900
) -> list:
    """
    Split long text into reasonably sized chunks.

    Tries to split at paragraph/sentence boundaries instead
    of cutting text randomly in the middle of a sentence.
    """

    if len(text) <= max_chars:
        return [text]

    # First split by paragraphs
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding the paragraph stays within the limit
        if len(current) + len(paragraph) + 2 <= max_chars:

            if current:
                current += "\n\n" + paragraph
            else:
                current = paragraph

            continue

        # Save current chunk
        if current:
            chunks.append(current)
            current = ""

        # Handle a paragraph that is itself too long
        if len(paragraph) > max_chars:

            sentences = re.split(
                r"(?<=[.!?।])\s+",
                paragraph
            )

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                if len(current) + len(sentence) + 1 <= max_chars:

                    if current:
                        current += " " + sentence
                    else:
                        current = sentence

                else:

                    if current:
                        chunks.append(current)

                    current = sentence

        else:
            current = paragraph

    # Add final chunk
    if current:
        chunks.append(current)

    return chunks


# ============================================================
# Hindi → English
# ============================================================

def translate_to_english(text: str) -> str:
    """
    Translate Hindi text to English.

    English text is returned unchanged.
    """

    if not text or not text.strip():
        return text

    lang = detect_language(text)

    if lang == "en":
        print("Text is already in English, skipping translation")
        return text

    print("Translating from Hindi to English...")

    try:

        # IMPORTANT:
        # Load Hindi → English model
        hi_en_tokenizer, hi_en_model, _, _ = (
            load_translation_models()
        )

        chunks = _split_text_into_chunks(text)

        translated_chunks = []

        for chunk in chunks:

            translated_chunk = _generate_translation(
                tokenizer=hi_en_tokenizer,
                model=hi_en_model,
                text=chunk,
                max_new_tokens=256
            )

            if translated_chunk:
                translated_chunks.append(
                    translated_chunk
                )

        translated_text = " ".join(
            translated_chunks
        ).strip()

        print("Translation completed")

        return translated_text if translated_text else text

    except Exception as e:

        print(
            f"Hindi to English translation error: {str(e)}"
        )

        # Fall back to original text
        return text


# ============================================================
# English → Hindi
# ============================================================

def translate_to_hindi(text: str) -> str:
    """
    Translate English text to Hindi.

    Hindi text is returned unchanged.
    """

    if not text or not text.strip():
        return text

    lang = detect_language(text)

    if lang == "hi":
        print("Text is already in Hindi, skipping translation")
        return text

    print("Translating from English to Hindi...")

    try:

        # IMPORTANT:
        # Load English → Hindi model
        _, _, en_hi_tokenizer, en_hi_model = (
            load_translation_models()
        )

        chunks = _split_text_into_chunks(text)

        translated_chunks = []

        for chunk in chunks:

            translated_chunk = _generate_translation(
                tokenizer=en_hi_tokenizer,
                model=en_hi_model,
                text=chunk,
                max_new_tokens=256
            )

            if translated_chunk:
                translated_chunks.append(
                    translated_chunk
                )

        translated_text = " ".join(
            translated_chunks
        ).strip()

        print("Translation completed")

        return translated_text if translated_text else text

    except Exception as e:

        print(
            f"English to Hindi translation error: {str(e)}"
        )

        # Fall back to original English text
        return text