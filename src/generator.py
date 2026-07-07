from groq import Groq
from src.retriever import retrieve_treatment_docs
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT_TEMPLATE = """You are CropPilot, an agricultural expert assistant helping Indian farmers treat crop diseases.
Answer only using the retrieved documents provided below.
If information is not in the documents, say so clearly. Never invent pesticide names or dosages.

DIAGNOSED DISEASE: {disease} in {crop}
CONFIDENCE: {confidence}%
FARMER CONTEXT: {user_context}

RETRIEVED FROM OFFICIAL DOCUMENTS:
{context}

Based only on the above documents, provide:
1. What this disease is and how it spreads
2. Organic / biological treatment options
3. Chemical treatment options with exact dosage
4. Prevention steps for next season
5. Urgency level: Low / Medium / High / Critical

Keep it practical and specific. Farmer should be able to act on this immediately."""


def generate_treatment(
    crop: str,
    disease: str,
    confidence: float,
    user_context: str = ""
) -> str:

    chunks = retrieve_treatment_docs(crop, disease)

    if not chunks:
        return "No treatment information found in knowledge base for this disease."

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Document {i+1} (Source: {chunk['source']}, Page: {chunk['page']}) ---\n"
        context += chunk["content"] + "\n"

    prompt = PROMPT_TEMPLATE.format(
        crop=crop,
        disease=disease,
        confidence=round(confidence * 100, 1),
        user_context=user_context or "Not specified",
        context=context
    )

    print("\nSending to Groq LLM...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content