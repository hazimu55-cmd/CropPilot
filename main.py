from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
from typing import Optional

from src.classifier import classify_disease, parse_label
from src.generator import generate_treatment
from src.retriever import retrieve_treatment_docs
from src.translator import translate_to_english, translate_to_hindi, detect_language
from src.retrieval_gate import apply_retrieval_gate
from src.faithfulness import check_faithfulness, format_faithfulness_warning
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI(title="CropPilot API", version="1.0.0")

# Initialize Groq client for general QA
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class CropPlanRequest(BaseModel):
    query: str
    context: Optional[str] = ""


class DiagnosisResponse(BaseModel):
    crop: str
    disease: str
    confidence: float
    diagnosis_text: str
    treatment_text: str
    alternatives: list
    faithful: bool
    faithfulness_score: float
    warning: Optional[str] = None


class CropPlanResponse(BaseModel):
    query: str
    response: str
    faithful: bool
    faithfulness_score: float
    warning: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "CropPilot API - Hindi/English Agricultural Assistant"}


@app.post("/api/diagnose/upload", response_model=DiagnosisResponse)
async def diagnose_crop(
    image: UploadFile = File(...),
    user_context: str = Form(""),
    language: str = Form("auto")  # auto, hi, en
):
    """
    Diagnose crop disease from uploaded image.
    Supports Hindi input/output with automatic translation.
    """
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(await image.read())
        temp_path = temp_file.name
    
    try:
        # Translate user context if in Hindi
        if language == "auto":
            detected_lang = detect_language(user_context)
            if detected_lang == "hi":
                user_context_en = translate_to_english(user_context)
            else:
                user_context_en = user_context
        elif language == "hi":
            user_context_en = translate_to_english(user_context)
        else:
            user_context_en = user_context
        
        # Classify disease
        result = classify_disease(temp_path)
        top = result["top_prediction"]
        crop, disease = parse_label(top["label"])
        confidence = top["confidence"]
        
        print(f"Classified: {crop} - {disease} ({confidence*100:.1f}%)")
        
        # Handle healthy case
        if disease.lower() == "healthy":
            diagnosis_text = (
                f"Crop      : {crop}\n"
                f"Status    : Healthy ✓\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"No disease detected. Your plant looks good!"
            )
            treatment_text = "✅ No treatment needed. Keep monitoring your plant regularly."
            
            # Translate output to Hindi if requested
            if language == "hi" or (language == "auto" and detect_language(user_context) == "hi"):
                diagnosis_text = translate_to_hindi(diagnosis_text)
                treatment_text = translate_to_hindi(treatment_text)
            
            return DiagnosisResponse(
                crop=crop,
                disease=disease,
                confidence=confidence,
                diagnosis_text=diagnosis_text,
                treatment_text=treatment_text,
                alternatives=[],
                faithful=True,
                faithfulness_score=1.0
            )
        
        # Retrieve treatment documents
        chunks = retrieve_treatment_docs(crop, disease)
        
        # Apply retrieval gate
        filtered_chunks = apply_retrieval_gate(chunks)
        
        if not filtered_chunks:
            treatment_text = "No treatment information found in knowledge base for this disease."
            
            # Translate to Hindi if needed
            if language == "hi" or (language == "auto" and detect_language(user_context) == "hi"):
                treatment_text = translate_to_hindi(treatment_text)
            
            diagnosis_text = (
                f"Crop      : {crop}\n"
                f"Disease   : {disease}\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"Other possibilities considered:\n"
                + "\n".join(
                    f"  • {parse_label(a['label'])[1]} ({a['confidence']*100:.1f}%)"
                    for a in result["alternatives"]
                )
            )
            
            return DiagnosisResponse(
                crop=crop,
                disease=disease,
                confidence=confidence,
                diagnosis_text=diagnosis_text,
                treatment_text=treatment_text,
                alternatives=result["alternatives"],
                faithful=False,
                faithfulness_score=0.0
            )
        
        # Generate treatment (using filtered chunks)
        treatment_en = generate_treatment(crop, disease, confidence, user_context_en, filtered_chunks)
        
        # Check faithfulness
        combined_context = " ".join(chunk.get('content', '') for chunk in filtered_chunks)
        is_faithful, faithfulness_score = check_faithfulness(treatment_en, filtered_chunks)
        
        # Add warning if not faithful
        warning = None
        if not is_faithful:
            warning = format_faithfulness_warning(faithfulness_score)
            treatment_en += warning
        
        # Format diagnosis
        alternatives = result["alternatives"]
        diagnosis_en = (
            f"Crop      : {crop}\n"
            f"Disease   : {disease}\n"
            f"Confidence: {confidence*100:.1f}%\n\n"
            f"Other possibilities considered:\n"
            + "\n".join(
                f"  • {parse_label(a['label'])[1]} ({a['confidence']*100:.1f}%)"
                for a in alternatives
            )
        )
        
        # Translate output to Hindi if requested
        if language == "hi" or (language == "auto" and detect_language(user_context) == "hi"):
            diagnosis_text = translate_to_hindi(diagnosis_en)
            treatment_text = translate_to_hindi(treatment_en)
        else:
            diagnosis_text = diagnosis_en
            treatment_text = treatment_en
        
        return DiagnosisResponse(
            crop=crop,
            disease=disease,
            confidence=confidence,
            diagnosis_text=diagnosis_text,
            treatment_text=treatment_text,
            alternatives=alternatives,
            faithful=is_faithful,
            faithfulness_score=faithfulness_score,
            warning=warning
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/crop-plan", response_model=CropPlanResponse)
async def crop_plan_qa(request: CropPlanRequest):
    """
    Answer general agricultural questions with crop planning advice.
    Supports Hindi input/output with automatic translation.
    """
    try:
        # Detect language and translate if needed
        detected_lang = detect_language(request.query)
        
        if detected_lang == "hi":
            query_en = translate_to_english(request.query)
            context_en = translate_to_english(request.context) if request.context else ""
        else:
            query_en = request.query
            context_en = request.context
        
        print(f"Processing query: {query_en}")
        
        # Retrieve relevant documents using FAISS
        # For general QA, we search the knowledge base
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from src.config import INDEX_PATH, EMBEDDING_MODEL, TOP_K
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.invoke(query_en)
        
        # Apply retrieval gate
        chunks = []
        for doc in docs:
            chunks.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })
        
        filtered_chunks = apply_retrieval_gate(chunks)
        
        if not filtered_chunks:
            response_en = "I couldn't find relevant information in the knowledge base for your question. Please try rephrasing or contact agricultural experts."
            
            # Translate back to Hindi if needed
            if detected_lang == "hi":
                response_hi = translate_to_hindi(response_en)
                return CropPlanResponse(
                    query=request.query,
                    response=response_hi,
                    faithful=False,
                    faithfulness_score=0.0
                )
            else:
                return CropPlanResponse(
                    query=request.query,
                    response=response_en,
                    faithful=False,
                    faithfulness_score=0.0
                )
        
        # Generate response using Groq LLM
        context = ""
        for i, chunk in enumerate(filtered_chunks):
            context += f"\n--- Document {i+1} (Source: {chunk['source']}, Page: {chunk['page']}) ---\n"
            context += chunk["content"] + "\n"
        
        prompt = f"""You are CropPilot, an agricultural expert assistant helping Indian farmers.
Answer the farmer's question using ONLY the retrieved documents provided below.
If information is not in the documents, say so clearly. Never invent information.

FARMER'S QUESTION: {query_en}
ADDITIONAL CONTEXT: {context_en}

RETRIEVED FROM OFFICIAL DOCUMENTS:
{context}

Provide a practical, specific answer that the farmer can act on immediately."""

        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        
        response_en = response.choices[0].message.content
        
        # Check faithfulness
        is_faithful, faithfulness_score = check_faithfulness(response_en, filtered_chunks)
        
        # Add warning if not faithful
        if not is_faithful:
            response_en += format_faithfulness_warning(faithfulness_score)
        
        # Translate back to Hindi if needed
        if detected_lang == "hi":
            response_hi = translate_to_hindi(response_en)
            return CropPlanResponse(
                query=request.query,
                response=response_hi,
                faithful=is_faithful,
                faithfulness_score=faithfulness_score
            )
        else:
            return CropPlanResponse(
                query=request.query,
                response=response_en,
                faithful=is_faithful,
                faithfulness_score=faithfulness_score
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qa")
async def general_qa(request: CropPlanRequest):
    """
    General agricultural Q&A without retrieval (for simple questions).
    Supports Hindi input/output.
    """
    try:
        # Detect language and translate if needed
        detected_lang = detect_language(request.query)
        
        if detected_lang == "hi":
            query_en = translate_to_english(request.query)
        else:
            query_en = request.query
        
        # Generate response using Groq LLM
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": "You are CropPilot, an agricultural expert assistant. Help farmers with general questions about crops, farming practices, and agriculture in India. Be practical and specific."},
                {"role": "user", "content": query_en}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        response_en = response.choices[0].message.content
        
        # Translate back to Hindi if needed
        if detected_lang == "hi":
            response_hi = translate_to_hindi(response_en)
            return {"query": request.query, "response": response_hi}
        else:
            return {"query": request.query, "response": response_en}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
