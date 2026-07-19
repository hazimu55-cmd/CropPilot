"""
FastAPI Backend API Layer
Provides REST endpoints for disease diagnosis, crop planning, and Q&A
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import shutil
import os
from services import classify_disease, parse_label, generate_cultivation_plan, answer_farming_question
from src.generator import generate_treatment

app = FastAPI(title="CropPilot API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class DiagnosisRequest(BaseModel):
    image_path: str
    user_context: Optional[str] = ""


class CropPlanRequest(BaseModel):
    crop: str
    region: Optional[str] = ""
    season: Optional[str] = ""
    soil_type: Optional[str] = ""
    user_context: Optional[str] = ""


class QuestionRequest(BaseModel):
    question: str


class GateResult(BaseModel):
    passed: bool
    confidence: float
    reason: str


class DiagnosisResponse(BaseModel):
    crop: str
    disease: str
    confidence: float
    diagnosis: str
    treatment: str
    gate_result: Optional[GateResult] = None


class CropPlanResponse(BaseModel):
    plan: str
    gate_results: Optional[List[Dict[str, Any]]] = None
    filtered_chunks_count: Optional[int] = None


class QuestionResponse(BaseModel):
    answer: str
    gate_results: Optional[List[Dict[str, Any]]] = None
    faithfulness_results: Optional[Dict[str, Dict[str, Any]]] = None
    filtered_chunks_count: Optional[int] = None


# Health check
@app.get("/")
async def root():
    return {"message": "CropPilot API is running", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Disease diagnosis endpoint
@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose_disease(request: DiagnosisRequest):
    """
    Diagnose crop disease from image
    """
    try:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=400, detail="Image file not found")
        
        result = classify_disease(request.image_path)
        top = result["top_prediction"]
        gate_result = result.get("gate_result")
        crop, disease = parse_label(top["label"])
        confidence = top["confidence"]
        
        # Check if confidence gate passed
        if gate_result and not gate_result.passed:
            diagnosis = (
                f"Crop      : {crop}\n"
                f"Disease   : {disease}\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"⚠️ Low confidence prediction. Diagnosis may not be reliable.\n"
                f"Reason: {gate_result.reason}"
            )
            treatment = "⚠️ Low confidence - cannot provide reliable treatment recommendation."
        elif disease.lower() == "healthy":
            diagnosis = (
                f"Crop      : {crop}\n"
                f"Status    : Healthy ✓\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"No disease detected. Your plant looks good!"
            )
            treatment = "✅ No treatment needed. Keep monitoring your plant regularly."
        else:
            alternatives = result["alternatives"]
            diagnosis = (
                f"Crop      : {crop}\n"
                f"Disease   : {disease}\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"Other possibilities considered:\n"
                + "\n".join(
                    f"  • {parse_label(a['label'])[1]} ({a['confidence']*100:.1f}%)"
                    for a in alternatives
                )
            )
            treatment = generate_treatment(crop, disease, confidence, request.user_context)
        
        return DiagnosisResponse(
            crop=crop,
            disease=disease,
            confidence=confidence,
            diagnosis=diagnosis,
            treatment=treatment,
            gate_result=GateResult(**gate_result.__dict__) if gate_result else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Upload image and diagnose endpoint
@app.post("/api/diagnose/upload", response_model=DiagnosisResponse)
async def diagnose_uploaded_image(
    file: UploadFile = File(...),
    user_context: Optional[str] = ""
):
    """
    Upload image and diagnose disease
    """
    try:
        # Save uploaded file
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Diagnose
        result = classify_disease(file_path)
        top = result["top_prediction"]
        gate_result = result.get("gate_result")
        crop, disease = parse_label(top["label"])
        confidence = top["confidence"]
        
        # Check if confidence gate passed
        if gate_result and not gate_result.passed:
            diagnosis = (
                f"Crop      : {crop}\n"
                f"Disease   : {disease}\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"⚠️ Low confidence prediction. Diagnosis may not be reliable.\n"
                f"Reason: {gate_result.reason}"
            )
            treatment = "⚠️ Low confidence - cannot provide reliable treatment recommendation."
        elif disease.lower() == "healthy":
            diagnosis = (
                f"Crop      : {crop}\n"
                f"Status    : Healthy ✓\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"No disease detected. Your plant looks good!"
            )
            treatment = "✅ No treatment needed. Keep monitoring your plant regularly."
        else:
            alternatives = result["alternatives"]
            diagnosis = (
                f"Crop      : {crop}\n"
                f"Disease   : {disease}\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"Other possibilities considered:\n"
                + "\n".join(
                    f"  • {parse_label(a['label'])[1]} ({a['confidence']*100:.1f}%)"
                    for a in alternatives
                )
            )
            treatment = generate_treatment(crop, disease, confidence, user_context)
        
        return DiagnosisResponse(
            crop=crop,
            disease=disease,
            confidence=confidence,
            diagnosis=diagnosis,
            treatment=treatment,
            gate_result=GateResult(**gate_result.__dict__) if gate_result else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Crop planning endpoint
@app.post("/api/crop-plan", response_model=CropPlanResponse)
async def create_crop_plan(request: CropPlanRequest):
    """
    Generate cultivation plan for a crop
    """
    try:
        result = generate_cultivation_plan(
            crop=request.crop,
            region=request.region,
            season=request.season,
            soil_type=request.soil_type,
            user_context=request.user_context
        )
        
        return CropPlanResponse(
            plan=result["plan"],
            gate_results=[gr.__dict__ for gr in result.get("gate_results", [])],
            filtered_chunks_count=len(result.get("filtered_chunks", []))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Q&A endpoint
@app.post("/api/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer farming question using RAG
    """
    try:
        result = answer_farming_question(request.question)
        
        faithfulness_dict = None
        if result.get("faithfulness_results"):
            faithfulness_dict = {
                key: value.__dict__ if hasattr(value, '__dict__') else value
                for key, value in result["faithfulness_results"].items()
            }
        
        return QuestionResponse(
            answer=result["answer"],
            gate_results=[gr.__dict__ for gr in result.get("gate_results", [])],
            faithfulness_results=faithfulness_dict,
            filtered_chunks_count=len(result.get("filtered_chunks", []))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
