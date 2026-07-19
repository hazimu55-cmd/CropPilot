"""
Disease AI Service - Leaf image analysis
Uses YOLOv8 detector (PlantDoc-trained) for disease detection
"""
from ultralytics import YOLO
from PIL import Image
import torch
from src.config import MODEL_NAME, CONFIDENCE_THRESHOLD
from reliability import ConfidenceGate, GateResult


# Load YOLOv8 model once when the file is imported
print(f"[Disease AI] Loading YOLOv8 model: {MODEL_NAME}")
try:
    # Try to load from local path or HuggingFace
    model = YOLO(MODEL_NAME)
except Exception as e:
    print(f"[Disease AI] Error loading model: {e}")
    print("[Disease AI] Falling back to default YOLOv8n model")
    model = YOLO("yolov8n.pt")

# Initialize confidence gate
confidence_gate = ConfidenceGate(threshold=CONFIDENCE_THRESHOLD)


def classify_disease(image_path: str) -> dict:
    """
    Classify disease from leaf image using YOLOv8
    
    Args:
        image_path: Path to the leaf image
        
    Returns:
        Dictionary with top_prediction, alternatives, and gate_result
    """
    # Run inference
    results = model(image_path, verbose=False)
    
    # Process results
    all_predictions = []
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names.get(class_id, "Unknown")
                
                # Skip invalid classes
                if class_name.lower() != "invalid":
                    all_predictions.append({
                        "label": class_name,
                        "confidence": round(confidence, 3)
                    })
    
    # If no detections, return unknown
    if not all_predictions:
        gate_result = GateResult(
            passed=False,
            confidence=0.0,
            reason="No disease detected in image"
        )
        return {
            "top_prediction": {"label": "Unknown", "confidence": 0.0},
            "alternatives": [],
            "gate_result": gate_result
        }
    
    # Sort by confidence and get top 3
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top3 = all_predictions[:3]
    
    # Apply confidence gate to top prediction
    gate_result = confidence_gate.check(top3[0])
    
    return {
        "top_prediction": top3[0] if top3 else {"label": "Unknown", "confidence": 0.0},
        "alternatives": top3[1:] if len(top3) > 1 else [],
        "gate_result": gate_result
    }


def parse_label(raw_label: str) -> tuple:
    """
    Parse model label into crop and disease
    
    Args:
        raw_label: Raw label from model
        
    Returns:
        Tuple of (crop, disease)
    """
    if "___" in raw_label:
        parts = raw_label.split("___")
        crop = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
    elif " - " in raw_label:
        parts = raw_label.split(" - ")
        crop = parts[0].strip()
        disease = parts[1].strip() if len(parts) > 1 else "Unknown"
    elif " with " in raw_label.lower():
        parts = raw_label.split(" with ")
        crop = parts[0].strip()
        disease = parts[1].strip() if len(parts) > 1 else "Unknown"
    else:
        full = raw_label.replace("_", " ").strip()
        if "healthy" in full.lower():
            crop = full.lower().replace("healthy", "").strip().title()
            disease = "Healthy"
        else:
            words = full.split()
            if len(words) >= 2:
                crop = words[0].title()
                disease = " ".join(words[1:])
            else:
                crop = "Unknown"
                disease = full
    
    # Handle "Invalid" labels
    if crop == "Invalid" or disease == "Invalid":
        crop = "Unknown"
        disease = "Unknown Disease"
    
    return crop, disease
