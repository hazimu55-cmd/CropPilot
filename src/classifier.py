from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from src.config import MODEL_NAME

# Global variables for model and processor
processor = None
model = None

def load_model():
    """Load model and processor - called on first use"""
    global processor, model
    if processor is None or model is None:
        print(f"Loading model: {MODEL_NAME}")
        processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        model = ViTForImageClassification.from_pretrained(MODEL_NAME)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print(f"Model loaded on device: {device}")
    
    return processor, model

def classify_disease(image_path: str) -> dict:
    """Classify plant disease from image"""
    # Load model if not already loaded
    processor, model = load_model()
    
    # Open and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    
    # Get all predictions and filter out "Invalid" class
    all_predictions = []
    for idx, prob in enumerate(probs):
        label = model.config.id2label[idx]
        if label != "Invalid":  # Skip the Invalid class
            all_predictions.append({
                "label": label,
                "confidence": round(prob.item(), 3)
            })
    
    # Sort by confidence and get top 3
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top3 = all_predictions[:3]
    
    for pred in top3:
        print(f"Raw label: {pred['label']}, Confidence: {pred['confidence']}")

    return {
        "top_prediction": top3[0] if top3 else {"label": "Unknown", "confidence": 0.0},
        "alternatives": top3[1:] if len(top3) > 1 else []
    }


def parse_label(raw_label: str) -> tuple:
    print(f"Parsing label: '{raw_label}'")
    
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
            # Try to extract crop and disease from the label
            # Common patterns: "Corn Rust", "Potato Early Blight", etc.
            words = full.split()
            if len(words) >= 2:
                # First word is likely the crop
                crop = words[0].title()
                disease = " ".join(words[1:])
            else:
                crop = "Unknown"
                disease = full
    
    # Handle "Invalid" labels
    if crop == "Invalid" or disease == "Invalid":
        crop = "Unknown"
        disease = "Unknown Disease"
    
    print(f"Parsed -> Crop: '{crop}', Disease: '{disease}'")
    return crop, disease