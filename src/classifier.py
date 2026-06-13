from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch
from src.config import MODEL_NAME


# Load model once when the file is imported
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()


def classify_disease(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    top3 = torch.topk(probs, 3)

    results = []
    for score, idx in zip(top3.values, top3.indices):
        label = model.config.id2label[idx.item()]
        results.append({
            "label": label,
            "confidence": round(score.item(), 3)
        })

    return {
        "top_prediction": results[0],
        "alternatives": results[1:]
    }


def parse_label(raw_label: str) -> tuple:
    parts = raw_label.split("___")
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
    return crop, disease