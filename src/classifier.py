from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    CLIPProcessor,
    CLIPModel
)

from PIL import Image
import torch
from src.config import MODEL_NAME


# ============================================================
# GLOBAL VARIABLES
# ============================================================

processor = None
model = None

# Validity gate
clip_processor = None
clip_model = None

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD DISEASE CLASSIFIER
# ============================================================

def load_model():
    """Load ViT disease classifier."""

    global processor, model

    if processor is None or model is None:

        print(f"Loading disease model: {MODEL_NAME}")

        processor = ViTImageProcessor.from_pretrained(
            MODEL_NAME
        )

        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME
        )

        model = model.to(device)

        model.eval()

        print(
            f"Disease model loaded on device: {device}"
        )

    return processor, model


# ============================================================
# LOAD VALIDITY GATE
# ============================================================

def load_validity_gate():
    """
    Load CLIP model used to determine whether the
    uploaded image looks like a supported crop leaf.
    """

    global clip_processor, clip_model

    if clip_processor is None or clip_model is None:

        print("Loading CropPilot validity gate...")

        clip_model_name = "openai/clip-vit-base-patch32"

        clip_processor = CLIPProcessor.from_pretrained(
            clip_model_name
        )

        clip_model = CLIPModel.from_pretrained(
            clip_model_name
        )

        clip_model = clip_model.to(device)

        clip_model.eval()

        print(
            f"Validity gate loaded on device: {device}"
        )

    return clip_processor, clip_model


# ============================================================
# VALIDITY GATE
# ============================================================

def check_image_validity(image):
    """
    Determine whether the uploaded image appears to be
    a supported crop leaf.

    Returns:
        dict containing:
            is_valid
            validity_confidence
            valid_score
            invalid_score
    """

    clip_processor, clip_model = load_validity_gate()

    image = image.convert("RGB")

    # --------------------------------------------------------
    # VALID IMAGE PROMPTS
    # --------------------------------------------------------

    valid_prompts = [
        "a clear photograph of a corn leaf",
        "a clear photograph of a potato leaf",
        "a clear photograph of a rice leaf",
        "a clear photograph of a wheat leaf",

        "a close-up photograph of a corn plant leaf",
        "a close-up photograph of a potato plant leaf",
        "a close-up photograph of a rice plant leaf",
        "a close-up photograph of a wheat plant leaf",

        "a photograph of a diseased crop leaf",
        "a photograph of a healthy crop leaf"
    ]

    # --------------------------------------------------------
    # INVALID IMAGE PROMPTS
    # --------------------------------------------------------

    invalid_prompts = [
        "a photograph of a dog",
        "a photograph of a cat",
        "a photograph of a person",
        "a photograph of a car",
        "a photograph of a building",
        "a photograph of food",
        "a photograph of cooked corn",
        "a photograph of a corn cob",
        "a photograph of harvested grains",
        "a photograph of a landscape",
        "a photograph of a random object",
        "a photograph that does not contain a plant leaf",
        "a photograph of an unrelated plant"
    ]

    all_prompts = valid_prompts + invalid_prompts

    # --------------------------------------------------------
    # CLIP PROCESSING
    # --------------------------------------------------------

    inputs = clip_processor(
        text=all_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = clip_model(**inputs)

    logits = outputs.logits_per_image[0]

    # --------------------------------------------------------
    # SPLIT VALID / INVALID SCORES
    # --------------------------------------------------------

    valid_logits = logits[:len(valid_prompts)]

    invalid_logits = logits[len(valid_prompts):]

    # Use strongest matching prompt from each group
    valid_score = valid_logits.max().item()
    invalid_score = invalid_logits.max().item()

    # Difference between valid and invalid
    score_difference = valid_score - invalid_score

    # Convert difference to 0-1 value
    validity_confidence = torch.sigmoid(
        torch.tensor(score_difference)
    ).item()

    # Conservative initial threshold
    is_valid = (
        validity_confidence >= 0.60
        and valid_score > invalid_score
    )

    print(
        f"Validity score: {validity_confidence:.3f}"
    )

    print(
        f"Valid CLIP score: {valid_score:.3f}"
    )

    print(
        f"Invalid CLIP score: {invalid_score:.3f}"
    )

    print(
        f"Image accepted: {is_valid}"
    )

    return {
        "is_valid": is_valid,
        "validity_confidence": round(
            validity_confidence,
            3
        ),
        "valid_score": valid_score,
        "invalid_score": invalid_score
    }


# ============================================================
# DISEASE CLASSIFICATION
# ============================================================

def classify_disease(image_path: str) -> dict:
    """
    Classify plant disease from image.

    Pipeline:

        Image
          ↓
        Validity Gate
          ↓
        ViT Disease Classifier
    """

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # ========================================================
    # STEP 1 — VALIDITY GATE
    # ========================================================

    validity_result = check_image_validity(
        image
    )

    # --------------------------------------------------------
    # INVALID IMAGE
    # --------------------------------------------------------

    if not validity_result["is_valid"]:

        print(
            "❌ Image rejected by validity gate."
        )

        return {
            "status": "invalid",

            "message": (
                "⚠️ Unable to Diagnose\n\n"

                "This image could not be identified as a\n"
                "supported crop leaf.\n\n"

                "Please upload a clear image of:\n\n"

                "🌽 Corn\n"
                "🥔 Potato\n"
                "🌾 Rice\n"
                "🌾 Wheat"
            ),

            "validity_confidence":
                validity_result[
                    "validity_confidence"
                ]
        }


    # ========================================================
    # STEP 2 — LOAD DISEASE MODEL
    # ========================================================

    processor, model = load_model()


    # ========================================================
    # STEP 3 — PROCESS IMAGE
    # ========================================================

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # ========================================================
    # STEP 4 — VIТ PREDICTION
    # ========================================================

    with torch.no_grad():

        logits = model(
            **inputs
        ).logits


    # ========================================================
    # STEP 5 — SOFTMAX
    # ========================================================

    probs = torch.softmax(
        logits,
        dim=-1
    )[0]


    # ========================================================
    # STEP 6 — CHECK INVALID CLASS
    # ========================================================

    invalid_id = None

    for idx, label in model.config.id2label.items():

        if label.lower().strip() == "invalid":

            invalid_id = int(idx)

            break


    # --------------------------------------------------------
    # If model itself predicts Invalid
    # --------------------------------------------------------

    if invalid_id is not None:

        invalid_probability = probs[
            invalid_id
        ].item()

        print(
            f"Model Invalid probability: "
            f"{invalid_probability:.3f}"
        )

    else:

        invalid_probability = 0.0


    # ========================================================
    # STEP 7 — GET TOP VALID CLASS
    # ========================================================

    valid_predictions = []

    for idx, prob in enumerate(probs):

        label = model.config.id2label[
            idx
        ]

        # DO NOT allow Invalid to become
        # a disease diagnosis
        if (
            label.lower().strip()
            == "invalid"
        ):
            continue

        valid_predictions.append({

            "label": label,

            "confidence": round(
                prob.item(),
                3
            )
        })


    # Sort highest confidence first

    valid_predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )


    # ========================================================
    # STEP 8 — NO VALID CLASS
    # ========================================================

    if not valid_predictions:

        return {

            "status": "invalid",

            "message": (
                "⚠️ Unable to Diagnose\n\n"
                "No supported crop disease "
                "could be identified."
            )
        }


    # ========================================================
    # STEP 9 — TOP PREDICTION
    # ========================================================

    top_prediction = valid_predictions[0]

    confidence = top_prediction[
        "confidence"
    ]


    print(
        f"Top prediction: "
        f"{top_prediction['label']}"
    )

    print(
        f"Confidence: {confidence}"
    )


    # ========================================================
    # STEP 10 — CLASSIFIER CONFIDENCE CHECK
    # ========================================================

    CLASSIFIER_THRESHOLD = 0.50

    if confidence < CLASSIFIER_THRESHOLD:

        return {

            "status": "uncertain",

            "message": (
                "⚠️ Diagnosis Uncertain\n\n"

                "The image appears to contain "
                "a supported crop leaf, but the "
                "disease classification is uncertain.\n\n"

                f"Confidence: {confidence:.1%}\n\n"

                "Please upload a clearer image "
                "of the affected leaf."
            ),

            "top_prediction": top_prediction,

            "confidence": confidence
        }


    # ========================================================
    # STEP 11 — TOP 3 ALTERNATIVES
    # ========================================================

    top3 = valid_predictions[:3]

    for pred in top3:

        print(
            f"Prediction: {pred['label']} "
            f"| Confidence: "
            f"{pred['confidence']}"
        )


    # ========================================================
    # STEP 12 — RETURN SUCCESS
    # ========================================================

    return {

        "status": "success",

        "top_prediction": top3[0],

        "alternatives": (
            top3[1:]
            if len(top3) > 1
            else []
        ),

        "confidence": confidence,

        "validity_confidence":
            validity_result[
                "validity_confidence"
            ]
    }


# ============================================================
# LABEL PARSER
# ============================================================

def parse_label(raw_label: str) -> tuple:

    print(
        f"Parsing label: '{raw_label}'"
    )

    if "___" in raw_label:

        parts = raw_label.split(
            "___"
        )

        crop = parts[0].replace(
            "_",
            " "
        ).strip()

        disease = parts[1].replace(
            "_",
            " "
        ).strip()

    elif " - " in raw_label:

        parts = raw_label.split(
            " - "
        )

        crop = parts[0].strip()

        disease = (
            parts[1].strip()
            if len(parts) > 1
            else "Unknown"
        )

    elif " with " in raw_label.lower():

        parts = raw_label.split(
            " with "
        )

        crop = parts[0].strip()

        disease = (
            parts[1].strip()
            if len(parts) > 1
            else "Unknown"
        )

    else:

        full = raw_label.replace(
            "_",
            " "
        ).strip()

        if "healthy" in full.lower():

            crop = (
                full.lower()
                .replace(
                    "healthy",
                    ""
                )
                .strip()
                .title()
            )

            disease = "Healthy"

        else:

            words = full.split()

            if len(words) >= 2:

                crop = words[0].title()

                disease = " ".join(
                    words[1:]
                )

            else:

                crop = "Unknown"

                disease = full


    # Handle Invalid

    if (
        crop.lower() == "invalid"
        or
        disease.lower() == "invalid"
    ):

        crop = "Unknown"

        disease = "Unknown Disease"


    print(
        f"Parsed -> Crop: '{crop}', "
        f"Disease: '{disease}'"
    )

    return crop, disease