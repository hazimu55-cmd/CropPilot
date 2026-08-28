from src.config import MIN_CHUNK_LENGTH


def apply_retrieval_gate(
    chunks: list,
    crop: str,
    disease: str
) -> list:
    """
    Filter retrieved chunks using basic quality and
    crop/disease relevance checks.
    """

    filtered_chunks = []

    crop_lower = crop.lower()
    disease_lower = disease.lower()

    # Disease name variations
    disease_terms = [
        disease_lower,
        disease_lower.replace(" leaf", ""),
        disease_lower.replace(" ", "_"),
        disease_lower.replace(" ", "-")
    ]

    for chunk in chunks:

        content = chunk.get("content", "")
        content_lower = content.lower()

        # -------------------------------------------------
        # Filter 1: Minimum length
        # -------------------------------------------------
        if len(content) < MIN_CHUNK_LENGTH:
            print(
                f"Filtering short chunk "
                f"(length: {len(content)})"
            )
            continue

        # -------------------------------------------------
        # Filter 2: Meaningful content
        # -------------------------------------------------
        meaningful_chars = sum(
            1 for c in content
            if c.isalnum() or c.isspace()
        )

        if len(content) == 0:
            continue

        if meaningful_chars / len(content) < 0.5:
            print("Filtering chunk with low meaningful content ratio")
            continue

        # -------------------------------------------------
        # Filter 3: Agricultural content
        # -------------------------------------------------
        agricultural_keywords = [
            "treatment",
            "disease",
            "pest",
            "chemical",
            "organic",
            "dosage",
            "prevention",
            "management",
            "control",
            "fungicide",
            "insecticide",
            "fertilizer",
            "symptom",
            "infection"
        ]

        has_agri_content = any(
            keyword in content_lower
            for keyword in agricultural_keywords
        )

        if not has_agri_content:
            print("Filtering non-agricultural chunk")
            continue

        # -------------------------------------------------
        # Filter 4: Crop relevance
        # -------------------------------------------------
        if crop_lower not in content_lower:
            print(
                f"Filtering chunk unrelated to crop: {crop}"
            )
            continue

        # -------------------------------------------------
        # Filter 5: Disease relevance
        # -------------------------------------------------
        has_disease = any(
            term in content_lower
            for term in disease_terms
            if term
        )

        if not has_disease:
            print(
                f"Filtering chunk unrelated to disease: {disease}"
            )
            continue

        filtered_chunks.append(chunk)

    print(
        f"Retrieval gate: "
        f"{len(chunks)} -> {len(filtered_chunks)} chunks"
    )

    return filtered_chunks