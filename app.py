import gradio as gr
from src.classifier import classify_disease, parse_label
from src.generator import generate_treatment


def analyze_crop(image, user_context):
    if image is None:
        return "Please upload an image.", ""

    print("\n--- New request ---")

    result = classify_disease(image)
    top = result["top_prediction"]
    crop, disease = parse_label(top["label"])
    confidence = top["confidence"]

    print(f"Classified: {crop} - {disease} ({confidence*100:.1f}%)")

    if disease.lower() == "healthy":
        diagnosis = (
            f"Crop      : {crop}\n"
            f"Status    : Healthy ✓\n"
            f"Confidence: {confidence*100:.1f}%\n\n"
            f"No disease detected. Your plant looks good!"
        )
        return diagnosis, "✅ No treatment needed. Keep monitoring your plant regularly."

    treatment = generate_treatment(crop, disease, confidence, user_context)

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

    return diagnosis, treatment


css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.diagnose-btn {
    background: #2d6a2d !important;
    border: none !important;
    font-size: 16px !important;
    height: 50px !important;
}
.diagnose-btn:hover {
    background: #1f4f1f !important;
}
footer { display: none !important; }
"""

with gr.Blocks(title="CropPilot", css=css) as demo:

    gr.Markdown("""
    <div style="text-align:center; padding: 20px 0 10px">
        <h1 style="font-size:2.5rem; margin-bottom:6px">🌿 CropPilot</h1>
        <p style="font-size:1rem; color:#888">AI-powered crop disease diagnosis · Backed by official NIPHM documents</p>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### Upload plant photo")
            image_input = gr.Image(
                type="filepath",
                label="",
                height=320
            )
            context_input = gr.Textbox(
                label="Additional context (optional)",
                placeholder="e.g. Maharashtra, Kharif season, irrigated field",
                lines=2
            )
            submit_btn = gr.Button(
                "🔍 Diagnose",
                variant="primary",
                elem_classes="diagnose-btn"
            )
            gr.Markdown("""
            <div style="margin-top:12px; padding:10px; background:#1a2e1a; border-radius:8px; font-size:13px; color:#aaa">
            <b style="color:#7ec87e">Supported crops:</b><br>
            Rice · Wheat · Maize · Potato · Cotton<br><br>
            <b style="color:#7ec87e">Knowledge base:</b><br>
            NIPHM IPM Packages — Govt. of India
            </div>
            """)

        with gr.Column(scale=1):
            gr.Markdown("### Results")

            with gr.Group():
                diagnosis_out = gr.Textbox(
                    label="Diagnosis",
                    lines=8,
                    interactive=False,
                    placeholder="Upload a plant photo and click Diagnose..."
                )

            with gr.Group():
                treatment_out = gr.Textbox(
                    label="Treatment Plan",
                    lines=18,
                    interactive=False,
                    placeholder="Treatment plan will appear here..."
                )

    submit_btn.click(
        fn=analyze_crop,
        inputs=[image_input, context_input],
        outputs=[diagnosis_out, treatment_out]
    )

demo.launch(share=True)