import gradio as gr
import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def analyze_crop(image, user_context):
    if image is None:
        return "Please upload an image.", ""

    print("\n--- New request ---")

    try:
        # Upload image and get diagnosis from API
        files = {"file": open(image, "rb")}
        data = {"user_context": user_context}
        
        response = requests.post(
            f"{API_BASE_URL}/api/diagnose/upload",
            files=files,
            data=data
        )
        files["file"].close()
        
        if response.status_code == 200:
            result = response.json()
            diagnosis = result["diagnosis"]
            treatment = result["treatment"]
            
            # Add gate result info if available
            if result.get("gate_result"):
                gate = result["gate_result"]
                if not gate["passed"]:
                    diagnosis += f"\n\n⚠️ Confidence Gate: {gate['reason']}"
            
            return diagnosis, treatment
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return error_msg, ""
            
    except Exception as e:
        error_msg = f"Error connecting to API: {str(e)}"
        print(error_msg)
        return error_msg, ""


def create_crop_plan(crop, region, season, soil_type, user_context):
    """Generate cultivation plan via API"""
    try:
        payload = {
            "crop": crop,
            "region": region,
            "season": season,
            "soil_type": soil_type,
            "user_context": user_context
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/crop-plan",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            plan = result["plan"]
            
            # Add gate result info if available
            if result.get("gate_results"):
                passed_count = sum(1 for gr in result["gate_results"] if gr.get("passed"))
                total_count = len(result["gate_results"])
                plan += f"\n\n📊 Retrieval Quality: {passed_count}/{total_count} chunks passed quality filters"
            
            return plan
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error connecting to API: {str(e)}"


def answer_question(question):
    """Answer farming question via API"""
    try:
        payload = {"question": question}
        
        response = requests.post(
            f"{API_BASE_URL}/api/qa",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["answer"]
            
            # Add gate and faithfulness info if available
            if result.get("gate_results"):
                passed_count = sum(1 for gr in result["gate_results"] if gr.get("passed"))
                total_count = len(result["gate_results"])
                answer += f"\n\n📊 Retrieval Quality: {passed_count}/{total_count} chunks passed quality filters"
            
            if result.get("faithfulness_results"):
                faith = result["faithfulness_results"]
                if faith.get("citations", {}).get("passed"):
                    answer += "\n✅ Response includes proper citations"
                else:
                    answer += "\n⚠️ Response may lack proper citations"
                
                if faith.get("faithfulness", {}).get("passed"):
                    answer += "\n✅ Response appears faithful to sources"
                else:
                    answer += "\n⚠️ Response may contain hallucinations"
            
            return answer
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error connecting to API: {str(e)}"


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
        <p style="font-size:1rem; color:#888">AI-powered agricultural assistant · Disease diagnosis · Crop planning · Expert Q&A</p>
    </div>
    """)

    with gr.Tabs():
        
        # Disease Diagnosis Tab
        with gr.Tab("🔍 Disease Diagnosis"):
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
        
        # Crop Planning Tab
        with gr.Tab("📋 Crop Planning"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Cultivation Plan Generator")
                    crop_input = gr.Textbox(
                        label="Crop",
                        placeholder="e.g. Rice, Wheat, Maize",
                        value="Rice"
                    )
                    region_input = gr.Textbox(
                        label="Region (optional)",
                        placeholder="e.g. Maharashtra, Punjab"
                    )
                    season_input = gr.Textbox(
                        label="Season (optional)",
                        placeholder="e.g. Kharif, Rabi"
                    )
                    soil_input = gr.Textbox(
                        label="Soil Type (optional)",
                        placeholder="e.g. Loamy, Clay, Sandy"
                    )
                    plan_context = gr.Textbox(
                        label="Additional context (optional)",
                        placeholder="Any specific requirements or constraints",
                        lines=2
                    )
                    plan_btn = gr.Button(
                        "📋 Generate Plan",
                        variant="primary",
                        elem_classes="diagnose-btn"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Cultivation Plan")
                    plan_output = gr.Textbox(
                        label="Plan",
                        lines=25,
                        interactive=False,
                        placeholder="Your cultivation plan will appear here..."
                    )

            plan_btn.click(
                fn=create_crop_plan,
                inputs=[crop_input, region_input, season_input, soil_input, plan_context],
                outputs=[plan_output]
            )
        
        # Expert Q&A Tab
        with gr.Tab("💬 Expert Q&A"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Ask an Agricultural Expert")
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. How do I control brown plant hopper in rice?",
                        lines=3
                    )
                    qa_btn = gr.Button(
                        "💬 Ask",
                        variant="primary",
                        elem_classes="diagnose-btn"
                    )
                    gr.Markdown("""
                    <div style="margin-top:12px; padding:10px; background:#1a2e1a; border-radius:8px; font-size:13px; color:#aaa">
                    <b style="color:#7ec87e">Knowledge sources:</b><br>
                    ICAR · NIPHM · Agricultural Universities
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("### Expert Answer")
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=20,
                        interactive=False,
                        placeholder="Expert answer will appear here..."
                    )

            qa_btn.click(
                fn=answer_question,
                inputs=[question_input],
                outputs=[answer_output]
            )

demo.launch(share=True)