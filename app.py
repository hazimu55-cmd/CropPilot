import spaces
import gradio as gr
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.classifier import classify_disease, parse_label
from src.generator import generate_treatment
from src.translator import translate_to_english, translate_to_hindi, detect_language
from src.retriever import retrieve_treatment_docs
from src.retrieval_gate import apply_retrieval_gate
from src.faithfulness import check_faithfulness, format_faithfulness_warning
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Custom CSS for elegant, responsive design
css = """
/* Modern, elegant styling */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #1a5f2a 0%, #2d8a4e 100%) !important;
    padding: 30px !important;
    border-radius: 16px !important;
    margin-bottom: 25px !important;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1) !important;
}

/* Button styling */
.diagnose-btn {
    background: linear-gradient(135deg, #2d6a2d 0%, #4caf50 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    height: 55px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(45, 106, 45, 0.3) !important;
}

.diagnose-btn:hover {
    background: linear-gradient(135deg, #1f4f1f 0%, #388e3c 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(45, 106, 45, 0.4) !important;
}

.diagnose-btn:active {
    transform: translateY(0) !important;
}

/* Card styling */
.info-card {
    background: #ffffff !important;
    border: 1px solid #e9ecef !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin: 15px 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}

.result-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%) !important;
    border: 2px solid #dee2e6 !important;
    border-radius: 12px !important;
    padding: 25px !important;
    margin: 15px 0 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}

/* Textarea styling */
.result-text {
    background: #ffffff !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 8px !important;
    padding: 18px !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    color: #212529 !important;
}

/* Image upload styling */
.upload-area {
    border: 3px dashed #2d6a2d !important;
    border-radius: 16px !important;
    padding: 30px !important;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%) !important;
    transition: all 0.3s ease !important;
    text-align: center !important;
}

.upload-area:hover {
    border-color: #4caf50 !important;
    background: linear-gradient(135deg, #f0f9f0 0%, #ffffff 100%) !important;
    transform: scale(1.02) !important;
}

/* Info box styling */
.info-box {
    background: linear-gradient(135deg, #1a2e1a 0%, #2d4a2d 100%) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    font-size: 14px !important;
    color: #e8f5e9 !important;
    border-left: 5px solid #4caf50 !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
}

.info-box b {
    color: #81c784 !important;
}

/* Tab styling */
.tab-nav button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover {
    transform: translateY(-1px) !important;
}

/* Chatbot styling */
.chatbot-container {
    border: 2px solid #dee2e6 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 1.5s ease-in-out infinite;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .gradio-container {
        max-width: 100% !important;
        padding: 15px !important;
    }
    
    .diagnose-btn {
        width: 100% !important;
    }
    
    .info-card, .result-card {
        padding: 15px !important;
    }
}

/* Footer removal */
footer { 
    display: none !important; 
}

/* Smooth transitions */
* {
    transition: all 0.2s ease !important;
}
"""

@spaces.GPU(duration=120)
def analyze_crop(image, user_context, language="Auto (स्वचालित)"):
    """Analyze crop disease with improved prediction logic"""
    
    if image is None:
        return "Please upload an image.", ""

    print("\n--- New request ---")

    # Detect language and translate user context if needed
    if language == "Auto (स्वचालित)":
        detected_lang = detect_language(user_context)
    elif language == "Hindi (हिंदी)":
        detected_lang = "hi"
    else:  # English
        detected_lang = "en"
    
    if detected_lang == "hi":
        user_context_en = translate_to_english(user_context)
        print(f"Translated context from Hindi to English")
    else:
        user_context_en = user_context

    try:
        result = classify_disease(image)
        top = result["top_prediction"]
        crop, disease = parse_label(top["label"])
        confidence = top["confidence"]

        print(f"Classified: {crop} - {disease} ({confidence*100:.1f}%)")

        # Confidence threshold for reliable predictions
        if confidence < 0.5:
            diagnosis_en = (
                f"⚠️ Low Confidence Detection\n\n"
                f"Crop      : {crop}\n"
                f"Disease   : {disease}\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"Confidence is below 50%. The image quality may be poor or the disease "
                f"may not be one of the supported types. Please try a clearer image of a "
                f"leaf from our supported crops: Corn, Potato, Rice, or Wheat."
            )
            treatment_en = "📸 Please upload a clearer leaf image for accurate diagnosis."
            
            # Translate to Hindi if needed
            if detected_lang == "hi":
                diagnosis = translate_to_hindi(diagnosis_en)
                treatment = translate_to_hindi(treatment_en)
            else:
                diagnosis = diagnosis_en
                treatment = treatment_en
            
            return diagnosis, treatment

        if disease.lower() == "healthy":
            diagnosis_en = (
                f"✅ Healthy Plant\n\n"
                f"Crop      : {crop}\n"
                f"Status    : Healthy ✓\n"
                f"Confidence: {confidence*100:.1f}%\n\n"
                f"Great news! No disease detected. Your plant looks healthy. "
                f"Continue with regular monitoring and good agricultural practices."
            )
            treatment_en = "🌱 No treatment needed. Keep up the good work with proper irrigation, fertilization, and pest monitoring."
            
            # Translate to Hindi if needed
            if detected_lang == "hi":
                diagnosis = translate_to_hindi(diagnosis_en)
                treatment = translate_to_hindi(treatment_en)
            else:
                diagnosis = diagnosis_en
                treatment = treatment_en

            return diagnosis, treatment

        # Retrieve treatment documents
        chunks = retrieve_treatment_docs(crop, disease)
        
        # Apply retrieval gate
        filtered_chunks = apply_retrieval_gate(
            chunks,
            crop,
            disease
        )
        
        if not filtered_chunks:
            treatment_en = (
                f"⚠️ Limited Information\n\n"
                f"No specific treatment information found in our knowledge base for {disease} in {crop}. "
                f"However, based on general agricultural practices:\n\n"
                f"1. Isolate affected plants to prevent spread\n"
                f"2. Remove severely infected leaves\n"
                f"3. Improve air circulation\n"
                f"4. Avoid overhead irrigation\n"
                f"5. Consult local agricultural extension office for specific treatment recommendations"
            )
        else:
            # Generate treatment with filtered chunks
            treatment_en = generate_treatment(crop, disease, confidence, user_context_en, filtered_chunks)
            
            # Check faithfulness
            is_faithful, faithfulness_score = check_faithfulness(treatment_en, filtered_chunks)
            
            # Add warning if not faithful
            if not is_faithful:
                treatment_en += format_faithfulness_warning(faithfulness_score)

        alternatives = result["alternatives"]
        
        # Build comprehensive diagnosis
        diagnosis_en = (
            f"🔍 Disease Detection\n\n"
            f"Crop      : {crop}\n"
            f"Disease   : {disease}\n"
            f"Confidence: {confidence*100:.1f}%\n\n"
        )
        
        if confidence >= 0.7:
            diagnosis_en += "✅ High confidence detection\n\n"
        elif confidence >= 0.5:
            diagnosis_en += "⚠️ Moderate confidence detection\n\n"
        
        if alternatives:
            diagnosis_en += "Other possibilities considered:\n"
            diagnosis_en += "\n".join(
                f"  • {parse_label(a['label'])[1]} ({a['confidence']*100:.1f}%)"
                for a in alternatives
            )
        
        # Translate to Hindi if needed
        if detected_lang == "hi":
            diagnosis = translate_to_hindi(diagnosis_en)
            treatment = translate_to_hindi(treatment_en)
        else:
            diagnosis = diagnosis_en
            treatment = treatment_en

        return diagnosis, treatment
        
    except Exception as e:
        error_msg = f"❌ Error during analysis: {str(e)}"
        print(f"Error: {e}")
        
        if detected_lang == "hi":
            error_msg = translate_to_hindi(error_msg)
        
        return error_msg, "Please try again with a clearer image."

@spaces.GPU(duration=120)
def chatbot_response(message, history, language="Auto (स्वचालित)"):
    """Chatbot for general agricultural questions with Hindi support"""

    if not message.strip():
        return history

    # Detect language
    if language == "Auto (स्वचालित)":
        detected_lang = detect_language(message)
    elif language == "Hindi (हिंदी)":
        detected_lang = "hi"
    else:
        detected_lang = "en"

    try:
        # Translate Hindi question to English
        if detected_lang == "hi":
            message_en = translate_to_english(message)
            print("Translated message from Hindi to English")
        else:
            message_en = message

        # Build conversation history for Groq
        messages = [
            {
                "role": "system",
                "content": (
                    "You are CropPilot, an agricultural expert assistant "
                    "for Indian farmers. Answer questions clearly, practically "
                    "and accurately. You can discuss crops, diseases, soil, "
                    "irrigation, fertilizers, pests and farming practices."
                )
            }
        ]

        # Add previous conversation
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current user question
        messages.append({
            "role": "user",
            "content": message_en
        })

        # Send complete conversation to Groq
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            reasoning_effort="none"
        )

        print("GROQ RESPONSE:")
        print(response)

        bot_response_en = response.choices[0].message.content

        # Translate response back to Hindi if needed
        if detected_lang == "hi":
            bot_response = translate_to_hindi(bot_response_en)
            print("Translated response from English to Hindi")
        else:
            bot_response = bot_response_en

        # Add user message to Gradio history
        history.append({
            "role": "user",
            "content": message
        })

        # Add assistant response to Gradio history
        history.append({
            "role": "assistant",
            "content": bot_response
        })

        return history

    except Exception as e:
        error_msg = f"Error: {str(e)}"

        if detected_lang == "hi":
            error_msg = translate_to_hindi(error_msg)

        history.append({
            "role": "user",
            "content": message
        })

        history.append({
            "role": "assistant",
            "content": error_msg
        })

        return history

def support_query(name, email, issue):
    """Support form for user issues"""
    if not name or not email or not issue:
        return "Please fill in all fields."
    
    support_message = f"""
Thank you for contacting CropPilot Support!

Name: {name}
Email: {email}
Issue: {issue}

We have received your query and will respond within 24-48 hours.

For urgent matters, please contact us directly at support@croppilot.com
"""
    return support_message

# Create the main interface
with gr.Blocks(title="CropPilot", css=css) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">🌿 CropPilot</h1>
        <p style="color: #e8f5e9; margin: 10px 0 0 0; font-size: 1.1em;">
            AI-Powered Crop Disease Diagnosis · Expert Agricultural Advice
        </p>
    </div>
    """)

    # Main tabs
    with gr.Tabs() as main_tabs:
        # Disease Diagnosis Tab
        with gr.Tab("🔍 Disease Diagnosis"):
            gr.Markdown("### Upload Plant Photo for Disease Detection")

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="filepath",
                        label="",
                        height=350,
                        elem_classes=["upload-area"]
                    )

                    context_input = gr.Textbox(
                        label="Additional Context (Optional)",
                        placeholder="e.g. Maharashtra, Kharif season, irrigated field",
                        lines=2,
                        max_lines=2
                    )

                    language_input = gr.Radio(
                        choices=[
                            "Auto (स्वचालित)",
                            "English",
                            "Hindi (हिंदी)"
                        ],
                        value="Auto (स्वचालित)",
                        label="Language / भाषा"
                    )

                    submit_btn = gr.Button(
                        "🔍 Diagnose Disease",
                        variant="primary",
                        elem_classes=["diagnose-btn"],
                        size="lg"
                    )

                    gr.HTML("""
                    <div class="info-box">
                        <b style="font-size: 1.1em;">🌾 Supported Crops:</b><br>
                        Corn · Potato · Rice · Wheat<br><br>
                        <b style="font-size: 1.1em;">📚 Knowledge Base:</b><br>
                        NIPHM IPM Packages — Govt. of India<br><br>
                        <b style="font-size: 1.1em;">💡 Tips:</b><br>
                        Upload clear leaf images for best results
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Diagnosis Results")

                    diagnosis_out = gr.Textbox(
                        label="",
                        lines=10,
                        interactive=False,
                        placeholder="Upload a plant photo and click Diagnose...",
                        elem_classes=["result-text"]
                    )

                    gr.Markdown("### 💊 Treatment Plan")

                    treatment_out = gr.Textbox(
                        label="",
                        lines=20,
                        interactive=False,
                        placeholder="Treatment plan will appear here...",
                        elem_classes=["result-text"]
                    )

            submit_btn.click(
                fn=analyze_crop,
                inputs=[image_input, context_input, language_input],
                outputs=[diagnosis_out, treatment_out]
            )

        # Chatbot Tab
        with gr.Tab("💬 Chatbot"):
            gr.Markdown("### 💬 Agricultural Expert Chatbot")
            gr.Markdown(
                "Ask general agricultural questions and get expert advice"
            )

            chat_language = gr.Radio(
                choices=[
                    "Auto (स्वचालित)",
                    "English",
                    "Hindi (हिंदी)"
                ],
                value="Auto (स्वचालित)",
                label="Language / भाषा"
            )

            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                elem_classes=["chatbot-container"]
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    label="Your Question / आपका प्रश्न",
                    placeholder="Ask about crops, farming practices, diseases...",
                    scale=4,
                    max_lines=1
                )

                chat_submit = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    size="lg"
                )

            chat_submit.click(
                fn=chatbot_response,
                inputs=[chat_input, chatbot, chat_language],
                outputs=[chatbot]
            )

            chat_input.submit(
                fn=chatbot_response,
                inputs=[chat_input, chatbot, chat_language],
                outputs=[chatbot]
            )

        # Support Tab
        with gr.Tab("🛠️ Support"):
            gr.Markdown("### 🛠️ Contact Support")
            gr.Markdown("Need help? Contact our support team")

            with gr.Row():
                with gr.Column(scale=1):
                    name_input = gr.Textbox(
                        label="Your Name",
                        placeholder="Enter your name"
                    )

                    email_input = gr.Textbox(
                        label="Email Address",
                        placeholder="your@email.com"
                    )

                    issue_input = gr.Textbox(
                        label="Describe Your Issue",
                        placeholder="Please describe the problem you're facing...",
                        lines=5
                    )

                    support_submit = gr.Button(
                        "Submit Support Request",
                        variant="primary",
                        elem_classes=["diagnose-btn"],
                        size="lg"
                    )

                with gr.Column(scale=1):
                    support_output = gr.Textbox(
                        label="Support Response",
                        lines=12,
                        interactive=False,
                        placeholder="Your support request confirmation will appear here...",
                        elem_classes=["result-text"]
                    )

            support_submit.click(
                fn=support_query,
                inputs=[name_input, email_input, issue_input],
                outputs=[support_output]
            )

# Launch the app
demo.launch()