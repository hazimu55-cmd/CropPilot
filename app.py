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




@spaces.GPU(duration=120)
def analyze_crop(image, user_context, language="Auto (स्वचालित)"):

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



    result = classify_disease(image)

    top = result["top_prediction"]

    crop, disease = parse_label(top["label"])

    confidence = top["confidence"]



    print(f"Classified: {crop} - {disease} ({confidence*100:.1f}%")



    if disease.lower() == "healthy":

        diagnosis_en = (

            f"Crop      : {crop}\n"

            f"Status    : Healthy ✓\n"

            f"Confidence: {confidence*100:.1f}%\n\n"

            f"No disease detected. Your plant looks good!"

        )

        treatment_en = "✅ No treatment needed. Keep monitoring your plant regularly."
        
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
    filtered_chunks = apply_retrieval_gate(chunks)
    
    if not filtered_chunks:
        treatment_en = "No treatment information found in knowledge base for this disease."
    else:
        # Generate treatment with filtered chunks
        treatment_en = generate_treatment(crop, disease, confidence, user_context_en, filtered_chunks)
        
        # Check faithfulness
        is_faithful, faithfulness_score = check_faithfulness(treatment_en, filtered_chunks)
        
        # Add warning if not faithful
        if not is_faithful:
            treatment_en += format_faithfulness_warning(faithfulness_score)



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
    
    # Translate to Hindi if needed
    if detected_lang == "hi":
        diagnosis = translate_to_hindi(diagnosis_en)
        treatment = translate_to_hindi(treatment_en)
    else:
        diagnosis = diagnosis_en
        treatment = treatment_en

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



with gr.Blocks(title="CropPilot") as demo:\


    gr.Markdown("# 🌿 CropPilot")
    gr.Markdown("AI-powered crop disease diagnosis · Backed by official NIPHM documents")



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

                placeholder="e.g. Maharashtra, Kharif season, irrigated field / उदाहरण: महाराष्ट्र, खरीफ मौसम, सिंचित खेत",

                lines=2

            )
            
            language_input = gr.Radio(
                choices=["Auto (स्वचालित)", "English", "Hindi (हिंदी)"],
                value="Auto (स्वचालित)",
                label="Language / भाषा"
            )

            submit_btn = gr.Button(

                "🔍 Diagnose",

                variant="primary",

                elem_classes="diagnose-btn"

            )

            gr.Markdown("""

            <div style="margin-top:12px; padding:10px; background:#1a2e1a; border-radius:8px; font-size:13px; color:#aaa">

            <b style="color:#7ec87e">Supported crops:</b><br>

            Rice · Wheat · Maize · Potato<br><br>

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

        inputs=[image_input, context_input, language_input],

        outputs=[diagnosis_out, treatment_out]

    )


# Chatbot Interface
with gr.Blocks(title="CropPilot Chatbot") as chatbot_tab:
    gr.Markdown("# 💬 CropPilot Chatbot")
    gr.Markdown("Ask general agricultural questions · Get expert advice")
    
    chat_language = gr.Radio(
        choices=["Auto (स्वचालित)", "English", "Hindi (हिंदी)"],
        value="Auto (स्वचालित)",
        label="Language / भाषा"
    )
    
    chatbot = gr.Chatbot(
        height=500
    )
    
    with gr.Row():
        chat_input = gr.Textbox(
            label="Your question / आपका प्रश्न",
            placeholder="Ask about crops, farming practices, diseases... / फसलों, खेती प्रथाओं, रोगों के बारे में पूछें...",
            scale=4
        )
        chat_submit = gr.Button("Send", variant="primary", scale=1)
    
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


# Support Interface
with gr.Blocks(title="CropPilot Support") as support_tab:
    gr.Markdown("# 🛠️ CropPilot Support")
    gr.Markdown("Need help? Contact our support team")
    
    with gr.Row():
        with gr.Column(scale=1):
            name_input = gr.Textbox(label="Your Name", placeholder="Enter your name")
            email_input = gr.Textbox(label="Email Address", placeholder="your@email.com")
            issue_input = gr.Textbox(
                label="Describe your issue",
                placeholder="Please describe the problem you're facing...",
                lines=5
            )
            support_submit = gr.Button("Submit Support Request", variant="primary", elem_classes="diagnose-btn")
        
        with gr.Column(scale=1):
            support_output = gr.Textbox(
                label="Support Response",
                lines=10,
                interactive=False,
                placeholder="Your support request confirmation will appear here..."
            )
    
    support_submit.click(
        fn=support_query,
        inputs=[name_input, email_input, issue_input],
        outputs=[support_output]
    )


# Combine all tabs
demo = gr.TabbedInterface(
    [demo, chatbot_tab, support_tab],
    ["🔍 Disease Diagnosis", "💬 Chatbot", "🛠️ Support"]
)

demo.launch()