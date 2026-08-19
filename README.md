# 🌿 CropPilot

> AI-powered crop disease diagnosis and treatment advisor — backed by official NIPHM government documents. Now with **Hindi language support**!

---

## What it does

Upload a photo of a diseased plant leaf. CropPilot identifies the disease and generates a treatment plan with organic options, chemical dosages, and prevention steps — sourced directly from NIPHM IPM packages.

**New Features:**
- 🇮🇳 **Hindi language support** - Ask questions in Hindi, get answers in Hindi
- 🔄 **Automatic translation** - Hindi ↔ English translation powered by AI
- 🚀 **FastAPI backend** - RESTful API for integration with other systems
- 🛡️ **Retrieval gate** - Filters weak/low-quality chunks for better accuracy
- ✅ **Faithfulness check** - Validates responses against source documents

---

## Tech stack

| Layer | Tool |
|-------|------|
| Disease classifier | ViT (HurudzaAI/plantdiseasedetection1) |
| Embeddings | sentence-transformers/multi-qa-mpnet-base-dot-v1 |
| Vector store | FAISS |
| LLM | Groq API — Qwen 3.6 27B |
| Translation | Helsinki-NLP/opus-mt (Hindi ↔ English) |
| API | FastAPI + Uvicorn |
| Orchestration | LangChain |
| UI | Gradio |

---

## Supported crops

Rice · Wheat · Maize · Potato · Cotton

> Upload **leaf images only** — not fruits, tubers, or stems.

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/hazimu55-cmd/CropPilot.git
cd CropPilot
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

**2. Add your Groq API key**

Create a `.env` file in the root:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**3. Build the knowledge base index** *(run once)*
```bash
python -m src.build_index
```

**4. Launch options**

**Option A: Gradio UI (Recommended for farmers)**
```bash
python app.py
```
Open `http://127.0.0.1:7863` in your browser.

**Option B: FastAPI Server (For developers/integration)**
```bash
python main.py
```
API documentation available at `http://localhost:8000/docs`

---

## How it works

```
Plant photo (Hindi/English)
    ↓
Language detection & translation (if needed)
    ↓
ViT classifier → disease label + confidence
    ↓
Query FAISS index (NIPHM PDFs, chunked + embedded)
    ↓
Retrieval gate (filter weak chunks)
    ↓
Top 5 relevant chunks retrieved
    ↓
Groq LLM generates treatment plan from retrieved docs
    ↓
Faithfulness check (word-overlap validation)
    ↓
Translation back to Hindi (if input was Hindi)
    ↓
Diagnosis + treatment displayed to farmer
```

The LLM never guesses — it only summarizes what the retrieved NIPHM documents say. Low-confidence predictions (< 70%) are flagged and not sent to the LLM.

---

## API Endpoints

### Disease Diagnosis
**POST** `/api/diagnose/upload`
- Upload plant image for disease diagnosis
- Supports Hindi context input
- Returns Hindi or English output based on input language

### Crop Planning Q&A
**POST** `/api/crop-plan`
- Ask general agricultural questions
- Uses FAISS retrieval for document-based answers
- Automatic Hindi/English translation

### General Q&A
**POST** `/api/qa`
- General agricultural questions without document retrieval
- Good for simple queries

---

## Project structure

```
CropPilot/
├── app.py                  ← Gradio UI entry point
├── main.py                 ← FastAPI application
├── src/
│   ├── config.py           ← all constants
│   ├── classifier.py       ← ViT inference
│   ├── translator.py       ← Hindi ↔ English translation
│   ├── retrieval_gate.py   ← Chunk quality filtering
│   ├── faithfulness.py     ← Response validation
│   ├── ingest.py           ← PDF loading and chunking
│   ├── build_index.py      ← FAISS index builder (run once)
│   ├── retriever.py        ← FAISS query interface
│   └── generator.py        ← prompt builder + Groq LLM
├── knowledge_base/         ← NIPHM PDFs (gitignored)
├── faiss_index/            ← generated index (gitignored)
├── .env                    ← API keys (gitignored)
└── requirements.txt
```

---

## Adding more crops

Drop any new NIPHM PDF into `knowledge_base/` and rebuild the index:
```bash
python -m src.build_index
```

No code changes needed.

---

## Knowledge base

All treatment data comes from official **NIPHM IPM Packages** — National Institute of Plant Health Management, Government of India.

| Crop | Source |
|------|--------|
| Rice | NIPHM Rice IPM Package |
| Wheat | NIPHM Wheat IPM Package |
| Maize | NIPHM Maize IPM Package |
| Potato | NIPHM Potato IPM Package |

---

## Hindi Language Support

The system now supports Hindi input and output:

**Gradio UI:**
- Select language: "Auto (स्वचालित)", "English", or "Hindi (हिंदी)"
- Type questions in Hindi or English
- System automatically detects and translates
- Responses are provided in the same language as input

**API:**
- Use `language` parameter: `"auto"`, `"hi"`, or `"en"`
- System handles translation automatically
- Faithfulness checks work on both languages

---

## Disclaimer

This tool provides recommendations based on official agricultural documents. Always consult a local agricultural expert before applying treatments. The developers are not responsible for crop damage or loss.

---

## License

MIT — code. Apache 2.0 — model weights. NIPHM documents are public domain (Government of India).