# 🌿 CropPilot

> AI-powered crop disease diagnosis and treatment advisor — backed by official NIPHM government documents.

---

## What it does

Upload a photo of a diseased plant leaf. CropPilot identifies the disease and generates a treatment plan with organic options, chemical dosages, and prevention steps — sourced directly from NIPHM IPM packages.

---

## Tech stack

| Layer | Tool |
|-------|------|
| Presentation Layer | Gradio / React UI |
| API Layer | FastAPI backend API |
| Orchestration Layer | LangGraph supervisor agent |
| Feature Service Layer | Disease AI, Crop Planner, AI Expert |
| Intelligence/Model Layer | YOLOv8 detector (PlantDoc), Embedding model (MPNet), LLM (Groq Llama 3.3 70B) |
| Reliability & Evaluation Layer | Confidence gate, Retrieval gate, Faithfulness check (RAGAS + citations) |
| Data/Knowledge Layer | FAISS / Qdrant vector store, Table-aware chunking, Knowledge base (ICAR, NIPHM, PlantDoc) |

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
API_BASE_URL=http://localhost:8000
```
Get a free key at [console.groq.com](https://console.groq.com)

**3. Build the knowledge base index** *(run once)*
```bash
python -m src.build_index
```

**4. Launch the FastAPI backend**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**5. Launch the Gradio UI** *(in a new terminal)*
```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## How it works

```
Plant photo
    ↓
MobileNetV2 → disease label + confidence
    ↓
Query FAISS index (NIPHM PDFs, chunked + embedded)
    ↓
Top 5 relevant chunks retrieved
    ↓
Groq LLM generates treatment plan from retrieved docs
    ↓
Diagnosis + treatment displayed to farmer
```

The LLM never guesses — it only summarizes what the retrieved NIPHM documents say. Low-confidence predictions (< 70%) are flagged and not sent to the LLM.

---

## Project structure

```
CropPilot/
├── app.py                  ← Gradio UI entry point (Presentation Layer)
├── api/
│   └── main.py             ← FastAPI backend API (API Layer)
├── orchestration/
│   └── supervisor.py      ← LangGraph supervisor agent (Orchestration Layer)
├── services/               ← Feature Service Layer
│   ├── disease_ai.py       ← Leaf image analysis (YOLOv8 detector)
│   ├── crop_planner.py     ← RAG cultivation plan generation
│   └── ai_expert.py        ← RAG farming Q&A
├── reliability/            ← Reliability & Evaluation Layer
│   └── gates.py            ← Confidence, retrieval, and faithfulness gates
├── src/
│   ├── config.py           ← all constants
│   ├── classifier.py       ← Legacy classifier (deprecated)
│   ├── ingest.py           ← PDF loading and table-aware chunking
│   ├── build_index.py      ← Vector index builder (run once)
│   ├── retriever.py        ← Vector store query interface
│   ├── generator.py        ← Prompt builder + Groq LLM (legacy)
│   └── vector_store.py     ← Vector store abstraction (FAISS/Qdrant)
├── knowledge_base/         ← NIPHM PDFs (gitignored)
├── faiss_index/            ← generated FAISS index (gitignored)
├── data/
│   └── uploads/            ← uploaded images (gitignored)
├── tests/
│   └── samples/            ← test images
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

## Reliability & Evaluation Features

The system includes multiple reliability gates to ensure high-quality outputs:

- **Confidence Gate**: Filters out low-confidence disease predictions (< 70%)
- **Retrieval Gate**: Filters out low-quality retrieved chunks based on relevance and content length
- **Faithfulness Check**: Verifies that LLM responses are faithful to retrieved context using RAGAS-inspired checks

These gates help prevent hallucinations and ensure only reliable information is presented to users.

---

## Vector Store Options

The system supports two vector store backends:

- **FAISS** (default): Local vector store, suitable for development and small deployments
- **Qdrant**: Production-ready vector database with better scalability

Configure in `src/config.py`:
```python
VECTOR_STORE_TYPE = "faiss"  # or "qdrant"
```

---

## Table-Aware Chunking

The ingestion pipeline includes table-aware chunking to preserve table structures from PDFs. Tables are detected and kept intact as separate chunks, preventing data loss during the chunking process.

---

## Knowledge base

All treatment data comes from official **NIPHM IPM Packages** — National Institute of Plant Health Management, Government of India.

| Crop | Source |
|------|--------|
| Rice | NIPHM Rice IPM Package |
| Wheat | NIPHM Wheat IPM Package |
| Maize | NIPHM Maize IPM Package |
| Potato | NIPHM Potato IPM Package |
| Cotton | NIPHM Cotton IPM Package |

---

## Disclaimer

This tool provides recommendations based on official agricultural documents. Always consult a local agricultural expert before applying treatments. The developers are not responsible for crop damage or loss.

---

## License

MIT — code. Apache 2.0 — model weights. NIPHM documents are public domain (Government of India).