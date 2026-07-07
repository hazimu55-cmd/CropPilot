# 🌿 CropPilot

> AI-powered crop disease diagnosis and treatment advisor — backed by official NIPHM government documents.

---

## What it does

Upload a photo of a diseased plant leaf. CropPilot identifies the disease and generates a treatment plan with organic options, chemical dosages, and prevention steps — sourced directly from NIPHM IPM packages.

---

## Tech stack

| Layer | Tool |
|-------|------|
| Disease classifier | MobileNetV2 (HuggingFace, pretrained on PlantVillage) |
| Embeddings | sentence-transformers/multi-qa-mpnet-base-dot-v1 |
| Vector store | FAISS |
| LLM | Groq API — Llama 3.3 70B |
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

**4. Launch**
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
├── app.py                  ← Gradio UI entry point
├── src/
│   ├── config.py           ← all constants
│   ├── classifier.py       ← MobileNetV2 inference
│   ├── ingest.py           ← PDF loading and chunking
│   ├── build_index.py      ← FAISS index builder (run once)
│   ├── retriever.py        ← FAISS query interface
│   └── generator.py        ← prompt builder + Groq LLM
├── knowledge_base/         ← NIPHM PDFs (gitignored)
├── faiss_index/            ← generated index (gitignored)
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