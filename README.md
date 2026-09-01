---

title: CropPilot
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
license: mit
------------

<div align="center">

# 🌿 CropPilot

### AI-powered crop disease diagnosis and grounded treatment guidance for Indian farmers

**Identify crop diseases from leaf images, retrieve relevant agricultural guidance, and get practical recommendations in English or Hindi.**

<br>

[![Live Demo](https://img.shields.io/badge/🤗_Spaces-Live_Demo-yellow)](https://huggingface.co/spaces/Hazzim010/CropPilot)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.49-orange)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<br>

**[🚀 Live Demo](https://huggingface.co/spaces/Hazzim010/CropPilot)** ·
**[🏗️ Architecture](ARCHITECTURE.md)** ·
**[📋 Project Status](PROJECT_STATUS.md)**

</div>

---

## 🌱 What is CropPilot?

CropPilot is an **AI-powered agricultural decision-support system** that combines computer vision, retrieval-augmented generation, and multilingual NLP to help identify common crop diseases and provide document-grounded treatment guidance.

The workflow is simple:

> **📷 Photograph a leaf → 🔍 Identify the disease → 📚 Retrieve relevant agricultural guidance → 💡 Generate a practical treatment plan**

Unlike a general-purpose chatbot that can freely generate agricultural advice, CropPilot's **disease-treatment workflow is grounded in official NIPHM Integrated Pest Management documentation from the Government of India**.

The LLM is therefore used primarily as a **reasoning and summarisation layer over retrieved evidence**, rather than as the source of treatment knowledge.

---

## ✨ Why CropPilot?

Agricultural advice isn't just informational. A wrong pesticide, dosage, or treatment recommendation can have real consequences for a farmer, a crop, and the environment.

CropPilot is designed around that constraint.

### 📚 Grounded treatment guidance

Treatment recommendations are generated from retrieved NIPHM documents rather than relying entirely on the model's internal knowledge.

### 🛡️ Multiple guardrails

The diagnosis pipeline applies confidence thresholds, retrieval filtering, prompt constraints, and a post-generation grounding check before presenting treatment guidance.

### 🇮🇳 Built for multilingual use

Hindi is supported throughout the diagnosis workflow, including Hindi input, English-language retrieval, and Hindi output.

### 🔍 Transparent architecture

The project separates **vision, retrieval, generation, translation, and verification** into independent components, making the system easier to understand, test, and improve.

---

# 🚀 Features

| Feature                             | Description                                                              |
| ----------------------------------- | ------------------------------------------------------------------------ |
| 🔍 **Disease Detection**            | Identify supported crop diseases from leaf photographs                   |
| 📊 **Confidence & Alternatives**    | View confidence and ranked alternative predictions                       |
| 📚 **Grounded Treatment Plans**     | Retrieve relevant guidance from NIPHM IPM documents                      |
| 🌱 **Organic & Biological Options** | Surface non-chemical treatment guidance where documented                 |
| 🧪 **Chemical Treatment Guidance**  | Provide documented pesticide recommendations and dosages when available  |
| 🛡️ **Grounding Checks**            | Flag responses with weak lexical overlap against retrieved evidence      |
| 💬 **Agricultural Chatbot**         | Ask general farming questions through a separate conversational workflow |
| 🇮🇳 **Hindi + English**            | Support both languages across the diagnosis experience                   |
| ⚡ **GPU Acceleration**              | Uses Hugging Face ZeroGPU for model inference                            |

---

# 🌾 Supported Crops & Diseases

CropPilot currently supports **four crops and eight disease classes**, alongside healthy-plant detection.

| Crop          | Supported classes                      |
| ------------- | -------------------------------------- |
| 🌽 **Corn**   | Common Rust · Gray Leaf Spot · Healthy |
| 🥔 **Potato** | Early Blight · Late Blight · Healthy   |
| 🌾 **Rice**   | Brown Spot · Leaf Blast · Healthy      |
| 🌿 **Wheat**  | Brown Rust · Yellow Rust · Healthy     |

The underlying classifier contains an additional `Invalid` class that is filtered before returning predictions.

---

# 🧠 How It Works

CropPilot has **two intentionally different AI workflows**.

## 1. 🔍 Disease Diagnosis

The diagnosis pipeline is the high-trust path.

```text
                   📷 Leaf Image
                        │
                        ▼
              ┌──────────────────┐
              │ Language Resolve │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Vision Model    │
              │      ViT         │
              │   13 classes     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Confidence Gate  │
              │      ≥ 50%       │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  FAISS Retrieval │
              │    Top-K = 5     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Retrieval Gate   │
              │ Quality +        │
              │ Crop + Disease   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   Qwen 3.6 27B   │
              │     via Groq     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Faithfulness     │
              │     Check        │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Hindi Translation│
              │   if required    │
              └────────┬─────────┘
                       │
                       ▼
                 💡 Treatment Plan
```

### The diagnosis pipeline in detail

**1. Classification**

A Vision Transformer analyzes the uploaded leaf image and produces ranked disease predictions.

**2. Confidence filtering**

Predictions below the current **50% confidence threshold** are rejected rather than presented as reliable diagnoses.

**3. Retrieval**

The predicted crop and disease are used to retrieve the top five relevant chunks from the FAISS knowledge base.

**4. Retrieval gate**

Retrieved chunks are filtered using deterministic quality and relevance checks before reaching the LLM.

**5. Grounded generation**

Qwen generates the treatment response using only the retrieved context.

**6. Faithfulness check**

The generated response is compared against the retrieved context. Weak lexical overlap produces a warning rather than silently presenting the answer as fully grounded.

**7. Translation**

When Hindi is selected, MarianMT handles the English ↔ Hindi translation required by the diagnosis workflow.

---

## 2. 💬 Conversational Q&A

The chatbot intentionally follows a different architecture:

```text
        💬 User Question
                │
                ▼
        Language Resolution
                │
                ▼
       Conversation History
                │
                ▼
          Qwen 3.6 27B
                │
                ▼
           💬 Response
```

The chatbot **does not use the disease-treatment knowledge base**.

This is deliberate.

The knowledge base is focused on crop disease and treatment protocols. General questions about soil, irrigation, weather, crop planning, or farming practices may not have useful matches in those documents.

Rather than forcing unrelated questions through retrieval, the chatbot uses Qwen directly and relies on its multilingual capabilities.

> **Trust model:** Disease diagnosis and treatment → grounded workflow.
> General farming conversation → direct LLM workflow.

---

# 🛡️ Safety & Grounding

CropPilot does not treat an LLM response as automatically trustworthy.

The diagnosis pipeline uses several layers of protection:

| Layer                        | Purpose                                                            |
| ---------------------------- | ------------------------------------------------------------------ |
| **Confidence Gate**          | Reject low-confidence visual predictions                           |
| **Retrieval Gate**           | Remove poor-quality or irrelevant document chunks                  |
| **Prompt Constraints**       | Restrict generation to supplied evidence                           |
| **Missing-Information Rule** | Tell the model to explicitly state when information is unavailable |
| **Reasoning Suppression**    | Prevent `<think>...</think>` content from reaching users           |
| **Faithfulness Check**       | Flag responses with weak overlap against retrieved context         |

The guardrails are intentionally **deterministic**.

They use thresholds and string-based rules rather than another LLM acting as a judge.

This makes them inexpensive, reproducible, and easy to test, although the current implementation has known precision limitations.

---

# 📚 Knowledge Base

The diagnosis workflow is backed by NIPHM Integrated Pest Management documents.

The current vector index contains:

* **1,109 vectors**
* **768 dimensions**
* **4 crop-specific document sets**
* **FAISS `IndexFlatL2`**
* `source` and `page` metadata retained for each chunk

### Index construction

```text
NIPHM PDFs
    │
    ▼
PyPDFLoader
    │
    ▼
RecursiveCharacterTextSplitter
    │
    ├── chunk_size: 512
    └── overlap: 64
    │
    ▼
HuggingFace Embeddings
    │
    ▼
FAISS Index
```

The prebuilt FAISS index is committed to the repository so the deployed application can start without rebuilding the knowledge base.

The source PDFs themselves are not committed.

---

# 🧩 Architecture

At a high level, CropPilot follows a **shared-core, multi-interface architecture**.

```text
                         CropPilot
                            │
              ┌─────────────┴─────────────┐
              │                           │
        ┌─────▼─────┐               ┌─────▼─────┐
        │  Gradio   │               │  FastAPI  │
        │  app.py   │               │  main.py  │
        └─────┬─────┘               └─────┬─────┘
              │                           │
              └─────────────┬─────────────┘
                            │
                    ┌───────▼───────┐
                    │   Shared Core │
                    │     src/      │
                    ├───────────────┤
                    │ Classifier    │
                    │ Retriever     │
                    │ Retrieval Gate│
                    │ Generator     │
                    │ Translator    │
                    │ Faithfulness  │
                    │ Configuration │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
       ┌──────▼──────┐             ┌──────▼──────┐
       │ FAISS Index │             │ External APIs│
       │ Model Files │             │ Groq / HF   │
       └─────────────┘             └─────────────┘
```

The two frontends do **not** communicate with each other.

* `app.py` executes the shared pipeline directly.
* `main.py` exposes REST endpoints for programmatic integration.
* The Hugging Face Space deploys `app.py`.

For the complete technical breakdown, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

# 🛠️ Tech Stack

| Layer               | Technology                                              |
| ------------------- | ------------------------------------------------------- |
| **Computer Vision** | Vision Transformer — `HurudzaAI/plantdiseasedetection1` |
| **Embeddings**      | `multi-qa-mpnet-base-dot-v1`                            |
| **Vector Search**   | FAISS                                                   |
| **LLM**             | Qwen 3.6 27B via Groq                                   |
| **Translation**     | Helsinki-NLP MarianMT                                   |
| **Frontend**        | Gradio 5.49                                             |
| **API**             | FastAPI                                                 |
| **ML Framework**    | PyTorch / Transformers                                  |
| **RAG**             | LangChain + FAISS                                       |
| **Deployment**      | Hugging Face Spaces                                     |
| **CI/CD**           | GitHub Actions                                          |

---

# 💻 Getting Started

## Prerequisites

* Python 3.10
* Git
* A [Groq API key](https://console.groq.com/)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CropPilot.git
cd CropPilot
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### 5. Start CropPilot

```bash
python app.py
```

The Gradio interface will be available at:

```text
http://127.0.0.1:7860
```

> **First launch:** expect approximately 30–60 seconds of startup time while the embedding model and FAISS index are loaded.

### Windows troubleshooting

If `python` launches the Microsoft Store or the environment reports missing modules, run:

```bash
.venv\Scripts\python.exe app.py
```

---

# 🔌 Optional REST API

CropPilot also includes a standalone FastAPI backend.

```bash
python main.py
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

| Endpoint               | Method | Description                        |
| ---------------------- | ------ | ---------------------------------- |
| `/`                    | GET    | Health check                       |
| `/api/diagnose/upload` | POST   | Image diagnosis + RAG              |
| `/api/crop-plan`       | POST   | Document-grounded agricultural Q&A |
| `/api/qa`              | POST   | Direct LLM Q&A                     |

The FastAPI backend runs independently of the Gradio application.

---

# 📦 Project Structure

```text
CropPilot/
│
├── app.py                      # Gradio application / deployed entry point
├── main.py                     # Optional FastAPI backend
│
├── src/
│   ├── config.py               # Configuration and thresholds
│   ├── classifier.py            # Vision inference
│   ├── retriever.py             # FAISS retrieval
│   ├── retrieval_gate.py        # Retrieval filtering
│   ├── generator.py             # LLM orchestration
│   ├── faithfulness.py          # Grounding verification
│   ├── translator.py            # Hindi ↔ English translation
│   ├── build_index.py           # Vector index builder
│   └── ingest.py                # PDF ingestion utilities
│
├── knowledge_base/              # Source NIPHM PDFs (not committed)
├── faiss_index/                 # Prebuilt FAISS index
│
├── .github/
│   └── workflows/
│       └── deploy.yml           # Hugging Face deployment
│
├── ARCHITECTURE.md              # Detailed technical architecture
├── PROJECT_STATUS.md            # Verified issues and project status
├── requirements.txt
└── README.md
```

---

# ⚙️ Configuration

Core parameters are centralized in `src/config.py`.

| Parameter                | Default | Description                     |
| ------------------------ | ------: | ------------------------------- |
| `TOP_K`                  |     `5` | Number of chunks retrieved      |
| `CHUNK_SIZE`             |   `512` | Chunk size used during indexing |
| `CHUNK_OVERLAP`          |    `64` | Chunk overlap during indexing   |
| `MIN_CHUNK_LENGTH`       |    `50` | Minimum accepted chunk length   |
| `FAITHFULNESS_THRESHOLD` |   `0.3` | Grounding warning threshold     |

Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` requires rebuilding the FAISS index.

---

# ⚠️ Limitations

CropPilot is an active project and has several known limitations.

### Model coverage

* Currently supports four crops.
* Designed for leaf images rather than fruits, stems, or tubers.
* Performance depends heavily on image quality.
* Low-confidence predictions are rejected.

### Retrieval

The current retrieval gate is intentionally strict, but its substring-based crop and disease matching causes uneven retrieval coverage across disease classes.

Some diagnoses therefore fall back to generic guidance instead of a fully document-grounded treatment response.

### Language

Automatic language detection can struggle with code-mixed Hinglish such as:

```text
wheat में rust
```

### Architecture

The diagnosis orchestration is currently duplicated between the Gradio and FastAPI frontends and is planned for consolidation.

### Retrieval metric

The current FAISS configuration uses an L2 index while the selected embedding model is optimized for dot-product similarity. This is planned for correction alongside the next index rebuild.

For the complete list of verified issues, see **[PROJECT_STATUS.md](PROJECT_STATUS.md)**.

---

# 🗺️ Roadmap

### Retrieval & Grounding

* [ ] Add crop synonym mapping (`Corn` ↔ `Maize`)
* [ ] Replace hard substring filters with weighted relevance scoring
* [ ] Align FAISS similarity with the embedding model's training objective
* [ ] Improve faithfulness evaluation

### Architecture

* [ ] Extract shared diagnosis orchestration into `src/`
* [ ] Remove duplicated ingestion logic
* [ ] Standardize LLM response sanitization across all endpoints

### Reliability

* [ ] Add regression tests for all disease classes
* [ ] Add linting and automated checks to CI
* [ ] Improve multilingual and code-mixed language detection

### Expansion

* [ ] Add more crops
* [ ] Expand disease coverage
* [ ] Add additional Indian regional languages

---

# 📋 Project Status

CropPilot is functional and deployed, but several architectural improvements are actively tracked.

The most important current priority is **improving retrieval coverage**, because the retrieval gate can currently reject relevant documents for several supported disease classes.

For verified findings, reproduction details, and remediation priorities:

**→ [PROJECT_STATUS.md](PROJECT_STATUS.md)**

For the complete system design:

**→ [ARCHITECTURE.md](ARCHITECTURE.md)**

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve CropPilot:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add or update tests where applicable
5. Open a pull request

For larger architectural changes, please open an issue first so the approach can be discussed.

---

# 🙏 Acknowledgements

CropPilot builds on the work of several open-source projects and organizations:

* **NIPHM, Government of India** — Integrated Pest Management packages of practice
* **HurudzaAI** — Plant disease classification model
* **Helsinki-NLP** — MarianMT translation models
* **Groq** — LLM inference
* **Hugging Face** — Models, tooling, and deployment infrastructure
* **FAISS** — Efficient vector similarity search

---

## ⚖️ Disclaimer

> **CropPilot is an AI-powered decision-support tool, not a substitute for professional agronomic advice.**
>
> AI predictions can be incorrect, and document retrieval can fail or return incomplete information. Always verify disease diagnoses, pesticide selection, dosage, and application instructions with a qualified agricultural professional or local agricultural extension office before taking action.

---

# 📄 License

CropPilot is released under the **MIT License**.
