# CropPilot — Architecture

Technical reference for the CropPilot system.

* **Setup & usage:** [README.md](README.md)
* **Current project status:** [PROJECT_STATUS.md](PROJECT_STATUS.md)

---

## Contents

1. [Design Goals](#1-design-goals)
2. [System Architecture](#2-system-architecture)
3. [Offline Pipeline — Index Construction](#3-offline-pipeline--index-construction)
4. [Online Pipeline — Disease Diagnosis](#4-online-pipeline--disease-diagnosis)
5. [Online Pipeline — Conversational Q&A](#5-online-pipeline--conversational-qa)
6. [Component Reference](#6-component-reference)
7. [Cross-Cutting Concerns](#7-cross-cutting-concerns)
8. [Deployment](#8-deployment)
9. [Design Decisions & Trade-offs](#9-design-decisions--trade-offs)
10. [Known Issues](#10-known-issues)

---

## 1. Design Goals

CropPilot is designed around a simple constraint: **incorrect agricultural advice can cause real-world harm.**

A fabricated pesticide dosage, for example, could damage a crop or put the person applying it at risk. The architecture therefore prioritizes grounding, explicit failure states, and multilingual accessibility.

### Grounded where it matters

The LLM is used to **summarize retrieved agricultural documents**, not as the primary source of treatment knowledge.

Treatment recommendations are expected to be traceable to official NIPHM documentation.

### Fail safely

When the system cannot produce a sufficiently reliable answer, it should say so rather than generate a plausible-looking response.

Examples include:

* Rejecting low-confidence disease classifications
* Declaring when relevant documentation cannot be retrieved
* Warning when generated content has weak overlap with retrieved context

### Support farmers in their language

Hindi is treated as a first-class input and output language throughout the diagnosis workflow rather than as an afterthought.

### Prefer deterministic guardrails

The primary safety and relevance gates use **deterministic rules rather than LLM-based judges**.

This keeps the system:

* Cheap to run
* Easy to debug
* Independently testable
* Deterministic

The trade-off is reduced precision, particularly for substring-based relevance checks. See [§10.1](#101-h1--retrieval-gate-discards-nearly-all-chunks).

---

# 2. System Architecture

CropPilot has **two independent frontends** backed by a shared `src/` package.

```text
                         ┌─────────────────────────────┐
                         │          CropPilot           │
                         │         Shared Core         │
                         │            src/             │
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 │                                             │
        ┌────────▼────────┐                           ┌────────▼────────┐
        │     app.py      │                           │    main.py      │
        │     Gradio      │                           │    FastAPI      │
        │                 │                           │                 │
        │  Port 7860      │                           │  Port 8000      │
        │  Deployed UI    │                           │  REST API       │
        └────────┬────────┘                           └────────┬────────┘
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   Shared Modules  │
                              │                   │
                              │ classifier        │
                              │ retriever         │
                              │ retrieval_gate    │
                              │ generator         │
                              │ translator        │
                              │ faithfulness      │
                              │ config            │
                              └─────────┬─────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                ┌────────▼────────┐          ┌────────▼────────┐
                │ Local Artifacts │          │ External APIs   │
                │                 │          │                 │
                │ FAISS Index     │          │ Groq            │
                │ Model Weights   │          │ Hugging Face    │
                └─────────────────┘          └─────────────────┘
```

### Frontends

| Frontend  | Technology | Port | Role                                           |
| --------- | ---------- | ---: | ---------------------------------------------- |
| `app.py`  | Gradio     | 7860 | Primary user-facing application                |
| `main.py` | FastAPI    | 8000 | Optional REST API for programmatic integration |

The two frontends **do not communicate over HTTP**.

The Gradio application imports `src/` directly and executes the pipeline in-process. The FastAPI application exposes similar functionality through REST endpoints.

The Hugging Face Space runs `app.py`; `main.py` is intended for programmatic integration.

---

## 2.1 Request Paths

The Gradio application exposes two intentionally different workflows.

|                     | Disease Diagnosis     | Conversational Q&A        |
| ------------------- | --------------------- | ------------------------- |
| Image input         | Yes                   | No                        |
| FAISS retrieval     | Yes                   | No                        |
| Retrieval gate      | Yes                   | No                        |
| Faithfulness check  | Yes                   | No                        |
| Hindi translation   | MarianMT              | LLM-native                |
| Knowledge grounding | NIPHM documents       | Ungrounded                |
| Primary purpose     | Diagnosis + treatment | General farming questions |

The two workflows have different trust requirements and therefore intentionally use different pipelines.

---

# 3. Offline Pipeline — Index Construction

`src/build_index.py` creates the vector database used by the diagnosis pipeline.

The process is run when the source documents change.

```text
knowledge_base/*.pdf
        │
        │  PyPDFLoader
        ▼
Document per page
        │
        │  RecursiveCharacterTextSplitter
        │  chunk_size=512
        │  overlap=64
        ▼
Text chunks
        │
        │  HuggingFaceEmbeddings
        │  multi-qa-mpnet-base-dot-v1
        │  768 dimensions
        ▼
Embeddings
        │
        │  FAISS
        ▼
faiss_index/
├── index.faiss
└── index.pkl
```

### Current index

* **Vectors:** 1,109
* **Dimensions:** 768
* **Index:** `IndexFlatL2`
* **Search:** Exhaustive

At this scale, exhaustive search is appropriate. With only 1,109 vectors, approximate indexes such as IVF or HNSW would add complexity without providing a meaningful performance benefit.

Each chunk retains:

* `source` — source PDF filename
* `page` — source page number

This metadata is passed into the generation pipeline so treatment recommendations can be traced back to the underlying NIPHM document.

> **Repository note:** `faiss_index/` is committed to Git even though it matches a `.gitignore` rule because the files were tracked before the rule was introduced. This allows the Hugging Face Space to start without rebuilding the index.
>
> `knowledge_base/` is not committed, so a fresh clone cannot regenerate the index without obtaining the source PDFs.

---

# 4. Online Pipeline — Disease Diagnosis

The primary diagnosis workflow starts in `analyze_crop()` inside `app.py`.

```text
Image + optional context + language
                 │
                 ▼
        ┌─────────────────┐
        │ Language Resolve │
        │                 │
        │ Auto-detect or  │
        │ user selection  │
        └────────┬────────┘
                 │
                 │ Hindi → English
                 ▼
        ┌─────────────────┐
        │   ViT Classifier│
        │                 │
        │ 13 logits       │
        │ → softmax       │
        │ → top-3         │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Confidence    │
        │     Gate        │
        └────────┬────────┘
                 │
        ┌────────┴─────────┐
        │                  │
   < 0.5 confidence    Healthy
        │                  │
        ▼                  ▼
   Early exit          Early exit
        │
        ▼
        ┌─────────────────┐
        │  Label Parsing  │
        │                 │
        │ Crop + Disease  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ FAISS Retrieval │
        │      top-5      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Retrieval Gate  │
        │                 │
        │ 5 deterministic  │
        │ filters         │
        └────────┬────────┘
                 │
           ┌─────┴─────┐
           │           │
        No chunks    Chunks
           │           │
           ▼           ▼
    Generic advice   Groq LLM
     No LLM call        │
                        ▼
                 Faithfulness Check
                        │
                        ▼
                 Hindi translation
                 if required
                        │
                        ▼
                  Final response
```

### Pipeline stages

#### 1. Language resolution

The system determines the requested language using the selector or automatic detection.

Automatic detection currently uses the ratio of Devanagari characters (`U+0900–097F`) to alphabetic characters.

Hindi input is translated from Hindi → English before entering the English-language retrieval pipeline.

#### 2. Disease classification

`src/classifier.py` uses the `HurudzaAI/plantdiseasedetection1` Vision Transformer.

The classifier:

1. Produces 13 logits
2. Applies softmax
3. Removes the `Invalid` class
4. Sorts predictions
5. Returns the top three classes

#### 3. Confidence gate

Predictions below **0.5 confidence** are rejected before retrieval or LLM generation.

Healthy classifications also exit early because treatment retrieval is unnecessary.

#### 4. Label parsing

Raw model labels such as:

```text
Corn___Common_Rust
```

are converted into:

```text
Crop: Corn
Disease: Common Rust
```

A fallback chain handles variations in upstream label formatting.

#### 5. Retrieval

The disease and crop are converted into a retrieval query:

```text
treatment and management of {disease} in {crop} plants
```

The FAISS retriever returns the top five chunks.

#### 6. Retrieval gate

Retrieved chunks pass through five deterministic filters.

| # | Filter            | Rule                             | Purpose                             |
| - | ----------------- | -------------------------------- | ----------------------------------- |
| 1 | Length            | `len(content) >= 50`             | Remove fragments                    |
| 2 | Legibility        | Alphanumeric/space ratio ≥ 0.5   | Remove extraction artefacts         |
| 3 | Domain            | Contains ≥1 agricultural keyword | Remove boilerplate                  |
| 4 | Crop relevance    | Crop appears in chunk            | Prevent cross-crop grounding        |
| 5 | Disease relevance | Disease variant appears in chunk | Prevent unrelated disease grounding |

A chunk must pass **all five filters** to reach the LLM.

#### 7. Generation

If relevant chunks remain, `src/generator.py` sends them to Groq using:

* **Model:** `qwen/qwen3.6-27b`
* **Temperature:** `0.2`
* **Max tokens:** `1000`
* **Reasoning effort:** `none`

The generation prompt requires the model to:

* Use only supplied documents
* Avoid inventing pesticide names or dosages
* State when information is unavailable
* Avoid exposing reasoning

The response is additionally sanitized for `<think>...</think>` blocks.

#### 8. Faithfulness check

`src/faithfulness.py` calculates a simple word-overlap score between the generated response and retrieved context.

If the score falls below `0.3`, a warning is appended.

The check is **advisory rather than blocking**.

#### 9. Translation

If Hindi output is requested, MarianMT translates the English response back into Hindi.

Long responses are split into approximately 900-character chunks at paragraph and sentence boundaries.

---

# 5. Online Pipeline — Conversational Q&A

The chatbot workflow starts in `chatbot_response()`.

```text
Message + conversation history
              │
              ▼
       Language resolution
              │
              ▼
       Build message array
              │
              ├── System prompt
              ├── Conversation history
              └── Current message
              │
              ▼
          Groq / Qwen
              │
              ▼
        Updated history
```

The chatbot intentionally differs from the diagnosis pipeline.

### No retrieval

The knowledge base focuses on disease treatment protocols.

General questions about topics such as:

* Soil
* Weather
* Crop planning
* Farming practices

may not have useful matches in the knowledge base.

Forcing every question through retrieval could therefore introduce irrelevant grounding.

The chatbot is consequently treated as a **general conversational assistant**, rather than a grounded treatment advisor.

### No translation layer

The chatbot sends the user's original message directly to Qwen.

Because Qwen is multilingual, translating Hindi → English → Hindi would introduce an unnecessary second translation step.

Language is instead enforced through the system prompt.

---

# 6. Component Reference

## `src/config.py`

Central configuration for:

* Model identifiers
* Chunking parameters
* Retrieval settings
* Confidence thresholds
* Faithfulness thresholds

| Constant                 | Value | Scope              |
| ------------------------ | ----: | ------------------ |
| `CHUNK_SIZE`             |   512 | Index construction |
| `CHUNK_OVERLAP`          |    64 | Index construction |
| `TOP_K`                  |     5 | Query time         |
| `MIN_CHUNK_LENGTH`       |    50 | Retrieval gate     |
| `FAITHFULNESS_THRESHOLD` |   0.3 | Grounding warning  |

Paths are relative to the project root.

---

## `src/classifier.py`

Vision inference using `HurudzaAI/plantdiseasedetection1`.

The model is loaded lazily and uses CUDA when available.

Supported classes:

| Crop   | Diseases / States                    |
| ------ | ------------------------------------ |
| Corn   | Common Rust, Gray Leaf Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy   |
| Rice   | Brown Spot, Leaf Blast, Healthy      |
| Wheat  | Brown Rust, Yellow Rust, Healthy     |
| —      | Invalid                              |

The classifier returns the top prediction and alternatives. Confidence gating is handled by the caller.

---

## `src/retriever.py`

Responsible for:

* Loading the embedding model
* Loading the FAISS index
* Performing vector search
* Returning content and source metadata

The embedding model and index are currently loaded eagerly at import time.

The FAISS index uses pickle-backed deserialization and therefore requires:

```text
allow_dangerous_deserialization=True
```

This is acceptable only because the index is a trusted, self-built artifact.

---

## `src/retrieval_gate.py`

Applies deterministic relevance and quality filtering to retrieved chunks.

```text
apply_retrieval_gate(chunks, crop, disease)
```

All three parameters are required.

See [§4](#4-online-pipeline--disease-diagnosis) for the filtering rules.

---

## `src/generator.py`

Responsible for LLM orchestration.

It:

1. Builds the grounded prompt
2. Calls Groq
3. Applies response constraints
4. Removes `<think>...</think>` blocks

The function accepts already-filtered chunks to avoid duplicate retrieval.

---

## `src/faithfulness.py`

Performs a lightweight grounding check using word overlap between the response and retrieved context.

Returns:

```text
(is_faithful, overlap_ratio)
```

The result is advisory and does not block the response.

---

## `src/translator.py`

Provides Hindi ↔ English translation using MarianMT.

### Detection

Devanagari ratio > `0.3` → Hindi.

### Translation

Long text is split into approximately 900-character chunks.

The decoder uses:

* `num_beams=4`
* `no_repeat_ngram_size=3`
* `early_stopping=True`

The n-gram constraint helps prevent repetitive translation loops.

### Failure handling

Translation exceptions return the original text rather than terminating the request.

The module also provides Windows UTF-8 stream handling.

---

## `src/build_index.py` / `src/ingest.py`

Responsible for document loading and chunking during index construction.

`ingest.py` currently duplicates part of the index-building logic and is not imported elsewhere.

---

## `app.py`

Gradio-based user interface.

The application contains two tabs:

### Disease Diagnosis

* Image upload
* Optional context
* Language selection
* Diagnosis output
* Treatment output

### Chatbot

* Message history
* Language selection
* Enter/click submission

The application launches on Gradio's default port:

```text
7860
```

---

## `main.py`

Optional FastAPI backend running through Uvicorn on port `8000`.

| Endpoint               | Method | Retrieval | Purpose               |
| ---------------------- | ------ | --------- | --------------------- |
| `/`                    | GET    | —         | Health check          |
| `/api/diagnose/upload` | POST   | Yes       | Image diagnosis       |
| `/api/crop-plan`       | POST   | Yes       | Document-grounded Q&A |
| `/api/qa`              | POST   | No        | Direct LLM Q&A        |

Uploads are stored in temporary files and cleaned up in a `finally` block.

---

# 7. Cross-Cutting Concerns

## Model Loading

| Component          | Loading        | Device              |
| ------------------ | -------------- | ------------------- |
| Embeddings + FAISS | Eager          | CPU                 |
| ViT classifier     | Lazy singleton | CUDA when available |
| MarianMT ×2        | Lazy           | CUDA when available |
| Groq client        | Lazy singleton | Remote              |

The embedding model currently dominates cold-start time because it is loaded even for chatbot-only sessions.

## Performance

Approximate current characteristics:

* **Cold start:** 30–60 seconds
* **Classification:** sub-second on GPU
* **LLM generation:** network-bound
* **Hindi output:** slowest path due to MarianMT beam search
* **Memory:** approximately 2–3 GB RAM in CPU mode with all models resident

## Error Handling

Handlers currently use broad exception handling and return user-facing fallback messages.

Examples:

* Translation failure → original text
* Retrieval failure → generic advice
* Classification failure → error response

This improves resilience but can also mask programming errors. See [§10.2](#102-h2-index-rebuild-is-broken) and the related API issues below.

## Security

### Secrets

`GROQ_API_KEY` is supplied through:

* `.env` locally
* Hugging Face Space secrets in production

Secrets are never stored in source code.

### FAISS deserialization

`allow_dangerous_deserialization=True` is safe only when loading a trusted, self-generated index.

### File uploads

Uploaded images are stored temporarily and removed after processing.

Explicit MIME-type and file-size validation is currently limited.

---

# 8. Deployment

| Aspect            | Configuration                           |
| ----------------- | --------------------------------------- |
| Platform          | Hugging Face Spaces                     |
| Space             | `Hazzim010/CropPilot`                   |
| SDK               | Gradio 5.49.1                           |
| Entry point       | `app.py`                                |
| CI/CD             | `.github/workflows/deploy.yml`          |
| Deployment action | `huggingface/hub-sync@v0.1.0`           |
| Trigger           | Push to `main`                          |
| Authentication    | `HF_TOKEN` secret                       |
| GPU               | ZeroGPU via `@spaces.GPU(duration=120)` |

The embedding model is intentionally pinned to CPU regardless of GPU availability.

---

# 9. Design Decisions & Trade-offs

### Shared core with two frontends

Gradio provides the primary farmer-facing experience while FastAPI enables programmatic integration.

Both reuse `src/` to avoid duplicating model implementations.

**Trade-off:** orchestration logic currently exists in both frontends and has begun to diverge.

### Grounded diagnosis, ungrounded chat

Treatment recommendations require traceable evidence, while general conversational questions benefit from the broader knowledge of the LLM.

The architecture therefore gives the two workflows different trust models.

### Translate for retrieval, not for chat

The knowledge base is English-language, so Hindi diagnosis input must be translated before retrieval.

The chatbot does not require this step because Qwen supports multilingual input directly.

### Deterministic guardrails

Thresholds and string matching are preferred over LLM-based evaluators.

**Benefits:**

* Predictable
* Cheap
* Fast
* Testable

**Trade-off:** deterministic substring matching can be brittle.

### Advisory faithfulness check

A weak grounding score produces a warning rather than suppressing the answer.

This preserves useful information while communicating uncertainty.

### Early confidence gate

Low-confidence classifications are rejected before retrieval and generation, reducing unnecessary latency and API usage.

### Flat FAISS index

At 1,109 vectors, exhaustive search provides simple and exact retrieval without the complexity of an approximate index.

---

# 10. Known Issues

The following issues were verified against the current codebase during the **2026-09-01** check.

Detailed reproduction steps and remediation priorities are documented in [PROJECT_STATUS.md](PROJECT_STATUS.md).

| ID        | Severity | Issue                                                                                |
| --------- | -------- | ------------------------------------------------------------------------------------ |
| **H1**    | High     | Retrieval gate discards most disease chunks                                          |
| **H2**    | High     | Index rebuild currently fails                                                        |
| **H3**    | High     | Two REST endpoints return HTTP 500                                                   |
| **M1**    | Medium   | Translator can fail during import under captured stdout                              |
| **M2**    | Medium   | `/api/qa` can expose raw `<think>` output                                            |
| **M3**    | Medium   | Code-mixed Hinglish can be detected as English                                       |
| **M4**    | Medium   | Embedding model and FAISS distance metric are mismatched                             |
| **M5**    | Medium   | Diagnosis orchestration is duplicated across frontends                               |
| **M6**    | Medium   | Faithfulness metric is a weak grounding proxy                                        |
| **M7**    | Medium   | `/api/crop-plan` reloads retrieval resources per request                             |
| **L1–L5** | Low      | Lint issues, environment pollution, deprecated imports, dead code, and missing tests |

---

## H1 — Retrieval Gate Discards Most Chunks

The retrieval gate requires the classifier's crop name to appear verbatim in the retrieved chunk.

The classifier emits:

```text
Corn
```

while the knowledge base uses:

```text
Maize
```

No crop synonym mapping currently exists.

Disease matching is similarly dependent on exact substring matches.

### Current results

| Label                   | Retrieved | After gate |
| ----------------------- | --------: | ---------: |
| `Corn___Common_Rust`    |         5 |      **0** |
| `Corn___Gray_Leaf_Spot` |         5 |      **0** |
| `Potato___Early_Blight` |         5 |      **1** |
| `Potato___Late_Blight`  |         5 |      **0** |
| `Rice___Brown_Spot`     |         5 |      **0** |
| `Rice___Leaf_Blast`     |         5 |      **0** |
| `Wheat___Brown_Rust`    |         5 |      **0** |
| `Wheat___Yellow_Rust`   |         5 |      **0** |

Six of eight disease classes currently retain no chunks.

When that happens, the diagnosis pipeline falls back to generic advice and skips the LLM.

**Impact:** the intended RAG pipeline is effectively bypassed for most supported diseases.

**Recommended direction:**

* Introduce crop synonym mapping
* Replace strict crop/disease substring filters with weighted relevance scoring
* Add a tunable relevance threshold

---

## H2 — Index Rebuild Currently Fails

`build_index.py` and `ingest.py` use the deprecated:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

With the current LangChain version, this results in:

```text
ModuleNotFoundError
```

The correct import is available through `langchain_text_splitters`.

Because the existing FAISS index is committed, this issue remains hidden during normal deployment.

**Impact:** the offline indexing pipeline cannot currently be reproduced from source.

**Recommended direction:**

Update the import to:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---

## H3 — Two REST Endpoints Return HTTP 500

`apply_retrieval_gate()` requires:

```text
(chunks, crop, disease)
```

but `main.py` currently calls it with only one argument in two locations.

This produces a `TypeError`, which is caught by the broad request handler and returned as HTTP 500.

Affected endpoints:

* `/api/diagnose/upload`
* `/api/crop-plan`

This issue is a direct consequence of the duplicated orchestration described in **M5**.

---

## M1 — Translator Import Failure

`translator.py` previously reassigned `sys.stdout` directly through `.buffer`.

This fails when stdout has been replaced by a capture or redirect object without a binary buffer.

Affected environments include:

* Pytest
* `redirect_stdout`
* Notebooks
* Some ASGI environments

The UTF-8 stream handling has since been isolated into defensive helpers, but this area should remain covered by tests.

---

## M2 — `/api/qa` Reasoning Leakage

The Gradio chatbot and `src/generator.py` suppress model reasoning through:

* `reasoning_effort="none"`
* `<think>...</think>` post-processing

The direct `/api/qa` implementation does not currently apply the same protections.

**Impact:** model-generated `<think>` content may be returned directly to API consumers.

**Recommended direction:** centralize LLM invocation and response sanitization in the shared core.

---

## M3 — Hinglish Detection Gap

Automatic language detection is based on the proportion of Devanagari characters.

For example:

```text
wheat में rust
```

contains enough English text that its Devanagari ratio falls below the current `0.3` threshold.

This causes it to be classified as English.

Pure Hindi, pure English, empty, and numeric inputs currently behave as expected.

**Recommended direction:** improve language detection to handle common code-mixed Hindi/English input.

---

## M4 — Embedding / Distance Metric Mismatch

`multi-qa-mpnet-base-dot-v1` is designed around dot-product similarity, while the current FAISS index uses:

```text
IndexFlatL2
```

with unnormalized embeddings.

This can produce rankings that do not align with the embedding model's intended similarity metric.

**Recommended direction:**

* Normalize embeddings and use an appropriate similarity configuration, or
* Use `DistanceStrategy.MAX_INNER_PRODUCT`

Changing the index requires rebuilding it, so **H2 should be addressed first**.

---

## M5 — Duplicated Diagnosis Orchestration

The diagnosis workflow is implemented separately in:

* `app.py`
* `main.py`

The two implementations have already diverged, resulting in issues such as H3 and M2 appearing in only one path.

`ingest.py` also duplicates part of the indexing logic.

**Recommended direction:**

Extract a shared orchestration function such as:

```text
src/diagnosis.py
```

Both frontends should call the same pipeline.

---

## M6 — Faithfulness Metric Is a Weak Grounding Proxy

The current faithfulness implementation uses simple set-based word overlap.

It does not account for:

* Stopwords
* Term frequency
* Inverse document frequency
* Semantic similarity

As a result, common words can inflate the score while a fluent fabricated response may still appear sufficiently grounded.

A `combined_context` variable in `main.py` is also currently computed but unused.

**Recommended direction:** replace or supplement lexical overlap with a stronger grounding metric.

---

## M7 — Per-Request Retrieval Resource Reloading

`/api/crop-plan` currently reconstructs the embedding model and reloads the FAISS index for each request instead of reusing the shared retriever.

**Impact:**

* Higher latency
* Increased memory pressure
* Unnecessary disk I/O

**Recommended direction:** reuse the retriever singleton from `src/retriever.py`.

---

## Low-Severity Issues

Additional low-priority issues include:

* Ruff lint findings
* Virtual-environment package pollution
* Deprecated imports
* Dead code
* Duplicate ingestion logic
* Missing automated test coverage
* Missing lint/test CI checks

The absence of automated checks is particularly important because several of the issues above could have been detected before deployment.

---

## Architecture Summary

CropPilot follows a **shared-core, multi-interface architecture**:

```text
                     ┌──────────────┐
                     │   Gradio UI  │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │              │
                     │  Shared Core │
                     │              │
                     │ Vision       │
                     │ Retrieval    │
                     │ Guardrails   │
                     │ Generation   │
                     │ Translation  │
                     │ Verification │
                     │              │
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │   FastAPI    │
                     │     API      │
                     └──────────────┘

        Diagnosis
            │
            ├── Vision classification
            ├── FAISS retrieval
            ├── Deterministic relevance gate
            ├── Grounded LLM generation
            └── Faithfulness verification

        Chat
            │
            └── Direct multilingual LLM conversation
```

The central architectural goal is to keep **high-risk treatment recommendations grounded and auditable**, while allowing the conversational interface to remain flexible for broader farming questions.

The most important architectural improvements are currently:

1. Fix the retrieval gate so the RAG pipeline is actually effective.
2. Fix the index-building pipeline.
3. Centralize diagnosis orchestration.
4. Apply consistent response sanitization across all LLM endpoints.
5. Add automated tests and CI checks.
