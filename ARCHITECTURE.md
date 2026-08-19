# CropPilot Architecture

## System Overview

CropPilot follows a dual-path architecture supporting both disease diagnosis from images and general agricultural Q&A, with full Hindi/English language support.

## Architecture Diagram

### Disease Diagnosis Path

```
Gradio UI (Hindi/English Input)
    ↓
Language Detection & Translation (if Hindi)
    ↓
FastAPI Endpoint: /api/diagnose/upload
    ↓
ViT Disease Classifier
    ↓
Confidence Gate (>70% threshold)
    ↓
FAISS Retrieval (Top 5 chunks)
    ↓
Retrieval Gate (Filter weak/short chunks)
    ↓
Groq LLM Generation (with source citations)
    ↓
Faithfulness Check (Word-overlap heuristic)
    ↓
English-to-Hindi Translation (if input was Hindi)
    ↓
Response to Farmer (Hindi/English)
```

### Q&A Path

```
Gradio UI (Hindi/English Question)
    ↓
Language Detection & Translation (if Hindi)
    ↓
FastAPI Endpoint: /api/crop-plan or /api/qa
    ↓
FAISS Retrieval (Top 5 chunks) [for /api/crop-plan]
    ↓
Retrieval Gate (Filter weak/short chunks)
    ↓
Groq LLM Generation (with source citations)
    ↓
Faithfulness Check (Word-overlap heuristic) [for /api/crop-plan]
    ↓
English-to-Hindi Translation (if input was Hindi)
    ↓
Response to Farmer (Hindi/English)
```

## Component Details

### 1. Language Translation Layer
- **Location**: `src/translator.py`
- **Models**: Helsinki-NLP/opus-mt-hi-en, Helsinki-NLP/opus-mt-en-hi
- **Function**: 
  - Automatic language detection (Devanagari character analysis)
  - Hindi → English translation for processing
  - English → Hindi translation for output
- **Fallback**: Returns original text if translation fails

### 2. Disease Classification
- **Location**: `src/classifier.py`
- **Model**: HurudzaAI/plantdiseasedetection1 (ViT-based)
- **Input**: Plant leaf image
- **Output**: Disease label + confidence score
- **Gate**: Rejects predictions <70% confidence

### 3. FAISS Retrieval
- **Location**: `src/retriever.py`
- **Embedding Model**: sentence-transformers/multi-qa-mpnet-base-dot-v1
- **Index**: Pre-built FAISS index from NIPHM documents
- **Output**: Top 5 most relevant document chunks

### 4. Retrieval Gate
- **Location**: `src/retrieval_gate.py`
- **Function**: Filters low-quality chunks
- **Criteria**:
  - Minimum length: 50 characters
  - Meaningful content ratio: >50%
  - Agricultural keyword presence
- **Purpose**: Ensures only relevant, high-quality chunks reach the LLM

### 5. LLM Generation
- **Location**: `src/generator.py`
- **Model**: Groq API (Qwen 3.6 27B)
- **Input**: Filtered chunks + disease context
- **Output**: Structured treatment plan
- **Features**:
  - Source citation requirement
  - Organic/chemical treatment options
  - Prevention steps
  - Urgency level

### 6. Faithfulness Check
- **Location**: `src/faithfulness.py`
- **Method**: Word-overlap heuristic
- **Threshold**: 30% minimum overlap
- **Purpose**: Validates that LLM response is grounded in retrieved documents
- **Output**: Faithfulness score + warning if low

### 7. API Layer
- **Location**: `main.py`
- **Framework**: FastAPI
- **Endpoints**:
  - `/api/diagnose/upload` - Image-based diagnosis
  - `/api/crop-plan` - Document-based Q&A
  - `/api/qa` - General Q&A
- **Features**: Automatic language handling, error handling, file management

### 8. UI Layer
- **Location**: `app.py`
- **Framework**: Gradio
- **Features**:
  - Three tabs: Diagnosis, Chatbot, Support
  - Language selection (Auto/English/Hindi)
  - Bilingual interface
  - Real-time translation feedback

## Data Flow

### Hindi Input Example
1. Farmer types: "मेरी गेहूं की फसल में पीला रतुआ है" (My wheat crop has yellow rust)
2. System detects Hindi (Devanagari characters)
3. Translates to English: "My wheat crop has yellow rust"
4. Processes through English pipeline
5. Generates English response
6. Translates back to Hindi: "आपकी गेहूं की फसल में पीला रतुआ है..."
7. Displays Hindi response to farmer

### Quality Assurance Pipeline
1. **Confidence Gate**: Only high-confidence disease predictions proceed
2. **Retrieval Gate**: Only relevant, substantial chunks are used
3. **Faithfulness Check**: Only well-grounded responses are returned
4. **Translation Fallback**: Original text returned if translation fails

## Performance Considerations

- **Translation Models**: Loaded once at startup (~500MB each)
- **FAISS Index**: Pre-built, loaded at startup
- **Classification Model**: Loaded once at startup (~300MB)
- **Cold Start Time**: ~30-60 seconds (model loading)
- **Inference Time**: ~2-5 seconds per request
- **Memory Usage**: ~2-3GB RAM (CPU mode)

## Security & Reliability

- **API Keys**: Stored in `.env` file (gitignored)
- **Input Validation**: File type checking, size limits
- **Error Handling**: Graceful fallbacks at each stage
- **No Hallucination**: LLM instructed to only use retrieved documents
- **Disclaimer**: Users advised to consult agricultural experts
