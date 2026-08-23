---
title: CropPilot
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: mit
---

# 🌿 CropPilot - AI-Powered Crop Disease Diagnosis

AI-powered crop disease diagnosis and treatment advisor for Indian farmers, with full Hindi/English language support.

## Features

- 🔍 **Disease Detection**: Upload plant leaf images for AI-powered disease diagnosis
- 💬 **Bilingual Chatbot**: Ask agricultural questions in Hindi or English
- 🛠️ **Support System**: Get help from agricultural experts
- 🌾 **Supported Crops**: Corn, Potato, Rice, Wheat
- 📚 **Knowledge Base**: Backed by official NIPHM IPM packages
- 🇮🇳 **Hindi Support**: Automatic translation and language detection

## How to Use

### Disease Diagnosis
1. Upload a clear photo of a plant leaf
2. Optionally add context (location, season, irrigation)
3. Select language (Auto/English/Hindi)
4. Click "Diagnose Disease"
5. Get detailed diagnosis and treatment plan

### Chatbot
- Ask general agricultural questions
- Get expert advice in your preferred language
- Covers crops, diseases, soil, irrigation, fertilizers

## Model Information

- **Disease Classifier**: HurudzaAI/plantdiseasedetection1 (ViT-based)
- **Translation**: Helsinki-NLP MarianMT models
- **LLM**: Groq API (Qwen 3.6 27B)
- **Embeddings**: sentence-transformers/multi-qa-mpnet-base-dot-v1

## Technical Details

- **Architecture**: Vision Transformer + RAG Pipeline
- **Supported Crops**: Corn, Potato, Rice, Wheat
- **Diseases**: Multiple diseases per crop + healthy detection
- **Language**: Hindi ↔ English automatic translation
- **Deployment**: GPU-accelerated on Hugging Face Spaces

## Limitations

- Only supports leaf images (not fruits, tubers, stems)
- Limited to 4 crop types
- Requires clear, well-lit images for best results
- Confidence threshold applied for reliable predictions

## Citation

If you use this model, please cite the original model and dataset:

```bibtex
@software{plantdiseasedetection1,
  title={Plant Disease Detection Model},
  author={HurudzaAI},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.