# Model configuration
MODEL_NAME = "yolov8n.pt"  # YOLOv8 model (can be replaced with PlantDoc-trained model)
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Chunking configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5

# Knowledge base configuration
KNOWLEDGE_BASE_DIR = "knowledge_base/"
INDEX_PATH = "faiss_index/"

# Vector store configuration
VECTOR_STORE_TYPE = "faiss"  # Options: "faiss" or "qdrant"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "croppilot_knowledge"

# Reliability layer configuration
CONFIDENCE_THRESHOLD = 0.70
RETRIEVAL_MIN_RELEVANCE = 0.5
RETRIEVAL_MIN_CONTENT_LENGTH = 50