MODEL_NAME = "HurudzaAI/plantdiseasedetection1"

EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

TRANSLATION_MODEL_HI_EN = "Helsinki-NLP/opus-mt-hi-en"
TRANSLATION_MODEL_EN_HI = "Helsinki-NLP/opus-mt-en-hi"


CHUNK_SIZE = 512

CHUNK_OVERLAP = 64

TOP_K = 5


MIN_CHUNK_LENGTH = 50  # Minimum chunk length for retrieval gate
FAITHFULNESS_THRESHOLD = 0.3  # Minimum word overlap for faithfulness check


KNOWLEDGE_BASE_DIR = "knowledge_base/"

INDEX_PATH = "faiss_index/"