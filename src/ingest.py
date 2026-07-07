from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
import os


def load_pdfs():
    documents = []
    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in knowledge_base/ folder")
        return []

    for pdf_file in pdf_files:
        path = os.path.join(KNOWLEDGE_BASE_DIR, pdf_file)
        print(f"Loading: {pdf_file}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def ingest():
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    return chunks