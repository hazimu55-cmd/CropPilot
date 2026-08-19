import os

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import (

    KNOWLEDGE_BASE_DIR,

    INDEX_PATH,

    CHUNK_SIZE,

    CHUNK_OVERLAP,

    EMBEDDING_MODEL

)





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





def build_faiss_index(chunks):

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={"device": "cpu"}

    )



    print("Building FAISS index — this takes 2-3 minutes...")

    vectorstore = FAISS.from_documents(chunks, embeddings)



    os.makedirs(INDEX_PATH, exist_ok=True)

    vectorstore.save_local(INDEX_PATH)

    print(f"\nIndex saved to {INDEX_PATH}")

    print("Done — never need to run this again unless you add new PDFs")





if __name__ == "__main__":

    docs = load_pdfs()

    if docs:

        chunks = chunk_documents(docs)

        build_faiss_index(chunks)