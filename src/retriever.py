from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import INDEX_PATH, EMBEDDING_MODEL, TOP_K

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

print("Loading FAISS index...")
vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)


def retrieve_treatment_docs(crop: str, disease: str) -> list:
    query = f"treatment and management of {disease} in {crop} plants"
    print(f"\nSearching for: {query}")

    docs = retriever.invoke(query)

    chunks = []
    for i, doc in enumerate(docs):
        chunks.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown")
        })
        print(f"Retrieved chunk {i+1} from {chunks[-1]['source']} page {chunks[-1]['page']}")

    return chunks