"""
AI Expert Service - RAG farming Q&A
Uses FAISS vector store and LLM to answer farming questions
"""
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import INDEX_PATH, EMBEDDING_MODEL, RETRIEVAL_MIN_RELEVANCE, RETRIEVAL_MIN_CONTENT_LENGTH
from reliability import RetrievalGate, FaithfulnessChecker
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("[AI Expert] Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

print("[AI Expert] Loading FAISS index...")
vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize retrieval gate and faithfulness checker
retrieval_gate = RetrievalGate(
    min_relevance=RETRIEVAL_MIN_RELEVANCE,
    min_content_length=RETRIEVAL_MIN_CONTENT_LENGTH
)
faithfulness_checker = FaithfulnessChecker()

PROMPT_TEMPLATE = """You are CropPilot, an agricultural expert assistant helping Indian farmers.
Answer the farmer's question using only the retrieved documents provided below.
If information is not in the documents, say so clearly. Never invent information.

FARMER'S QUESTION: {question}

RETRIEVED FROM OFFICIAL DOCUMENTS (ICAR, NIPHM, Universities):
{context}

Based only on the above documents, provide a clear, practical answer.
If appropriate, include:
- Direct answer to the question
- Specific steps or recommendations
- Any precautions or warnings
- References to the source documents

Keep it simple and actionable for farmers."""


def answer_farming_question(question: str) -> dict:
    """
    Answer farming question using RAG with retrieval gate and faithfulness check
    
    Args:
        question: Farmer's question
        
    Returns:
        Dictionary with answer, gate results, and faithfulness results
    """
    print(f"[AI Expert] Processing question: {question}")
    
    docs = retriever.invoke(question)
    
    if not docs:
        return {
            "answer": "No relevant information found in the knowledge base. Please try rephrasing your question or ensure relevant documents are available.",
            "gate_results": [],
            "faithfulness_results": None,
            "filtered_chunks": []
        }

    # Convert docs to chunks format
    chunks = []
    for doc in docs:
        chunks.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown")
        })

    # Apply retrieval gate
    filtered_chunks, gate_results = retrieval_gate.filter_chunks(chunks)
    
    if not filtered_chunks:
        return {
            "answer": "No relevant information passed the quality filters. Please try rephrasing your question.",
            "gate_results": gate_results,
            "faithfulness_results": None,
            "filtered_chunks": []
        }

    context = ""
    for i, chunk in enumerate(filtered_chunks):
        context += f"\n--- Document {i+1} (Source: {chunk['source']}, Page: {chunk['page']}) ---\n"
        context += chunk["content"] + "\n"

    prompt = PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )

    print("[AI Expert] Generating answer...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    
    # Apply faithfulness check
    faithfulness_results = faithfulness_checker.comprehensive_check(answer, filtered_chunks)

    return {
        "answer": answer,
        "gate_results": gate_results,
        "faithfulness_results": faithfulness_results,
        "filtered_chunks": filtered_chunks
    }
