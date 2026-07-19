"""
Crop Planner Service - RAG cultivation plan generation
Uses FAISS vector store and LLM to generate cultivation plans
"""
from groq import Groq
from src.retriever import retrieve_treatment_docs
from src.config import RETRIEVAL_MIN_RELEVANCE, RETRIEVAL_MIN_CONTENT_LENGTH
from reliability import RetrievalGate
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize retrieval gate
retrieval_gate = RetrievalGate(
    min_relevance=RETRIEVAL_MIN_RELEVANCE,
    min_content_length=RETRIEVAL_MIN_CONTENT_LENGTH
)

PROMPT_TEMPLATE = """You are CropPilot, an agricultural expert assistant helping Indian farmers plan crop cultivation.
Answer only using the retrieved documents provided below.
If information is not in the documents, say so clearly. Never invent information.

CROP: {crop}
REGION: {region}
SEASON: {season}
SOIL TYPE: {soil_type}
FARMER CONTEXT: {user_context}

RETRIEVED FROM OFFICIAL DOCUMENTS:
{context}

Based only on the above documents, provide a comprehensive cultivation plan including:
1. Variety selection recommendations
2. Land preparation steps
3. Sowing time and method
4. Irrigation schedule
5. Fertilizer application (organic and chemical)
6. Pest and disease management schedule
7. Harvesting guidelines
8. Expected yield

Keep it practical and specific. Farmer should be able to follow this plan."""


def generate_cultivation_plan(
    crop: str,
    region: str = "",
    season: str = "",
    soil_type: str = "",
    user_context: str = ""
) -> dict:
    """
    Generate cultivation plan using RAG with retrieval gate
    
    Args:
        crop: Crop name
        region: Geographic region
        season: Growing season
        soil_type: Type of soil
        user_context: Additional farmer context
        
    Returns:
        Dictionary with plan and gate results
    """
    # Build query for retrieval
    query_parts = [f"cultivation practices for {crop}"]
    if region:
        query_parts.append(f"in {region}")
    if season:
        query_parts.append(f"during {season} season")
    query = " ".join(query_parts)
    
    chunks = retrieve_treatment_docs(crop, "cultivation") if crop else []
    
    # If no chunks found with crop, try broader search
    if not chunks:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from src.config import INDEX_PATH, EMBEDDING_MODEL
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        
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
            "plan": f"No cultivation information found in knowledge base for {crop}. Please ensure relevant documents are in the knowledge base.",
            "gate_results": gate_results,
            "filtered_chunks": []
        }

    context = ""
    for i, chunk in enumerate(filtered_chunks):
        context += f"\n--- Document {i+1} (Source: {chunk['source']}, Page: {chunk['page']}) ---\n"
        context += chunk["content"] + "\n"

    prompt = PROMPT_TEMPLATE.format(
        crop=crop,
        region=region or "Not specified",
        season=season or "Not specified",
        soil_type=soil_type or "Not specified",
        user_context=user_context or "Not specified",
        context=context
    )

    print("[Crop Planner] Generating cultivation plan...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )

    return {
        "plan": response.choices[0].message.content,
        "gate_results": gate_results,
        "filtered_chunks": filtered_chunks
    }
