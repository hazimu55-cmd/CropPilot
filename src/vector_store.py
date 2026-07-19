"""
Vector Store Abstraction Layer
Supports both FAISS and Qdrant as vector stores
"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import (
    INDEX_PATH, 
    EMBEDDING_MODEL, 
    VECTOR_STORE_TYPE,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME
)
from typing import List, Optional
import os


class VectorStore:
    """
    Abstract vector store interface
    """
    
    def __init__(self, store_type: str = None):
        """
        Initialize vector store
        
        Args:
            store_type: Type of vector store ("faiss" or "qdrant")
        """
        self.store_type = store_type or VECTOR_STORE_TYPE
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        
        if self.store_type == "faiss":
            self._init_faiss()
        elif self.store_type == "qdrant":
            self._init_qdrant()
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def _init_faiss(self):
        """Initialize FAISS vector store"""
        print(f"[Vector Store] Initializing FAISS from {INDEX_PATH}")
        if os.path.exists(INDEX_PATH):
            self.store = FAISS.load_local(
                INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.store.as_retriever(search_kwargs={"k": 5})
        else:
            print("[Vector Store] FAISS index not found. Create it first using build_index.py")
            self.store = None
            self.retriever = None
    
    def _init_qdrant(self):
        """Initialize Qdrant vector store"""
        print(f"[Vector Store] Initializing Qdrant at {QDRANT_URL}")
        try:
            self.client = QdrantClient(url=QDRANT_URL)
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if QDRANT_COLLECTION_NAME not in collection_names:
                print(f"[Vector Store] Creating Qdrant collection: {QDRANT_COLLECTION_NAME}")
                self.client.create_collection(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=768,  # MPNet embedding dimension
                        distance=Distance.COSINE
                    )
                )
            
            self.store = Qdrant(
                client=self.client,
                collection_name=QDRANT_COLLECTION_NAME,
                embeddings=self.embeddings
            )
            self.retriever = self.store.as_retriever(search_kwargs={"k": 5})
            
        except Exception as e:
            print(f"[Vector Store] Error initializing Qdrant: {e}")
            print("[Vector Store] Falling back to FAISS")
            self._init_faiss()
    
    def add_documents(self, documents: List, ids: Optional[List[str]] = None):
        """
        Add documents to vector store
        
        Args:
            documents: List of documents to add
            ids: Optional list of document IDs
        """
        if self.store is None:
            raise ValueError("Vector store not initialized")
        
        if self.store_type == "faiss":
            if ids:
                self.store.add_documents(documents, ids=ids)
            else:
                self.store.add_documents(documents)
            # Save FAISS index
            self.store.save_local(INDEX_PATH)
        elif self.store_type == "qdrant":
            if ids:
                self.store.add_documents(documents, ids=ids)
            else:
                self.store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized")
        
        return self.retriever.invoke(query)[:k]
    
    def as_retriever(self, **kwargs):
        """
        Get retriever instance
        
        Args:
            **kwargs: Additional arguments for retriever
            
        Returns:
            Retriever instance
        """
        if self.store is None:
            raise ValueError("Vector store not initialized")
        
        return self.store.as_retriever(**kwargs)


def get_vector_store(store_type: str = None) -> VectorStore:
    """
    Factory function to get vector store instance
    
    Args:
        store_type: Type of vector store ("faiss" or "qdrant")
        
    Returns:
        VectorStore instance
    """
    return VectorStore(store_type)
