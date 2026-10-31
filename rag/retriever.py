"""
Retriever module for Techpaathshala Terms & Conditions RAG.
Responsible for:
1. Loading t&c.txt from the documents folder.
2. Chunking text into semantically coherent segments.
3. Generating embeddings (Google GenAI, OpenAI, or HuggingFace/fallback).
4. Persisting and loading FAISS vector index.
5. Providing similarity search retrieval interface.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load local environment if present
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

CURRENT_DIR = Path(__file__).resolve().parent
DOCS_DIR = CURRENT_DIR / "documents"
DEFAULT_FILE_PATH = DOCS_DIR / "t&c.txt"
DEFAULT_INDEX_DIR = CURRENT_DIR / "faiss_index"


def get_embedding_model(api_key: Optional[str] = None, provider: str = "auto"):
    """
    Returns an appropriate embedding model.
    Prioritizes Google Generative AI embeddings when GEMINI_API_KEY is available,
    then OpenAI, then HuggingFace/Local embeddings.
    """
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    
    if (provider in ["gemini", "auto"]) and gemini_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=gemini_key
            )
            # Quick verification
            embeddings.embed_query("test")
            return embeddings
        except Exception as e:
            print(f"[Retriever] Google embeddings init error: {e}. Trying fallback.")
            
    if (provider in ["openai", "auto"]) and openai_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=openai_key)
        except Exception as e:
            print(f"[Retriever] OpenAI embeddings init error: {e}. Trying fallback.")
            
    # Try HuggingFace Embeddings if installed
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[Retriever] HuggingFace embeddings init error: {e}. Using deterministic local embedding.")

    # Lightweight deterministic embedding fallback (works with FAISS without external network/weights)
    from langchain_core.embeddings import Embeddings
    import hashlib
    import numpy as np

    class SimpleDeterministicEmbeddings(Embeddings):
        """A lightweight zero-dependency embedding fallback generating 128-dim normalized vectors."""
        def __init__(self, dim: int = 128):
            self.dim = dim

        def _embed_text(self, text: str) -> List[float]:
            words = text.lower().split()
            vec = np.zeros(self.dim, dtype=np.float32)
            for w in words:
                h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self._embed_text(t) for t in texts]

        def embed_query(self, text: str) -> List[float]:
            return self._embed_text(text)

    return SimpleDeterministicEmbeddings()


def load_documents(file_path: Path = DEFAULT_FILE_PATH) -> List[Document]:
    """Loads the terms and conditions text document."""
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sections or use RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = text_splitter.create_documents(
        texts=[content],
        metadatas=[{"source": file_path.name, "company": "Techpaathshala"}]
    )
    return docs


def build_or_load_vectorstore(
    force_rebuild: bool = False,
    api_key: Optional[str] = None,
    index_dir: Path = DEFAULT_INDEX_DIR
) -> FAISS:
    """
    Builds a FAISS vectorstore if not present or forced, otherwise loads the saved index.
    """
    embeddings = get_embedding_model(api_key=api_key)

    if not force_rebuild and index_dir.exists() and (index_dir / "index.faiss").exists():
        try:
            return FAISS.load_local(
                folder_path=str(index_dir),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"[Retriever] Failed to load cached FAISS index: {e}. Rebuilding...")

    docs = load_documents()
    vectorstore = FAISS.from_documents(docs, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def retrieve_relevant_documents(
    query: str,
    k: int = 3,
    vectorstore: Optional[FAISS] = None,
    api_key: Optional[str] = None
) -> List[Document]:
    """
    Queries FAISS database and returns the top k relevant Document objects.
    """
    if vectorstore is None:
        vectorstore = build_or_load_vectorstore(api_key=api_key)
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    print("[Testing Retriever] Building/loading FAISS vectorstore...")
    vs = build_or_load_vectorstore(force_rebuild=True)
    test_query = "What is Techpaathshala's refund policy?"
    results = retrieve_relevant_documents(test_query, k=2, vectorstore=vs)
    print(f"\nQuery: {test_query}")
    print(f"Retrieved {len(results)} chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content)
