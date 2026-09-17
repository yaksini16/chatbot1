"""
Vector Knowledge Base (FAISS + Multilingual Embeddings)
Stores and indexes Legal Metrology statutory notifications, gazettes, and FAQs (Hindi + English).
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ComplianceVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._index = None
        self._encoder = None
        self._documents: List[Dict[str, Any]] = []

    def _init_components(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                self._encoder = SentenceTransformer(self.model_name)
                # get_embedding_dimension() is the new name (sentence-transformers >= 3.x)
                # fall back to get_sentence_embedding_dimension() for older installs
                dim_fn = (
                    getattr(self._encoder, "get_embedding_dimension", None)
                    or self._encoder.get_sentence_embedding_dimension
                )
                dim = dim_fn()
                self._index = faiss.IndexFlatL2(dim)
            except Exception as e:
                logger.warning(f"Vector store dependencies could not be loaded: {e}. Running in degraded mode.")
                self._encoder = None
                self._index = None

    def add_documents(self, docs: List[Dict[str, Any]]):
        self._init_components()
        self._documents.extend(docs)
        if self._encoder is not None and self._index is not None and docs:
            texts = [d["text"] for d in docs]
            embeddings = self._encoder.encode(texts, convert_to_numpy=True)
            self._index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        self._init_components()
        if self._encoder is not None and self._index is not None and self._index.ntotal > 0:
            query_emb = self._encoder.encode([query], convert_to_numpy=True)
            distances, indices = self._index.search(query_emb, min(top_k, self._index.ntotal))
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._documents):
                    results.append(self._documents[idx])
            return results

        # Fallback keyword match
        query_words = set(query.lower().split())
        matched = []
        for doc in self._documents:
            doc_text = doc.get("text", "").lower()
            if any(w in doc_text for w in query_words):
                matched.append(doc)
            if len(matched) >= top_k:
                break
        return matched
