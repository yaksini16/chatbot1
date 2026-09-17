"""
Legal Metrology Statutory RAG Chatbot Router
Supports Google GenAI (Gemini) synthesis with statutory fallback.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.knowledge.vector_store import ComplianceVectorStore
from app.knowledge.corpus_loader import load_corpus_into_vector_store
from app.chatbot.rag_assistant import RAGAssistant
from app.services.chat_service import answer_compliance_question as keyword_fallback

router = APIRouter(prefix="/api/chat", tags=["Legal Metrology Chatbot"])

# Build vector store and seed it with the full gazette corpus (backend/data/rules_corpus.json),
# then add the 4 statutory FAQ entries from RAGAssistant._seed_default_knowledge().
_vector_store = ComplianceVectorStore()
_corpus_chunk_count = load_corpus_into_vector_store(_vector_store)
_rag_assistant = RAGAssistant(_vector_store)
IST = timezone(timedelta(hours=5, minutes=30))


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    text: Optional[str] = None
    citations: List[str] = []
    sources: List[str] = []  # gazette-provenance labels: "G.S.R. 779(E) -- PCR_Amendment_...pdf"
    corpus_chunks_loaded: Optional[int] = None  # informational: how many corpus chunks are active
    llm_generated: Optional[bool] = None  # True if synthesized by Gemini, False if template/fallback
    gemini_configured: Optional[bool] = None  # True if GEMINI_API_KEY is present; never exposes key value
    timestamp: str


@router.get("/status")
def chat_status():
    """
    Diagnostic status endpoint for Legal Metrology Chatbot.
    SECURITY: Detects whether GEMINI_API_KEY is configured, but never prints or returns its value.
    """
    return {
        "status": "active",
        "gemini_configured": settings.is_gemini_configured,
        "corpus_chunks_loaded": _corpus_chunk_count,
        "llm_model": settings.GEMINI_MODEL if settings.is_gemini_configured else None,
    }


@router.post("", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    RAG-powered conversational assistant for enforcement officers & packagers.
    Retrieves Legal Metrology (Packaged Commodities) statutory sections and citations,
    synthesizing with Gemini if configured, or falling back to local statutory templates.
    """
    query = req.message.strip()
    now_iso = datetime.now(IST).isoformat()
    llm_generated = False

    try:
        res = _rag_assistant.answer_query(query)
        reply = res.get("answer", "")
        citations = res.get("citations", [])
        llm_generated = res.get("llm_generated", False)
    except Exception:
        # Fallback to keyword matcher if vector store or assistant fails
        reply, citations = keyword_fallback(query)
        llm_generated = False

    # If RAG returned generic no-match, consult keyword service as backup
    if "No relevant" in reply:
        fallback_reply, fallback_citations = keyword_fallback(query)
        if fallback_reply and "I don't have" not in fallback_reply:
            reply = fallback_reply
            citations = fallback_citations or citations
            llm_generated = False

    return ChatResponse(
        reply=reply,
        text=reply,
        citations=citations,
        sources=citations,
        corpus_chunks_loaded=_corpus_chunk_count,
        llm_generated=llm_generated,
        gemini_configured=settings.is_gemini_configured,
        timestamp=now_iso,
    )
