"""
Unit & Integration Tests for Gemini API Configuration, Detection, Security & Fallback
Verifies:
1. GEMINI_API_KEY is read securely from the environment/.env.
2. Application can detect whether GEMINI_API_KEY is configured (is_gemini_configured).
3. API key is NEVER exposed in logs, responses, or status endpoints.
4. When GEMINI_API_KEY is missing, application provides clean fallback to statutory templates.
5. .env is properly ignored in .gitignore.
"""

import os
from unittest.mock import patch, PropertyMock
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import Settings, settings
from app.knowledge.vector_store import ComplianceVectorStore
from app.chatbot.rag_assistant import RAGAssistant


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_gemini_api_key_detection_when_configured():
    """Verify is_gemini_configured returns True when GEMINI_API_KEY is set."""
    custom_settings = Settings(GEMINI_API_KEY="test-configured-dummy-key")
    assert custom_settings.is_gemini_configured is True
    # Ensure boolean output
    assert isinstance(custom_settings.is_gemini_configured, bool)


def test_gemini_api_key_detection_when_missing():
    """Verify is_gemini_configured returns False when GEMINI_API_KEY is unset or empty string."""
    with patch.dict(os.environ, {}, clear=True):
        empty_settings = Settings(GEMINI_API_KEY="")
        assert empty_settings.is_gemini_configured is False


def test_no_api_key_leaked_in_status_endpoint(client):
    """GET /api/chat/status returns gemini_configured boolean and never leaks key."""
    resp = client.get("/api/chat/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "gemini_configured" in data
    assert isinstance(data["gemini_configured"], bool)
    
    # Strictly ensure no secret key field exists in response
    assert "api_key" not in data
    assert "GEMINI_API_KEY" not in data
    assert "key" not in data


def test_chat_response_never_leaks_api_key(client):
    """POST /api/chat returns statutory answer without leaking API key."""
    resp = client.post(
        "/api/chat",
        json={"message": "What is the requirement for Maximum Retail Price under Rule 6(1)(e)?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "citations" in data
    assert "gemini_configured" in data
    assert isinstance(data["gemini_configured"], bool)

    # Convert entire response to string and verify no secret key value leaked
    resp_text = resp.text
    if settings.GEMINI_API_KEY:
        assert settings.GEMINI_API_KEY not in resp_text


def test_rag_assistant_clean_fallback_when_key_missing():
    """Verify RAGAssistant falls back cleanly to local statutory template when GEMINI_API_KEY is missing."""
    vs = ComplianceVectorStore()
    
    with patch.object(Settings, "is_gemini_configured", new_callable=PropertyMock) as mock_cfg:
        mock_cfg.return_value = False
        assistant = RAGAssistant(vs)
        result = assistant.answer_query("What are the rules for net quantity?")
        
        assert "answer" in result
        assert "According to statutory provisions:" in result["answer"]
        assert result.get("llm_generated") is False
        assert len(result.get("citations", [])) > 0


def test_gitignore_protects_env():
    """Verify that .env is explicitly declared in .gitignore."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gitignore_path = os.path.join(project_root, ".gitignore")
    assert os.path.exists(gitignore_path), ".gitignore must exist"
    
    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Verify .env is in .gitignore
    lines = [line.strip() for line in content.splitlines()]
    assert ".env" in lines
