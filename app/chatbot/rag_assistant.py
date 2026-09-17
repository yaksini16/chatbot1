"""
Compliance RAG Assistant
Answers inspector and packager questions on Legal Metrology Act & Packaged Commodities Rules 2011.
Supports Google GenAI (Gemini) LLM synthesis when GEMINI_API_KEY is configured,
with robust non-LLM statutory template fallback when the key is missing or unavailable.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from ..knowledge.vector_store import ComplianceVectorStore
from ..config import settings

logger = logging.getLogger(__name__)


class RAGAssistant:
    def __init__(self, vector_store: ComplianceVectorStore):
        self.vector_store = vector_store
        self._seed_default_knowledge()
        self._client = None
        self._init_gemini_client()

    def _init_gemini_client(self):
        """
        Initializes the official Google GenAI client if GEMINI_API_KEY is configured.
        SECURITY: Never logs, returns, or prints the API key value.
        """
        if not settings.is_gemini_configured:
            logger.info("GEMINI_API_KEY is not configured. Retaining non-LLM template fallback mode.")
            return

        try:
            from google import genai
            key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            self._client = genai.Client(api_key=key)
            logger.info("Google GenAI client successfully initialized for RAG chatbot.")
        except Exception as e:
            logger.warning("Failed to initialize Google GenAI client (%s). Retaining non-LLM fallback.", type(e).__name__)
            self._client = None

    def _seed_default_knowledge(self):
        # Citations verified against the consolidated LMPC Rules 2011 text (as amended 2017/2021)
        # and the FSS (Labelling and Display) Regulations 2020.
        faq_data = [
            {
                "title": "MRP Rule 6(1)(e)",
                "text": "Under Rule 6(1)(e) of Legal Metrology (Packaged Commodities) Rules 2011, the retail sale price shall clearly mention '(inclusive of all taxes)' and must be stated in Indian Rupees (₹ or Rs.). Unit Sale Price is additionally mandatory per the 2021 amendment (G.S.R. 779(E)).",
                "source": "Legal Metrology Rules 2011, Rule 6(1)(e)",
            },
            {
                "title": "Net Quantity Rule 6(1)(c)",
                "text": "Net quantity shall be declared in standard metric units: mass in grams (g) or kilograms (kg), volume in milliliters (ml) or liters (l), length in meters (m) or centimeters (cm). Non-metric units like lbs or oz are strictly prohibited.",
                "source": "Legal Metrology Rules 2011, Rule 6(1)(c)",
            },
            {
                "title": "Consumer Care Rule 6(2)",
                "text": "Every package shall bear the name, address, telephone number, and email address of the designated officer or grievance redressal cell. This requirement was inserted as sub-rule 6(2) by the 2017 Amendment (G.S.R. 629(E)).",
                "source": "Legal Metrology Rules 2011, Rule 6(2)",
            },
            {
                "title": "Country of Origin Rule 6(1)(aa)",
                "text": "For imported packages, the name of the country of origin or manufacture or assembly must be declared conspicuously on the package (clause inserted by the 2017 Amendment).",
                "source": "Legal Metrology Rules 2011, Rule 6(1)(aa)",
            },
        ]
        self.vector_store.add_documents(faq_data)

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answers a user query using vector search over statutory provisions,
        enriched with Gemini LLM generation if GEMINI_API_KEY is configured,
        or statutory template fallback if unconfigured or unavailable.
        """
        retrieved_docs: List[Dict[str, Any]] = self.vector_store.search(query, top_k=3)
        if not retrieved_docs:
            return {
                "answer": "No relevant Legal Metrology clause found in the local knowledge base.",
                "citations": [],
                "llm_generated": False,
            }

        citations = [d.get("source") for d in retrieved_docs if d.get("source")]
        context = "\n".join(f"- {d.get('title')}: {d.get('text')}" for d in retrieved_docs)
        template_answer = f"According to statutory provisions:\n{context}"

        # If GEMINI_API_KEY is not configured, log a clear message and return fallback
        if not settings.is_gemini_configured:
            logger.info("GEMINI_API_KEY is not configured in environment; retaining non-LLM statutory template.")
            return {
                "answer": template_answer,
                "citations": citations,
                "llm_generated": False,
            }

        # Lazy initialize client if not already done
        if self._client is None:
            self._init_gemini_client()

        if self._client is not None:
            try:
                prompt = (
                    "You are an expert Legal Metrology Compliance Officer assistant for PRAMAN.\n"
                    f"User Question: \"{query}\"\n\n"
                    f"Statutory Context & Gazette Provisions:\n{context}\n\n"
                    "Instructions:\n"
                    "- Provide a clear, authoritative, and concise compliance answer based strictly on the statutory provisions and gazette rules above.\n"
                    "- Cite the relevant Rule numbers (e.g. Rule 6(1)(e), Rule 6(2)) and gazette notifications where applicable.\n"
                    "- If the context does not fully answer the question, state what the rules specify and clarify the limits."
                )
                response = self._client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                )
                if response and hasattr(response, "text") and response.text:
                    return {
                        "answer": response.text.strip(),
                        "citations": citations,
                        "llm_generated": True,
                    }
            except Exception as e:
                logger.warning(
                    "Gemini LLM generation failed (%s). Retaining non-LLM statutory template fallback.",
                    type(e).__name__
                )

        # Fallback to local template
        return {
            "answer": template_answer,
            "citations": citations,
            "llm_generated": False,
        }
