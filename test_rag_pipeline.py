"""
test_rag_pipeline.py
--------------------
Fast functional and integration tests for the HR Policy RAG Pipeline.
Verifies PDF loading, vector search accuracy, and guardrails without external APIs.

Run:
    pytest test_rag_pipeline.py -v
"""

import pytest
from hr_rag_chatbot import (
    load_hr_documents,
    split_documents,
    build_rag_chain,
    query_hr_bot,
    HR_DOCS_DIR,
)

# Shared RAG chain across tests in this module
@pytest.fixture(scope="module")
def rag():
    chain, retriever = build_rag_chain(model_name="llama3.1")
    return {"chain": chain, "retriever": retriever}


def test_document_ingestion():
    """Verify that all HR PDF documents are discovered and loaded."""
    docs = load_hr_documents(HR_DOCS_DIR)
    assert len(docs) > 0, "No pages loaded from hr_documents directory"
    # Ensure source metadata contains PDF paths
    sources = {doc.metadata.get("source") for doc in docs}
    assert any("Maternity-Policy.pdf" in str(s) for s in sources), "Maternity policy not found"
    assert any("Alcohol-Policy.pdf" in str(s) for s in sources), "Alcohol policy not found"


def test_text_splitting():
    """Verify text chunking preserves content and overlap."""
    sample_docs = load_hr_documents(HR_DOCS_DIR)[:3]
    chunks = split_documents(sample_docs, chunk_size=1200, chunk_overlap=250)
    assert len(chunks) >= len(sample_docs), "Chunks must be created from pages"
    for c in chunks:
        assert len(c.page_content) <= 1500, "Chunk exceeded maximum expected character size"


def test_retrieval_relevance(rag):
    """Verify that retrieval accurately finds the correct policy document."""
    retriever = rag["retriever"]
    docs = retriever.invoke("What is the maternity leave entitlement?")
    assert len(docs) > 0, "Retriever returned 0 chunks"
    top_sources = [d.metadata.get("source", "") for d in docs]
    assert any("Maternity-Policy.pdf" in s for s in top_sources), (
        f"Expected Maternity-Policy.pdf in retrieved chunks, got: {top_sources}"
    )


def test_rag_in_scope_answer(rag):
    """Verify end-to-end question answering on an in-scope question."""
    result = query_hr_bot(
        "What is the maternity leave policy?",
        rag["chain"],
        rag["retriever"],
    )
    assert result["answer"], "Answer was empty"
    assert len(result["source_documents"]) > 0, "No source documents returned"
    assert "cannot find this in the HR policy" not in result["answer"].lower(), (
        "Bot erroneously triggered out-of-policy fallback on in-scope question"
    )


def test_strict_guardrails_out_of_scope(rag):
    """Verify that out-of-scope questions strictly trigger the required guardrail phrase."""
    result = query_hr_bot(
        "What is the company stock option vesting schedule?",
        rag["chain"],
        rag["retriever"],
    )
    assert "I cannot find this in the HR policy." in result["answer"], (
        f"Guardrail failed! Expected 'I cannot find this in the HR policy.', got: {result['answer']}"
    )
