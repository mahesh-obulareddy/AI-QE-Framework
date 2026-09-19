"""
test_hr_rag_chatbot.py
----------------------
DeepEval pytest eval suite for the HR Policy RAG Chatbot.

Uses the LangChain CallbackHandler integration so every LLM call and retriever
call is traced automatically — no manual @observe decorators needed.

Run with:
    deepeval test run tests/evals/test_hr_rag_chatbot.py \\
      --identifier "iterating-on-retrieval-round-1" \\
      --ignore-errors \\
      --skip-on-missing-params

Do NOT run with raw `pytest` — DeepEval's runner handles metric collection,
caching, and (optionally) Confident AI reporting.
"""

import sys
import os

import pytest

# Ensure project root is on the path so hr_rag_chatbot can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing import next_retriever_span

from hr_rag_chatbot import build_rag_chain, query_hr_bot

from deepeval.test_case import LLMTestCase

# Import shared metric lists (keeps thresholds in one place)
from tests.evals.metrics import RAG_TRACE_METRICS


# ---------------------------------------------------------------------------
# One-time RAG chain initialisation — built once, shared across all tests.
# Building the chain is slow (embeds PDFs, loads Chroma) so we do it at
# module load time rather than inside each test function.
# ---------------------------------------------------------------------------

_rag_chain, _retriever = build_rag_chain()


# ---------------------------------------------------------------------------
# Dataset — loaded from the generated golden file
# ---------------------------------------------------------------------------

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    file_path="tests/evals/dataset.json"
)


# ---------------------------------------------------------------------------
# Eval test — one parametrized invocation per golden
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("golden", dataset.goldens)
def test_hr_rag_response_quality(golden: Golden):
    """
    Run a single HR policy question through the RAG chain, collect the trace
    via the LangChain CallbackHandler, construct the LLMTestCase with the
    retrieved source documents, and evaluate with DeepEval metrics.
    """
    handler = CallbackHandler(
        name="hr-rag-chatbot",
        tags=["hr", "rag", "deepeval-loop"],
    )

    result = query_hr_bot(
        question=golden.input,
        rag_chain=_rag_chain,
        retriever=_retriever,
        callbacks=[handler],
    )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result["answer"],
        expected_output=golden.expected_output,
        retrieval_context=result["source_documents"],
    )

    assert_test(test_case=test_case, metrics=RAG_TRACE_METRICS)

