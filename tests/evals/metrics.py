"""
metrics.py
----------
Shared DeepEval metric instances for the HR RAG Chatbot eval suite.

All metrics are explicitly configured with OllamaModel (llama3.1)
running locally with zero rate-limit or quota issues.

Import these lists in the test file rather than constructing metrics inline.
Keep all thresholds here so they are easy to track and adjust across rounds.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from deepeval.models import OllamaModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
)
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

# ---------------------------------------------------------------------------
# Shared evaluation model — Ollama llama3.1 (local, robust, no 429 quota limits)
# ---------------------------------------------------------------------------

_eval_model = OllamaModel(model="llama3.1")

# ---------------------------------------------------------------------------
# End-to-end trace-level metrics (scored against the full RAG response)
# ---------------------------------------------------------------------------

RAG_TRACE_METRICS = [
    # Does the answer actually address the question asked?
    AnswerRelevancyMetric(threshold=0.7, model=_eval_model),

    # Is the answer grounded in the retrieved chunks (no hallucination)?
    FaithfulnessMetric(threshold=0.7, model=_eval_model),

    # Is the retrieved context semantically relevant to the question?
    ContextualRelevancyMetric(threshold=0.7, model=_eval_model),

    # Are the most relevant chunks ranked at the top of the retrieved set?
    ContextualPrecisionMetric(threshold=0.5, model=_eval_model),

    # Custom GEval: HR policy accuracy
    # Checks that the answer correctly cites/applies the policy (not just relevant).
    GEval(
        name="HR Policy Accuracy",
        criteria=(
            "The answer correctly and accurately reflects the HR policy content "
            "in the retrieval context. It does not fabricate policy details, "
            "omit critical caveats, or present placeholder text (e.g. '[insert X]') "
            "as if it were real policy. It gracefully states when the policy does "
            "not cover a topic rather than inventing an answer."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.7,
        model=_eval_model,
    ),
]

# ---------------------------------------------------------------------------
# Retriever span metrics (scored at the Chroma retriever span level)
# ---------------------------------------------------------------------------

RETRIEVER_SPAN_METRICS = [
    # Is the retrieved context relevant to what the user asked?
    ContextualRelevancyMetric(threshold=0.7, model=_eval_model),

    # Are the most relevant chunks ranked first (retrieval ordering quality)?
    ContextualPrecisionMetric(threshold=0.5, model=_eval_model),
]
