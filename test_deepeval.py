"""
test_deepeval.py
----------------
DeepEval benchmark test suite for the HR RAG Chatbot.

This file demonstrates how to wire `query_hr_bot()` output directly into
DeepEval's evaluation pipeline.

Metrics tested:
  - AnswerRelevancyMetric   : Is the answer relevant to the question?
  - FaithfulnessMetric      : Is the answer grounded in retrieved context?
  - ContextualRecallMetric  : Did we retrieve the right context?
  - ContextualPrecisionMetric: Are the retrieved chunks actually useful?

Run:
    # Run with pytest (standard DeepEval integration)
    pytest test_deepeval.py -v

    # Or run directly
    python test_deepeval.py
"""

import os
import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.dataset import EvaluationDataset

from hr_rag_chatbot import build_rag_chain, query_hr_bot
from langchain_ollama import ChatOllama


# ---------------------------------------------------------------------------
# Build the RAG chain ONCE for the entire test session (expensive operation)
# ---------------------------------------------------------------------------

# NOTE: Change `model_name` here to benchmark a different Ollama model.
# All models must be pulled first: `ollama pull <model_name>`
MODEL_NAME = "llama3.1"   # <- swap to "mistral", "phi3", "gemma2", etc.

print(f"\n[SETUP] Building RAG chain with model: {MODEL_NAME}")
RAG_CHAIN, RETRIEVER = build_rag_chain(model_name=MODEL_NAME)


import requests
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Gemini Judge for DeepEval (Fast, reliable, cloud-based judge)
# ---------------------------------------------------------------------------

class GeminiJudge(DeepEvalBaseLLM):
    """Gemini LLM judge for DeepEval using Google's Generative Language API."""

    def __init__(self, model_name: str = "gemini-3.5-flash", api_key: str = None):
        self.model_name = model_name
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
        super().__init__(model=model_name)

    def load_model(self):
        return None

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        gen_config = {"temperature": 0}
        if schema is not None:
            gen_config["responseMimeType"] = "application/json"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        texts = [p["text"] for p in parts if "text" in p]
        return "\n".join(texts)

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


# ---------------------------------------------------------------------------
# Local Ollama Judge for DeepEval (Fallback if no Gemini/OpenAI key)
# ---------------------------------------------------------------------------

class OllamaJudge(DeepEvalBaseLLM):
    """Local LLM judge using Ollama, avoiding external API keys."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return ChatOllama(model=self.model_name, temperature=0)

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        res = self.model.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return self.model_name


def get_eval_model():
    """Return Gemini as judge if key is available, else OpenAI, else local Ollama."""
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return GeminiJudge(model_name="gemini-3.5-flash")
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    return OllamaJudge(model_name=MODEL_NAME)


# ---------------------------------------------------------------------------
# Golden Q&A dataset — ground_truth is used by Recall/Precision metrics
# ---------------------------------------------------------------------------

GOLDEN_DATASET = [
    {
        "question": "What is the company's maternity leave policy?",
        "expected_output": (
            "Eligible employees are entitled to maternity leave as outlined "
            "in the Maternity Policy document."
        ),
    },
    {
        "question": "How many days notice is required for resignation?",
        "expected_output": (
            "Notice periods vary by role and tenure, as specified in the "
            "Notice Periods Policy."
        ),
    },
    {
        "question": "What is the probationary period for new employees?",
        "expected_output": (
            "New employees typically serve a probationary period as outlined "
            "in the Probationary Periods Policy."
        ),
    },
    {
        "question": "What is the company's stance on equal opportunity?",
        "expected_output": (
            "The company is committed to equal opportunity and diversity as "
            "detailed in the Equal Opportunity and Diversity Policy."
        ),
    },
    {
        "question": "Can employees use personal devices for work?",
        "expected_output": (
            "Personal device usage for work is governed by the Bring Your Own "
            "Device (BYOD) Policy."
        ),
    },
    {
        "question": "What is the policy for employees working from home?",
        "expected_output": (
            "The Home Working Policy outlines the conditions and expectations "
            "for remote work arrangements."
        ),
    },
    {
        "question": "What is the alcohol policy in the workplace?",
        "expected_output": (
            "The Alcohol Policy prohibits being under the influence of alcohol "
            "in the workplace."
        ),
    },
    {
        "question": "What are the grievance procedures?",
        "expected_output": (
            "Employees can raise grievances through the formal procedure "
            "outlined in the Grievance Procedures document."
        ),
    },
    # --- Out-of-scope question — bot MUST reply with the guardrail phrase ---
    {
        "question": "What is the company's stock option vesting schedule?",
        "expected_output": "I cannot find this in the HR policy.",
    },
]


# ---------------------------------------------------------------------------
# Helper: run a single question through the bot and build an LLMTestCase
# ---------------------------------------------------------------------------

def make_test_case(item: dict) -> LLMTestCase:
    """
    Query the HR bot and construct a DeepEval LLMTestCase from the result.

    Args:
        item: Dict with keys "question" and "expected_output".

    Returns:
        A populated LLMTestCase ready for metric evaluation.
    """
    result = query_hr_bot(item["question"], RAG_CHAIN, RETRIEVER)

    return LLMTestCase(
        # The user's question
        input=result["question"],

        # The bot's actual answer (what we're evaluating)
        actual_output=result["answer"],

        # The expected / reference answer (used by Recall & Precision metrics)
        expected_output=item["expected_output"],

        # The raw retrieved chunks — CRITICAL for Faithfulness metric
        # DeepEval checks that actual_output only contains claims from these chunks
        retrieval_context=result["source_documents"],
    )


# ---------------------------------------------------------------------------
# pytest-style test functions (run with: pytest test_deepeval.py -v)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item", GOLDEN_DATASET)
def test_answer_relevancy(item):
    """Each answer must be relevant to the question asked."""
    test_case = make_test_case(item)
    eval_model = get_eval_model()
    metric = AnswerRelevancyMetric(threshold=0.7, model=eval_model, async_mode=False)
    metric.measure(test_case)
    assert metric.score >= metric.threshold, (
        f"AnswerRelevancy too low ({metric.score:.2f}): {metric.reason}"
    )


@pytest.mark.parametrize("item", GOLDEN_DATASET)
def test_faithfulness(item):
    """Each answer must be grounded in the retrieved context (no hallucination)."""
    test_case = make_test_case(item)
    eval_model = get_eval_model()
    metric = FaithfulnessMetric(threshold=0.7, model=eval_model, async_mode=False)
    metric.measure(test_case)
    assert metric.score >= metric.threshold, (
        f"Faithfulness too low ({metric.score:.2f}): {metric.reason}"
    )


# ---------------------------------------------------------------------------
# Standalone batch evaluation (run with: python test_deepeval.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n[INFO] Building test cases from golden dataset...")
    test_cases = [make_test_case(item) for item in GOLDEN_DATASET]

    eval_model = get_eval_model()
    # Define all metrics to evaluate
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=eval_model),
        FaithfulnessMetric(threshold=0.7, model=eval_model),
        ContextualRecallMetric(threshold=0.7, model=eval_model),
        ContextualPrecisionMetric(threshold=0.7, model=eval_model),
    ]

    print(f"[INFO] Running DeepEval on {len(test_cases)} test case(s) "
          f"with {len(metrics)} metric(s)...\n")

    # Run the full evaluation suite
    results = evaluate(test_cases=test_cases, metrics=metrics)

    # Print a summary table
    print("\n" + "=" * 60)
    print("  DeepEval Results Summary")
    print("=" * 60)
    for tc in test_cases:
        print(f"\nQuestion: {tc.input[:80]}")
        print(f"Answer  : {tc.actual_output[:120]}")
