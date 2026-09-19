"""
tests/evals/test_hr_rag_chatbot.py
----------------------------------
DeepEval build-loop test suite for the HR Policy RAG Chatbot.

Uses:
  - LangChain integration (deepeval.integrations.langchain.CallbackHandler)
    to auto-instrument chain invocations, LLM calls, and retriever spans.
  - DeepEval RAG metrics: AnswerRelevancy, Faithfulness, ContextualRecall,
    and ContextualPrecision.
  - assert_test() to produce structured trace output, pass/fail status,
    and metric failure explanations for the agent build loop.

Run via DeepEval CLI:
    deepeval test run tests/evals/test_hr_rag_chatbot.py
"""

import os
import time
import pytest
import requests
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from hr_rag_chatbot import build_rag_chain, query_hr_bot
from langchain_ollama import ChatOllama

load_dotenv()

# ---------------------------------------------------------------------------
# Judge Model Setup (Gemini fast judge with retries, Ollama fallback)
# ---------------------------------------------------------------------------

class GeminiJudge(DeepEvalBaseLLM):
    """Gemini judge for DeepEval using Google's Generative Language API with retry logic."""

    def __init__(self, model_name: str = "gemini-flash-latest"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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

        for attempt in range(5):
            try:
                res = requests.post(url, json=payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    texts = [p["text"] for p in parts if "text" in p]
                    return "\n".join(texts)
                elif res.status_code in (429, 503):
                    time.sleep(2 * (attempt + 1))
                else:
                    res.raise_for_status()
            except Exception as e:
                if attempt == 4:
                    raise e
                time.sleep(2)
        return ""

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


class OllamaJudge(DeepEvalBaseLLM):
    """Local Ollama judge fallback."""

    def __init__(self, model_name: str = "llama3.1"):
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


def get_eval_judge():
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return GeminiJudge(model_name="gemini-flash-latest")
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    return OllamaJudge(model_name="llama3.1")


# ---------------------------------------------------------------------------
# Shared RAG Chain Fixture (session-scoped to avoid rebuilding per test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hr_bot():
    """Build the RAG chain and retriever once for the evaluation session."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    rag_chain, retriever = build_rag_chain(model_name=model_name)
    return {"chain": rag_chain, "retriever": retriever}


@pytest.fixture(scope="session")
def eval_judge():
    """Return configured judge model."""
    return get_eval_judge()


# ---------------------------------------------------------------------------
# Golden Q&A Dataset for Build Loop
# ---------------------------------------------------------------------------

GOLDEN_EVAL_CASES = [
    {
        "id": "maternity_policy",
        "question": "What is the company's maternity leave policy?",
        "expected_output": (
            "Eligible employees are entitled to statutory and contractual maternity leave "
            "as outlined in the Maternity Policy document."
        ),
    },
    {
        "id": "resignation_notice",
        "question": "How many days notice is required for resignation?",
        "expected_output": (
            "Notice periods vary depending on employee grade, role, and length of service "
            "as set out in the Notice Periods Policy."
        ),
    },
    {
        "id": "probationary_period",
        "question": "What is the probationary period for new employees?",
        "expected_output": (
            "New employees typically serve a probationary period as outlined in the "
            "Probationary Periods Policy, during which suitability is assessed."
        ),
    },
    {
        "id": "out_of_scope_guardrail",
        "question": "What is the company's stock option vesting schedule?",
        "expected_output": "I cannot find this in the HR policy.",
    },
]


# ---------------------------------------------------------------------------
# Evaluated Tests with LangChain CallbackHandler Tracing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", GOLDEN_EVAL_CASES, ids=[c["id"] for c in GOLDEN_EVAL_CASES])
def test_hr_rag_response_quality(hr_bot, eval_judge, case):
    """
    Test each golden input using LangChain auto-instrumentation and DeepEval metrics.
    """
    # 1. Instantiate DeepEval LangChain CallbackHandler for tracing
    cb = CallbackHandler()

    # 2. Invoke the RAG chain with callback tracking
    result = query_hr_bot(
        question=case["question"],
        rag_chain=hr_bot["chain"],
        retriever=hr_bot["retriever"],
        callbacks=[cb],
    )

    # 3. Formulate DeepEval LLMTestCase
    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result["answer"],
        expected_output=case["expected_output"],
        retrieval_context=result["source_documents"],
    )

    # 4. Configure metrics
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=eval_judge, async_mode=False),
        FaithfulnessMetric(threshold=0.7, model=eval_judge, async_mode=False),
    ]

    # For policy questions that have retrieved context, also test recall
    if case["id"] != "out_of_scope_guardrail":
        metrics.append(ContextualRecallMetric(threshold=0.7, model=eval_judge, async_mode=False))

    # 5. Evaluate and assert
    assert_test(test_case=test_case, metrics=metrics)
