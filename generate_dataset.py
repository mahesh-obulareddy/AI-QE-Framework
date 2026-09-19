"""
generate_dataset.py
-------------------
Generates ~30 single-turn goldens from the HR policy PDFs using DeepEval's
Synthesizer with GeminiModel (matching the eval model already configured).

Run once:
    source .venv/bin/activate && python generate_dataset.py

Output: tests/evals/.dataset.json
"""

import os
from dotenv import load_dotenv

load_dotenv()

# GeminiModel needs GOOGLE_API_KEY (or GEMINI_API_KEY) in env
from deepeval.synthesizer import Synthesizer
from deepeval.models import GeminiModel
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.synthesizer.config import ContextConstructionConfig
from sentence_transformers import SentenceTransformer


class HFEmbedder(DeepEvalBaseEmbeddingModel):
    """Thin DeepEval wrapper around the same sentence-transformer model
    the HR RAG chatbot uses for Chroma — all-MiniLM-L6-v2, already cached locally."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")

    def load_model(self):
        return self._model

    def embed_text(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    async def a_embed_text(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return "all-MiniLM-L6-v2"

DOCS_DIR = "./hr_documents"
OUTPUT_PATH = "./tests/evals/.dataset.json"
TARGET_GOLDENS = 30  # ~2 per document across 16 PDFs

# Collect all PDF paths
pdf_paths = [
    os.path.join(DOCS_DIR, f)
    for f in sorted(os.listdir(DOCS_DIR))
    if f.lower().endswith(".pdf")
]

print(f"[INFO] Found {len(pdf_paths)} HR policy PDFs")
print(f"[INFO] Targeting ~{TARGET_GOLDENS} goldens")

model = GeminiModel(model="gemini-2.0-flash")

synthesizer = Synthesizer(
    model=model,
    async_mode=False,   # sync for stability on local runs
)

synthesizer.generate_goldens_from_docs(
    document_paths=pdf_paths,
    include_expected_output=True,
    max_goldens_per_context=2,    # ~2 per context chunk across 16 docs → ~30 goldens
    context_construction_config=ContextConstructionConfig(
        # Use the same local sentence-transformer model the app uses for retrieval
        # — no OpenAI or external embedding API needed.
        embedder=HFEmbedder(),
        critic_model=model,          # reuse the Gemini model as quality critic
        max_contexts_per_document=2,
        chunk_size=1024,
        chunk_overlap=100,
        # Lower threshold: HR PDFs contain template placeholders like [Insert X]
        # which score low on quality but are still valid policy content.
        context_quality_threshold=0.0,
        context_similarity_threshold=0.0,
    ),
)

print(f"\n[INFO] Generated {len(synthesizer.synthetic_goldens)} goldens")

synthesizer.save_as(file_type="json", directory="./tests/evals", file_name="dataset")
print(f"[INFO] Saved to ./tests/evals/dataset.json")
