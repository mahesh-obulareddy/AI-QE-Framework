# AI-QE-Framework — HR Policy RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers employee HR questions
by reading local PDF policy documents. Built for benchmarking with the **DeepEval** framework.

---

## Architecture

```
hr_documents/  (PDF files)
     ↓
PyPDFDirectoryLoader  →  RecursiveCharacterTextSplitter
     ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
     ↓
Chroma Vector DB (local, persisted in ./chroma_db/)
     ↓
Retriever (MMR search, top-5 chunks)
     ↓
ChatOllama (llama3.2, temp=0)  +  Strict System Prompt
     ↓
query_hr_bot() → { answer, source_documents, source_metadata }
     ↓
DeepEval Metrics (Faithfulness, AnswerRelevancy, ContextualRecall, ContextualPrecision)
```

---

## Prerequisites

### 1. Python 3.11+
```bash
python --version
```

### 2. Ollama (local LLM server)
```bash
# Install: https://ollama.com/download
ollama pull llama3.2        # default model
ollama pull mistral         # optional: test with other models
ollama pull phi3
ollama pull gemma2
```

### 3. HR Documents
Place all PDF policy files in a directory named **exactly** `hr_documents/` at the project root.
```bash
mkdir -p hr_documents
# Copy your PDF files in here
cp /path/to/your/pdfs/*.pdf hr_documents/
```

> **Note:** The existing `hr-documents/` folder (with a hyphen) needs to be copied or symlinked:
> ```bash
> cp -r hr-documents/* hr_documents/
> ```

---

## Installation

```bash
# Clone / navigate to project
cd AI-QE-Framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Smoke Test
```bash
# Default model (llama3.2), default question
python hr_rag_chatbot.py

# Custom model and question
python hr_rag_chatbot.py --model mistral --question "What is the paternity leave policy?"

# Force re-index PDFs (run after adding/updating documents)
python hr_rag_chatbot.py --reindex
```

### Use as a Module
```python
from hr_rag_chatbot import build_rag_chain, query_hr_bot

# Build once per session
rag_chain, retriever = build_rag_chain(model_name="llama3.2")

# Query
result = query_hr_bot("What is the maternity leave policy?", rag_chain, retriever)

print(result["answer"])           # The grounded answer
print(result["source_documents"]) # Raw chunks for DeepEval
print(result["source_metadata"])  # File + page references
```

---

## Running DeepEval Benchmarks

DeepEval metrics (Faithfulness, AnswerRelevancy, etc.) use an LLM judge
(default: `gpt-4o-mini`). You need an **OpenAI API key** for the judge:

```bash
export OPENAI_API_KEY=sk-...
```

### Run full evaluation suite
```bash
python test_deepeval.py
```

### Run with pytest (recommended)
```bash
pytest test_deepeval.py -v

# Run only faithfulness tests
pytest test_deepeval.py -v -k "faithfulness"
```

### Benchmark multiple Ollama models
Edit `MODEL_NAME` in `test_deepeval.py` and re-run:
```python
MODEL_NAME = "mistral"   # change this line
```

---

## Key Design Decisions

| Design Choice | Rationale |
|---|---|
| `PyPDFDirectoryLoader` | Loads all PDFs from a folder in one call, preserving page metadata |
| `RecursiveCharacterTextSplitter` | Splits on semantic boundaries (paragraph → sentence → word) |
| `all-MiniLM-L6-v2` | Fast, lightweight local embedding model; no API key needed |
| Chroma (persistent) | Re-uses index across restarts; call `--reindex` after doc changes |
| MMR retrieval | Reduces redundancy in retrieved chunks vs. pure cosine similarity |
| `temperature=0` | Deterministic outputs required for reproducible DeepEval scores |
| Guardrail phrase | `"I cannot find this in the HR policy."` — testable exact string for out-of-scope detection |
| `query_hr_bot()` returns `dict` | Exposes `source_documents` as `list[str]` for DeepEval `retrieval_context` |

---

## File Structure

```
AI-QE-Framework/
├── hr_documents/           <- Place all HR PDF files here
│   ├── Alcohol-Policy.pdf
│   ├── Maternity-Policy.pdf
│   └── ...
├── chroma_db/              <- Auto-created; Chroma vector index (gitignore this)
├── hr_rag_chatbot.py       <- Core RAG pipeline + query_hr_bot()
├── test_deepeval.py        <- DeepEval benchmark test suite
├── requirements.txt        <- Python dependencies
└── README.md
```
