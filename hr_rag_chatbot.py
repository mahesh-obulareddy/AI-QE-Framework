"""
hr_rag_chatbot.py
-----------------
HR Policy RAG Chatbot built with LangChain + Ollama + ChromaDB.

Designed to be benchmarked with the DeepEval framework.
The core function `query_hr_bot()` returns both the final answer
AND the raw retrieved source documents so that DeepEval metrics
like Faithfulness and ContextualRelevancy can be evaluated.

Usage:
    from hr_rag_chatbot import build_rag_chain, query_hr_bot

    # Build once at startup (indexes PDFs into Chroma)
    rag_chain, retriever = build_rag_chain()

    # Query at any time
    result = query_hr_bot("What is the maternity leave policy?", rag_chain, retriever)
    print(result["answer"])
    print(result["source_documents"])
"""

import os
import shutil
import argparse
from typing import Any

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Constants — tweak these without touching the rest of the code
# ---------------------------------------------------------------------------

# Path to the directory containing your HR policy PDFs.
# Place all PDFs inside a folder named `hr_documents` at the project root.
HR_DOCS_DIR: str = "./hr_documents"

# HuggingFace embedding model (runs locally, no API key required).
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Chroma vector store persistence directory.
CHROMA_PERSIST_DIR: str = "./chroma_db"

# Number of relevant chunks to retrieve per query.
TOP_K_RESULTS: int = 7

# Chunk size (in characters) and overlap for the text splitter.
CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 250


# ---------------------------------------------------------------------------
# System prompt — strictly grounds the model to the retrieved HR context
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are an accurate and helpful HR Policy assistant for an organization.
Your job is to answer employee questions thoroughly and accurately, strictly based on the HR policy \
documents provided as context below.

Rules you MUST follow without exception:
1. Base your answer strictly and exclusively on the provided context below.
2. Do NOT use external assumptions, prior knowledge, or information outside the context.
3. Provide clear, comprehensive, and well-structured answers using bullet points where appropriate, citing specific numbers, timeframes, entitlements, and conditions found in the context.
4. If the context does not contain sufficient information to answer the question, you MUST reply with exactly: "I cannot find this in the HR policy."
5. Be professional, objective, and clear.

Context from HR Policy Documents:
{context}
"""


# ---------------------------------------------------------------------------
# Step 1 — Document Ingestion
# ---------------------------------------------------------------------------

def load_hr_documents(docs_dir: str = HR_DOCS_DIR) -> list:
    """
    Load all PDF files from the specified directory using PyPDFDirectoryLoader.

    Args:
        docs_dir: Path to the folder containing HR policy PDF files.

    Returns:
        A list of LangChain Document objects, one per PDF page.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If the directory contains no PDF files.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(
            f"HR documents directory not found: '{docs_dir}'\n"
            f"Please create the directory and place your HR policy PDFs inside it."
        )

    print(f"[INFO] Loading PDFs from: {docs_dir}")
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()

    if not documents:
        raise ValueError(
            f"No PDF documents found in '{docs_dir}'. "
            f"Ensure the directory contains at least one .pdf file."
        )

    print(f"[INFO] Loaded {len(documents)} page(s) from PDF documents.")
    return documents


# ---------------------------------------------------------------------------
# Step 2 — Text Splitting
# ---------------------------------------------------------------------------

def split_documents(
    documents: list,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list:
    """
    Split raw PDF pages into smaller, overlapping chunks for embedding.

    RecursiveCharacterTextSplitter splits on paragraph -> sentence -> word
    boundaries in order, keeping semantically coherent chunks.

    Args:
        documents:     List of Document objects from the PDF loader.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks
                       (preserves context across chunk boundaries).

    Returns:
        A list of smaller Document chunks ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split on paragraph -> sentence -> word -> character (priority order)
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(
        f"[INFO] Split into {len(chunks)} chunk(s) "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Embeddings & Vector Store
# ---------------------------------------------------------------------------

def build_vector_store(
    chunks: list,
    persist_dir: str = CHROMA_PERSIST_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    """
    Embed document chunks and persist them in a local Chroma vector database.

    If a persisted Chroma database already exists at `persist_dir`, it loads
    that instead of re-indexing (avoids expensive re-embedding on restart).

    Args:
        chunks:          List of text chunks to embed.
        persist_dir:     Directory where Chroma will persist the vector index.
        embedding_model: HuggingFace sentence-transformer model name.

    Returns:
        A Chroma vector store instance ready for similarity search.
    """
    print(f"[INFO] Initializing HuggingFace embeddings with model: {embedding_model}")

    # HuggingFaceEmbeddings runs the model locally via sentence-transformers.
    # No API key required. First run downloads the model (~90 MB).
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},       # change to "cuda" if GPU is available
        encode_kwargs={"normalize_embeddings": True},
    )

    # Check if a persisted DB already exists to avoid re-indexing
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"[INFO] Loading existing Chroma DB from: {persist_dir}")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    else:
        print(f"[INFO] Building new Chroma DB and persisting to: {persist_dir}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

    return vector_store


# ---------------------------------------------------------------------------
# Step 4 — RAG Chain Assembly
# ---------------------------------------------------------------------------

def build_rag_chain(
    model_name: str = "llama3.1",
    top_k: int = TOP_K_RESULTS,
    force_reindex: bool = False,
) -> tuple:
    """
    Build and return the full RAG chain and retriever.

    This function orchestrates the full pipeline:
        PDF Loader -> Text Splitter -> Chroma Vector Store ->
        Retriever -> Prompt -> ChatOllama -> Answer

    Build this once at startup, then reuse the returned objects
    for every call to `query_hr_bot()`.

    Args:
        model_name:    Ollama model to use. Must be pulled first via:
                           ollama pull llama3.1
                       Other tested options: "mistral", "phi3", "gemma2",
                       "deepseek-r1".
        top_k:         Number of document chunks to retrieve per query.
        force_reindex: If True, deletes any existing Chroma DB and re-indexes
                       all PDFs. Use this whenever HR documents are updated.

    Returns:
        A tuple of (rag_chain, retriever):
            rag_chain  — LangChain Runnable; call .invoke(question_str)
            retriever  — Vector store retriever; call .invoke(question_str)
                         to get raw source Documents for DeepEval.
    """
    # --- Optional: force a clean re-index ---
    if force_reindex and os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("[INFO] Existing Chroma DB deleted. Re-indexing from scratch.")

    # --- Load, split, embed ---
    raw_docs = load_hr_documents()
    chunks = split_documents(raw_docs)
    vector_store = build_vector_store(chunks)

    # --- Retriever ---
    # Similarity search retrieves the most semantically relevant policy passages.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # --- LLM ---
    # temperature=0 -> deterministic outputs, required for reliable DeepEval scoring.
    print(f"[INFO] Initializing ChatOllama with model: '{model_name}' (temperature=0)")
    llm = ChatOllama(model=model_name, temperature=0)

    # --- Prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # --- Helper: format retrieved docs into a readable context string ---
    def format_docs(docs: list) -> str:
        """
        Concatenate retrieved document chunks into a single context string.
        Includes source filename and page number for traceability.
        """
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')} | "
            f"Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    # --- Assemble the LCEL chain ---
    # Flow: question -> {context: retrieve+format, question: passthrough}
    #       -> prompt -> llm -> parse string output
    rag_chain = (
        {
            # Retrieve top_k relevant chunks and format them as {context}
            "context": retriever | format_docs,
            # Pass the question string through unchanged as {question}
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[INFO] RAG chain built and ready.\n")
    return rag_chain, retriever


# ---------------------------------------------------------------------------
# Step 5 — Public Query Function (DeepEval-compatible interface)
# ---------------------------------------------------------------------------

def query_hr_bot(
    question: str,
    rag_chain: Any,
    retriever: Any,
) -> dict:
    """
    Query the HR RAG chatbot and return a structured result for DeepEval.

    This is the PRIMARY interface for DeepEval benchmarking. It deliberately
    exposes the raw retrieved source documents as a list[str] so you can
    directly pass them into DeepEval's `retrieval_context` parameter for
    Faithfulness, ContextualPrecision, and AnswerRelevancy metrics.

    Args:
        question:   The employee's HR policy question (plain string).
        rag_chain:  The RAG chain returned by `build_rag_chain()`.
        retriever:  The vector store retriever returned by `build_rag_chain()`.

    Returns:
        dict with the following keys:

        {
            "question":         str         original question
            "answer":           str         model's grounded answer
            "source_documents": list[str]   page_content of each retrieved chunk
                                            -> use as DeepEval `retrieval_context`
            "source_metadata":  list[dict]  metadata per chunk (file, page number)
                                            -> useful for debugging / tracing
        }

    -----------------------------------------------------------------------
    DeepEval integration example:

        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

        result = query_hr_bot(question, rag_chain, retriever)

        test_case = LLMTestCase(
            input=result["question"],
            actual_output=result["answer"],
            retrieval_context=result["source_documents"],
        )

        evaluate(
            test_cases=[test_case],
            metrics=[
                FaithfulnessMetric(threshold=0.7),
                AnswerRelevancyMetric(threshold=0.7),
            ],
        )
    -----------------------------------------------------------------------
    """
    # Retrieve source documents independently so we have the raw Document
    # objects BEFORE they are formatted into the context string.
    # This gives DeepEval the unmodified chunk text for evaluation.
    retrieved_docs: list = retriever.invoke(question)

    # Run the full RAG chain to produce the grounded answer.
    answer: str = rag_chain.invoke(question)

    return {
        # Original question -> DeepEval `input`
        "question": question,

        # Model's final answer -> DeepEval `actual_output`
        "answer": answer,

        # List of raw chunk strings -> DeepEval `retrieval_context`
        # DeepEval expects list[str], NOT list[Document]
        "source_documents": [doc.page_content for doc in retrieved_docs],

        # Metadata for debugging: which file + page each chunk came from
        "source_metadata": [doc.metadata for doc in retrieved_docs],
    }


# ---------------------------------------------------------------------------
# CLI Smoke Test — run `python hr_rag_chatbot.py` to verify setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HR RAG Chatbot — Smoke Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1",
        help="Ollama model name (must be pulled first: ollama pull <model>)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the maternity leave policy?",
        help="Question to ask the chatbot",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-index of all PDFs (use when HR documents change)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  HR Policy RAG Chatbot — Smoke Test")
    print("=" * 60)

    # Build the RAG chain once (expensive; do this once per session)
    chain, ret = build_rag_chain(model_name=args.model, force_reindex=args.reindex)

    # Run the query and print the structured result
    result = query_hr_bot(args.question, chain, ret)

    print(f"\n{'='*60}")
    print(f"Question : {result['question']}")
    print(f"{'='*60}")
    print(f"Answer   :\n{result['answer']}")
    print(f"\n--- Retrieved {len(result['source_documents'])} source chunk(s) ---")
    for i, (text, meta) in enumerate(
        zip(result["source_documents"], result["source_metadata"]), start=1
    ):
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        preview = text[:300] + "..." if len(text) > 300 else text
        print(f"\n[Chunk {i}] {src} | Page {page}")
        print(preview)
