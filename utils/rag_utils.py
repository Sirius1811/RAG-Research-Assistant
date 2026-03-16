"""
utils/rag_utils.py
------------------
Handles all RAG operations (PDF ingestion, vector search)
and real-time web search fallback via DuckDuckGo.

Web search is integrated here per project requirements.
"""

import os
import tempfile
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from googlesearch import search
from newspaper import Article
import requests
from bs4 import BeautifulSoup
from readability import Document

from tavily import TavilyClient

from models.embeddings import get_embedding_model

# Use config constants everywhere — never hardcode values that live in config.py
from config.config import (
    SIMILARITY_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    TAVILY_API_KEY,
    WEB_SEARCH_MAX_RESULTS,
)


# -------------------------------------------------------
# Embedding model — cached so it loads only once
# -------------------------------------------------------

@lru_cache(maxsize=1)
def _cached_embeddings():
    return get_embedding_model()


# -------------------------------------------------------
# PDF Processing
# -------------------------------------------------------

def process_pdfs(uploaded_files):
    """
    Ingest a list of Streamlit UploadedFile objects.

    Returns:
        vector_db   (FAISS)  — for similarity search
        paper_texts (dict)   — {filename: full_text} for analysis features
    """
    documents   = []
    paper_texts = {}

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs   = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name
            documents.extend(docs)
            paper_texts[file.name] = "\n\n".join(d.page_content for d in docs)
        finally:
            os.unlink(tmp_path)

    if not documents:
        raise ValueError("No content could be extracted from the uploaded PDFs.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(documents)

    vector_db = FAISS.from_documents(chunks, _cached_embeddings())
    return vector_db, paper_texts


# -------------------------------------------------------
# RAG Retrieval — helpers
# -------------------------------------------------------

def _get_page_num(metadata: dict) -> str:
    """
    PyPDFLoader stores the page index under different keys depending on the
    langchain-community version installed:
      - older versions  → "page"        (0-indexed int)
      - newer versions  → "page_number" (0-indexed int)
    Try both; fall back to "?" so citations are never silently blank.
    Add 1 to convert 0-indexed to the human-readable page number.
    """
    for key in ("page", "page_number"):
        val = metadata.get(key)
        if isinstance(val, int):
            return str(val + 1)
    return "?"


# -------------------------------------------------------
# RAG Retrieval
# -------------------------------------------------------

def retrieve_context(vector_db, query):
    """
    Run similarity search against the vector store.

    Returns:
        context_text  (str)   — chunks that passed the relevance threshold (fed to LLM)
        sources       (list)  — citation card for every unique page that was examined,
                                 regardless of threshold, so the caller can decide
                                 whether to surface them (e.g. only when relevant=True)
        is_relevant   (bool)  — True if ≥1 chunk passed SIMILARITY_THRESHOLD
    """
    try:
        retrieved = vector_db.similarity_search_with_score(query, k=TOP_K_RESULTS)
    except Exception as e:
        print(f"[retrieve_context] vector search error: {e}")
        return "", [], False

    if not retrieved:
        return "", [], False

    context_chunks = []
    sources        = []
    # Deduplicate per (filename, page) — not on the full citation string —
    # so multiple chunks from the same page produce exactly one citation card.
    seen_pages     = set()
    is_relevant    = False

    for doc, score in retrieved:
        source   = doc.metadata.get("source", "Unknown")
        page_num = _get_page_num(doc.metadata)

        # Only create citations for relevant chunks
        if score < SIMILARITY_THRESHOLD:

            is_relevant = True
            context_chunks.append(doc.page_content)

            dedup_key = f"{source}::p{page_num}"

            if dedup_key not in seen_pages:

                seen_pages.add(dedup_key)

                flat_text = " ".join(doc.page_content.split())
                excerpt = flat_text[:160] + ("…" if len(flat_text) > 160 else "")

                sources.append(
                    f"📄 **{source}** — Page {page_num}\n"
                    f"> *\"{excerpt}\"*"
        )

    return "\n\n".join(context_chunks), sources, is_relevant


# -------------------------------------------------------
# Real-time Web Search
# -------------------------------------------------------

tavily = TavilyClient(api_key=TAVILY_API_KEY)


def web_search(query):
    """
    Uses Tavily search API optimized for LLM applications.

    Returns:
        context_text (str) — text snippets used for answering
        sources (list)     — formatted source cards with links
    """

    try:

        response = tavily.search(
            query=query,
            max_results=5
        )

        results = []
        sources = []

        for r in response.get("results", []):

            content = r.get("content", "").strip()
            url = r.get("url", "")
            title = r.get("title", url)

            if not content:
                continue

            results.append(content)

            sources.append(
                f"🌐 **[{title}]({url})**\n"
                f"> *\"{content[:160]}...\"*"
            )

        return "\n\n".join(results), sources

    except Exception as e:
        print("[web_search] Tavily error:", e)
        return "", []
