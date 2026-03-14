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
from duckduckgo_search import DDGS

from models.embeddings import get_embedding_model

# Use config constants everywhere — never hardcode values that live in config.py
from config.config import (
    SIMILARITY_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
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

        # ── Citation card (collected for every examined page) ────────────────
        # Collecting regardless of threshold means the caller gets a full picture
        # of what was searched even for follow-up questions whose score sits just
        # above the threshold.  The caller (app.py) only exposes these when
        # is_relevant=True, so off-topic questions never surface stale PDF cites.
        dedup_key = f"{source}::p{page_num}"
        if dedup_key not in seen_pages:
            seen_pages.add(dedup_key)
            flat_text = " ".join(doc.page_content.split())
            excerpt   = flat_text[:160] + ("…" if len(flat_text) > 160 else "")
            sources.append(
                f"📄 **{source}** — Page {page_num}\n"
                f"> *\"{excerpt}\"*"
            )

        # ── LLM context (only high-confidence chunks) ───────────────────────
        if score < SIMILARITY_THRESHOLD:
            is_relevant = True
            context_chunks.append(doc.page_content)

    return "\n\n".join(context_chunks), sources, is_relevant


# -------------------------------------------------------
# Real-time Web Search
# -------------------------------------------------------

def web_search(query):
    """
    Perform a real-time web search using DuckDuckGo.
    Called when RAG finds no relevant context in the uploaded PDFs.

    Returns:
        context_text  (str)   — concatenated result snippets (fed to LLM)
        sources       (list)  — formatted citation cards with clickable links
    """
    results = []
    sources = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS):
                body  = r.get("body",  "").strip()
                link  = r.get("href",  "").strip()
                title = r.get("title", link).strip()

                # Compute excerpt at the top of the loop — before either `if`
                # block — so it is always defined when sources.append runs,
                # even when body is empty for a particular result.
                excerpt = (body[:120] + "…") if len(body) > 120 else body

                if body:
                    results.append(body)

                if link:
                    if excerpt:
                        sources.append(
                            f"🌐 **[{title}]({link})**\n"
                            f"> *\"{excerpt}\"*"
                        )
                    else:
                        sources.append(f"🌐 **[{title}]({link})**")

    except Exception as e:
        print(f"[web_search] DuckDuckGo error: {e}")
        return "", []

    return "\n\n".join(results), sources
