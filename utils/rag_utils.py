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
        # Write to a named temp file so PyPDFLoader can read it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs   = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name   # tag with original filename
            documents.extend(docs)
            paper_texts[file.name] = "\n\n".join(d.page_content for d in docs)
        finally:
            os.unlink(tmp_path)   # clean up temp file immediately

    if not documents:
        raise ValueError("No content could be extracted from the uploaded PDFs.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    ).split_documents(documents)

    vector_db = FAISS.from_documents(chunks, _cached_embeddings())
    return vector_db, paper_texts


# -------------------------------------------------------
# RAG Retrieval
# -------------------------------------------------------

def retrieve_context(vector_db, query):
    """
    Run similarity search against the vector store.

    Returns:
        context_text  (str)   — concatenated relevant chunks
        sources       (list)  — e.g. ["paper.pdf | Page 4", ...]
        is_relevant   (bool)  — True if at least one chunk passed threshold
    """
    try:
        retrieved = vector_db.similarity_search_with_score(query, k=3)
    except Exception as e:
        print(f"[retrieve_context] vector search error: {e}")
        return "", [], False

    if not retrieved:
        return "", [], False

    context_chunks = []
    sources        = []
    seen           = set()
    is_relevant    = False

    for doc, score in retrieved:
        # FAISS L2 distance — lower is more similar; 1.0 is a safe cutoff
        if score < 1.0:
            is_relevant = True
            context_chunks.append(doc.page_content)

            source   = doc.metadata.get("source", "Unknown")
            raw_page = doc.metadata.get("page", None)
            # PyPDFLoader is 0-indexed → +1 for human-readable page numbers
            page_num = (raw_page + 1) if isinstance(raw_page, int) else "?"

            citation = f"{source} | Page {page_num}"
            if citation not in seen:
                seen.add(citation)
                sources.append(citation)

    return "\n\n".join(context_chunks), sources, is_relevant


# -------------------------------------------------------
# Real-time Web Search  ← integrated here per requirements
# -------------------------------------------------------

def web_search(query):
    """
    Perform a real-time web search using DuckDuckGo.
    Called automatically when RAG finds no relevant context.

    Returns:
        context_text  (str)   — concatenated result snippets
        sources       (list)  — formatted markdown links, e.g. ["[Title](url)", ...]
    """
    results = []
    sources = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                body  = r.get("body",  "").strip()
                link  = r.get("href",  "").strip()
                title = r.get("title", link).strip()

                if body:
                    results.append(body)
                if link:
                    # Markdown link so it renders as clickable in Streamlit
                    sources.append(f"[{title}]({link})")

    except Exception as e:
        print(f"[web_search] DuckDuckGo error: {e}")
        return "", []

    return "\n\n".join(results), sources
