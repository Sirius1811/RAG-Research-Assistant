import os

# ==============================
# API KEYS
# ==============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERP_API_KEY = os.getenv("SERP_API_KEY", "")  # optional fallback

# ==============================
# MODEL SETTINGS
# ==============================

GROQ_MODEL = "llama-3.1-8b-instant"

# ==============================
# EMBEDDING SETTINGS
# ==============================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==============================
# VECTOR STORE SETTINGS
# ==============================

VECTOR_DB_PATH = "vector_store"

# ==============================
# RAG SETTINGS
# ==============================

TOP_K_RESULTS = 4
SIMILARITY_THRESHOLD = 1.0   # FAISS L2 distance — lower = more similar; 1.0 is a reasonable cutoff

# ==============================
# WEB SEARCH SETTINGS
# ==============================

WEB_SEARCH_MAX_RESULTS = 3

# ==============================
# CHUNK SETTINGS
# ==============================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100