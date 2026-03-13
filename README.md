# 🔬 AI Research Assistant

> An intelligent, multi-feature research assistant built with Streamlit and LangChain that lets you chat with your research papers, perform real-time web searches, analyze paper structure, compare studies, identify research gaps, and get peer-reviewer style critiques — all powered by Groq's ultra-fast LLaMA inference.

---

## 📌 GitHub Description

> AI-powered research assistant that enables RAG-based Q&A over uploaded PDFs with automatic web search fallback, paper analysis, multi-paper comparison, critical review generation, and research gap identification — built with Streamlit, LangChain, FAISS, and Groq.

---

## 📸 Features at a Glance

| Page | What it does |
|---|---|
| 💬 **Chat** | Ask questions about your papers. RAG retrieval first, DuckDuckGo web search fallback if no relevant context found. Sources always shown. |
| 📊 **Paper Analyzer** | Auto-extracts problem statement, methodology, results, limitations, datasets, and a reproducibility checklist from any uploaded paper. |
| ⚖️ **Compare Papers** | Generates a structured side-by-side markdown comparison table across all uploaded papers. |
| 🔬 **Critical Review** | Produces a peer-reviewer style critique: weaknesses, missing baselines, statistical concerns, and pointed questions for authors. |
| 🗺️ **Research Gaps** | Synthesizes across all loaded papers to surface open problems, contradictions, and unexplored directions. |
| 🧠 **Explain Concept** | Explains any technical term at three levels — Simple, Graduate, and Expert — optionally grounded in a specific paper. |

---

## 🗂️ Project Structure

```
your_project/
│
├── app.py                        # Main Streamlit app — all pages and UI
│
├── models/
│   ├── llm.py                    # Groq LLaMA model initialisation
│   └── embeddings.py             # HuggingFace sentence-transformer embeddings
│
└── utils/
    ├── rag_utils.py              # PDF ingestion, FAISS vector search, web search
    └── analysis_utils.py         # All LLM-powered analysis features
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the root of the project:

```env
GROQ_API_KEY=your_groq_api_key_here
SERP_API_KEY=your_serp_api_key_here
```

> **How to get your keys:**
> - **GROQ_API_KEY** — Sign up at [console.groq.com](https://console.groq.com), create a new API key from the dashboard. Free tier is available and sufficient for this project.
> - **SERP_API_KEY** — Sign up at [serpapi.com](https://serpapi.com). Used as an optional enhanced web search backend. The app uses DuckDuckGo by default (no key needed), but SERP API provides more reliable results.

Alternatively, export them directly in your terminal:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
export SERP_API_KEY="your_serp_api_key_here"
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
streamlit
python-dotenv
langchain
langchain-core
langchain-community
langchain-groq
faiss-cpu
sentence-transformers
pypdf
duckduckgo-search
```

---

## 🚀 How to Use

### Chat with your papers
1. Upload one or more PDF research papers using the **sidebar file uploader**
2. Wait for the **"✅ Indexed N paper(s)"** confirmation
3. Type your question in the chat input
4. The assistant will:
   - Search your documents first (RAG)
   - Fall back to **real-time web search** if no relevant content is found
   - Always display sources — 📄 for PDFs, 🌐 for web results

### Paper Analyzer
1. Upload a PDF
2. Navigate to **📊 Paper Analyzer**
3. Click **"Analyze → filename.pdf"**
4. View the structured breakdown including reproducibility checklist

### Compare Papers
1. Upload **at least 2 PDFs**
2. Navigate to **⚖️ Compare Papers**
3. Click **"Generate Comparison Table"**
4. Export the result as a `.md` file

### Critical Review
1. Upload a PDF
2. Navigate to **🔬 Critical Review**
3. Select the paper from the dropdown
4. Click **"Generate Review"** for a structured NeurIPS/ICML style critique

### Research Gaps
1. Upload one or more PDFs
2. Navigate to **🗺️ Research Gaps**
3. Click **"Find Research Gaps"** to surface open problems across your loaded papers

### Explain Concept
1. Navigate to **🧠 Explain Concept**
2. Type any technical term (e.g. `attention mechanism`, `LoRA`, `RLHF`)
3. Choose your explanation depth: Simple / Graduate / Expert
4. Optionally ground the explanation in one of your uploaded papers

---

## 🔍 How RAG + Web Search Works

```
User Question
     │
     ▼
Is a PDF uploaded?
     │
    YES ──► FAISS similarity search over paper chunks
     │           │
     │      Relevant? ──► YES ──► Answer from PDF  ──► 📄 Show page sources
     │           │
     │           NO
     │
    NO / Not Relevant
     │
     ▼
DuckDuckGo real-time web search (utils/rag_utils.py)
     │
     ▼
Answer from web  ──► 🌐 Show clickable web sources
```

Web search is handled entirely inside `utils/rag_utils.py` via the `web_search()` function using the `duckduckgo-search` library — no extra API key required for basic usage.

---

## 🧠 Model Details

| Component | Model / Library |
|---|---|
| LLM | `llama-3.1-8b-instant` via Groq API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (in-memory, per session) |
| Web Search | DuckDuckGo Search (`duckduckgo-search`) |
| PDF Loader | LangChain `PyPDFLoader` |

---

## ⚠️ Important Notes

- The FAISS vector index is **in-memory only** — it resets when you restart the app or clear papers. There is no persistent disk storage by design.
- PDFs are processed on upload and **temporarily written to disk** during ingestion, then immediately deleted.
- The app uses **0-indexed page numbers internally** from PyPDFLoader, which are converted to **1-indexed** for display so "Page 1" in the source citation correctly refers to the first page of your PDF.
- For very large PDFs, paper analysis features trim content to fit within the LLM context window (first ~9000 chars + last ~3000 chars), capturing abstract, introduction, methodology, and conclusion.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE)
