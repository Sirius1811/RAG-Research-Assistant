"""
app.py — AI Research Assistant
================================
Imports all RAG + web search logic from utils/rag_utils.py
Imports all analysis features from utils/analysis_utils.py
"""

import os
import sys
import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.llm import get_chatgroq_model

# All RAG + web search logic lives in utils/rag_utils.py
from utils.rag_utils import process_pdfs, retrieve_context, web_search

# All analysis features live in utils/analysis_utils.py
from utils.analysis_utils import (
    analyze_paper,
    compare_papers,
    find_research_gaps,
    critical_review,
    suggest_follow_ups,
    explain_concept,
)


# ═══════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════

def init_state():
    defaults = {
        "messages":          [],
        "vector_db":         None,
        "paper_texts":       {},       # {filename: full_text}
        "indexed_files":     [],
        "paper_analyses":    {},       # {filename: analysis_dict} — cached
        "follow_up_qs":      [],
        "comparison_result": None,
        "gaps_result":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════
# LLM helper
# ═══════════════════════════════════════════════════════

def get_chat_response(chat_model, messages, system_prompt):
    try:
        formatted = [SystemMessage(content=system_prompt)]
        for m in messages:
            cls = HumanMessage if m["role"] == "user" else AIMessage
            formatted.append(cls(content=m["content"]))
        return chat_model.invoke(formatted).content
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}"


# ═══════════════════════════════════════════════════════
# Source Rendering
# ═══════════════════════════════════════════════════════

def render_sources(sources: list, source_type: str):
    """
    Renders source citations below an assistant reply.

    Each entry in `sources` is a two-line markdown string:
      PDF → "📄 **paper.pdf** — Page N\\n> *\\"excerpt…\\"*"
      Web → "🌐 **[Title](url)**\\n> *\\"excerpt…\\"*"

    Each citation must be its own st.markdown() call — wrapping in a bullet
    (`- {s}`) causes Streamlit to print the blockquote `>` literally instead
    of rendering it as an indented quote.

    IMPORTANT: always call this OUTSIDE any st.spinner() block.
    """
    if not sources:
        return
    if source_type == "pdf":
        st.markdown("**📄 Sources — from your documents:**")
    elif source_type == "web":
        st.markdown("**🌐 Sources — from web search:**")
    elif source_type == "mixed":
        st.markdown("**📚 Sources — documents + web:**")
    else:
        st.markdown("**🔍 Sources:**")
    for s in sources:
        st.markdown(s)
        st.markdown("---")


# ═══════════════════════════════════════════════════════
# Sidebar (shared across all pages)
# ═══════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.divider()

        # Navigation
        page = st.radio("Navigate", [
            "💬 Chat", "📊 Paper Analyzer", "⚖️ Compare Papers",
            "🔬 Critical Review", "🗺️ Research Gaps", "🧠 Explain Concept",
        ], label_visibility="collapsed")

        st.divider()
        st.markdown("### 📂 Upload Papers")

        uploaded = st.file_uploader(
            "PDFs", type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            new_names = sorted(f.name for f in uploaded)
            if new_names != st.session_state.indexed_files:
                with st.spinner("Indexing…"):
                    try:
                        vdb, ptexts = process_pdfs(uploaded)
                        st.session_state.vector_db     = vdb
                        st.session_state.paper_texts   = ptexts
                        st.session_state.indexed_files = new_names
                        st.session_state.paper_analyses = {}
                        st.success(f"✅ Indexed {len(uploaded)} paper(s)")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

        if st.session_state.indexed_files:
            st.markdown("**📚 Loaded:**")
            for n in st.session_state.indexed_files:
                st.markdown(f"- `{n}`")
            if st.button("🗑️ Remove All Papers"):
                st.session_state.vector_db      = None
                st.session_state.paper_texts    = {}
                st.session_state.indexed_files  = []
                st.session_state.paper_analyses = {}
                st.rerun()
        else:
            st.info("No papers loaded — web search will be used.")

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages     = []
            st.session_state.follow_up_qs = []
            st.rerun()

        if st.session_state.messages:
            lines = [f"Chat Export {datetime.datetime.now():%Y-%m-%d %H:%M}\n{'='*50}"]
            for m in st.session_state.messages:
                lines.append(f"\n[{'You' if m['role']=='user' else 'Assistant'}]\n{m['content']}")
            st.download_button("💾 Export Chat", "\n".join(lines),
                               "chat_export.txt", "text/plain")

    return page


# ═══════════════════════════════════════════════════════
# Page 1 — 💬 Chat
# ═══════════════════════════════════════════════════════

def page_chat():
    st.title("💬 Chat with Your Papers")
    mode = st.radio("Response style", ["Concise", "Detailed"], horizontal=True)
    st.divider()

    # ── Replay history with sources ────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []), msg.get("source_type", ""))

    # ── Follow-up suggestions ──────────────────────────
    if st.session_state.follow_up_qs:
        st.markdown("**💡 Suggested follow-ups — click to ask:**")
        cols = st.columns(3)
        for i, q in enumerate(st.session_state.follow_up_qs):
            if cols[i].button(q, key=f"fup_{i}"):
                st.session_state._pending = q
                st.rerun()

    pending = st.session_state.pop("_pending", None)
    prompt  = st.chat_input("Ask anything about your research papers…") or pending

    if not prompt:
        return

    # ── User turn ──────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Assistant turn ─────────────────────────────────
    # ALL computation happens inside ONE spinner so there is no
    # inter-spinner re-render that can wipe intermediate output.
    context, sources, source_type, response = "", [], "", ""
    # Must be initialised here — if vector_db is None the retrieve_context
    # block is skipped entirely, and `if not relevant` below would NameError.
    relevant = False
    model = get_chatgroq_model()

    with st.spinner("Thinking…"):
        # --- PDF Retrieval ---
        pdf_ctx, pdf_sources, relevant = "", [], False

        if st.session_state.vector_db is not None:
            pdf_ctx, pdf_sources, relevant = retrieve_context(
                st.session_state.vector_db, prompt
            )

        # --- Web Search ---
        web_ctx, web_sources = web_search(prompt)

        # --- Combine Contexts ---
        context_parts = []

        if pdf_ctx:
            context_parts.append("PDF Context:\n" + pdf_ctx)

        if web_ctx:
            context_parts.append("Web Context:\n" + web_ctx)

        context = "\n\n".join(context_parts)

        # --- Combine Sources ---
        sources = []
        source_type = "mixed"

        if pdf_sources:
            sources.extend(pdf_sources)

        if web_sources:
            sources.extend(web_sources)

        # --- LLM response ---
        instruction = (
            "Answer briefly in 2-3 sentences." if mode == "Concise"
            else "Provide a thorough, structured explanation."
        )
        system_prompt = f"""
        You are an expert AI research assistant.

        Use the provided context which may include:
        - excerpts from research papers
        - information retrieved from web articles.

        Base your answer on the provided context when possible.

        Style: {instruction}

        Context:
        {context}
        """

        response = get_chat_response(model, st.session_state.messages, system_prompt)

        # --- Follow-up suggestions (computed here, displayed on next render) ---
        try:
            st.session_state.follow_up_qs = suggest_follow_ups(prompt, response, model)
        except Exception:
            st.session_state.follow_up_qs = []

    # Render response and sources AFTER the spinner has fully closed.
    # Nothing between here and session_state.append can trigger a rerender.
    with st.chat_message("assistant"):
        st.markdown(response)
        render_sources(sources, source_type)

    # Persist the full message WITH sources so history replay shows them too.
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     response,
        "sources":     sources,
        "source_type": source_type,
    })


# ═══════════════════════════════════════════════════════
# Page 2 — 📊 Paper Analyzer
# ═══════════════════════════════════════════════════════

def page_analyzer():
    st.title("📊 Paper Analyzer")
    st.caption("Auto-extracts problem, methodology, results, limitations, reproducibility.")

    if not st.session_state.paper_texts:
        st.warning("⬅️ Upload at least one PDF first.")
        return

    model = get_chatgroq_model()

    for filename, text in st.session_state.paper_texts.items():
        st.subheader(f"📄 {filename}")

        if filename not in st.session_state.paper_analyses:
            if st.button(f"Analyze → {filename}", key=f"btn_{filename}"):
                with st.spinner(f"Analyzing {filename}…"):
                    st.session_state.paper_analyses[filename] = analyze_paper(
                        text, filename, model
                    )
                st.rerun()
            continue

        a = st.session_state.paper_analyses[filename]
        st.info(f"💡 **{a.get('one_line_summary', '')}**")

        c1, c2 = st.columns(2)
        with c1:
            with st.expander("🎯 Problem", expanded=True):
                st.markdown(a.get("problem", "—"))
            with st.expander("🔬 Methodology"):
                st.markdown(a.get("methodology", "—"))
            with st.expander("📈 Results"):
                st.markdown(a.get("results", "—"))
            with st.expander("🗺️ Future Work"):
                st.markdown(a.get("future_work", "—"))
        with c2:
            with st.expander("✨ Contributions", expanded=True):
                for c in a.get("contributions", []):
                    st.markdown(f"- {c}")
            with st.expander("⚠️ Limitations"):
                for l in a.get("limitations", []):
                    st.markdown(f"- {l}")
            with st.expander("🗃️ Datasets"):
                for d in a.get("datasets", []):
                    st.markdown(f"- `{d}`")
            with st.expander("🔁 Reproducibility"):
                r = a.get("reproducibility", {})
                for label, key in [("Code mentioned",        "code_mentioned"),
                                   ("Hyperparameters given", "hyperparameters_given"),
                                   ("Datasets public",       "datasets_public")]:
                    st.markdown(f"{'✅' if r.get(key) else '❌'} {label}")
                if r.get("notes"):
                    st.caption(r["notes"])

        if st.button("🔄 Re-analyze", key=f"re_{filename}"):
            del st.session_state.paper_analyses[filename]
            st.rerun()
        st.divider()


# ═══════════════════════════════════════════════════════
# Page 3 — ⚖️ Compare Papers
# ═══════════════════════════════════════════════════════

def page_compare():
    st.title("⚖️ Compare Papers")

    if len(st.session_state.paper_texts) < 2:
        st.warning("Upload at least 2 papers to compare.")
        return

    st.markdown("Comparing: " + ", ".join(f"`{n}`" for n in st.session_state.paper_texts))

    if st.button("⚖️ Generate Comparison"):
        with st.spinner("Comparing…"):
            st.session_state.comparison_result = compare_papers(
                st.session_state.paper_texts, get_chatgroq_model()
            )

    if st.session_state.comparison_result:
        st.markdown(st.session_state.comparison_result)
        st.download_button("💾 Export", st.session_state.comparison_result,
                           "comparison.md", "text/markdown")


# ═══════════════════════════════════════════════════════
# Page 4 — 🔬 Critical Review
# ═══════════════════════════════════════════════════════

def page_review():
    st.title("🔬 Critical Reviewer")
    st.caption("Peer-reviewer style critique.")

    if not st.session_state.paper_texts:
        st.warning("⬅️ Upload at least one PDF first.")
        return

    selected = st.selectbox("Select paper", list(st.session_state.paper_texts.keys()))
    key = f"review_{selected}"

    if st.button("🔬 Generate Review"):
        with st.spinner(f"Reviewing {selected}…"):
            st.session_state[key] = critical_review(
                st.session_state.paper_texts[selected], selected, get_chatgroq_model()
            )

    if st.session_state.get(key):
        st.markdown(st.session_state[key])
        st.download_button("💾 Export Review", st.session_state[key],
                           f"review_{selected.replace('.pdf','')}.md", "text/markdown")


# ═══════════════════════════════════════════════════════
# Page 5 — 🗺️ Research Gaps
# ═══════════════════════════════════════════════════════

def page_gaps():
    st.title("🗺️ Research Gap Finder")

    if not st.session_state.paper_texts:
        st.warning("⬅️ Upload at least one PDF first.")
        return

    if st.button("🗺️ Find Research Gaps"):
        with st.spinner("Analyzing gaps…"):
            st.session_state.gaps_result = find_research_gaps(
                st.session_state.paper_texts, get_chatgroq_model()
            )

    if st.session_state.gaps_result:
        st.markdown(st.session_state.gaps_result)
        st.download_button("💾 Export", st.session_state.gaps_result,
                           "research_gaps.md", "text/markdown")


# ═══════════════════════════════════════════════════════
# Page 6 — 🧠 Explain Concept
# ═══════════════════════════════════════════════════════

def page_explain():
    st.title("🧠 Concept Explainer")

    concept = st.text_input("Concept / term / equation",
                            placeholder="e.g. attention mechanism, LoRA, RLHF…")

    level = st.radio("Depth", [
        "🟢 Simple (non-specialist)",
        "🟡 Graduate (ML student)",
        "🔴 Expert (researcher)",
    ], horizontal=True)

    level_map = {
        "🟢 Simple (non-specialist)": "simple",
        "🟡 Graduate (ML student)":   "graduate",
        "🔴 Expert (researcher)":     "expert",
    }

    paper_ctx = ""
    if st.session_state.paper_texts:
        choice = st.selectbox("Ground in paper (optional)",
                              ["None"] + list(st.session_state.paper_texts.keys()))
        if choice != "None":
            paper_ctx = st.session_state.paper_texts[choice]

    if st.button("🧠 Explain") and concept.strip():
        with st.spinner("Explaining…"):
            result = explain_concept(concept.strip(), paper_ctx,
                                     level_map[level], get_chatgroq_model())
        st.divider()
        st.markdown(result)


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔬",
        layout="wide",
    )
    init_state()
    page = render_sidebar()

    {
        "💬 Chat":            page_chat,
        "📊 Paper Analyzer":  page_analyzer,
        "⚖️ Compare Papers":  page_compare,
        "🔬 Critical Review": page_review,
        "🗺️ Research Gaps":   page_gaps,
        "🧠 Explain Concept": page_explain,
    }.get(page, page_chat)()


if __name__ == "__main__":
    main()
