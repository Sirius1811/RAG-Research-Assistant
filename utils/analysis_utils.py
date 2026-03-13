"""
utils/analysis_utils.py
-----------------------
LLM-powered analysis features for the AI Research Assistant.
All functions take paper text(s) + a chat model and return results.
"""

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage


# -------------------------------------------------------
# Internal helper — single-shot LLM call
# -------------------------------------------------------

def _llm_call(chat_model, system: str, user: str) -> str:
    try:
        resp = chat_model.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return resp.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def _trim(text: str, max_chars: int = 12000) -> str:
    """Keep head + tail to stay within token limits."""
    if len(text) <= max_chars:
        return text
    return text[:9000] + "\n\n[...middle truncated...]\n\n" + text[-3000:]


def _parse_json(raw: str) -> dict | list | None:
    """Strip markdown fences and parse JSON."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except Exception:
        return None


# -------------------------------------------------------
# 1. Paper Analyzer
# -------------------------------------------------------

def analyze_paper(paper_text: str, filename: str, chat_model) -> dict:
    """
    Returns a structured dict with keys:
    one_line_summary, problem, contributions, methodology, results,
    datasets, limitations, reproducibility, future_work
    """
    system = """You are an expert AI researcher.
Analyze the research paper and return a JSON object with EXACTLY these keys:
{
  "one_line_summary": "One sentence capturing the core idea",
  "problem": "What specific problem does this paper solve?",
  "contributions": ["contribution 1", "contribution 2"],
  "methodology": "How do they solve it?",
  "results": "Key quantitative results and benchmarks",
  "datasets": ["dataset1", "dataset2"],
  "limitations": ["limitation 1", "limitation 2"],
  "reproducibility": {
    "code_mentioned": true,
    "hyperparameters_given": false,
    "datasets_public": true,
    "notes": "any notes"
  },
  "future_work": "What next steps do authors suggest?"
}
Return ONLY valid JSON. No markdown fences, no extra text."""

    raw    = _llm_call(chat_model, system, f"Paper: {filename}\n\n{_trim(paper_text)}")
    parsed = _parse_json(raw)

    if isinstance(parsed, dict):
        return parsed

    # Fallback if JSON parsing fails
    return {
        "one_line_summary": "Could not parse analysis — see raw output below.",
        "problem": raw, "contributions": [], "methodology": "",
        "results": "", "datasets": [], "limitations": [],
        "reproducibility": {
            "code_mentioned": False, "hyperparameters_given": False,
            "datasets_public": False, "notes": "Analysis failed."
        },
        "future_work": ""
    }


# -------------------------------------------------------
# 2. Multi-Paper Comparison
# -------------------------------------------------------

def compare_papers(paper_texts: dict, chat_model) -> str:
    """
    Returns a markdown comparison table + key takeaways.
    paper_texts: {filename: full_text}
    """
    if len(paper_texts) < 2:
        return "Please upload at least 2 papers to compare."

    parts = [f"=== {fn} ===\n{_trim(txt, 6000)}"
             for fn, txt in paper_texts.items()]

    system = """You are an expert AI researcher comparing multiple papers.
Create a markdown comparison table with papers as columns covering:
Problem, Approach/Architecture, Datasets, Best Metric, Compute Cost, Code Available, Main Limitation, Novel Contribution.
After the table, add ## Key Takeaways with 3–5 bullet points on the most important differences."""

    return _llm_call(chat_model, system, "\n\n".join(parts))


# -------------------------------------------------------
# 3. Research Gap Finder
# -------------------------------------------------------

def find_research_gaps(paper_texts: dict, chat_model) -> str:
    """Synthesizes across all papers to surface open problems."""
    parts = [f"=== {fn} ===\n{_trim(txt, 5000)}"
             for fn, txt in paper_texts.items()]

    system = """You are a senior AI researcher identifying gaps in a body of literature.
Identify and explain:
1. **Open Problems** — questions explicitly left unanswered
2. **Contradictions** — places where papers disagree
3. **Unexplored Combinations** — promising ideas not yet combined
4. **Missing Experiments** — obvious ablations/baselines not tested
5. **Scalability Questions** — methods not validated at scale
6. **Real-World Applicability Gaps** — what's missing for deployment

Rate each gap: 🔴 High / 🟡 Medium / 🟢 Low importance.
Suggest a concrete next step for each. Reference paper names."""

    return _llm_call(chat_model, system, "\n\n".join(parts))


# -------------------------------------------------------
# 4. Critical Reviewer
# -------------------------------------------------------

def critical_review(paper_text: str, filename: str, chat_model) -> str:
    """Peer-reviewer style critique of a single paper."""
    system = """You are a rigorous peer reviewer for NeurIPS/ICML/ICLR.
Structure your review exactly as:
## ⚠️ Potential Weaknesses
## ❓ Questionable Assumptions
## 🧪 Missing Experiments / Baselines
## 📊 Statistical Concerns
## 🔄 Reproducibility Red Flags
## 💬 5 Questions to Ask the Authors
## ✅ Genuine Strengths
Be specific and reference actual claims from the paper."""

    return _llm_call(chat_model, system,
                     f"Review this paper: {filename}\n\n{_trim(paper_text)}")


# -------------------------------------------------------
# 5. Follow-up Question Suggester
# -------------------------------------------------------

def suggest_follow_ups(question: str, answer: str, chat_model) -> list[str]:
    """Returns 3 smart follow-up questions based on the last Q&A."""
    system = """Generate exactly 3 smart follow-up questions a researcher would ask next.
Rules: each under 15 words, distinct topics, genuinely useful.
Return ONLY a JSON array: ["q1", "q2", "q3"]. No other text."""

    raw    = _llm_call(chat_model, system, f"Q: {question}\nA: {answer[:800]}")
    parsed = _parse_json(raw)

    if isinstance(parsed, list) and len(parsed) >= 3:
        return [str(q) for q in parsed[:3]]

    # Fallback: extract lines containing "?"
    lines = [l.strip().lstrip("-•123456789. ") for l in raw.split("\n") if "?" in l]
    return lines[:3] if len(lines) >= 3 else [
        "What are the key limitations of this approach?",
        "How does this compare to prior work?",
        "What experiments would further validate this?",
    ]


# -------------------------------------------------------
# 6. Concept Explainer (3 levels)
# -------------------------------------------------------

def explain_concept(concept: str, paper_context: str, level: str, chat_model) -> str:
    """
    level: "simple" | "graduate" | "expert"
    """
    instructions = {
        "simple":   "Explain to a curious non-specialist using an analogy. No jargon. Under 150 words.",
        "graduate": "Explain to an ML grad student: proper terminology, intuition, key equations. ~200 words.",
        "expert":   "Explain to a domain researcher: precise, nuances, trade-offs, connections to related work. ~250 words.",
    }
    system = f"You are an expert AI educator. {instructions.get(level, instructions['graduate'])} Ground the explanation in the paper context where relevant."
    user   = f"Concept: {concept}\n\nPaper context:\n{paper_context[:3000] or 'None provided.'}"

    return _llm_call(chat_model, system, user)
