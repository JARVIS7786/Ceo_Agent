# nodes.py
import asyncio
import json
import re
from pathlib import Path

from duckduckgo_search import DDGS
from langchain_ollama import OllamaLLM
from ingest_tree import load_tree
from memory_manager import load_memory, save_memory
from state import AgentState

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = OllamaLLM(model=" qwen3.5:2b", temperature=0)

CONTEXT_SEPARATOR = "\n\n...[NEXT CONTEXT CHUNK]...\n\n"


async def _llm(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: llm.invoke(prompt))


# ── NODE 1: Load Memory ───────────────────────────────────────────────────────
async def node_load_memory(state: AgentState) -> AgentState:
    ceo_profile, memory = await load_memory()
    return {**state, "ceo_profile": ceo_profile, "memory": memory}


# ── NODE 1.5: Multi-Turn Query Rewriter ──────────────────────────────────────
async def node_rewrite_query(state: AgentState) -> AgentState:
    history = state.get("conversation_history", "")
    if not history or not history.strip():
        return state

    prompt = f"""You are a query rewriter for a CEO assistant. Your job is to make the user's
latest query self-contained by resolving pronouns, references, and implicit context from the
conversation history.

CONVERSATION HISTORY:
{history[-2000:]}

LATEST QUERY: {state['query']}

If the latest query is already self-contained, output it unchanged.
If it contains references like "that", "it", "the same", "drill into this", "more details",
"the other one", etc., rewrite it into a fully explicit standalone query.

Output ONLY the rewritten query. Nothing else."""

    rewritten = await _llm(prompt)
    return {**state, "query": rewritten.strip()}


# ── NODE 2: LLM Tree Search Retrieval ────────────────────────────────────────
async def node_retrieve(state: AgentState) -> AgentState:
    tree = load_tree()

    if not tree:
        return {
            **state,
            "retrieved_context": "",
            "sql_context_id":    None,
            "vector_context_id": None,
        }

    # ── Step 1: Build Table of Contents (summaries only, no full_text) ────────
    toc_lines = [
        f"[ID: {n['node_id']}] {n['title']} — {n['summary']}"
        for n in tree
    ]
    toc = "\n".join(toc_lines)

    # ── Step 2: Ask LLM to identify relevant Node IDs ─────────────────────────
    selection_prompt = f"""You are a document retrieval assistant.

USER QUERY: {state['query']}

TABLE OF CONTENTS:
{toc}

Task: Identify which Node IDs are most likely to contain information relevant to the query above.

IMPORTANT: If the query asks to COMPARE, CONTRAST, or CROSS-REFERENCE multiple documents
(e.g. "Compare Q1 Report to Q2 Report"), you MUST select Node IDs from EACH distinct source
document mentioned. Do not limit your selection to a single source — pull from all relevant sources
so the downstream answer can perform a fair comparison.

Output ONLY a comma-separated list of Node IDs. No explanation, no punctuation, no extra text.
Example output: 3f9a1c02b7e4, 8a2b1c99d4e1

Relevant Node IDs:"""

    raw_ids = await _llm(selection_prompt)

    # ── Step 3: Parse Node IDs ────────────────────────────────────────────────
    parsed_ids = {
        token.strip()
        for token in re.split(r"[,\s]+", raw_ids.strip())
        if token.strip()
    }

    # ── Step 4: Extract full_text for matched nodes ───────────────────────────
    node_map = {n["node_id"]: n for n in tree}
    matched  = [
        node_map[nid]["full_text"]
        for nid in parsed_ids
        if nid in node_map
    ]

    # Fallback: no IDs matched → top-3 by index
    if not matched:
        matched = [n["full_text"] for n in tree[:3]]

    # ── Step 5: Assemble context ──────────────────────────────────────────────
    retrieved_context = CONTEXT_SEPARATOR.join(matched)

    return {
        **state,
        "retrieved_context": retrieved_context,
        # Nullify old DB keys — no longer used
        "sql_context_id":    None,
        "vector_context_id": None,
    }


# ── NODE 3: CRAG — Grade Retrieved Context ────────────────────────────────────
async def node_crag(state: AgentState) -> AgentState:
    context = state.get("retrieved_context", "")

    if not context.strip():
        return {**state, "crag_verdict": "REJECT", "raw_context": context}

    prompt = f"""You are a context quality judge for an executive AI assistant.

USER QUERY: {state['query']}

RETRIEVED CONTEXT:
{context[:3000]}

Question: Does this context contain enough relevant information to answer the query?
Reply with EXACTLY one word — ACCEPT or REJECT. Nothing else."""

    verdict = (await _llm(prompt)).strip().upper()
    verdict = verdict if verdict in ("ACCEPT", "REJECT") else "REJECT"

    return {**state, "crag_verdict": verdict, "raw_context": context}


# ── NODE 4: External Fallback (Live Web Search) ─────────────────────────────
async def node_fallback(state: AgentState) -> AgentState:
    query = state["query"]
    loop = asyncio.get_event_loop()

    def _search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=3))

    results = await loop.run_in_executor(None, _search)

    if not results:
        return {**state, "fallback_context": "No web results found."}

    web_context = "\n\n".join(
        f"[{r['title']}]\n{r['body']}" for r in results
    )

    prompt = f"""You are an executive AI assistant with access to live web search results.
The internal knowledge base did not have relevant information, so a web search was performed.
Synthesize the search results below into a concise, board-level executive response.

QUERY: {query}

WEB SEARCH RESULTS:
{web_context}

Provide a concise response grounded only in the search results above."""

    fallback = await _llm(prompt)
    return {**state, "fallback_context": fallback}


# ── NODE 5: Generate Response ─────────────────────────────────────────────────
async def node_generate(state: AgentState) -> AgentState:
    context = (
        state.get("fallback_context", "")
        if state["crag_verdict"] == "REJECT"
        else state.get("raw_context", "")
    )

    prompt = f"""You are a senior executive AI assistant briefing a CEO/CTO.

CEO PROFILE:
{state.get('ceo_profile', 'N/A')}

AGENT MEMORY:
{state.get('memory', 'N/A')}

RELEVANT CONTEXT:
{context}

CEO QUERY: {state['query']}

Instructions:
- Be concise, structured, and board-level in tone.
- Cite specific facts from the context where possible.
- Do not hallucinate figures or data not present in the context.

CHART INSTRUCTION:
If your answer involves concrete numerical data (financial metrics, performance comparisons,
quarterly figures, trends with real numbers), you MUST append a single JSON chart block at
the very end of your response using this exact format:

|||CHART|||{{"chart_type": "bar_chart", "title": "Chart Title", "data": {{"Category": [10, 20, 30], "Revenue": [100, 200, 300]}}, "index": ["Q1", "Q2", "Q3"]}}|||END_CHART|||

Rules for chart data:
- chart_type must be "bar_chart" or "line_chart"
- data must be a dict of column names to lists of numbers (all same length)
- index must be a list of labels (same length as the value lists)
- Only include a chart if you have REAL numbers from the context. Never fabricate chart data.
- If no numerical data is available, do NOT include a chart block.

Response:"""

    raw_response = await _llm(prompt)

    chart_data = None
    clean_response = raw_response

    chart_match = re.search(
        r"\|\|\|CHART\|\|\|(.*?)\|\|\|END_CHART\|\|\|",
        raw_response,
        re.DOTALL,
    )
    if chart_match:
        try:
            chart_json = chart_match.group(1).strip()
            json.loads(chart_json)
            chart_data = chart_json
        except (json.JSONDecodeError, ValueError):
            pass
        clean_response = raw_response[:chart_match.start()].strip()

    return {**state, "response": clean_response, "chart_data": chart_data}


# ── NODE 6: Self-RAG Eval (Single LLM Call) ───────────────────────────────────
async def node_eval(state: AgentState) -> AgentState:
    prompt = f"""You are a strict quality auditor for executive AI responses.

ORIGINAL QUERY:
{state['query']}

CONTEXT USED:
{state.get('raw_context', state.get('fallback_context', ''))[:2000]}

AGENT RESPONSE:
{state.get('response', '')}

Evaluate the response and return ONLY valid JSON (no markdown, no backticks):
{{
  "summary": "<one-sentence executive summary of the response>",
  "score": <integer 0-10; 10 = perfectly accurate, grounded, and complete>,
  "reasoning": "<one sentence explaining the score>",
  "is_hallucination": <true if response contains claims absent from context, else false>
}}"""

    raw = await _llm(prompt)

    # Strip markdown fences if model disobeys
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {
            "summary":          "Evaluation parse error.",
            "score":            0,
            "reasoning":        "LLM returned malformed JSON.",
            "is_hallucination": True,
        }

    return {
        **state,
        "eval_summary":      str(parsed.get("summary", "")),
        "eval_score":        int(parsed.get("score", 0)),
        "eval_reasoning":    str(parsed.get("reasoning", "")),
        "is_hallucination":  bool(parsed.get("is_hallucination", True)),
    }


# ── NODE 7: Hard Abort ────────────────────────────────────────────────────────
async def node_hard_abort(state: AgentState) -> AgentState:
    score  = state.get("eval_score", 0)
    is_hal = state.get("is_hallucination", False)
    reason = (
        f"Hallucination detected (score: {score}/10)."
        if is_hal
        else f"Quality score {score}/10 is below the required threshold of 9."
    )
    return {
        **state,
        "hard_abort":   True,
        "abort_reason": reason,
        "response":     f"[HARD ABORT] Pipeline terminated. Reason: {reason}",
    }


# ── NODE 8: Persist Memory ────────────────────────────────────────────────────
async def node_save_memory(state: AgentState) -> AgentState:
    entry = (
        f"\n---\n"
        f"## Query\n{state['query']}\n\n"
        f"## Response\n{state.get('response', '')}\n\n"
        f"## Eval\nScore: {state.get('eval_score')}/10 | "
        f"CRAG: {state.get('crag_verdict')} | "
        f"Hallucination: {state.get('is_hallucination')}\n"
        f"Reasoning: {state.get('eval_reasoning', '')}\n"
    )
    await save_memory(entry)
    return state