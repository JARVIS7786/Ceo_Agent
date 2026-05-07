# state.py
from typing import TypedDict, Optional, Literal

class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query:             str

    # ── Memory ────────────────────────────────────────────────────────────────
    ceo_profile:       str
    memory:            str

    # ── Retrieval (Vectorless) ────────────────────────────────────────────────
    retrieved_context: Optional[str]   # joined full_text from LLM tree search

    # ── CRAG ──────────────────────────────────────────────────────────────────
    crag_verdict:      Optional[Literal["ACCEPT", "REJECT"]]
    raw_context:       Optional[str]
    fallback_context:  Optional[str]

    # ── Generation ────────────────────────────────────────────────────────────
    response:          Optional[str]

    # ── Self-RAG Eval ─────────────────────────────────────────────────────────
    eval_summary:      Optional[str]
    eval_score:        Optional[int]
    eval_reasoning:    Optional[str]
    is_hallucination:  Optional[bool]

    # ── Chart Data ────────────────────────────────────────────────────────────
    chart_data:        Optional[str]

    # ── Conversation History (Multi-Turn) ─────────────────────────────────────
    conversation_history: Optional[str]

    # ── Control Flow ──────────────────────────────────────────────────────────
    hard_abort:        bool
    abort_reason:      Optional[str]