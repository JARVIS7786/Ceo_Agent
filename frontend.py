# frontend.py
import asyncio
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from ingest_tree import process_uploaded_file, load_tree

# ── Config ────────────────────────────────────────────────────────────────────
API_URL     = "http://localhost:8000/query"
MEMORY_PATH = Path("MEMORY.md")
CEO_PATH    = Path("CEO_PROFILE.md")
TREE_PATH   = Path("tree_index.json")

st.set_page_config(
    page_title="CEO Intelligence Agent",
    page_icon="⚡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0d0d; }
  [data-testid="stSidebar"]          { background: #111418; border-right: 1px solid #1f2937; }
  [data-testid="stChatMessage"]      { background: #161b22; border: 1px solid #1f2937;
                                       border-radius: 12px; margin-bottom: 8px; }
  .score-badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; margin-left: 8px;
  }
  .score-high  { background: #14532d; color: #4ade80; }
  .score-mid   { background: #713f12; color: #fbbf24; }
  .score-low   { background: #450a0a; color: #f87171; }
  .abort-banner {
    background: #450a0a; border: 1px solid #7f1d1d; border-radius: 8px;
    padding: 10px 16px; color: #fca5a5; font-size: 0.85rem; margin-top: 6px;
  }
  .meta-row { font-size: 0.75rem; color: #6b7280; margin-top: 4px; }
  .tree-node {
    background: #0f172a; border: 1px solid #1e293b; border-radius: 6px;
    padding: 6px 10px; margin-bottom: 4px; font-size: 0.75rem; color: #94a3b8;
  }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helpers ───────────────────────────────────────────────────────────────────
# FIX: Using Optional[int] for Python 3.9 compatibility
def _score_badge(score: Optional[int]) -> str:
    if score is None:
        return ""
    css = "score-high" if score >= 9 else "score-mid" if score >= 6 else "score-low"
    return f'<span class="score-badge {css}">⚡ {score}/10</span>'

def _render_assistant(content: str, meta: dict) -> None:
    badge = _score_badge(meta.get("eval_score"))
    st.markdown(f"{content}{badge}", unsafe_allow_html=True)

    chart_raw = meta.get("chart_data")
    if chart_raw:
        try:
            chart = json.loads(chart_raw)
            df = pd.DataFrame(chart["data"], index=chart.get("index"))
            st.markdown(f"**{chart.get('title', 'Chart')}**")
            if chart.get("chart_type") == "line_chart":
                st.line_chart(df)
            else:
                st.bar_chart(df)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    if meta.get("hard_abort"):
        st.markdown(
            f'<div class="abort-banner">🚨 <b>Hard Abort</b> — '
            f'{meta.get("abort_reason", "Pipeline terminated.")}</div>',
            unsafe_allow_html=True,
        )
    if meta.get("eval_summary"):
        st.markdown(
            f'<div class="meta-row">📋 {meta["eval_summary"]} &nbsp;|&nbsp; '
            f'CRAG: <b>{meta.get("crag_verdict", "—")}</b> &nbsp;|&nbsp; '
            f'Hallucination: <b>{meta.get("is_hallucination", "—")}</b></div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ CEO Intelligence Agent")
    st.caption("Vectorless RAG · LangGraph · Ollama · JSON Tree")
    st.divider()

    # ── Document Ingestion ────────────────────────────────────────────────────
    st.markdown("### 📂 Data Pipeline")
    uploaded = st.file_uploader(
        "Upload document",
        type=["pdf", "txt"],
        help="Parsed into a logical tree and saved to tree_index.json.",
    )
    if uploaded is not None:
        with st.spinner(f"Building tree for `{uploaded.name}` …"):
            try:
                result = asyncio.run(
                    process_uploaded_file(
                        file_bytes=uploaded.read(),
                        filename=uploaded.name,
                    )
                )
                if result["status"] == "success":
                    st.success(
                        f"✅ **{result['filename']}** — "
                        f"**{result['chunks_ingested']}** nodes added to tree."
                    )
                else:
                    st.error(f"❌ {result.get('detail', 'Unknown error.')}")
            except Exception as e:
                st.error(f"❌ {e}")

    st.divider()

    # ── Tree Index Viewer ─────────────────────────────────────────────────────
    st.markdown("### 🌲 Knowledge Tree")
    with st.expander(
        f"View tree_index.json "
        f"({'exists' if TREE_PATH.exists() else 'empty'})",
        expanded=False,
    ):
        tree = load_tree()
        if tree:
            for node in tree:
                st.markdown(
                    f'<div class="tree-node">'
                    f'<b>[{node["node_id"]}]</b> {node["title"]} '
                    f'<i>({node["source"]})</i><br/>{node["summary"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("_No nodes yet. Upload a document._")

    st.divider()

    # ── Memory Viewer ─────────────────────────────────────────────────────────
    st.markdown("### 🧠 Agent Memory")
    with st.expander("View MEMORY.md", expanded=False):
        st.markdown(
            MEMORY_PATH.read_text() if MEMORY_PATH.exists() else "_No memory yet._"
        )
    with st.expander("View CEO_PROFILE.md", expanded=False):
        st.markdown(
            CEO_PATH.read_text() if CEO_PATH.exists() else "_No profile yet._"
        )

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("## 💬 Executive Briefing")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant(msg["content"], msg.get("meta", {}))
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask your agent anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history from prior turns
    history_lines = []
    for msg in st.session_state.messages[:-1]:
        role = "CEO" if msg["role"] == "user" else "AGENT"
        history_lines.append(f"{role}: {msg['content']}")
    conversation_history = "\n".join(history_lines[-20:])

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge tree…"):
            try:
                res = requests.post(
                    API_URL,
                    json={
                        "query": prompt,
                        "conversation_history": conversation_history,
                    },
                    timeout=120,
                )
                res.raise_for_status()
                data = res.json()

                response_text = data.get("response") or "_No response._"
                meta = {
                    "eval_score":       data.get("eval_score"),
                    "eval_summary":     data.get("eval_summary"),
                    "eval_reasoning":   data.get("eval_reasoning"),
                    "crag_verdict":     data.get("crag_verdict"),
                    "hard_abort":       data.get("hard_abort", False),
                    "abort_reason":     data.get("abort_reason"),
                    "is_hallucination": data.get("is_hallucination"),
                    "chart_data":       data.get("chart_data"),
                }
                _render_assistant(response_text, meta)

            except requests.exceptions.ConnectionError:
                response_text = "❌ Cannot reach agent API. Is `uvicorn` running?"
                meta = {}
                st.error(response_text)
            except Exception as e:
                response_text = f"❌ Unexpected error: {e}"
                meta = {}
                st.error(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "meta": meta}
    )