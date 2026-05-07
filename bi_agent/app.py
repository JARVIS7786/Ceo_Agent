# app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from bi_agent.setup_db import setup_database
from bi_agent.sql_agent import query_bi, get_schema

setup_database()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BI SQL Agent",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS (matches CEO Agent dark theme) ────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0d0d; }
  [data-testid="stSidebar"]          { background: #111418; border-right: 1px solid #1f2937; }
  [data-testid="stChatMessage"]      { background: #161b22; border: 1px solid #1f2937;
                                       border-radius: 12px; margin-bottom: 8px; }
  .sql-badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
    background: #14532d; color: #4ade80; margin-left: 8px;
  }
  .sql-error {
    background: #450a0a; border: 1px solid #7f1d1d; border-radius: 8px;
    padding: 10px 16px; color: #fca5a5; font-size: 0.85rem; margin-top: 6px;
  }
  .meta-row { font-size: 0.75rem; color: #6b7280; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 BI SQL Agent")
    st.caption("Natural Language → SQL → Executive Insight")
    st.divider()

    st.markdown("### 🗄️ Database Schema")
    st.code(get_schema(), language="sql")

    st.divider()

    st.markdown("### 💡 Example Queries")
    st.markdown(
        "- *Total revenue by region*\n"
        "- *Top 3 products by units sold*\n"
        "- *Average revenue per category in 2025*\n"
        "- *Compare Tech vs Healthcare total revenue*"
    )

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("## 💬 SQL Analytics Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
            meta = msg.get("meta", {})
            if meta.get("sql_query"):
                with st.expander("🔍 View SQL Query & Raw Data"):
                    st.code(meta["sql_query"], language="sql")
                    if meta.get("raw_result"):
                        st.text(meta["raw_result"])
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask your database a question..."):
    
    # 1. Show the user's message immediately
    st.chat_message("user").write(prompt)
    
    # 2. Open the "Starbucks" Background Status Box
    with st.status("🤖 Analyzing Enterprise Data...", expanded=True) as status:
        
        st.write("🔍 Inspecting database schema...")
        # (Imagine the LLM is reading the tables here)
        
        st.write("⚙️ Writing SQL query...")
        # Call your backend function
        response = query_bi(prompt) 
        
        st.write("📊 Executing query against database...")
        
        # 3. Mark the status as complete and collapse it!
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # 4. Show the final polished answer outside the loading box
    with st.chat_message("assistant"):
        st.write(response["answer"])
        
        # Add a dropdown so the CEO can verify the raw SQL if they want
        with st.expander("View Raw SQL & Data"):
            st.code(response["sql_query"], language="sql")
            st.write(response["raw_result"])