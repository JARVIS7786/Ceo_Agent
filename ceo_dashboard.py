import streamlit as st
import os
import asyncio
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from graph import agent_graph
from bi_agent.sql_agent import query_bi

# 1. Page Config (MUST be the first Streamlit command)
st.set_page_config(page_title="Master CEO Dashboard", layout="wide", page_icon="👔")

# 2. Helper to bridge Sync (Streamlit) and Async (LangGraph)
async def run_rag_agent(query):
    return await agent_graph.ainvoke({"query": query})

st.title("👔 Enterprise Master Dashboard")
st.markdown("One chat box. Two autonomous AI agents. Infinite answers.")

# 3. Initialize the Router Brain
# Make sure your GROQ_API_KEY is in your environment variables!
router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

router_prompt = PromptTemplate.from_template("""
You are a highly intelligent router for an Enterprise CEO. 
You have access to two expert systems:
1. DATABASE: Contains structured SQL data about sales, revenue, products, units sold, and regions.
2. DOCUMENT: Contains unstructured PDF textbook data about LangChain, AI, Agents, and programming.

Given the user's prompt, reply with exactly ONE word: 'DATABASE' or 'DOCUMENT'.
Do not explain your reasoning. Output only the single word.

User Prompt: {question}
""")

router_chain = router_prompt | router_llm

# 4. The Chat Interface
if prompt := st.chat_input("Ask about our sales data, or how to build AI..."):
    st.chat_message("user").write(prompt)
    
    with st.status("🧠 Analyzing request intent...", expanded=True) as status:
        st.write("🚦 Routing to the correct AI department...")
        
        # The Router decides where to go
        decision = router_chain.invoke({"question": prompt}).content.strip().upper()
        
        # --- ROUTE 1: THE SQL AGENT ---
        if "DATABASE" in decision:
            st.write("📊 Route chosen: **Quantitative Data Analyst (SQL)**")
            st.write("⚙️ Translating English to SQL and querying enterprise.db...")
            
            response = query_bi(prompt)
            status.update(label="SQL Analysis Complete!", state="complete", expanded=False)
            
            with st.chat_message("assistant"):
                st.write(response["answer"])
                with st.expander("View Raw SQL & Verification"):
                    st.code(response["sql_query"], language="sql")
                    st.write(response["raw_result"])
                    
        # --- ROUTE 2: THE VECTORLESS RAG AGENT ---
        else:
            st.write("📚 Route chosen: **Qualitative Researcher (RAG)**")
            st.write("🔍 Searching the JSON Knowledge Tree...")
            
            # Bridge the sync/async gap here
            final_state = asyncio.run(run_rag_agent(prompt))
            
            status.update(label="Document Search Complete!", state="complete", expanded=False)
            
            with st.chat_message("assistant"):
                # Using "response" as the key from your state
                st.write(final_state["response"])