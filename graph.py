# graph.py
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    node_load_memory,
    node_rewrite_query,
    node_retrieve,
    node_crag,
    node_fallback,
    node_generate,
    node_eval,
    node_hard_abort,
    node_save_memory,
)

# ── Conditional Edges ─────────────────────────────────────────────────────────
def edge_crag_router(state: AgentState) -> str:
    return "fallback" if state["crag_verdict"] == "REJECT" else "generate"

def edge_eval_router(state: AgentState) -> str:
    # Change the 9 to a 7. This allows solid B-grade 
    # answers through while still blocking garbage.
    if state.get("eval_score", 0) < 7 or state.get("is_hallucination", True):
        return "abort_node"
    return "save_memory"
# ── Build ─────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("load_memory",    node_load_memory)
    g.add_node("rewrite_query", node_rewrite_query)
    g.add_node("retrieve",      node_retrieve)
    g.add_node("crag",        node_crag)
    g.add_node("fallback",    node_fallback)
    g.add_node("generate",    node_generate)
    g.add_node("eval",        node_eval)
    
    # Changed the node name to "abort_node" to prevent collision
    g.add_node("abort_node",  node_hard_abort)
    
    g.add_node("save_memory", node_save_memory)

    g.set_entry_point("load_memory")

    g.add_edge("load_memory",    "rewrite_query")
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve",      "crag")

    g.add_conditional_edges(
        "crag", edge_crag_router,
        {"fallback": "fallback", "generate": "generate"},
    )
    g.add_edge("fallback", "generate")
    g.add_edge("generate",  "eval")

    g.add_conditional_edges(
        "eval", edge_eval_router,
        {"abort_node": "abort_node", "save_memory": "save_memory"},
    )

    g.add_edge("abort_node",  END)
    g.add_edge("save_memory", END)

    return g.compile()

agent_graph = build_graph()