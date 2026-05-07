# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional  # <-- Added this import
from graph import agent_graph
from state import AgentState


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No DB init required — fully JSON-based now
    yield

app = FastAPI(
    title="CEO Intelligence Agent — Vectorless RAG",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[str] = None

# FIX: Swapped `| None` to `Optional[]` for Python 3.9 compatibility
class QueryResponse(BaseModel):
    response:        str
    eval_score:      Optional[int] = None
    eval_summary:    Optional[str] = None
    eval_reasoning:  Optional[str] = None
    crag_verdict:    Optional[str] = None
    is_hallucination: Optional[bool] = None
    hard_abort:      bool
    abort_reason:    Optional[str] = None
    chart_data:      Optional[str] = None


# ── State Factory ─────────────────────────────────────────────────────────────
def _initial_state(query: str, conversation_history: str = "") -> AgentState:
    return AgentState(
        query              = query,
        ceo_profile        = "",
        memory             = "",
        retrieved_context  = None,
        crag_verdict       = None,
        raw_context        = None,
        fallback_context   = None,
        response           = None,
        eval_summary       = None,
        eval_score         = None,
        eval_reasoning     = None,
        is_hallucination   = None,
        hard_abort         = False,
        abort_reason       = None,
        chart_data         = None,
        conversation_history = conversation_history,
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_agent(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        final: AgentState = await agent_graph.ainvoke(
            _initial_state(req.query, req.conversation_history or "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return QueryResponse(
        response       = final.get("response")       or "",
        eval_score     = final.get("eval_score"),
        eval_summary   = final.get("eval_summary"),
        eval_reasoning = final.get("eval_reasoning"),
        crag_verdict   = final.get("crag_verdict"),
        is_hallucination = final.get("is_hallucination"),
        hard_abort     = final.get("hard_abort",   False),
        abort_reason   = final.get("abort_reason"),
        chart_data     = final.get("chart_data"),
    )

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mode": "vectorless-rag"}


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)