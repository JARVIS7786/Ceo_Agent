# ingest_tree.py
import asyncio
import json
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# ── Config ────────────────────────────────────────────────────────────────────
TREE_INDEX_PATH = Path("tree_index.json")
CHUNK_SIZE      = 2000   # large chunks to respect logical boundaries
CHUNK_OVERLAP   = 100

llm = OllamaLLM(model=" qwen3.5:2b", temperature=0)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n\n", "\n\n", "\n", ". "],
)


# ── Loader Factory ────────────────────────────────────────────────────────────
def _get_loader(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    elif ext == ".txt":
        return TextLoader(str(path), encoding="utf-8")
    raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt")


# ── LLM Summarise (blocking → executor) ──────────────────────────────────────
async def _summarise(text: str) -> str:
    prompt = (
        "In exactly ONE sentence, summarise what the following section is about. "
        "Be specific. Output only the sentence, nothing else.\n\n"
        f"SECTION:\n{text[:1500]}"          # cap to avoid prompt bloat
    )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: llm.invoke(prompt))


# ── Derive a short title from page metadata or chunk index ────────────────────
def _derive_title(doc, idx: int) -> str:
    meta = doc.metadata or {}
    page = meta.get("page")                 # PyPDFLoader provides this
    if page is not None:
        return f"Page {int(page) + 1}"
    source = meta.get("source", "")
    return f"{Path(source).stem} — Section {idx + 1}" if source else f"Section {idx + 1}"


# ── Build Tree for ONE document ───────────────────────────────────────────────
async def build_tree(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Returns a list of node dicts (not yet saved — caller merges & saves).
    """
    import tempfile, os
    ext = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        loader = _get_loader(Path(tmp_path))
        docs   = await loop.run_in_executor(None, loader.load)
        chunks = splitter.split_documents(docs)

        if not chunks:
            return []

        # Summarise all chunks concurrently
        summaries = await asyncio.gather(
            *[_summarise(c.page_content) for c in chunks]
        )

        nodes = [
            {
                "node_id":   uuid.uuid4().hex[:12],
                "source":    filename,
                "title":     _derive_title(chunk, i),
                "summary":   summaries[i].strip(),
                "full_text": chunk.page_content.strip(),
            }
            for i, chunk in enumerate(chunks)
        ]
        return nodes

    finally:
        os.unlink(tmp_path)


# ── Load / Save tree_index.json ───────────────────────────────────────────────
def load_tree() -> list[dict]:
    if TREE_INDEX_PATH.exists():
        return json.loads(TREE_INDEX_PATH.read_text(encoding="utf-8"))
    return []


def save_tree(nodes: list[dict]) -> None:
    TREE_INDEX_PATH.write_text(
        json.dumps(nodes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Public API (called by Streamlit / FastAPI) ────────────────────────────────
async def process_uploaded_file(file_bytes: bytes, filename: str) -> dict:
    """
    Drop-in replacement for the old ingest.py — same signature.
    Appends new nodes into tree_index.json (deduplicates by source+title).
    """
    new_nodes = await build_tree(file_bytes, filename)
    if not new_nodes:
        return {"status": "error", "detail": "No text extracted.", "chunks": 0}

    existing   = load_tree()
    seen       = {(n["source"], n["title"]) for n in existing}
    to_add     = [n for n in new_nodes if (n["source"], n["title"]) not in seen]
    save_tree(existing + to_add)

    return {
        "status":          "success",
        "filename":        filename,
        "chunks_ingested": len(to_add),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
async def _cli(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}")
        return
    print(f"Building tree for: {path.name} …")
    result = await process_uploaded_file(path.read_bytes(), path.name)
    print(result)
    print(f"File size: {path.stat().st_size:,} bytes")
    if result["status"] == "success":
        tree = load_tree()
        print(f"\n── tree_index.json preview (first node) ──")
        print(json.dumps(tree[0], indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest_tree.py <file.pdf|.txt>")
        sys.exit(1)
    asyncio.run(_cli(sys.argv[1]))