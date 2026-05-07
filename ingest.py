# ingest.py
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from db.sql_client import store_context
from db.vector_client import upsert_vector

# ── Embeddings ────────────────────────────────────────────────────────────────
embedder = OllamaEmbeddings(model="nomic-embed-text")

# ── Splitter ──────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=250,
    separators=["\n\n", "\n", ".", " "],
)


# ── Loader Factory ────────────────────────────────────────────────────────────
def _get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt")


# ── Embed a single chunk (run_in_executor for blocking Ollama call) ───────────
async def _embed_chunk(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: embedder.embed_query(text))


# ── Core Pipeline ─────────────────────────────────────────────────────────────
async def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    source_label: Optional[str] = None,
) -> dict:
    """
    Ingests a PDF or TXT file end-to-end.
    - Writes temp file → loads → splits → embeds → stores SQL + ChromaDB.
    - Returns a summary dict for the caller (FastAPI / Streamlit).
    """
    ext = Path(filename).suffix.lower()
    source = source_label or filename

    # ── 1. Write bytes to a named temp file (loaders need a real path) ────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # ── 2. Load ───────────────────────────────────────────────────────────
        loop = asyncio.get_event_loop()
        loader = _get_loader(tmp_path)
        docs = await loop.run_in_executor(None, loader.load)

        # ── 3. Split ──────────────────────────────────────────────────────────
        chunks = splitter.split_documents(docs)
        if not chunks:
            return {"status": "error", "detail": "No text extracted.", "chunks": 0}

        # ── 4. Embed + Store (Sequential to prevent SQLite locking) ───────────
        async def _ingest_chunk(chunk, idx: int):
            text = chunk.page_content.strip()
            if not text:
                return

            # 1. Get the math vector from Ollama
            embedding = await _embed_chunk(text)
            
            # 2. Save to SQLite securely
            sql_id = await store_context(
                query=f"{source}::chunk_{idx}",
                chunks=text,
            )

            # 3. Save to ChromaDB
            await upsert_vector(text=text, embedding=embedding)

        # Sequential loop prevents the database from locking
        for i, chunk in enumerate(chunks):
            await _ingest_chunk(chunk, i)

        return {
            "status": "success",
            "filename": filename,
            "chunks_ingested": len(chunks),
        }

    finally:
        os.unlink(tmp_path)  # Always clean up the temp file!


# ── CLI helper: ingest a local file directly ──────────────────────────────────
async def ingest_local_file(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "detail": f"File not found: {file_path}"}
    file_bytes = path.read_bytes()
    return await process_uploaded_file(file_bytes, path.name, source_label=path.stem)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path/to/file.pdf|.txt>")
        sys.exit(1)
    result = asyncio.run(ingest_local_file(sys.argv[1]))
    print(result)