# db/vector_client.py
import uuid
import asyncio
import chromadb

# Initialize ChromaDB client 
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="ceo_agent")

async def upsert_vector(text: str, embedding: list[float]) -> str:
    """Upsert a document into ChromaDB; returns its context_id."""
    ctx_id = uuid.uuid4().hex
    loop = asyncio.get_event_loop()
    
    await loop.run_in_executor(
        None,
        lambda: collection.add(
            ids=[ctx_id],
            embeddings=[embedding],
            documents=[text]
        )
    )
    return ctx_id

# FIX: Increased top_k to 5 so we get a much broader view of the document
async def query_vector(embedding: list[float], top_k: int = 5) -> str:
    """ANN search; returns context_ids combined as a comma-separated string."""
    loop = asyncio.get_event_loop()
    
    results = await loop.run_in_executor(
        None,
        lambda: collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
    )
    
    # If we found matches, combine all their IDs into one string
    if results and results["ids"] and len(results["ids"][0]) > 0:
        return ",".join(results["ids"][0])
        
    return f"vec_noresult_{uuid.uuid4().hex[:8]}"

async def resolve_vector_context(context_id: str) -> str:
    """Fetch raw text from ChromaDB for all context_ids and remove duplicates."""
    if context_id.startswith("vec_noresult"):
        return ""
        
    ids_list = context_id.split(",")
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: collection.get(ids=ids_list)
    )
    
    if results and results["documents"] and len(results["documents"]) > 0:
        # FIX: Deduplicate the chunks so the LLM doesn't read the same thing 5 times!
        unique_chunks = []
        for doc in results["documents"]:
            if doc not in unique_chunks:
                unique_chunks.append(doc)
                
        # Stitch the UNIQUE chunks together
        stitched_text = "\n\n...[NEXT CONTEXT CHUNK]...\n\n".join(unique_chunks)
        print(f"\n--- WHAT CHROMADB FOUND (DEDUPLICATED) ---\n{stitched_text}\n---------------------------\n")
        return stitched_text
        
    return ""