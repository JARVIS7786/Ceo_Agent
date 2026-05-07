# clients/fallback_client.py
import asyncio
from duckduckgo_search import DDGS

async def fetch_fallback(query: str) -> str:
    """Async external web/news search using DuckDuckGo (100% Free, No API Key)."""
    
    def _perform_search():
        try:
            # We add "news" to the query to bias DuckDuckGo toward recent events/articles
            search_query = f"{query} news"
            results = DDGS().text(search_query, max_results=3)
            
            if not results:
                return "[FALLBACK] No external data found."
            
            return "\n".join(
                f"- {r.get('title')}: {r.get('body')}" for r in results
            )
        except Exception as e:
            return f"[FALLBACK UNAVAILABLE] {str(e)}"

    loop = asyncio.get_event_loop()
    # The DDGS library is synchronous, so we run it in an executor 
    # to ensure it doesn't block your blazing fast FastAPI server!
    return await loop.run_in_executor(None, _perform_search)