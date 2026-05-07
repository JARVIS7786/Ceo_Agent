# memory_manager.py
import asyncio
from pathlib import Path

CEO_PROFILE_PATH = Path("CEO_PROFILE.md")
MEMORY_PATH = Path("MEMORY.md")

async def _read(path: Path) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: path.read_text(encoding="utf-8") if path.exists() else ""
    )

async def _write(path: Path, content: str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: path.write_text(content, encoding="utf-8")
    )

async def load_memory() -> tuple[str, str]:
    """Returns (ceo_profile, memory) concurrently."""
    ceo_profile, memory = await asyncio.gather(
        _read(CEO_PROFILE_PATH),
        _read(MEMORY_PATH),
    )
    return ceo_profile, memory

async def save_memory(memory_content: str) -> None:
    """Appends a new entry to MEMORY.md."""
    existing = await _read(MEMORY_PATH)
    updated = existing + "\n" + memory_content
    await _write(MEMORY_PATH, updated)

async def update_ceo_profile(profile_content: str) -> None:
    await _write(CEO_PROFILE_PATH, profile_content)