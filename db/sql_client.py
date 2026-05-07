# db/sql_client.py
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, Text, select
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./ceo_agent.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ContextRecord(Base):
    __tablename__ = "contexts"
    id     = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    query  = Column(String, index=True)
    chunks = Column(Text)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def store_context(query: str, chunks: str) -> str:
    async with AsyncSessionLocal() as session:
        record = ContextRecord(id=uuid.uuid4().hex, query=query, chunks=chunks)
        session.add(record)
        await session.commit()
        return record.id


async def fetch_context(context_id: str) -> str:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ContextRecord).where(ContextRecord.id == context_id)
        )
        record = result.scalar_one_or_none()
        return record.chunks if record else ""


async def query_sql(query: str) -> str:
    """Full-text keyword search; returns context_id."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ContextRecord).where(ContextRecord.query.ilike(f"%{query}%"))
        )
        record = result.scalars().first()
        if record:
            return record.id
        # Store a stub if no match — replace with real ETL data
        return await store_context(query, f"[SQL stub for: {query}]")