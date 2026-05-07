# sql_agent.py
import re
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
import os
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq # NEW IMPORT

# Set your API key securely
GROQ_API_KEY= os.getenv("GROQ_API_KEY")
DB_PATH = Path(__file__).parent / "enterprise.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# SWAP OLLAMA FOR GROQ (Using Llama 3 8B running on Groq's insanely fast chips)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# DB_PATH = Path(__file__).parent / "enterprise.db"

# db  = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
# llm = OllamaLLM(model="gemma3:4b", temperature=0)


def get_schema() -> str:
    return db.get_table_info()


def _clean_sql(raw: str) -> str:
    cleaned = re.sub(r"```(?:sql)?|```", "", raw).strip()
    cleaned = cleaned.replace("\\_", "_")
    lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("--")]
    cleaned = " ".join(lines).strip()
    if not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


def query_bi(question: str) -> dict:
    schema = get_schema()

    sql_prompt = f"""You are a SQLite SQL expert. Given the database schema below and a user question,
write a single SELECT query that answers the question.

SCHEMA:
{schema}

RULES:
- Output ONLY the raw SQL query. No explanation, no markdown, no backticks.
- Use SQLite syntax only (use LIKE not ILIKE, use || for concat, etc.).
- Use single quotes for string literals.
- Always end the query with a semicolon.
- If aggregating, include meaningful column aliases.

QUESTION: {question}

SQL:"""

    raw_sql = llm.invoke(sql_prompt).content
    sql_query = _clean_sql(raw_sql)

    try:
        raw_result = db.run(sql_query)
    except Exception as e:
        return {
            "sql_query":  sql_query,
            "raw_result": f"ERROR: {e}",
            "answer":     f"The generated SQL failed to execute: {e}",
        }

    answer_prompt = f"""You are a senior business intelligence analyst presenting to a CEO.
Given the original question, the SQL query used, and the raw database results,
provide a clear and concise executive summary.

QUESTION: {question}

SQL QUERY: {sql_query}

RAW RESULTS:
{raw_result}

RULES:
- Write 2-3 sentences maximum.
- Include specific numbers and figures from the results.
- Do not mention SQL or technical details.
- If results are empty, say so clearly.

ANSWER:"""

    answer = llm.invoke(answer_prompt).content
    return {
        "sql_query":  sql_query,
        "raw_result": str(raw_result),
        "answer":     answer.strip(),
    }
