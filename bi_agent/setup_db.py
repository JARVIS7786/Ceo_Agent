# setup_db.py
import sqlite3
import random
from pathlib import Path
from datetime import date, timedelta

DB_PATH = Path(__file__).parent / "enterprise.db"

PRODUCTS = [
    (1,  "SmartSensor X1",    "Tech",            45.00),
    (2,  "CloudSync Pro",     "Tech",           120.00),
    (3,  "BioTrack Wearable", "Healthcare",      85.00),
    (4,  "MedFlow Analyzer",  "Healthcare",     210.00),
    (5,  "SolarGrid Panel",   "Energy",         300.00),
    (6,  "WindTurbine Mini",  "Energy",         550.00),
    (7,  "HomeComfort Hub",   "Consumer Goods",  35.00),
    (8,  "AquaPure Filter",   "Consumer Goods",  18.00),
    (9,  "FinLedger Suite",   "Finance",         95.00),
    (10, "RiskShield AI",     "Finance",        160.00),
]

REGIONS = ["North", "South", "East", "West"]

START_DATE = date(2024, 1, 1)
END_DATE   = date(2025, 12, 31)
DATE_RANGE = (END_DATE - START_DATE).days


def _generate_sales(n: int = 50) -> list[tuple]:
    random.seed(42)
    rows = []
    for i in range(1, n + 1):
        d = START_DATE + timedelta(days=random.randint(0, DATE_RANGE))
        rows.append((
            i,
            d.isoformat(),
            random.randint(1, 10),
            random.choice(REGIONS),
            round(random.uniform(500, 50000), 2),
            random.randint(10, 500),
        ))
    return rows


def setup_database() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS sales")
    cur.execute("DROP TABLE IF EXISTS products")

    cur.execute("""
        CREATE TABLE products (
            id                  INTEGER PRIMARY KEY,
            name                TEXT    NOT NULL,
            category            TEXT    NOT NULL,
            cost_to_manufacture REAL    NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE sales (
            id         INTEGER PRIMARY KEY,
            date       TEXT    NOT NULL,
            product_id INTEGER NOT NULL,
            region     TEXT    NOT NULL,
            revenue    REAL    NOT NULL,
            units_sold INTEGER NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    cur.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        PRODUCTS,
    )
    cur.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?)",
        _generate_sales(),
    )

    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")
    print(f"  products: {len(PRODUCTS)} rows")
    print(f"  sales:    50 rows")


if __name__ == "__main__":
    setup_database()
