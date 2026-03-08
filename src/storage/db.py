import sqlite3
import pandas as pd
from contextlib import contextmanager

# Constants
DB_PATH = '.refinery/facts.db'

# Schema Definition
FACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT,
    entity_name TEXT,
    attribute TEXT,
    value REAL,
    unit TEXT,
    year INTEGER,
    page_number INTEGER,
    bbox TEXT,
    content_hash TEXT,
    confidence REAL,
    source_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

DOCUMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT,
    ingestion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(FACTS_SCHEMA)
    cursor.execute(DOCUMENTS_SCHEMA)
    conn.commit()
    conn.close()
    
def insert_fact(conn, fact_dict):
    """
    Inserts a single fact into the database.
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO facts (
            doc_id, entity_name, attribute, value, unit, year, 
            page_number, bbox, content_hash, confidence, source_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        fact_dict.get('doc_id'),
        fact_dict.get('entity_name'),
        fact_dict.get('attribute'),
        fact_dict.get('value'),
        fact_dict.get('unit'),
        fact_dict.get('year'),
        fact_dict.get('page_number'),
        str(fact_dict.get('bbox')),
        fact_dict.get('content_hash'),
        fact_dict.get('confidence'),
        fact_dict.get('source_text')
    ))

def safe_query(query, params=()):
    """
    Executes a read-only query and returns a DataFrame.
    """
    # Simple check for read-only via keyword, but SQLite permissions are better managed via connection URI if needed
    query_lower = query.lower().strip()
    if not query_lower.startswith("select"):
        raise ValueError("Only SELECT queries are allowed via safe_query.")
        
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
