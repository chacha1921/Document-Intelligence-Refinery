import logging
import sqlite3
import os
from src.models import ExtractedDocument
from src.agents.fact_extractor import FactTableExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestor:
    """
    Handles ingestion of ExtractedDocuments into:
    1. Vector Store (ChromaDB/FAISS) - for text chunks
    2. SQL Database (SQLite) - for structured facts
    """
    
    def __init__(self, db_path="refinery.db", vector_store_path="chroma_db"):
        self.db_path = db_path
        self.vector_store_path = vector_store_path
        self._init_sql_db()
        self.fact_extractor = FactTableExtractor()

    def _init_sql_db(self):
        """Initialize SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT,
                    entity TEXT,
                    attribute TEXT,
                    value TEXT,
                    unit TEXT,
                    page_num INTEGER,
                    extraction_method TEXT,
                    confidence REAL,
                    timestamp TEXT
                )
            ''')
            conn.commit()

    def ingest_document(self, doc_id: str, document: ExtractedDocument):
        """
        Main ingestion pipeline.
        """
        logger.info(f"Ingesting document {doc_id}...")
        
        # 1. Fact Extraction & SQL Ingestion
        facts = self.fact_extractor.extract_facts(document)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for fact in facts:
                cursor.execute('''
                    INSERT INTO facts (doc_id, entity, attribute, value, unit, page_num, extraction_method, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id, fact.entity, fact.attribute, str(fact.value), fact.unit,
                    fact.provenance.page_number, fact.provenance.extraction_method,
                    fact.provenance.confidence_score, fact.provenance.timestamp
                ))
            conn.commit()
        logger.info(f"Ingested {len(facts)} facts into SQL.")

        # 2. Vector Store Ingestion (Mocked per requirement)
        self._ingest_to_vector_store(doc_id, document)

    def _ingest_to_vector_store(self, doc_id: str, document: ExtractedDocument):
        """
        Chunk text and add to vector DB.
        """
        # from chromadb import Client...
        # collection = client.get_or_create_collection("refinery_docs")
        
        # Placeholder
        chunks = len(document.text_blocks) # Simplified
        logger.info(f"Simulated ingestion of {chunks} chunks into Vector Store at {self.vector_store_path}")

if __name__ == "__main__":
    ingestor = DataIngestor()
    # Mock document passing would happen here
    logger.info("Ingestor initialized and DB schema created.")
