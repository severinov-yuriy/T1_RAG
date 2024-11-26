import sqlite3
import os
from core.database import init_db, save_chunks, fetch_chunks_by_ids


def test_database_operations(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)

    chunks = [
        {"chunk_id": "1", "chunk_text": "Test chunk 1", "file_path": "file1.txt"},
        {"chunk_id": "2", "chunk_text": "Test chunk 2", "file_path": "file2.txt"}
    ]

    save_chunks(chunks, db_path)
    retrieved_chunks = fetch_chunks_by_ids(["1", "2"], db_path)

    assert len(retrieved_chunks) == 2
    assert retrieved_chunks[0]["chunk_text"] == "Test chunk 1"
