import sqlite3
from typing import List, Dict

# Создаем таблицу в SQLite для хранения метаданных
def init_database(db_path: str = "data/chunks.db") -> None:
    """
    Инициализация базы данных: создание таблицы для хранения кусочков текста.

    :param db_path: Путь к файлу базы данных.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS text_chunks (
        id TEXT PRIMARY KEY,
        file_path TEXT,
        chunk_text TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_chunks_to_db(chunks: List[Dict[str, str]], db_path: str = "data/chunks.db") -> None:
    """
    Сохранение кусочков текста в базу данных.

    :param chunks: Список словарей с кусочками текста.
    :param db_path: Путь к файлу базы данных.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for chunk in chunks:
        cursor.execute("""
        INSERT OR IGNORE INTO text_chunks (id, file_path, chunk_text)
        VALUES (?, ?, ?)
        """, (chunk["id"], chunk["file_path"], chunk["chunk_text"]))

    conn.commit()
    conn.close()


def fetch_all_chunks(db_path: str = "data/chunks.db") -> List[Dict[str, str]]:
    """
    Извлечение всех кусочков текста из базы данных.

    :param db_path: Путь к файлу базы данных.
    :return: Список словарей с кусочками текста.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, file_path, chunk_text FROM text_chunks")
    rows = cursor.fetchall()

    conn.close()

    return [{"id": row[0], "file_path": row[1], "chunk_text": row[2]} for row in rows]


def fetch_chunks_by_ids(ids: List[str], db_path: str = "data/chunks.db") -> List[Dict[str, str]]:
    """
    Извлечение кусочков текста по их идентификаторам.

    :param ids: Список идентификаторов.
    :param db_path: Путь к файлу базы данных.
    :return: Список словарей с кусочками текста.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT id, file_path, chunk_text FROM text_chunks WHERE id IN ({','.join(['?'] * len(ids))})"
    cursor.execute(query, ids)
    rows = cursor.fetchall()

    conn.close()

    return [{"id": row[0], "file_path": row[1], "chunk_text": row[2]} for row in rows]
