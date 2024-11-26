from core.vector_store import VectorStore


def test_vector_store(tmp_path):
    vector_dim = 384
    store_path = tmp_path / "test.index"

    store = VectorStore(vector_dim, store_path)
    chunks = [
        {"chunk_id": "1", "chunk_text": "Hello world", "file_path": "file1.txt"},
        {"chunk_id": "2", "chunk_text": "This is a test", "file_path": "file2.txt"}
    ]

    # Векторизация и индексация
    store.index_chunks(chunks)
    store.save_index()

    # Поиск
    query_vector = [0.1] * vector_dim  # Простой тестовый вектор
    results = store.search(query_vector, top_k=1)

    assert len(results) == 1
    assert results[0][0] == 0  # Первый элемент — ID ближайшего вектора
