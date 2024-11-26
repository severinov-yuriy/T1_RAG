from core.rag import RAGPipeline
from core.api_clients import AbstractAPIClient


class MockAPIClient(AbstractAPIClient):
    def generate(self, prompt: str, **kwargs) -> str:
        return f"Mock response to: {prompt}"


def test_rag_pipeline(tmp_path):
    db_path = tmp_path / "test.db"
    vector_store_path = tmp_path / "test.index"
    vector_dim = 384

    # Инициализация RAG
    rag = RAGPipeline(vector_store_path, db_path, vector_dim)

    # Mock контекст
    chunks = [
        {"chunk_id": "1", "chunk_text": "Test chunk 1", "file_path": "file1.txt"},
        {"chunk_id": "2", "chunk_text": "Test chunk 2", "file_path": "file2.txt"}
    ]
    from core.database import init_db, save_chunks
    init_db(db_path)
    save_chunks(chunks, db_path)

    # Mock API
    api_client = MockAPIClient()
    response = rag.get_response("What is the test?", api_client, top_k=1)

    assert "Mock response" in response["answer"]
    assert len(response["context_files"]) == 1
