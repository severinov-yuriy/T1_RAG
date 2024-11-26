from fastapi.testclient import TestClient
from app.main import app


def test_query_endpoint():
    client = TestClient(app)

    # Mock запрос
    request_data = {
        "query": "What is the test?",
        "api_type": "openai",
        "api_key": "mock_key",
        "model_name": "gpt-3.5-turbo",
        "top_k": 1
    }
    response = client.post("/api/query/", json=request_data)

    assert response.status_code == 200
    assert "answer" in response.json()
