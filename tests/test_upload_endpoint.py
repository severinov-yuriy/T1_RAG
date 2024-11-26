from fastapi.testclient import TestClient
from app.main import app


def test_upload_endpoint(tmp_path):
    client = TestClient(app)

    # Создаем архив
    test_file_path = tmp_path / "test.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test document. " * 50)

    archive_path = tmp_path / "test.zip"
    import zipfile
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.write(test_file_path, "test.txt")

    # Тестируем загрузку
    with open(archive_path, "rb") as archive:
        response = client.post("/api/upload/", files={"file": archive})

    assert response.status_code == 200
    assert "total_chunks" in response.json()
