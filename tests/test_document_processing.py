import os
from core.document_processing import process_archive


def test_process_archive(tmp_path):
    # Создаем временный архив
    test_file_path = tmp_path / "test.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test document. " * 50)

    archive_path = tmp_path / "test.zip"
    import zipfile
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.write(test_file_path, "test.txt")

    # Тестируем обработку архива
    chunks = process_archive(archive_path, max_chunk_size=50)

    assert len(chunks) > 1
    assert chunks[0]["chunk_text"].startswith("This is a test")
    assert chunks[0]["file_path"] == "test.txt"
