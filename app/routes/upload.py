import os
from fastapi import APIRouter, UploadFile, HTTPException
from core.document_processing import process_archive
from core.database import init_database, save_chunks_to_db
from core.vector_store import VectorStore

router = APIRouter()

# Путь для сохранения временных файлов
TEMP_FOLDER = "data/uploads"
DB_PATH = "data/chunks.db"
VECTOR_STORE_PATH = "data/vector_store.index"
VECTOR_DIM = 384


@router.post("/upload/")
async def upload_archive(file: UploadFile):
    """
    Эндпоинт для загрузки архива документов.
    """
    # Создаем временную папку, если она не существует
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    # Сохраняем загруженный файл
    archive_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(archive_path, "wb") as f:
        f.write(await file.read())

    try:
        # Обрабатываем архив и получаем текстовые кусочки
        chunks = process_archive(archive_path, max_chunk_size=500)

        # Инициализируем базу данных и сохраняем кусочки
        init_database(DB_PATH)
        save_chunks_to_db(chunks, DB_PATH)

        # Инициализируем векторное хранилище и индексируем кусочки
        vector_store = VectorStore(VECTOR_DIM, VECTOR_STORE_PATH)
        vector_store.index_chunks(chunks)
        vector_store.save_index()

        return {"message": "Archive processed successfully", "total_chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing archive: {str(e)}")
    finally:
        # Удаляем временный файл
        os.remove(archive_path)
