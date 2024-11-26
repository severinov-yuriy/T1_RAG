import os

# Настройки приложения
class Config:
    DATA_FOLDER = "data"
    UPLOAD_FOLDER = os.path.join(DATA_FOLDER, "uploads")
    DB_PATH = os.path.join(DATA_FOLDER, "chunks.db")
    VECTOR_STORE_PATH = os.path.join(DATA_FOLDER, "vector_store.index")
    VECTOR_DIM = 384
    ALLOWED_EXTENSIONS = {".zip"}
