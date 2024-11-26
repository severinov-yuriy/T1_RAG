from fastapi import FastAPI
from app.routes import upload, query

# Инициализация приложения
app = FastAPI(
    title="RAG Service",
    description="A Retrieval-Augmented Generation (RAG) Service for document-based Q&A",
    version="1.0.0"
)

# Регистрация маршрутов
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(query.router, prefix="/api", tags=["Query"])

# Точка входа для проверки статуса
@app.get("/")
async def root():
    return {"message": "RAG Service is running"}
