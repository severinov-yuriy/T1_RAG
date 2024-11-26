from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.rag import RAGPipeline
from core.api_clients import OpenAIAPIClient, CustomAPIClient, OpenRouterAPIClient

router = APIRouter()

VECTOR_STORE_PATH = "data/vector_store.index"
DB_PATH = "data/chunks.db"
VECTOR_DIM = 384

# Конфигурация клиентов API
API_CLIENTS = {
    "openai": OpenAIAPIClient,
    "custom": CustomAPIClient,
    "openrouter": OpenRouterAPIClient
}


class QueryRequest(BaseModel):
    query: str
    api_type: str
    api_key: str
    api_url: str = None  # Для кастомных моделей
    model_name: str = "meta-llama/llama-3.1-70b-instruct:free"
    top_k: int = 10


@router.post("/query/")
async def handle_query(request: QueryRequest):
    """
    Эндпоинт для обработки запросов пользователя с использованием RAG.
    """
    try:
        # Инициализируем RAG-пайплайн
        rag = RAGPipeline(
            vector_store_path=VECTOR_STORE_PATH,
            db_path=DB_PATH,
            vector_dim=VECTOR_DIM
        )

        # Инициализируем API-клиент
        if request.api_type not in API_CLIENTS:
            raise HTTPException(status_code=400, detail="Unsupported API type")

        if request.api_type == "custom":
            if not request.api_url:
                raise HTTPException(status_code=400, detail="API URL is required for custom API type")
            api_client = API_CLIENTS[request.api_type](base_url=request.api_url, api_key=request.api_key)
        else:
            api_client = API_CLIENTS[request.api_type](api_key=request.api_key)

        # Обрабатываем запрос
        response = rag.get_response(
            query=request.query,
            api_client=api_client,
            top_k=request.top_k,
            model=request.model_name
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
