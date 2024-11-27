from typing import List, Dict
from core.vectorization import TextVectorizer
from core.vector_store import VectorStore
from core.database import fetch_chunks_by_ids


class RAGPipeline:
    def __init__(self, vector_store_path: str, db_path: str, vector_dim: int = 384, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализация RAG-пайплайна.
        
        :param vector_store_path: Путь к файлу векторного индекса.
        :param db_path: Путь к SQLite базе данных.
        :param vector_dim: Размерность векторов.
        :param model_name: Имя модели SentenceTransformers для векторизации.
        """
        self.vector_store = VectorStore(vector_dim, vector_store_path)
        self.vector_store.load_index()
        self.vectorizer = TextVectorizer(model_name)
        self.db_path = db_path

    def retrieve_context(self, query: str, top_k: int = 10) -> List[Dict[str, str]]:
        """
        Извлечение релевантного контекста на основе пользовательского запроса.
        
        :param query: Запрос пользователя.
        :param top_k: Количество релевантных текстов для извлечения.
        :return: Список текстовых кусочков с метаданными.
        """
        # Векторизация запроса
        query_vector = self.vectorizer.vectorize([query])[0]

        # Поиск ближайших соседей
        results = self.vector_store.search(query_vector, top_k)

        print([(idx, _) for idx, _ in results])
        print([idx for idx in self.vector_store.metadata])
        # Извлечение метаданных из базы данных
        chunk_ids = [self.vector_store.metadata[idx] for idx, _ in results]
        context_chunks = fetch_chunks_by_ids(chunk_ids, self.db_path)

        return context_chunks

    def generate_prompt(self, query: str, context_chunks: List[Dict[str, str]]) -> str:
        """
        Формирование промта для языковой модели.
        
        :param query: Запрос пользователя.
        :param context_chunks: Список текстовых кусочков для контекста.
        :return: Готовый промт.
        """
        context_texts = "\n\n".join([f"[{chunk['file_path']}]: {chunk['chunk_text']}" for chunk in context_chunks])
        prompt = f"""
            You are a helpful assistant. Use the context provided to answer the user's query.
            Context:
            {context_texts}
            User Query: {query}
            Answer:
            """
        return prompt

    def get_response(self, query: str, api_client, top_k: int = 10, **kwargs) -> Dict[str, str]:
        """
        Получение ответа от языковой модели на основе RAG.
        
        :param query: Запрос пользователя.
        :param api_client: Клиент для взаимодействия с API модели (объект, реализующий метод `generate`).
        :param top_k: Количество релевантных текстов для извлечения.
        :param kwargs: Дополнительные параметры для метода `generate`.
        :return: Ответ и информация о контексте.
        """
        # Извлечение контекста
        context_chunks = self.retrieve_context(query, top_k)
        # Формирование промта
        prompt = self.generate_prompt(query, context_chunks)
        print(prompt)

        # Получение ответа от модели
        answer = api_client.generate(prompt, **kwargs)

        file_paths = [chunk['file_path'] for chunk in context_chunks]

        return {
            "query": query,
            "answer": answer,
            "context_files": file_paths
        }
