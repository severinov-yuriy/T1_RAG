import faiss
import numpy as np
from typing import List, Tuple

class VectorStore:
    def __init__(self, vector_dim: int, index_path: str = "data/vector_store.index"):
        """
        Инициализация векторного хранилища.
        :param vector_dim: Размерность векторов.
        :param index_path: Путь к файлу векторного индекса.
        """
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(vector_dim)

    def add_vectors(self, vectors: List[List[float]], metadata: List[str]) -> None:
        """
        Добавление векторов в хранилище.
        :param vectors: Список векторных представлений.
        :param metadata: Список метаданных (например, идентификаторы текстов).
        """
        vectors_np = np.array(vectors, dtype="float32")
        self.index.add(vectors_np)
        # Сохраняем метаданные для привязки к вектору (на данном этапе локально)
        self.metadata = metadata

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Поиск ближайших соседей по вектору.
        :param query_vector: Вектор запроса.
        :param top_k: Количество ближайших соседей.
        :return: Список индексов и расстояний для ближайших соседей.
        """
        query_np = np.array([query_vector], dtype="float32")
        distances, indices = self.index.search(query_np, top_k)
        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

    def save_index(self) -> None:
        """
        Сохранение индекса на диск.
        """
        faiss.write_index(self.index, self.index_path)

    def load_index(self) -> None:
        """
        Загрузка индекса с диска.
        """
        self.index = faiss.read_index(self.index_path)
