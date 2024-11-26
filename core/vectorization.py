from typing import List
from sentence_transformers import SentenceTransformer

class TextVectorizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализация векторизатора текста.
        :param model_name: Имя модели SentenceTransformers для векторизации.
        """
        self.model = SentenceTransformer(model_name)

    def vectorize(self, texts: List[str]) -> List[List[float]]:
        """
        Векторизация списка текстов.
        :param texts: Список текстовых строк.
        :return: Список векторных представлений.
        """
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
