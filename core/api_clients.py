from abc import ABC, abstractmethod


class AbstractAPIClient(ABC):
    """
    Абстрактный класс для API-клиентов языковых моделей.
    """
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Метод для получения ответа от модели.
        :param prompt: Промт для генерации.
        :param kwargs: Дополнительные параметры.
        :return: Сгенерированный ответ.
        """
        pass


class OpenAIAPIClient(AbstractAPIClient):
    """
    Клиент для взаимодействия с OpenAI API.
    """
    def __init__(self, api_key: str):
        import openai
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate(self, prompt: str, model: str = "gpt-3.5-turbo", **kwargs) -> str:
        import openai
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": prompt}]
        )
        return response['choices'][0]['message']['content']


class OpenRouterAPIClient(AbstractAPIClient):
    """
    Клиент для взаимодействия с OpenAI API.
    """
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            )

    def generate(self, prompt: str, model: str = "meta-llama/llama-3.1-70b-instruct:free", **kwargs) -> str:
        from openai import OpenAI
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']


class CustomAPIClient(AbstractAPIClient):
    """
    Пример кастомного клиента для другого API.
    """
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"prompt": prompt, **kwargs}
        response = requests.post(f"{self.base_url}/generate", headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("text", "")
