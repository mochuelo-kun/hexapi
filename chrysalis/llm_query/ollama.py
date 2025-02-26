import requests
from .base import LLMQueryBase
from ..config import Config

class OllamaQuery(LLMQueryBase):
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.base_url = Config.OLLAMA_HOST
    
    def query_llm(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                **kwargs
            }
        )
        return response.json()["response"] 