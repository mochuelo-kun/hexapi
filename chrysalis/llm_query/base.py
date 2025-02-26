from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class LLMQueryBase(ABC):
    @abstractmethod
    def query_llm(self, prompt: str, **kwargs) -> str:
        """Query a single LLM model."""
        pass

class MultiLLMQuery:
    def __init__(self, models: Dict[str, LLMQueryBase]):
        self.models = models
    
    def query_llms(self, prompt: str, model_ids: Optional[List[str]] = None, **kwargs) -> Dict[str, str]:
        """Query multiple LLM models in parallel."""
        if model_ids is None:
            model_ids = list(self.models.keys())
            
        results = {}
        # TODO: Make this actually parallel using asyncio
        for model_id in model_ids:
            if model_id in self.models:
                results[model_id] = self.models[model_id].query_llm(prompt, **kwargs)
        return results 