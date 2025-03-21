from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger('chrysalis.llm_query.base')

class LLMQueryBase(ABC):
    """Base class for LLM query implementations"""
    
    def __init__(self, implementation: str, model_name: str, **kwargs):
        """
        Initialize LLM query implementation
        
        Args:
            implementation: Name of the implementation (e.g., "ollama", "openai")
            model_name: Name of the model to use (e.g., "llama2", "gpt-4")
            **kwargs: Additional implementation-specific parameters
        """
        self.implementation = implementation
        self.model_name = model_name
        self.kwargs = kwargs
        logger.info("Initialized %s LLM query with model: %s", implementation, model_name)
    
    @abstractmethod
    def query(self, prompt: str, system_prompt: str = None) -> str:
        """
        Query the LLM with the given prompt
        
        Args:
            prompt: The user's prompt
            system_prompt: Optional system prompt to set context
            
        Returns:
            The model's response
        """
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
                results[model_id] = self.models[model_id].query(prompt, **kwargs)
        return results 