import requests
import logging
import ollama
from typing import Optional
from .base import LLMQueryBase
from ..config import Config

logger = logging.getLogger('chrysalis.llm_query.ollama')

DEFAULT_LLM = "qwen2.5:1.5b"

class OllamaQuery(LLMQueryBase):
    """Ollama LLM query implementation"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize Ollama query implementation
        
        Args:
            model_name: Name of the Ollama model to use
            **kwargs: Additional parameters
        """
        super().__init__("ollama", model_name, **kwargs)
        self.api_url = Config.OLLAMA_HOST
        logger.debug("Using Ollama API at: %s", self.api_url)
    
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Query Ollama with the given prompt
        
        Args:
            prompt: The user's prompt
            system_prompt: Optional system prompt to set context
            
        Returns:
            The model's response
        """
        logger.debug("Querying Ollama model %s with prompt: %s", 
                    self.model_name, prompt[:100] + "..." if len(prompt) > 100 else prompt)
        
        # TODO: implement system_prompt
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
            )
            
            logger.debug("Received response from Ollama: %s", response)
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error("Error querying Ollama: %s", e)
            raise 