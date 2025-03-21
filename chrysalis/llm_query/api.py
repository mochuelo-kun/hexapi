import logging
from typing import Dict, List, Optional, Type
from .base import LLMQueryBase
from .ollama import OllamaQuery
# Import other implementations here

logger = logging.getLogger('chrysalis.llm_query.api')

# Map of implementation names to their classes
IMPLEMENTATIONS: Dict[str, Type[LLMQueryBase]] = {
    "ollama": OllamaQuery,
    # Add other implementations here
}

class LLMQueryAPI:
    """Factory for creating LLM query implementations"""
    
    def __init__(self, implementation: str, model_name: str, **kwargs):
        """
        Initialize LLM query API
        
        Args:
            implementation: Name of the implementation to use
            model_name: Name of the model to use
            **kwargs: Additional parameters for the implementation
        """
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"Unknown LLM implementation: {implementation}")
            
        self.implementation = implementation
        self.model_name = model_name
        self.kwargs = kwargs
        
        logger.info("Creating LLM query API with implementation: %s, model: %s", 
                   implementation, model_name)
        
        try:
            self._impl = IMPLEMENTATIONS[implementation](
                model_name=model_name,
                **kwargs
            )
        except Exception as e:
            logger.error("Failed to initialize LLM implementation %s: %s", 
                        implementation, e, exc_info=True)
            raise
    
    def query(self, prompt: str, system_prompt: str = None) -> str:
        """
        Query the LLM with the given prompt
        
        Args:
            prompt: The user's prompt
            system_prompt: Optional system prompt to set context
            
        Returns:
            The model's response
        """
        logger.debug("Querying LLM with prompt: %s", 
                    prompt[:100] + "..." if len(prompt) > 100 else prompt)
        
        try:
            return self._impl.query(prompt, system_prompt)
        except Exception as e:
            logger.error("Error during LLM query: %s", e, exc_info=True)
            raise
    
    # def query_multiple(self, 
    #                   prompt: str,
    #                   models: Optional[List[str]] = None,
    #                   **kwargs) -> Dict[str, str]:
    #     """
    #     Query multiple LLM models in parallel
        
    #     Args:
    #         prompt: Input text prompt
    #         models: List of model identifiers
    #         **kwargs: Additional model-specific parameters
            
    #     Returns:
    #         Dict mapping model names to responses
    #     """
    #     if models is None:
    #         models = ["ollama:llama2"]  # Default model
            
    #     implementations = {
    #         model: self.get_implementation(model)
    #         for model in models
    #     }
        
    #     multi_query = MultiLLMQuery(implementations)
    #     return multi_query.query_llms(prompt, **kwargs) 