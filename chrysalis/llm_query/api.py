from typing import Optional, Dict, List
from .base import LLMQueryBase, MultiLLMQuery
from .ollama import OllamaQuery
# Import other implementations here

class LLMQueryAPI:
    def __init__(self):
        self.implementation_map = {
            "ollama": OllamaQuery,
            # Add other implementations here
        }
        
        self.instances: Dict[str, LLMQueryBase] = {}
    
    def get_implementation(self, model_name: str) -> LLMQueryBase:
        """Get or create an implementation instance based on model name prefix"""
        if model_name in self.instances:
            return self.instances[model_name]
            
        # Parse model name to determine implementation
        # Example: "ollama:llama2" -> use OllamaQuery with model "llama2"
        impl_name, *model_parts = model_name.split(":")
        actual_model = model_parts[0] if model_parts else model_name
        
        if impl_name not in self.implementation_map:
            raise ValueError(f"Unknown implementation: {impl_name}")
            
        instance = self.implementation_map[impl_name](model_name=actual_model)
        self.instances[model_name] = instance
        return instance
    
    def query(self, prompt: str, model: str = "ollama:llama2", **kwargs) -> str:
        """
        Query a single LLM model
        
        Args:
            prompt: Input text prompt
            model: Model identifier (e.g. "ollama:llama2", "openai:gpt-4")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model response
        """
        implementation = self.get_implementation(model)
        return implementation.query_llm(prompt, **kwargs)
    
    def query_multiple(self, 
                      prompt: str,
                      models: Optional[List[str]] = None,
                      **kwargs) -> Dict[str, str]:
        """
        Query multiple LLM models in parallel
        
        Args:
            prompt: Input text prompt
            models: List of model identifiers
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict mapping model names to responses
        """
        if models is None:
            models = ["ollama:llama2"]  # Default model
            
        implementations = {
            model: self.get_implementation(model)
            for model in models
        }
        
        multi_query = MultiLLMQuery(implementations)
        return multi_query.query_llms(prompt, **kwargs) 