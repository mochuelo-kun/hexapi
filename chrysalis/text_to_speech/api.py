from typing import Optional
from .base import TextToSpeechBase
from .elevenlabs import ElevenLabsTTS
from .piper_local import PiperLocalTTS
import logging

logger = logging.getLogger('chrysalis.tts')

class TextToSpeechAPI:
    def __init__(self,
                 implementation: str,
                 tts_model: str,
                 **kwargs):
        self.implementation_map = {
            "elevenlabs": ElevenLabsTTS,
            "piper": PiperLocalTTS,
            # Add other implementations here
        }
        
        if implementation not in self.implementation_map:
            raise ValueError(f"Unknown implementation: {implementation}")
        self.tts_model = tts_model
        self.implementation = self.implementation_map[implementation]()
    
    def synthesize(self, 
                  text: str,
                  output_file: str,
                  **kwargs) -> None:
        """
        Synthesize text to speech
        
        Args:
            text: Input text
            output_file: Path to save audio file
            **kwargs: Additional implementation-specific parameters
        """
            
        self.implementation.synthesize(text, output_file, **kwargs) 