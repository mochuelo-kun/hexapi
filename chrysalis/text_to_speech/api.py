from typing import Optional
from .base import TextToSpeechBase
from .elevenlabs import ElevenLabsTTS
from .piper_local import PiperLocalTTS
import logging

logger = logging.getLogger('chrysalis.tts')

class TextToSpeechAPI:
    def __init__(self, implementation: str = "elevenlabs", **kwargs):
        self.implementation_map = {
            "elevenlabs": ElevenLabsTTS,
            "piper": PiperLocalTTS,
            # Add other implementations here
        }
        
        if implementation not in self.implementation_map:
            raise ValueError(f"Unknown implementation: {implementation}")
            
        self.implementation = self.implementation_map[implementation]()
    
    def synthesize(self, 
                  text: str,
                  output_file: str,
                  voice_id: Optional[str] = None,
                  **kwargs) -> None:
        """
        Synthesize text to speech
        
        Args:
            text: Input text
            output_file: Path to save audio file
            voice_id: Voice identifier (implementation specific)
            **kwargs: Additional implementation-specific parameters
        """
        if voice_id and hasattr(self.implementation, 'voice_id'):
            self.implementation.voice_id = voice_id
            
        self.implementation.synthesize(text, output_file, **kwargs) 