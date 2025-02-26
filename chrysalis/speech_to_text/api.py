from typing import Optional, Tuple
import numpy as np
from .base import SpeechToTextBase
from .whisper_api import WhisperAPI

class SpeechToTextAPI:
    def __init__(self, implementation: str = "whisper_api"):
        self.implementation_map = {
            "whisper_api": WhisperAPI,
            # Add other implementations here
        }
        
        if implementation not in self.implementation_map:
            raise ValueError(f"Unknown implementation: {implementation}")
            
        self.implementation = self.implementation_map[implementation]()
    
    def transcribe(self, 
                  audio_file: Optional[str] = None,
                  audio_data: Optional[Tuple[np.ndarray, int]] = None,
                  use_mic: bool = False) -> str:
        """
        Transcribe audio from file, numpy array, or microphone
        
        Args:
            audio_file: Path to audio file
            audio_data: Tuple of (audio_array, sample_rate)
            use_mic: Whether to record from microphone
            
        Returns:
            Transcribed text
        """
        if use_mic:
            audio_data = self.implementation.record_audio()
            
        if audio_data is not None:
            return self.implementation.transcribe_audio(audio_data[0], audio_data[1])
        elif audio_file is not None:
            return self.implementation.transcribe_file(audio_file)
        else:
            raise ValueError("Must provide audio_file, audio_data or use_mic=True") 