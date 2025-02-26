from abc import ABC, abstractmethod
from typing import Optional

class TextToSpeechBase(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_file: str, **kwargs) -> None:
        """Synthesize text to speech and save to file."""
        pass 