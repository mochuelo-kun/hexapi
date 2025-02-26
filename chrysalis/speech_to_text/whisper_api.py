import openai
from .base import SpeechToTextBase
from ..config import Config
import numpy as np
import soundfile as sf
import tempfile

class WhisperAPI(SpeechToTextBase):
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def transcribe_file(self, audio_file: str) -> str:
        with open(audio_file, "rb") as audio:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        return transcript.text
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            return self.transcribe_file(temp_file.name) 