import logging
import time
from .base import SpeechToTextBase
from ..config import Config
import numpy as np
import soundfile as sf
import tempfile

logger = logging.getLogger('chrysalis.stt.whisper_api')

class WhisperAPI(SpeechToTextBase):
    def __init__(self):
        """Initialize Whisper API client"""
        logger.info("Initializing Whisper API client")
        try:
            import openai
            self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info("Whisper API client initialized successfully")
        except ImportError as e:
            logger.error("Failed to import OpenAI dependencies: %s", e)
            raise ImportError(
                "Missing required dependency: openai. "
                "Please check the README for installation instructions."
            )
    
    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe audio from a file"""
        logger.debug("Transcribing file with Whisper API: %s", audio_file)
        start = time.time()
        
        with open(audio_file, "rb") as audio:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        
        duration = time.time() - start
        logger.debug("File transcription completed in %.2fs", duration)
        return transcript.text
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio from numpy array"""
        logger.debug("Transcribing audio data with Whisper API")
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            return self.transcribe_file(temp_file.name) 