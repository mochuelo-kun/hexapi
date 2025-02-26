import time
import logging
import whisper
import numpy as np
from typing import Optional
from .base import SpeechToTextBase

logger = logging.getLogger('chrysalis.stt.whisper_local')

class WhisperLocal(SpeechToTextBase):
    def __init__(self, model_name: str = "base"):
        """
        Initialize local Whisper model
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        logger.info("Loading Whisper model: %s", model_name)
        load_start = time.time()
        self.model = whisper.load_model(model_name)
        logger.info("Whisper model loaded in %.2fs", time.time() - load_start)
        
        self.model_name = model_name
    
    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe audio from a file"""
        logger.debug("Transcribing file with %s model: %s", self.model_name, audio_file)
        start = time.time()
        
        result = self.model.transcribe(audio_file)
        
        duration = time.time() - start
        logger.debug("File transcription completed in %.2fs", duration)
        return result["text"].strip()
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio from numpy array"""
        logger.debug("Transcribing audio data with %s model", self.model_name)
        start = time.time()
        
        # Whisper expects float32 in range [-1, 1]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0  # Assuming 16-bit audio
            
        result = self.model.transcribe(audio_data)
        
        duration = time.time() - start
        logger.debug("Audio data transcription completed in %.2fs", duration)
        return result["text"].strip() 