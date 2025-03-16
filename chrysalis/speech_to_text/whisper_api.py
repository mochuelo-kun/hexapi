import logging
import time
from typing import Union, Dict, Any
from .base import SpeechToTextBase
from ..config import Config
import numpy as np
import tempfile
from .. import audio_utils

logger = logging.getLogger('chrysalis.stt.whisper_api')

class WhisperAPI(SpeechToTextBase):
    def __init__(self, enable_diarization: bool = False, **kwargs):
        """Initialize Whisper API client
        
        Args:
            enable_diarization: Whether to enable speaker diarization (not supported)
            **kwargs: Additional parameters (for compatibility)
        """
        if enable_diarization:
            logger.warning("Diarization is not supported by the Whisper API. The parameter will be ignored.")
            
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
        
    
    def transcribe_array(self, audio_data: np.ndarray, sample_rate: int) -> Union[str, Dict[str, Any]]:
        """Transcribe audio data directly from numpy array
        
        Since the Whisper API requires a file, this method creates a temporary file.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data in Hz
            
        Returns:
            Transcribed text as string
        """
        logger.debug("Transcribing audio array with Whisper API (%.2fs at %dHz)",
                    len(audio_data)/sample_rate, sample_rate)
        start = time.time()
        
        # The API requires a file, so save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            # Save the audio data
            audio_utils.save_audio_file(audio_data, temp_file.name, sample_rate=sample_rate)
            
            # Send to API
            with open(temp_file.name, "rb") as audio:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
        
        duration = time.time() - start
        text = transcript.text.strip()
        logger.debug("Audio array transcription completed in %.2fs: %s",
                    duration, text[:50] + "..." if len(text) > 50 else text)
        
        return text 