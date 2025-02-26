import time
import logging
from typing import Optional, Tuple
import numpy as np
from .base import SpeechToTextBase
from .whisper_api import WhisperAPI
from .whisper_local import WhisperLocal

logger = logging.getLogger('chrysalis.stt')

class SpeechToTextAPI:
    def __init__(self, implementation: str = "whisper_api", **kwargs):
        self.implementation_map = {
            "whisper_api": WhisperAPI,
            "whisper_local": WhisperLocal,
            # Add other implementations here
        }
        
        if implementation not in self.implementation_map:
            raise ValueError(f"Unknown implementation: {implementation}")
        
        logger.info("Initializing STT with implementation: %s", implementation)
        start = time.time()
        self.implementation = self.implementation_map[implementation](**kwargs)
        logger.debug("STT implementation initialized in %.2fs", time.time() - start)
    
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
        start = time.time()
        
        if use_mic:
            logger.info("Recording from microphone")
            mic_start = time.time()
            audio_data = self.implementation.record_audio()
            logger.debug("Recording completed in %.2fs", time.time() - mic_start)
            
        if audio_data is not None:
            logger.info("Transcribing audio data")
            result = self.implementation.transcribe_audio(audio_data[0], audio_data[1])
        elif audio_file is not None:
            logger.info("Transcribing audio file: %s", audio_file)
            result = self.implementation.transcribe_file(audio_file)
        else:
            raise ValueError("Must provide audio_file, audio_data or use_mic=True")
        
        duration = time.time() - start
        logger.info("Transcription completed in %.2fs: %s", duration, result)
        return result 