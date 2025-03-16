import time
import logging
import numpy as np
from typing import Union, Dict, Any
from .base import SpeechToTextBase

logger = logging.getLogger('chrysalis.stt.whisper_local')

class WhisperLocal(SpeechToTextBase):
    def __init__(self, 
                model_name: str = "base", 
                enable_diarization: bool = False,
                **kwargs):
        """
        Initialize local Whisper model
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
            enable_diarization: Whether to enable speaker diarization (not supported in Whisper)
            **kwargs: Additional parameters (for compatibility with other STT backends)
        """
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        
        if enable_diarization:
            logger.warning("Diarization is not supported by the Whisper local model. The parameter will be ignored.")
        
        logger.info("Loading Whisper model: %s", model_name)
        try:
            import whisper
            load_start = time.time()
            self.model = whisper.load_model(model_name)
            logger.info("Whisper model loaded in %.2fs", time.time() - load_start)
        except ImportError as e:
            logger.error("Failed to import Whisper dependencies: %s", e)
            raise ImportError(
                "Missing required dependency: whisper. "
                "Please check the README for installation instructions."
            )
    
    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe audio from a file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text
        """
        logger.debug("Transcribing file with %s model: %s", self.model_name, audio_file)
        start = time.time()
        
        result = self.model.transcribe(audio_file)
        
        duration = time.time() - start
        logger.debug("File transcription completed in %.2fs", duration)
        return result["text"].strip()
    
    def transcribe_array(self, audio_data: np.ndarray, sample_rate: int) -> Union[str, Dict[str, Any]]:
        """Transcribe audio data directly from numpy array
        
        This method implements the SpeechToTextBase interface for direct array processing.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data in Hz
            
        Returns:
            Transcribed text as string (Whisper doesn't support diarization)
        """
        logger.debug("Transcribing audio array with %s model (%.2fs at %dHz)", 
                    self.model_name, len(audio_data)/sample_rate, sample_rate)
        start = time.time()
        
        # Whisper expects float32 in range [-1, 1]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0  # Assuming 16-bit audio
        
        # Process with Whisper model
        result = self.model.transcribe(audio_data, sr=sample_rate)
        
        duration = time.time() - start
        
        # Log some details about the transcription
        text = result["text"].strip()
        logger.debug("Audio array transcription completed in %.2fs: %s", 
                    duration, text[:50] + "..." if len(text) > 50 else text)
        
        return text
