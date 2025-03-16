import time
import logging
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import numpy as np

logger = logging.getLogger('chrysalis.speech_to_text.base')

class SpeechToTextBase(ABC):
    """Base class for all speech-to-text implementations
    
    This abstract class defines the interface that all
    speech-to-text backends should implement.
    """
    
    @abstractmethod
    def __init__(self, enable_diarization: bool = False, **kwargs):
        """Initialize the STT implementation
        
        Args:
            enable_diarization: Whether to enable speaker diarization
            **kwargs: Additional implementation-specific parameters
        """
        pass
    
    @abstractmethod
    def transcribe_array(self, audio_data: np.ndarray, sample_rate: int) -> Union[str, Dict[str, Any]]:
        """Transcribe audio data directly from numpy array
        
        This is the core method that all backend implementations must provide.
        All other methods (transcribe_file, etc.) will eventually call this method.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data in Hz
            
        Returns:
            Transcription result (string or dict with diarization info)
            
            If diarization is disabled:
                A string containing the transcribed text
                
            If diarization is enabled:
                A dictionary with:
                - "text": Full transcribed text
                - "segments": List of segments with speaker info
                  Each segment has:
                  - "speaker": Speaker identifier
                  - "start": Start time in seconds
                  - "end": End time in seconds
                  - "text": Text for this segment
        """
        pass
    
    def transcribe_file(self, file_path: str) -> Union[str, Dict[str, Any]]:
        """Transcribe audio from a file
        
        Default implementation loads the file and calls transcribe_array.
        Override this method if your backend has a more efficient way
        to process files directly.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcription result (string or dict with diarization info)
        """
        from .. import audio_utils
        
        start_time = time.time()
        logger.debug(f"Loading audio file for transcription: {file_path}")
        
        # Load audio file
        audio_data, sample_rate = audio_utils.load_audio_file(file_path)
        load_time = time.time() - start_time
        logger.debug(f"File loaded in {load_time:.3f}s: {len(audio_data)/sample_rate:.2f}s at {sample_rate}Hz")
        
        # Transcribe the audio data
        return self.transcribe_array(audio_data, sample_rate)
    