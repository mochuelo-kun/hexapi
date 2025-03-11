from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any
import sounddevice as sd
import soundfile as sf
import numpy as np

class SpeechToTextBase(ABC):
    """Base class for speech-to-text implementations"""
    
    @abstractmethod
    def __init__(self, enable_diarization: bool = False, **kwargs):
        """Initialize the STT implementation
        
        Args:
            enable_diarization: Whether to enable speaker diarization
            **kwargs: Additional implementation-specific parameters
        """
        pass
    
    @abstractmethod
    def transcribe_file(self, audio_file: str) -> Union[str, Dict[str, Any]]:
        """Transcribe audio from a file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            If diarization is disabled: transcribed text
            If diarization is enabled: dict with keys:
                - text: full transcribed text
                - segments: list of diarized segments with speaker info
        """
        pass
    
    @abstractmethod  
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Union[str, Dict[str, Any]]:
        """Transcribe audio from numpy array
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            If diarization is disabled: transcribed text
            If diarization is enabled: dict with keys:
                - text: full transcribed text
                - segments: list of diarized segments with speaker info
        """
        pass
    
    def record_audio(self) -> tuple[np.ndarray, int]:
        """Record audio from microphone."""
        sample_rate = 16000
        print("Press SPACE to start recording, SPACE again to stop...")
        
        recording = []
        is_recording = False
        
        def callback(indata, frames, time, status):
            if is_recording:
                recording.append(indata.copy())
        
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            while True:
                if input() == ' ':
                    if not is_recording:
                        print("Recording started...")
                        is_recording = True
                    else:
                        print("Recording stopped...")
                        break
        
        return np.concatenate(recording), sample_rate 