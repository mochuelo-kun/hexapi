from abc import ABC, abstractmethod
from typing import Optional, Union
import sounddevice as sd
import soundfile as sf
import numpy as np

class SpeechToTextBase(ABC):
    @abstractmethod
    def transcribe_file(self, audio_file: str) -> str:
        """Transcribe audio from a file."""
        pass
    
    @abstractmethod
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio from numpy array."""
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