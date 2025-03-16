import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
from .. import audio_utils
from .base import TextToSpeechBase
from .elevenlabs import ElevenLabsTTS
from .piper_local import PiperLocalTTS
from .bark import BarkTTS
from .sherpa_local import SherpaLocalTTS

logger = logging.getLogger('chrysalis.text_to_speech.api')

class TextToSpeechAPI:
    """API for text-to-speech operations with different backend implementations"""
    
    def __init__(self, 
                implementation: str = "sherpa_local",
                tts_model: Optional[str] = None,
                speaker_id: int = 0,
                speed: float = 1.0,
                **kwargs):
        """Initialize the text-to-speech API
        
        Args:
            implementation: Backend implementation to use ('sherpa_local', 'elevenlabs', etc.)
            tts_model: Model name for the implementation
            speaker_id: Speaker ID for multi-speaker models
            speed: Speech speed factor
            **kwargs: Additional implementation-specific parameters
        """
        self.implementation = implementation
        self.tts_model = tts_model
        self.speaker_id = speaker_id
        self.speed = speed
        self.kwargs = kwargs
        
        # Initialize the appropriate backend
        if implementation == "sherpa_local":
            from .sherpa_local import SherpaLocalTTS
            self._backend = SherpaLocalTTS(
                tts_model=tts_model,
                speaker_id=speaker_id,
                speed=speed,
                **kwargs
            )
        elif implementation == "elevenlabs":
            from .elevenlabs import ElevenLabsTTS
            self._backend = ElevenLabsTTS(
                tts_model=tts_model,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown TTS implementation: {implementation}")
            
        logger.info(f"Initialized {implementation} text-to-speech engine")
    
    def synthesize_to_array(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio array
        
        Args:
            text: Text to synthesize
            **kwargs: Additional parameters for synthesis (override instance values)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        text_preview = text[:50] + "..." if len(text) > 50 else text
        logger.debug(f"Synthesizing text to audio array: '{text_preview}'")
        
        # Merge instance parameters with any overrides
        params = {
            "speaker_id": self.speaker_id,
            "speed": self.speed
        }
        params.update(kwargs)
        
        return self._backend.synthesize_to_array(text, **params)
    
    def synthesize(self, text: str, output_file: str, **kwargs) -> str:
        """Synthesize text to audio file
        
        Args:
            text: Text to synthesize
            output_file: Path to save output audio file
            **kwargs: Additional parameters for synthesis
            
        Returns:
            Path to the output audio file
        """
        # Get audio data array
        audio_data, sample_rate = self.synthesize_to_array(text, **kwargs)
        
        # Save to file
        return audio_utils.save_audio_file(audio_data, output_file, sample_rate=sample_rate) 