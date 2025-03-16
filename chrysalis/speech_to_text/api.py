import time
import logging
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
from .base import SpeechToTextBase
from .whisper_api import WhisperAPI
from .whisper_local import WhisperLocal
from .sherpa_local import SherpaLocalSTT
from .. import audio_utils

logger = logging.getLogger('chrysalis.speech_to_text.api')

class SpeechToTextAPI:
    """API for speech-to-text operations with different backend implementations"""
    
    def __init__(self, 
                implementation: str = "sherpa_local",
                model_name: Optional[str] = None,
                enable_diarization: bool = False,
                recognition_model: Optional[str] = None,
                segmentation_model: Optional[str] = None,
                use_int8: bool = False,
                **kwargs):
        """Initialize the speech-to-text API
        
        Args:
            implementation: Backend implementation to use ('sherpa_local', 'whisper_api', 'whisper_local', etc.)
            model_name: Model name for the implementation
            enable_diarization: Whether to enable speaker diarization
            recognition_model: Speaker recognition model for diarization
            segmentation_model: Speaker segmentation model for diarization
            use_int8: Whether to use int8 quantized models
            **kwargs: Additional implementation-specific parameters
        """
        init_start = time.time()
        self.implementation = implementation
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        self.recognition_model = recognition_model
        self.segmentation_model = segmentation_model
        self.use_int8 = use_int8
        self.kwargs = kwargs
        
        # Initialize the appropriate backend
        if implementation == "sherpa_local":
            from .sherpa_local import SherpaLocalSTT
            self._backend = SherpaLocalSTT(
                model_name=model_name,
                enable_diarization=enable_diarization,
                recognition_model=recognition_model,
                segmentation_model=segmentation_model,
                use_int8=use_int8,
                **kwargs
            )
        elif implementation == "whisper_api":
            from .whisper_api import WhisperAPI
            self._backend = WhisperAPI(model_name=model_name, **kwargs)
        elif implementation == "whisper_local":
            from .whisper_local import WhisperLocal
            self._backend = WhisperLocal(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unknown STT implementation: {implementation}")
        
        init_time = time.time() - init_start    
        logger.info(f"Initialized {implementation} speech-to-text engine in {init_time:.3f}s")
    
    def transcribe(self,
                  audio_file: Optional[str] = None,
                  use_microphone: bool = False,
                  microphone_duration: float = audio_utils.DEFAULT_MIC_DURATION,
                  audio_array: Optional[np.ndarray] = None,
                  sample_rate: Optional[int] = None,
                  target_sample_rate: Optional[int] = None) -> Union[str, Dict[str, Any]]:
        """Transcribe audio to text
        
        This method accepts audio in various forms:
        1. Audio file path
        2. Microphone recording
        3. Audio array with sample rate
        
        Args:
            audio_file: Path to audio file
            use_microphone: Whether to record from microphone
            microphone_duration: Duration to record if using microphone (seconds)
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio array (required if audio_array is provided)
            target_sample_rate: Target sample rate for processing (if None, uses original)
            
        Returns:
            Transcription result (string or dict with diarization info)
            
        Raises:
            ValueError: If no valid audio input is provided
        """
        start_time = time.time()
        logger.debug("Starting audio transcription")
        
        try:
            # Get audio data from any of the available sources
            audio_loading_start = time.time()
            audio_data, audio_sample_rate = audio_utils.get_audio_array(
                audio_file=audio_file,
                use_microphone=use_microphone,
                microphone_duration=microphone_duration,
                target_sample_rate=target_sample_rate,
                audio_array=audio_array,
                array_sample_rate=sample_rate
            )
            audio_loading_time = time.time() - audio_loading_start
            logger.debug(f"Audio obtained in {audio_loading_time:.3f}s: {len(audio_data)/audio_sample_rate:.2f}s at {audio_sample_rate}Hz")
            
            # Transcribe the audio data
            transcription_start = time.time()
            result = self._backend.transcribe_array(audio_data, audio_sample_rate)
            transcription_time = time.time() - transcription_start
            
            # Log the result
            total_time = time.time() - start_time
            
            if isinstance(result, dict) and "text" in result:
                text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                logger.info(f"Transcription completed in {total_time:.3f}s (loading: {audio_loading_time:.3f}s, processing: {transcription_time:.3f}s)")
                logger.debug(f"Transcribed text: {text_preview}")
            else:
                text_preview = result[:100] + "..." if isinstance(result, str) and len(result) > 100 else result
                logger.info(f"Transcription completed in {total_time:.3f}s (loading: {audio_loading_time:.3f}s, processing: {transcription_time:.3f}s)")
                logger.debug(f"Transcribed text: {text_preview}")
                
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed after {time.time() - start_time:.3f}s: {str(e)}")
            raise 