import time
import logging
from typing import Union, Dict, Any, Optional
import numpy as np
import pprint
import torch
import whisperx
from .base import SpeechToTextBase
from ..config import Config


logger = logging.getLogger('chrysalis.stt.whisper_local')

DEFAULT_STT_MODEL = "base"
DEFAULT_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_COMPUTE_TYPE = "float32"
# DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_DIARIZATION_MODEL="pyannote/speaker-diarization-3.1"

class WhisperLocal(SpeechToTextBase):
    def __init__(self, 
                model_name: Optional[str] = None, 
                enable_diarization: bool = False,
                device: Optional[str] = None,
                compute_type: Optional[str] = None,
                **kwargs):
        """
        Initialize local WhisperX model
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
            enable_diarization: Whether to enable speaker diarization
            device: Device to run the model on ("cuda" or "cpu")
            compute_type: Compute type for faster-whisper ("float16", "float32", "int8")
            **kwargs: Additional parameters (for compatibility with other STT backends)
        """
        self.model_name = model_name or DEFAULT_STT_MODEL
        self.enable_diarization = enable_diarization
        self.device = device or DEFAULT_DEVICE_TYPE
        self.compute_type = compute_type or DEFAULT_COMPUTE_TYPE
        
        logger.info("Loading WhisperX model: %s (device: %s, compute_type: %s)", 
                   self.model_name, self.device, self.compute_type)
        try:
            # import whisperx
            # import gc
            # import torch
            
            load_start = time.time()
            # Load the ASR model
            self.model = whisperx.load_model(
                self.model_name, 
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("WhisperX model loaded in %.2fs", time.time() - load_start)
            
            # Initialize diarization pipeline if enabled
            if enable_diarization:
                try:
                    self.diarization_model = whisperx.DiarizationPipeline(
                        use_auth_token=Config.HUGGING_FACE_TOKEN,  # You'll need to set this if using pyannote
                        model_name=DEFAULT_DIARIZATION_MODEL,
                        device=self.device
                    )
                    logger.info("Diarization model loaded successfully")
                except Exception as e:
                    logger.error("Failed to load diarization model: %s", e)
                    raise ImportError(
                        "Failed to initialize diarization. "
                        "Please check if you have the required dependencies and permissions."
                    )
            
        except ImportError as e:
            logger.error("Failed to import WhisperX dependencies: %s", e)
            raise ImportError(
                "Missing required dependency: whisperx. "
                "Please check the README for installation instructions."
            )
    
    def transcribe_file(self, audio_file: str) -> Union[str, Dict[str, Any]]:
        """Transcribe audio from a file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or dict with transcription and speaker info if diarization enabled
        """
        logger.debug("Transcribing file with %s model: %s", self.model_name, audio_file)
        start = time.time()
        
        # Transcribe with WhisperX
        # result = self.model.transcribe(audio_file)
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio)

        duration = time.time() - start
        logger.debug("File basic transcription completed in %.2fs", duration)
        pprint.pp(result, depth=2)
        
        # Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_file,
            self.device,
            return_char_alignments=False
        )
        duration = time.time() - start
        logger.debug("File alignment completed in %.2fs", duration)
        pprint.pp(result, depth=2)
        
        # Perform diarization if enabled
        if self.enable_diarization:
            diarization_segments = self.diarization_model(audio)
            result = whisperx.assign_word_speakers(diarization_segments, result)
            duration = time.time() - start
            logger.debug("File diarization completed in %.2fs", duration)
            pprint.pp(result, depth=2)
        
        duration = time.time() - start
        logger.debug("File full transcription completed in %.2fs", duration)
        
        # Return appropriate format based on diarization
        if self.enable_diarization:
            return result
        else:
            return " ".join(segment["text"] for segment in result["segments"]).strip()
    
    def transcribe_array(self, audio_data: np.ndarray, sample_rate: int) -> Union[str, Dict[str, Any]]:
        """Transcribe audio data directly from numpy array
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio data in Hz
            
        Returns:
            Transcribed text or dict with transcription and speaker info if diarization enabled
        """
        logger.debug("Transcribing audio array with %s model (%.2fs at %dHz)", 
                    self.model_name, len(audio_data)/sample_rate, sample_rate)
        start = time.time()
        
        # WhisperX expects float32 in range [-1, 1]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0  # Assuming 16-bit audio
        
        # Process with WhisperX model
        # result = self.model.transcribe(audio_data, sample_rate=sample_rate)
        result = self.model.transcribe(audio_data)
        
        duration = time.time() - start
        logger.debug("Basic transcription completed in %.2fs", duration)
        pprint.pp(result, depth=2)
        
        # Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_data,
            self.device,
            return_char_alignments=False
        )

        duration = time.time() - start
        logger.debug("Alignment completed in %.2fs", duration)
        pprint.pp(result, depth=2)
        
        # Perform diarization if enabled
        if self.enable_diarization:
            diarization_segments = self.diarization_model(audio_data)
            result = whisperx.assign_word_speakers(diarization_segments, result)
        
        duration = time.time() - start
        
        # Log some details about the transcription
        if self.enable_diarization:
            text = " ".join(segment["text"] for segment in result["segments"]).strip()
        else:
            text = " ".join(segment["text"] for segment in result["segments"]).strip()
            
        logger.debug("Full audio array transcription completed in %.2fs: %s", 
                    duration, text[:50] + "..." if len(text) > 50 else text)
        pprint.pp(result, depth=2)
        
        # Return appropriate format based on diarization
        if self.enable_diarization:
            return result
        else:
            return text
