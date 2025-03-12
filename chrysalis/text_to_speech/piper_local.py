import time
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
from .base import TextToSpeechBase
from ..config import Config

logger = logging.getLogger('chrysalis.tts.piper_local')

# Piper uses ONNX models from https://huggingface.co/rhasspy/piper-voices
# Example English models:
#   en_US-amy-low
#   en_GB-semaine-medium
# Full list at: https://huggingface.co/rhasspy/piper-voices
# Voice samples available at: https://rhasspy.github.io/piper-samples/

DEFAULT_SAMPLE_RATE = 22050

class PiperLocalTTS(TextToSpeechBase):
    def __init__(self, 
                 tts_model: str,
                 sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Initialize Piper TTS
        
        Args:
            tts_model: Name of voice model (e.g. 'en_US-amy-medium')
            sample_rate: Output audio sample rate
        """
        model_dir = Path(Config.ONNX_MODEL_DIR) / "tts"
        model_path = model_dir / f"{tts_model}.onnx"
        config_path = model_dir / f"{tts_model}.onnx.json"
        
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {model_dir}. "
                f"Expected {model_path.name} and {config_path.name}"
            )
        
        logger.info("Loading Piper model: %s from %s", tts_model, model_dir)
        try:
            from piper import PiperVoice
            
            load_start = time.time()
            self.voice = PiperVoice.load(str(model_path), str(config_path))
            logger.info("Piper model loaded in %.2fs", time.time() - load_start)
            
        except ImportError as e:
            logger.error("Failed to import Piper dependencies: %s", e)
            raise ImportError(
                "Missing required dependency: piper-tts. "
                "Please check the README for installation instructions."
            )
        
        self.model_name = tts_model
        self.sample_rate = sample_rate
    
    def synthesize(self, text: str, output_file: str, **kwargs) -> None:
        """
        Synthesize text to speech and save to file
        
        Args:
            text: Input text
            output_file: Path to save audio file
            **kwargs: Additional synthesis parameters
        """
        logger.debug("Synthesizing text with %s: %s", 
                    self.model_name, 
                    text[:50] + "..." if len(text) > 50 else text)
        start_time = time.time()
        
        # Generate audio
        audio_data = self.voice.synthesize(text)
        
        # Save to file
        sf.write(output_file, audio_data, self.sample_rate)
        
        duration = time.time() - start_time
        audio_duration = len(audio_data) / self.sample_rate
        logger.info(
            "Synthesis completed in %.2fs (audio length: %.2fs, %.1fx realtime)", 
            duration, audio_duration, audio_duration/duration
        )
