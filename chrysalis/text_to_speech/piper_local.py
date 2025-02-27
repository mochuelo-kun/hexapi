import time
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
from piper import PiperVoice
from .base import TextToSpeechBase

logger = logging.getLogger('chrysalis.tts.piper_local')

# Piper uses ONNX models from https://huggingface.co/rhasspy/piper-voices
# Example English models:
#   en_US-amy-low
#   en_GB-semaine-medium
# Full list at: https://huggingface.co/rhasspy/piper-voices
# Voice samples available at: https://rhasspy.github.io/piper-samples/

class PiperLocalTTS(TextToSpeechBase):
    def __init__(self, 
                 onnx_model_dir: str,
                 onnx_model_name: str,
                 sample_rate: int = 22050):
        """
        Initialize Piper TTS
        
        Args:
            onnx_model_dir: Directory containing the .onnx and .json files
            onnx_model_name: Name of model (e.g. 'en_US-amy-medium')
            sample_rate: Output audio sample rate
        """
        model_dir = Path(onnx_model_dir)
        model_path = model_dir / f"{onnx_model_name}.onnx"
        config_path = model_dir / f"{onnx_model_name}.json"
        
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {model_dir}. "
                f"Expected {model_path.name} and {config_path.name}"
            )
        
        logger.info("Loading Piper model: %s from %s", onnx_model_name, model_dir)
        load_start = time.time()
        self.voice = PiperVoice.load(str(model_path), str(config_path))
        logger.info("Piper model loaded in %.2fs", time.time() - load_start)
        
        self.model_name = onnx_model_name
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
        start = time.time()
        
        # Generate audio
        audio_data = self.voice.synthesize(text)
        
        # Save to file
        sf.write(output_file, audio_data, self.sample_rate)
        
        duration = time.time() - start
        audio_duration = len(audio_data) / self.sample_rate
        logger.info(
            "Synthesis completed in %.2fs (audio length: %.2fs, %.1fx realtime)", 
            duration, audio_duration, audio_duration/duration
        ) 