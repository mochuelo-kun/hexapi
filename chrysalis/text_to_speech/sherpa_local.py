import logging
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from .base import TextToSpeechBase
from ..config import Config

logger = logging.getLogger('chrysalis.tts.sherpa_local')

# Default model selections
DEFAULT_TTS_MODEL = "en_US-amy-medium"
DEFAULT_SAMPLE_RATE = 22050

# Default synthesis parameters
DEFAULT_SPEAKER_ID = 0
DEFAULT_SPEED = 1.0

class SherpaLocalTTS(TextToSpeechBase):
    def __init__(self, 
                 tts_model: str = DEFAULT_TTS_MODEL,
                 sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Initialize Sherpa-ONNX TTS
        
        Args:
            tts_model: Name of the model to use
            sample_rate: Output audio sample rate
        """
        self.model_name = tts_model
        self.sample_rate = sample_rate
        
        try:
            import sherpa_onnx
            
            model_dir = Path(Config.ONNX_MODEL_DIR) / "tts"
            model_path = model_dir / f"{tts_model}.onnx"
            config_path = model_dir / f"{tts_model}.json"
            
            if not model_path.exists() or not config_path.exists():
                available_models = [p.stem for p in model_dir.glob("*.onnx")]
                raise FileNotFoundError(
                    f"Model files not found in {model_dir}. "
                    f"Expected {model_path.name} and {config_path.name}\n"
                    f"Available models: {', '.join(available_models)}"
                )
            
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=str(model_path)
                ),
                tokens=str(config_path),
                data_dir=str(model_dir),
                sample_rate=sample_rate
            )
            
            logger.info("Loading Sherpa-ONNX TTS model: %s", tts_model)
            load_start = time.time()
            self.synthesizer = sherpa_onnx.OfflineTts(config)
            logger.info("TTS model loaded in %.2fs", time.time() - load_start)
            
        except ImportError as e:
            logger.error("Failed to import Sherpa-ONNX dependencies: %s", e)
            raise ImportError(
                "Missing required dependency: sherpa-onnx. "
                "Please check the README for installation instructions."
            )
    
    def synthesize(self, text: str, output_file: str, **kwargs) -> None:
        """
        Synthesize text to speech and save to file
        
        Args:
            text: Input text
            output_file: Path to save audio file
            **kwargs: Additional parameters including:
                speaker_id: Speaker identity to use (if model supports it)
                speed: Speech speed factor (default: 1.0)
        """
        logger.debug("Synthesizing text with %s: %s", 
                    self.model_name, 
                    text[:50] + "..." if len(text) > 50 else text)
        start_time = time.time()
        
        # Get optional parameters
        speaker_id = kwargs.get("speaker_id", DEFAULT_SPEAKER_ID)
        speed = kwargs.get("speed", DEFAULT_SPEED)
        
        # Generate audio
        audio_data = self.synthesizer.generate(
            text=text,
            speaker=speaker_id,
            speed=speed
        )
        
        # Save to file
        sf.write(output_file, audio_data, self.sample_rate)
        
        # Calculate metrics
        duration = time.time() - start_time
        audio_duration = len(audio_data) / self.sample_rate
        logger.info(
            "Synthesis completed in %.2fs (audio length: %.2fs, %.1fx realtime)", 
            duration, audio_duration, audio_duration/duration
        ) 