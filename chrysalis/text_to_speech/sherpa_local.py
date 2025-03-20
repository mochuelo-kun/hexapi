import logging
import time
from pathlib import Path
from typing import Optional
from .base import TextToSpeechBase
from ..config import Config

logger = logging.getLogger('chrysalis.tts.sherpa_local')

# Default model selections
DEFAULT_TTS_MODEL = "en_US-amy-medium"
DEFAULT_SAMPLE_RATE = 22050

# Default synthesis parameters
# DEFAULT_SPEAKER_ID = 0
DEFAULT_SPEED = 1.0

class SherpaLocalTTS(TextToSpeechBase):
    def __init__(self, 
                 tts_model: Optional[str] = None,
                 sample_rate: Optional[int] = None,
                 **kwargs):
        """
        Initialize Sherpa-ONNX TTS
        
        Args:
            tts_model: Name of the model to use
            sample_rate: Output audio sample rate
        """
        self.model_name = tts_model or DEFAULT_TTS_MODEL
        self.sample_rate = sample_rate or DEFAULT_SAMPLE_RATE
        
        try:
            import sherpa_onnx
            
            model_dir = Path(Config.ONNX_MODEL_DIR) / "tts"
            model_path = model_dir / f"{self.model_name}.onnx"

            # Look for tokens file - might be named {model}.tokens.txt or just tokens.txt
            tokens_path = model_dir / f"{self.model_name}.tokens.txt"

            if not tokens_path.exists():
                # Try original config file
                tokens_path = model_dir / f"{self.model_name}.json"
                
                # If that doesn't exist, try .onnx.json (Piper format)
                if not tokens_path.exists():
                    tokens_path = model_dir / f"{self.model_name}.onnx.json"
            
            if not model_path.exists():
                available_models = [p.stem for p in model_dir.glob("*.onnx")]
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"Available models: {', '.join(available_models)}"
                )
                
            if not tokens_path.exists():
                raise FileNotFoundError(
                    f"Tokens file not found for model {self.model_name}. "
                    f"Expected one of: {self.model_name}.tokens.txt, {self.model_name}.json, or {self.model_name}.onnx.json"
                )
            
            logger.info("Loading Sherpa-ONNX TTS model: %s", self.model_name)
            logger.info("  Model: %s", model_path)
            logger.info("  Tokens: %s", tokens_path)
            
            # # Simplified config creation
            # # Use direct parameter passing instead of building config objects manually
            # # This matches the pattern seen in sherpa-onnx examples
            tts_config = sherpa_onnx.OfflineTtsConfig()
            tts_config.model.vits.model = str(model_path)
            tts_config.model.vits.tokens = str(tokens_path)
            tts_config.model.vits.data_dir = Config.ESPEAK_DATA_PATH
            # tts_config.model.num_threads = 1
            tts_config.model.debug = True

            load_start = time.time()
            self.synthesizer = sherpa_onnx.OfflineTts(tts_config)
            logger.info("TTS model loaded in %.2fs", time.time() - load_start)
            
            
        except ImportError as e:
            logger.error("Failed to import Sherpa-ONNX dependencies: %s", e)
            raise ImportError(
                "Missing required dependency: sherpa-onnx. "
                "Please check the README for installation instructions."
            )
    
    # def synthesize(self, text: str, output_file: str, **kwargs) -> None:
    def synthesize(self, text: str, output_file: str, **kwargs) -> None:
        """
        Synthesize text to speech
        
        Args:
            text: Input text
            **kwargs: Additional parameters including:
                speaker_id: Speaker identity to use (if model supports it)
                speed: Speech speed factor (default: 1.0)
        """
        logger.debug("Synthesizing text with %s: %s", 
                    self.model_name, 
                    text[:50] + "..." if len(text) > 50 else text)
        start_time = time.time()
        
        # Get optional parameters
        # speaker_id = kwargs.get("speaker_id", DEFAULT_SPEAKER_ID)
        speed = kwargs.get("speed", DEFAULT_SPEED)
        
        # Generate audio
        audio_data = self.synthesizer.generate(
            text=text,
            # speaker=speaker_id,
            speed=speed
        )
        
        # Calculate metrics
        duration = time.time() - start_time
        audio_duration = len(audio_data.samples) / audio_data.sample_rate
        logger.info(
            "Synthesis completed in %.2fs (audio length: %.2fs, %.1fx realtime)", 
            duration, audio_duration, audio_duration/duration
        )
        return audio_data.samples, audio_data.sample_rate