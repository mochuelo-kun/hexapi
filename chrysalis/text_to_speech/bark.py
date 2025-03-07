import logging
import torch
import scipy.io.wavfile as wavfile
from .base import TextToSpeechBase

logger = logging.getLogger('chrysalis.tts.bark')

class BarkTTS(TextToSpeechBase):
    def __init__(self, tts_model: str = "suno/bark"):
        """
        Initialize Bark TTS
        
        Args:
            tts_model: Name of the Bark model to use
        """
        self.tts_model = tts_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("Loading Bark model: %s on %s", tts_model, self.device)
        try:
            from transformers import BarkModel, BarkProcessor
            
            self.model = BarkModel.from_pretrained(self.tts_model)
            self.model.to(self.device)
            self.processor = BarkProcessor.from_pretrained(self.tts_model)
            logger.info("Bark model loaded successfully")
            
        except ImportError as e:
            logger.error("Failed to import Bark dependencies: %s", e)
            raise ImportError(
                "Missing required dependency: transformers. "
                "Please check the README for installation instructions."
            )
        
    def synthesize(self, text: str, output_file: str, **kwargs) -> None:
        """
        Synthesize text to speech and save to file
        
        Args:
            text: Input text
            output_file: Path to save audio file
            **kwargs: Additional parameters including:
                speaker_id: Speaker identity to use
                voice_preset: Voice style/preset to use
        """
        logger.debug("Synthesizing text with Bark: %s", 
                    text[:50] + "..." if len(text) > 50 else text)
        
        # Get optional parameters
        speaker_id = kwargs.get("speaker_id", "v2/en_speaker_6")
        voice_preset = kwargs.get("voice_preset", None)
        
        # Prepare inputs
        inputs = self.processor(
            text=text,
            voice_preset=voice_preset or speaker_id,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate audio
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            
            if start_time:
                start_time.record()
                
            audio_array = self.model.generate(**inputs, do_sample=True)
            audio_array = audio_array.cpu().numpy().squeeze()
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                duration = start_time.elapsed_time(end_time) / 1000
            else:
                duration = 0  # Not timing on CPU
        
        # Get the sample rate from the model config or use default
        sample_rate = getattr(self.model.generation_config, "sample_rate", 24000)
        
        # Save to file
        wavfile.write(output_file, rate=sample_rate, data=audio_array)
        
        # Calculate metrics
        audio_duration = len(audio_array) / sample_rate
        logger.info(
            "Bark synthesis completed in %.2fs (audio length: %.2fs, %.1fx realtime)", 
            duration, audio_duration, audio_duration/duration if duration > 0 else 0
        ) 