import logging
import time
from typing import Optional
from dataclasses import dataclass
from .speech_to_text.api import SpeechToTextAPI
from .llm_query.api import LLMQueryAPI
from .text_to_speech.api import TextToSpeechAPI
from .logging import setup_logging

logger = logging.getLogger('chrysalis.pipeline')

@dataclass
class ChrysalisConfig:
    """Configuration for the Chrysalis pipeline"""
    stt_implementation: str = "whisper_api"
    stt_model: Optional[str] = None  # For whisper_local
    llm_model: str = "ollama:llama2"
    tts_implementation: str = "elevenlabs"
    voice_id: Optional[str] = None
    tts_model_dir: Optional[str] = None   # For piper
    tts_model_name: Optional[str] = None  # For piper

class ChrysalisPipeline:
    def __init__(self, config: Optional[ChrysalisConfig] = None):
        self.config = config or ChrysalisConfig()
        
        logger.info("Initializing Chrysalis pipeline with config: %s", self.config)
        
        # Initialize components
        start = time.time()
        self.stt = SpeechToTextAPI(
            implementation=self.config.stt_implementation,
            model_name=self.config.stt_model
        )
        self.llm = LLMQueryAPI()
        self.tts = TextToSpeechAPI(
            implementation=self.config.tts_implementation,
            model_path=self.config.tts_model_path,
            config_path=self.config.tts_config_path
        )
        logger.debug("Pipeline components initialized in %.2fs", time.time() - start)
    
    def run(self,
            audio_file: Optional[str] = None,
            use_mic: bool = False,
            output_file: str = "output.mp3") -> dict:
        """
        Run the full pipeline: STT -> LLM -> TTS
        
        Args:
            audio_file: Path to input audio file
            use_mic: Whether to record from microphone
            output_file: Path to save output audio file
            
        Returns:
            Dictionary containing intermediate results
        """
        pipeline_start = time.time()
        logger.info("Starting pipeline run")
        
        # Speech to text
        stt_start = time.time()
        transcription = self.stt.transcribe(
            audio_file=audio_file,
            use_mic=use_mic
        )
        stt_time = time.time() - stt_start
        logger.info("Speech-to-text completed in %.2fs: %s", stt_time, transcription)
        
        # Query LLM
        llm_start = time.time()
        llm_response = self.llm.query(
            prompt=transcription,
            model=self.config.llm_model
        )
        llm_time = time.time() - llm_start
        logger.info("LLM query completed in %.2fs: %s", llm_time, llm_response)
        
        # Text to speech
        tts_start = time.time()
        self.tts.synthesize(
            text=llm_response,
            output_file=output_file,
            voice_id=self.config.voice_id
        )
        tts_time = time.time() - tts_start
        logger.info("Text-to-speech completed in %.2fs", tts_time)
        
        total_time = time.time() - pipeline_start
        logger.info("Pipeline completed in %.2fs (STT: %.2fs, LLM: %.2fs, TTS: %.2fs)",
                   total_time, stt_time, llm_time, tts_time)
        
        return {
            "transcription": transcription,
            "llm_response": llm_response,
            "output_file": output_file,
            "timings": {
                "stt": stt_time,
                "llm": llm_time,
                "tts": tts_time,
                "total": total_time
            }
        }
    
    def transcribe_only(self,
                       audio_file: Optional[str] = None,
                       use_mic: bool = False) -> str:
        """Run only the speech-to-text component"""
        return self.stt.transcribe(audio_file=audio_file, use_mic=use_mic)
    
    def query_only(self, text: str) -> str:
        """Run only the LLM query component"""
        return self.llm.query(prompt=text, model=self.config.llm_model)
    
    def synthesize_only(self,
                       text: str,
                       output_file: str = "output.mp3") -> None:
        """Run only the text-to-speech component"""
        self.tts.synthesize(
            text=text,
            output_file=output_file,
            voice_id=self.config.voice_id
        ) 