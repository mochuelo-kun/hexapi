import logging
import time
from typing import Optional, Union, Dict, Any
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
    tts_model: Optional[str] = None

class Pipeline:
    def __init__(self,
                 stt_engine: str = "sherpa_local",
                 tts_engine: str = "sherpa_local",
                 llm_engine: str = "openai",
                 enable_diarization: bool = False,
                 **kwargs):
        """Initialize the pipeline
        
        Args:
            stt_engine: Speech-to-text engine to use
            tts_engine: Text-to-speech engine to use
            llm_engine: Language model engine to use
            enable_diarization: Whether to enable speaker diarization
            **kwargs: Additional engine-specific parameters including:
                stt_params: Dict with optional keys:
                    - recognition_model: Name of speaker recognition model
                    - segmentation_model: Name of speaker segmentation model
                tts_params: Dict of TTS parameters
                llm_params: Dict of LLM parameters
        """
        self.enable_diarization = enable_diarization
        
        # Extract STT parameters
        stt_params = kwargs.get("stt_params", {})
        if enable_diarization:
            # Add diarization-specific parameters if provided
            stt_params["enable_diarization"] = True
            if "recognition_model" in stt_params:
                stt_params["recognition_model"] = stt_params["recognition_model"]
            if "segmentation_model" in stt_params:
                stt_params["segmentation_model"] = stt_params["segmentation_model"]
        
        # Initialize STT
        if stt_engine == "sherpa_local":
            from .speech_to_text.sherpa_local import SherpaLocalSTT
            self.stt = SherpaLocalSTT(**stt_params)
        else:
            raise ValueError(f"Unknown STT engine: {stt_engine}")
            
        # Initialize TTS
        if tts_engine == "sherpa_local":
            from .text_to_speech.sherpa_local import SherpaLocalTTS
            self.tts = SherpaLocalTTS(**kwargs.get("tts_params", {}))
        else:
            raise ValueError(f"Unknown TTS engine: {tts_engine}")
            
        # Initialize LLM
        if llm_engine == "openai":
            from .llm_query.openai import OpenAIQuery
            self.llm = OpenAIQuery(**kwargs.get("llm_params", {}))
        else:
            raise ValueError(f"Unknown LLM engine: {llm_engine}")

    def transcribe_to_text(self, audio_input: str) -> Union[str, Dict[str, Any]]:
        """Transcribe audio file to text
        
        Args:
            audio_input: Path to audio file
            
        Returns:
            If diarization disabled: transcribed text
            If diarization enabled: dict with text and speaker segments
        """
        logger.info("Transcribing audio to text: %s", audio_input)
        return self.stt.transcribe_file(audio_input)

    def process_audio_query(self, 
                          audio_input: str,
                          system_prompt: Optional[str] = None,
                          output_audio: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """Process an audio query through the full pipeline
        
        Args:
            audio_input: Path to input audio file
            system_prompt: Optional system prompt for LLM
            output_audio: Optional path to save response audio
            **kwargs: Additional parameters for TTS
            
        Returns:
            Dict containing:
                - transcription: Input transcription (str or dict if diarized)
                - response: LLM response text
                - audio_path: Path to response audio if generated
        """
        # Transcribe audio
        transcription = self.transcribe_to_text(audio_input)
        
        # Extract text for LLM query
        query_text = transcription["text"] if self.enable_diarization else transcription
        
        # Query LLM
        response = self.llm.query(query_text, system_prompt)
        
        # Generate audio response if requested
        audio_path = None
        if output_audio:
            self.tts.synthesize(response, output_audio, **kwargs)
            audio_path = output_audio
            
        return {
            "transcription": transcription,
            "response": response,
            "audio_path": audio_path
        }

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
            tts_model=self.config.tts_model,
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
            tts_model=self.config.tts_model
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
            tts_model=self.config.tts_model
        ) 