import logging
import time
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
import psutil
import os
from pathlib import Path
from .speech_to_text.api import SpeechToTextAPI
from .llm_query.api import LLMQueryAPI
from .text_to_speech.api import TextToSpeechAPI
from .logging import setup_logging

logger = logging.getLogger('chrysalis.pipeline')

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_change(start_mem: float, operation: str):
    """Log memory change after an operation"""
    current_mem = get_memory_usage()
    delta = current_mem - start_mem
    logger.debug("Memory after %s: %.1fMB (%.1fMB change)", 
                operation, current_mem, delta)
    return current_mem

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
        init_start = time.time()
        start_mem = get_memory_usage()
        logger.info("Initializing Pipeline configuration")
        
        # Store configuration for lazy initialization
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.llm_engine = llm_engine
        self.enable_diarization = enable_diarization
        self.kwargs = kwargs
        
        # Initialize components as None
        self._stt = None
        self._tts = None
        self._llm = None
        
        logger.debug("Pipeline configuration initialized in %.3fs", time.time() - init_start)
        log_memory_change(start_mem, "pipeline initialization")
    
    @property
    def stt(self):
        """Lazy initialization of STT component"""
        if self._stt is None:
            init_start = time.time()
            start_mem = get_memory_usage()
            logger.info("Initializing STT engine: %s", self.stt_engine)
            
            # Extract STT parameters
            stt_params = self.kwargs.get("stt_params", {})
            if self.enable_diarization:
                stt_params["enable_diarization"] = True
                logger.debug("Enabling diarization with params: %s", stt_params)
            
            # Initialize STT
            try:
                if self.stt_engine == "sherpa_local":
                    from .speech_to_text.sherpa_local import SherpaLocalSTT
                    self._stt = SherpaLocalSTT(**stt_params)
                else:
                    raise ValueError(f"Unknown STT engine: {self.stt_engine}")
                
                init_time = time.time() - init_start
                logger.info("STT engine initialized successfully in %.3fs", init_time)
                log_memory_change(start_mem, "STT initialization")
            except Exception as e:
                logger.error("Failed to initialize STT engine: %s", e, exc_info=True)
                raise
        
        return self._stt
    
    @property
    def tts(self):
        """Lazy initialization of TTS component"""
        if self._tts is None:
            init_start = time.time()
            start_mem = get_memory_usage()
            logger.info("Initializing TTS engine: %s", self.tts_engine)
            
            try:
                if self.tts_engine == "sherpa_local":
                    from .text_to_speech.sherpa_local import SherpaLocalTTS
                    self._tts = SherpaLocalTTS(**self.kwargs.get("tts_params", {}))
                else:
                    raise ValueError(f"Unknown TTS engine: {self.tts_engine}")
                
                init_time = time.time() - init_start
                logger.info("TTS engine initialized successfully in %.3fs", init_time)
                log_memory_change(start_mem, "TTS initialization")
            except Exception as e:
                logger.error("Failed to initialize TTS engine: %s", e, exc_info=True)
                raise
        
        return self._tts
    
    @property
    def llm(self):
        """Lazy initialization of LLM component"""
        if self._llm is None:
            init_start = time.time()
            start_mem = get_memory_usage()
            logger.info("Initializing LLM engine: %s", self.llm_engine)
            
            try:
                if self.llm_engine == "openai":
                    from .llm_query.openai import OpenAIQuery
                    self._llm = OpenAIQuery(**self.kwargs.get("llm_params", {}))
                else:
                    raise ValueError(f"Unknown LLM engine: {self.llm_engine}")
                
                init_time = time.time() - init_start
                logger.info("LLM engine initialized successfully in %.3fs", init_time)
                log_memory_change(start_mem, "LLM initialization")
            except Exception as e:
                logger.error("Failed to initialize LLM engine: %s", e, exc_info=True)
                raise
        
        return self._llm

    def transcribe_to_text(self, audio_input: str) -> Union[str, Dict[str, Any]]:
        """Transcribe audio file to text
        
        Args:
            audio_input: Path to audio file
            
        Returns:
            If diarization disabled: transcribed text
            If diarization enabled: dict with text and speaker segments
        """
        start_time = time.time()
        start_mem = get_memory_usage()
        logger.info("Starting transcription of: %s", audio_input)
        
        try:
            result = self.stt.transcribe_file(audio_input)
            duration = time.time() - start_time
            
            if isinstance(result, dict):
                text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                logger.info("Transcription completed in %.3fs with diarization", duration)
                logger.debug("Transcribed text: %s", text_preview)
            else:
                text_preview = result[:100] + "..." if len(result) > 100 else result
                logger.info("Transcription completed in %.3fs", duration)
                logger.debug("Transcribed text: %s", text_preview)
            
            log_memory_change(start_mem, "transcription")
            return result
            
        except Exception as e:
            logger.error("Transcription failed after %.3fs: %s", time.time() - start_time, e, exc_info=True)
            raise

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
        pipeline_start = time.time()
        start_mem = get_memory_usage()
        logger.info("Starting audio query pipeline")
        timings = {}
        memory_changes = {}
        
        try:
            # Transcribe audio
            stt_start = time.time()
            stt_start_mem = get_memory_usage()
            transcription = self.transcribe_to_text(audio_input)
            timings["stt"] = time.time() - stt_start
            memory_changes["stt"] = get_memory_usage() - stt_start_mem
            
            # Extract text for LLM query
            query_text = transcription["text"] if self.enable_diarization else transcription
            
            # Query LLM
            llm_start = time.time()
            llm_start_mem = get_memory_usage()
            logger.info("Querying LLM with transcribed text")
            response = self.llm.query(query_text, system_prompt)
            timings["llm"] = time.time() - llm_start
            memory_changes["llm"] = get_memory_usage() - llm_start_mem
            logger.info("LLM response received in %.3fs", timings["llm"])
            logger.debug("LLM response: %s", response[:100] + "..." if len(response) > 100 else response)
            
            # Generate audio response if requested
            audio_path = None
            if output_audio:
                tts_start = time.time()
                tts_start_mem = get_memory_usage()
                logger.info("Generating audio response to: %s", output_audio)
                self.tts.synthesize(response, output_audio, **kwargs)
                timings["tts"] = time.time() - tts_start
                memory_changes["tts"] = get_memory_usage() - tts_start_mem
                logger.info("Audio response generated in %.3fs", timings["tts"])
                audio_path = output_audio
            
            total_time = time.time() - pipeline_start
            total_mem_change = get_memory_usage() - start_mem
            
            logger.info("Pipeline completed in %.3fs (memory change: %.1fMB)", 
                       total_time, total_mem_change)
            
            for component, timing in timings.items():
                logger.debug("%s: %.3fs (%.1f%% of total time, %.1fMB memory change)", 
                           component.upper(), timing, 
                           (timing/total_time)*100,
                           memory_changes[component])
            
            return {
                "transcription": transcription,
                "response": response,
                "audio_path": audio_path,
                "timings": timings,
                "memory_changes": memory_changes
            }
            
        except Exception as e:
            logger.error("Pipeline failed after %.3fs: %s", 
                        time.time() - pipeline_start, e, exc_info=True)
            raise
    
    def cleanup(self):
        """Clean up resources and unload models"""
        start_mem = get_memory_usage()
        logger.info("Starting pipeline cleanup")
        
        # Clean up STT
        if self._stt is not None:
            logger.debug("Unloading STT models")
            self._stt = None
        
        # Clean up TTS
        if self._tts is not None:
            logger.debug("Unloading TTS models")
            self._tts = None
        
        # Clean up LLM
        if self._llm is not None:
            logger.debug("Unloading LLM models")
            self._llm = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        log_memory_change(start_mem, "cleanup")
        logger.info("Pipeline cleanup completed")

class ChrysalisPipeline:
    def __init__(self, config: Optional[ChrysalisConfig] = None):
        self.config = config or ChrysalisConfig()
        
        logger.info("Initializing Chrysalis pipeline with config: %s", self.config)
        
        # Store configuration for lazy initialization
        self._stt = None
        self._llm = None
        self._tts = None
    
    @property
    def stt(self):
        """Lazy initialization of STT component"""
        if self._stt is None:
            logger.info("Initializing STT component: %s", self.config.stt_implementation)
            self._stt = SpeechToTextAPI(
                implementation=self.config.stt_implementation,
                model_name=self.config.stt_model
            )
        return self._stt
    
    @property
    def llm(self):
        """Lazy initialization of LLM component"""
        if self._llm is None:
            logger.info("Initializing LLM component")
            self._llm = LLMQueryAPI()
        return self._llm
    
    @property
    def tts(self):
        """Lazy initialization of TTS component"""
        if self._tts is None:
            logger.info("Initializing TTS component: %s", self.config.tts_implementation)
            self._tts = TextToSpeechAPI(
                implementation=self.config.tts_implementation,
                tts_model=self.config.tts_model,
            )
        return self._tts
    
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