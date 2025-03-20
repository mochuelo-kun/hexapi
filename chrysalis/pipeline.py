import logging
import time
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
import psutil
import os
from .speech_to_text.api import SpeechToTextAPI
from .llm_query.api import LLMQueryAPI
from .text_to_speech.api import TextToSpeechAPI
from . import audio_utils

DEFAULT_STT_IMPLEMENTATION="whisper_local"
DEFAULT_LLM_IMPLEMENTATION="openai"
DEFAULT_TTS_IMPLEMENTATION="sherpa_local"

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
class PipelineConfig:
    """Configuration for the Chrysalis pipeline"""
    # Speech-to-text configuration
    stt_implementation: str = DEFAULT_STT_IMPLEMENTATION  # "whisper_api", "whisper_local", "sherpa_local", etc.
    stt_model: Optional[str] = None  # Name of STT model if applicable 
    enable_diarization: bool = False  # Whether to enable speaker diarization
    recognition_model: Optional[str] = None  # Speaker recognition model for diarization
    segmentation_model: Optional[str] = None  # Speaker segmentation model for diarization
    use_int8: bool = False  # Whether to use int8 quantized models
    num_speakers: Optional[int] = None  # (For diarizatino) How many speakers if known
    
    # LLM configuration
    llm_implementation: str = DEFAULT_LLM_IMPLEMENTATION  # "openai", "ollama", etc.
    llm_model: Optional[str] = None  # Model identifier
    
    # TTS configuration
    tts_implementation: str = DEFAULT_TTS_IMPLEMENTATION  # "elevenlabs", "sherpa_local", etc.
    tts_model: Optional[str] = None  # Name of TTS model if applicable
    speaker_id: int = 0  # Speaker ID for multi-speaker TTS models
    speed: float = 1.0  # Speech speed factor

class Pipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, **kwargs):
        """Initialize the Chrysalis pipeline
        
        Args:
            config: Optional PipelineConfig object
            **kwargs: Configuration options that override config if provided
        """
        init_start = time.time()
        start_mem = get_memory_usage()
        
        # Initialize configuration
        self.config = config or PipelineConfig()
        
        # Override config with any kwargs provided
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Store any additional parameters
        self.kwargs = {k: v for k, v in kwargs.items() if not hasattr(self.config, k)}
        
        # Initialize components as None for lazy loading
        self._stt = None
        self._llm = None
        self._tts = None
        
        logger.info("Initialized Pipeline with config: %s", self.config)
        logger.debug("Pipeline initialization completed in %.3fs", time.time() - init_start)
        log_memory_change(start_mem, "pipeline initialization")
    
    @property
    def stt(self):
        """Lazy initialization of STT component"""
        if self._stt is None:
            init_start = time.time()
            start_mem = get_memory_usage()
            logger.info("Initializing STT engine: %s", self.config.stt_implementation)
            
            # Construct STT parameters
            stt_params = {
                "model_name": self.config.stt_model,
                "use_int8": self.config.use_int8
            }
            
            # Add diarization parameters if enabled
            if self.config.enable_diarization:
                stt_params.update({
                    "enable_diarization": True,
                    "recognition_model": self.config.recognition_model,
                    "segmentation_model": self.config.segmentation_model,
                    "num_speakers": self.config.num_speakers,
                })
                
            # Additional parameters from kwargs
            stt_params.update(self.kwargs.get("stt_params", {}))
            
            try:
                self._stt = SpeechToTextAPI(
                    implementation=self.config.stt_implementation,
                    **stt_params
                )
                
                init_time = time.time() - init_start
                logger.info("STT engine initialized successfully in %.3fs", init_time)
                log_memory_change(start_mem, "STT initialization")
            except Exception as e:
                logger.error("Failed to initialize STT engine: %s", e, exc_info=True)
                raise
        
        return self._stt
    
    @property
    def llm(self):
        """Lazy initialization of LLM component"""
        if self._llm is None:
            init_start = time.time()
            start_mem = get_memory_usage()
            logger.info("Initializing LLM engine: %s", self.config.llm_implementation)
            
            llm_params = {
                "model": self.config.llm_model
            }
            
            # Additional parameters from kwargs
            llm_params.update(self.kwargs.get("llm_params", {}))
            
            try:
                self._llm = LLMQueryAPI(
                    implementation=self.config.llm_implementation,
                    **llm_params
                )
                
                init_time = time.time() - init_start
                logger.info("LLM engine initialized successfully in %.3fs", init_time)
                log_memory_change(start_mem, "LLM initialization")
            except Exception as e:
                logger.error("Failed to initialize LLM engine: %s", e, exc_info=True)
                raise
        
        return self._llm
    
    @property
    def tts(self):
        """Lazy initialization of TTS component"""
        if self._tts is None:
            init_start = time.time()
            start_mem = get_memory_usage()
            logger.info("Initializing TTS engine: %s", self.config.tts_implementation)
            
            tts_params = {
                "tts_model": self.config.tts_model,
                "speaker_id": self.config.speaker_id,
                "speed": self.config.speed
            }
            
            # Additional parameters from kwargs
            tts_params.update(self.kwargs.get("tts_params", {}))
            
            try:
                self._tts = TextToSpeechAPI(
                    implementation=self.config.tts_implementation,
                    **tts_params
                )
                
                init_time = time.time() - init_start
                logger.info("TTS engine initialized successfully in %.3fs", init_time)
                log_memory_change(start_mem, "TTS initialization")
            except Exception as e:
                logger.error("Failed to initialize TTS engine: %s", e, exc_info=True)
                raise
        
        return self._tts

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
            result = self.stt.transcribe(audio_file=audio_input)
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
                - timings: Dictionary with timing information
                - memory_changes: Dictionary with memory change information
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
            query_text = transcription["text"] if isinstance(transcription, dict) else transcription
            
            # Query LLM
            llm_start = time.time()
            llm_start_mem = get_memory_usage()
            logger.info("Querying LLM with transcribed text")
            response = self.llm.query(prompt=query_text, system_prompt=system_prompt)
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
                
                # Merge kwargs with existing TTS parameters
                tts_kwargs = {
                    "speaker_id": self.config.speaker_id,
                    "speed": self.config.speed
                }
                tts_kwargs.update(kwargs)
                
                self.tts.synthesize(
                    text=response, 
                    output_file=output_audio,
                    **tts_kwargs
                )
                
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
    
    def run(self,
            audio_file: Optional[str] = None,
            use_microphone: bool = False,
            output_file: str = "output.mp3",
            system_prompt: Optional[str] = None) -> dict:
        """
        Run the full pipeline: STT -> LLM -> TTS
        
        Args:
            audio_file: Path to input audio file
            use_microphone: Whether to record from microphone
            output_file: Path to save output audio file
            system_prompt: Optional system prompt for LLM
            
        Returns:
            Dictionary containing intermediate results
        """
        pipeline_start = time.time()
        start_mem = get_memory_usage()
        logger.info("Starting pipeline run")
        timings = {}
        memory_changes = {}
        
        # Speech to text
        stt_start = time.time()
        stt_start_mem = get_memory_usage()
        transcription = self.stt.transcribe(
            audio_file=audio_file,
            use_microphone=use_microphone
        )
        timings["stt"] = time.time() - stt_start
        memory_changes["stt"] = get_memory_usage() - stt_start_mem
        logger.info("Speech-to-text completed in %.2fs", timings["stt"])
        text_preview = transcription[:100] + "..." if isinstance(transcription, str) and len(transcription) > 100 else transcription
        logger.debug("Transcription: %s", text_preview)
        
        # Query LLM
        llm_start = time.time()
        llm_start_mem = get_memory_usage()
        query_text = transcription["text"] if isinstance(transcription, dict) else transcription
        llm_response = self.llm.query(
            prompt=query_text,
            system_prompt=system_prompt
        )
        timings["llm"] = time.time() - llm_start
        memory_changes["llm"] = get_memory_usage() - llm_start_mem
        logger.info("LLM query completed in %.2fs", timings["llm"])
        response_preview = llm_response[:100] + "..." if len(llm_response) > 100 else llm_response
        logger.debug("LLM response: %s", response_preview)
        
        # Text to speech
        tts_start = time.time()
        tts_start_mem = get_memory_usage()
        self.tts.synthesize(
            text=llm_response,
            output_file=output_file,
        )
        timings["tts"] = time.time() - tts_start
        memory_changes["tts"] = get_memory_usage() - tts_start_mem
        logger.info("Text-to-speech completed in %.2fs", timings["tts"])
        
        total_time = time.time() - pipeline_start
        total_mem_change = get_memory_usage() - start_mem
        
        logger.info("Pipeline completed in %.2fs (memory change: %.1fMB)", 
                   total_time, total_mem_change)
        
        for component, timing in timings.items():
            logger.debug("%s: %.3fs (%.1f%% of total time, %.1fMB memory change)", 
                       component.upper(), timing, 
                       (timing/total_time)*100,
                       memory_changes[component])
        
        return {
            "transcription": transcription,
            "llm_response": llm_response,
            "output_file": output_file,
            "timings": timings,
            "memory_changes": memory_changes
        }
    
    def transcribe_only(self,
                       audio_file: Optional[str] = None,
                       use_microphone: bool = False) -> str:
        """Run only the speech-to-text component"""
        start_time = time.time()
        start_mem = get_memory_usage()
        logger.info("Running transcribe_only")
        
        result = self.stt.transcribe(
            audio_file=audio_file, 
            use_microphone=use_microphone
        )
        
        logger.info("Transcribe_only completed in %.3fs", time.time() - start_time)
        log_memory_change(start_mem, "transcribe_only")
        
        return result
    
    def query_only(self, text: str, system_prompt: Optional[str] = None) -> str:
        """Run only the LLM query component"""
        start_time = time.time()
        start_mem = get_memory_usage()
        logger.info("Running query_only")
        
        result = self.llm.query(prompt=text, system_prompt=system_prompt)
        
        logger.info("Query_only completed in %.3fs", time.time() - start_time)
        log_memory_change(start_mem, "query_only")
        
        return result
    
    def synthesize_only(self,
                       text: str,
                       output_file: Optional[str] = None,
                       play_audio: bool = True,
                       **kwargs) -> None:
        """Run only the text-to-speech component
        
        Args:
            text: Text to synthesize
            output_file: Optional path to save audio file. If None and play_audio is True,
                        audio will be played through speakers
            play_audio: Whether to play audio through speakers if no output file specified
            **kwargs: Additional parameters for TTS
        """
        start_time = time.time()
        start_mem = get_memory_usage()
        logger.info("Running synthesize_only")
        
        # Merge kwargs with existing TTS parameters
        tts_kwargs = {
            "speaker_id": self.config.speaker_id,
            "speed": self.config.speed
        }
        tts_kwargs.update(kwargs)
        
        # Generate audio
        audio_data, sample_rate = self.tts.synthesize(
            text=text,
            output_file=None,  # Always get audio data back
            **tts_kwargs
        )
        
        # Save to file if specified
        if output_file:
            audio_utils.save_audio_file(
                audio_data=audio_data,
                output_path=output_file,
                sample_rate=sample_rate
            )
            logger.info("Audio saved to: %s", output_file)
        # Play through speakers if enabled and no output file
        elif play_audio:
            logger.info("Playing audio through speakers")
            audio_utils.play_audio(audio_data, sample_rate)
    
        logger.info("Synthesize_only completed in %.3fs", time.time() - start_time)
        log_memory_change(start_mem, "synthesize_only")
    
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