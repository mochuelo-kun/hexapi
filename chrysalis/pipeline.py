from typing import Optional
from dataclasses import dataclass
from .speech_to_text.api import SpeechToTextAPI
from .llm_query.api import LLMQueryAPI
from .text_to_speech.api import TextToSpeechAPI

@dataclass
class ChrysalisConfig:
    """Configuration for the Chrysalis pipeline"""
    stt_implementation: str = "whisper_api"
    llm_model: str = "ollama:llama2"
    tts_implementation: str = "elevenlabs"
    voice_id: Optional[str] = None

class ChrysalisPipeline:
    def __init__(self, config: Optional[ChrysalisConfig] = None):
        self.config = config or ChrysalisConfig()
        
        # Initialize components
        self.stt = SpeechToTextAPI(implementation=self.config.stt_implementation)
        self.llm = LLMQueryAPI()
        self.tts = TextToSpeechAPI(implementation=self.config.tts_implementation)
    
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
        # Speech to text
        transcription = self.stt.transcribe(
            audio_file=audio_file,
            use_mic=use_mic
        )
        
        # Query LLM
        llm_response = self.llm.query(
            prompt=transcription,
            model=self.config.llm_model
        )
        
        # Text to speech
        self.tts.synthesize(
            text=llm_response,
            output_file=output_file,
            voice_id=self.config.voice_id
        )
        
        return {
            "transcription": transcription,
            "llm_response": llm_response,
            "output_file": output_file
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