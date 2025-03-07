import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Whisper Local
    WHISPER_MODEL = os.getenv("WHISPER_MODEL")
    
    # Piper
    PIPER_MODEL_DIR = os.getenv("PIPER_MODEL_DIR")
    # PIPER_MODEL_NAME = os.getenv("PIPER_MODEL_NAME", "en_US-amy-medium")
    # PIPER_SAMPLE_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))
    
    # ElevenLabs
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    # Local Server
    # LOCAL_LLM_SERVER_URL = os.getenv("LOCAL_LLM_SERVER_URL", "http://localhost:8000")
    
    # Ollama
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")