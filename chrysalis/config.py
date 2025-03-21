import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Whisper Local
    WHISPER_MODEL = os.getenv("WHISPER_MODEL")
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    
    # ONNX Models (Piper, Sherpa)
    ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR")
    ESPEAK_DATA_PATH = os.getenv("ESPEAK_DATA_PATH")
    
    # ElevenLabs
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    # Local Server
    # LOCAL_LLM_SERVER_URL = os.getenv("LOCAL_LLM_SERVER_URL", "http://localhost:8000")
    
    # Ollama
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")