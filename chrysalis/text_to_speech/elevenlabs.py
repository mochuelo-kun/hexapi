import requests
from .base import TextToSpeechBase
from ..config import Config

class ElevenLabsTTS(TextToSpeechBase):
    def __init__(self, tts_model: str = "21m00Tcm4TlvDq8ikWAM"):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.tts_model = tts_model
        
    def synthesize(self, text: str, output_file: str, **kwargs) -> None:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.tts_model}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            **kwargs
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        with open(output_file, 'wb') as f:
            f.write(response.content) 