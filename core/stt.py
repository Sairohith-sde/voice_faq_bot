import os
import mimetypes
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class SpeechToTextEngine:
    """
    Transcribes spoken voice audio into text using Gemini 3.6 Flash Multimodal Audio.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-3.6-flash"

    def transcribe(self, audio_source, mime_type: Optional[str] = None) -> str:
        """
        Transcribes audio from a filepath (str) or raw audio bytes.
        Returns the clean transcribed text string.
        """
        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                raise FileNotFoundError(f"Audio file not found: {audio_source}")
            
            with open(audio_source, "rb") as f:
                audio_bytes = f.read()
            
            guessed_mime, _ = mimetypes.guess_type(audio_source)
            clean_mime = guessed_mime or ("audio/mp3" if audio_source.endswith(".mp3") else "audio/wav")

        elif isinstance(audio_source, bytes):
            audio_bytes = audio_source
            if mime_type:
                clean_mime = mime_type.split(";")[0].strip()
            else:
                clean_mime = "audio/webm"
        else:
            raise TypeError("audio_source must be a file path string or raw bytes.")

        prompt = (
            "Listen carefully to this audio and transcribe the exact words spoken into text. "
            "Output ONLY the accurate text transcript with proper punctuation. "
            "Do not add any explanations, prefixes, or commentary."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1000
                )
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Transcription failed with {self.model_name}: {str(e)}")


# Self-test
if __name__ == "__main__":
    stt = SpeechToTextEngine()
    test_audio = os.path.join("temp_audio", "test_voice.mp3")
    if os.path.exists(test_audio):
        print(f"Transcribing with {stt.model_name}...")
        result = stt.transcribe(test_audio)
        print(f"Transcript: {result}")