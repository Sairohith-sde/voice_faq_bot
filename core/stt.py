import os
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class SpeechToTextEngine:
    """
    Transcribes spoken voice audio into text using Gemini Multimodal Audio.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"

    def transcribe(self, audio_source) -> str:
        """
        Transcribes audio from a filepath (str) or raw audio bytes.
        Returns the clean transcribed text string.
        """
        if isinstance(audio_source, str):
            # Audio source is a file path
            if not os.path.exists(audio_source):
                raise FileNotFoundError(f"Audio file not found: {audio_source}")
            
            with open(audio_source, "rb") as f:
                audio_bytes = f.read()
            
            # Determine mime type based on extension
            mime_type, _ = mimetypes.guess_type(audio_source)
            if not mime_type:
                mime_type = "audio/mp3" if audio_source.endswith(".mp3") else "audio/wav"

        elif isinstance(audio_source, bytes):
            # Audio source is raw audio bytes from browser recording
            audio_bytes = audio_source
            mime_type = "audio/webm"  # Standard browser media recording format
        else:
            raise TypeError("audio_source must be a file path string or raw bytes.")

        prompt = (
            "Listen carefully to this audio recording and transcribe the exact words spoken. "
            "Output ONLY the accurate text transcript with proper punctuation. "
            "Do not include any commentary, prefixes, or explanations."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    prompt
                ]
            )
            return response.text.strip()
        except Exception as e:
            # Fallback to gemini-2.0-flash if needed
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        prompt
                    ]
                )
                return response.text.strip()
            except Exception as e2:
                raise RuntimeError(f"Failed to transcribe audio with Gemini: {str(e2)}")


# Self-test when run directly using the audio file generated in Step 6
if __name__ == "__main__":
    stt = SpeechToTextEngine()
    test_audio_path = os.path.join("temp_audio", "test_voice.mp3")
    
    if os.path.exists(test_audio_path):
        print("\n================ SPEECH-TO-TEXT TEST ================")
        print(f"🎧 Transcribing audio file: {test_audio_path}")
        transcript = stt.transcribe(test_audio_path)
        print(f"\n📝 Transcribed Text Output:\n\"{transcript}\"")
        print("=====================================================")
    else:
        print("Please run Step 6 (python -m core.tts) first to generate a sample audio file.")