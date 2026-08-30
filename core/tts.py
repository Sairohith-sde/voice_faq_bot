import os
import asyncio
import edge_tts
from typing import Optional


class TextToSpeechEngine:
    """
    Converts AI response text into natural neural speech audio using Edge-TTS.
    """
    # High-quality neural voices
    AVAILABLE_VOICES = {
        "US Male (Guy)": "en-US-GuyNeural",
        "US Male (Christopher)": "en-US-ChristopherNeural",
        "US Female (Aria)": "en-US-AriaNeural",
        "India Male (Prabhat)": "en-IN-PrabhatNeural",
        "UK Male (Ryan)": "en-GB-RyanNeural",
    }

    def __init__(self, voice: str = "en-US-GuyNeural", output_dir: str = "temp_audio"):
        self.voice = voice
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def _synthesize_async(self, text: str, output_filepath: str) -> str:
        """Asynchronously calls edge-tts to generate audio file."""
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate="+0%",     # Speed: e.g. +10% or -10%
            pitch="+0Hz"    # Pitch adjustment
        )
        await communicate.save(output_filepath)
        return output_filepath

    def synthesize(self, text: str, filename: Optional[str] = None) -> str:
        """
        Synchronous wrapper to convert text to an MP3 audio file.
        Returns the absolute path to the generated audio file.
        """
        if not text or not text.strip():
            raise ValueError("Text to synthesize cannot be empty.")

        if filename is None:
            filename = "response.mp3"
            
        output_filepath = os.path.join(self.output_dir, filename)

        # Run async TTS inside standard synchronous execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If running inside an existing event loop (e.g. FastAPI/Tornado)
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._synthesize_async(text, output_filepath))
            else:
                return loop.run_until_complete(self._synthesize_async(text, output_filepath))
        except RuntimeError:
            return asyncio.run(self._synthesize_async(text, output_filepath))


# Self-test when run directly
if __name__ == "__main__":
    tts = TextToSpeechEngine()
    test_text = (
        "Hello! I am Sai Rohith's AI voice assistant. "
        "I can answer questions about his software engineering projects, technical skills, and background."
    )
    
    print("\n================ TEXT-TO-SPEECH TEST ================")
    print(f"🎙️ Generating speech for: \"{test_text}\"")
    output_file = tts.synthesize(test_text, filename="test_voice.mp3")
    print(f"✅ Audio generated successfully at: {output_file}")
    print("=====================================================")