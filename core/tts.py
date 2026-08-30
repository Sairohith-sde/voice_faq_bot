import os
import asyncio
import edge_tts
from typing import Optional


class TextToSpeechEngine:
    """
    Converts AI response text into natural neural speech audio using Edge-TTS.
    """
    AVAILABLE_VOICES = {
        "US Male Executive (Christopher)": "en-US-ChristopherNeural",
        "US Male (Guy)": "en-US-GuyNeural",
        "US Female (Aria)": "en-US-AriaNeural",
        "India Male (Prabhat)": "en-IN-PrabhatNeural",
        "UK Male (Ryan)": "en-GB-RyanNeural",
    }

    def __init__(self, voice: str = "en-US-ChristopherNeural", output_dir: str = "temp_audio"):
        self.voice = voice
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def synthesize_to_base64_async(self, text: str) -> str:
        """Synthesizes speech audio in-memory and returns base64 data URL (zero disk writes, 100% serverless)."""
        if not text or not text.strip():
            raise ValueError("Text to synthesize cannot be empty.")

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate="-3%",
            pitch="+0Hz"
        )
        audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])

        import base64
        b64_data = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:audio/mp3;base64,{b64_data}"

    async def synthesize_async(self, text: str, filename: Optional[str] = None) -> str:
        """Asynchronously generates speech audio with articulate, professional pacing."""
        if not text or not text.strip():
            raise ValueError("Text to synthesize cannot be empty.")

        if filename is None:
            filename = "response.mp3"

        output_filepath = os.path.join(self.output_dir, filename)

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate="-3%",    # Slightly calmer rate for clear, executive clarity
            pitch="+0Hz"
        )
        await communicate.save(output_filepath)
        return output_filepath

    def synthesize(self, text: str, filename: Optional[str] = None) -> str:
        """Synchronous wrapper for offline testing."""
        return asyncio.run(self.synthesize_async(text, filename))


if __name__ == "__main__":
    tts = TextToSpeechEngine()
    test_text = (
        "Sai architected Agentflow AI, a distributed multi-agent automation platform "
        "featuring an interactive visual canvas and an autonomous self-healing execution engine."
    )
    output_file = tts.synthesize(test_text, filename="test_voice.mp3")
    print(f"✅ Professional audio generated at: {output_file}")