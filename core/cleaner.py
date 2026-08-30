import re


class TranscriptCleaner:
    """
    Cleans spoken transcripts by removing filler words, hesitation markers,
    stutters, and conversational noise.
    """
    FILLER_PATTERNS = [
        r"\b(uh|um|umm|er|ah|like|you know|so basically|basically|actually|sort of|kind of|i mean|so yeah|well|anyway|right)\b",
        r"\b(can you please|could you please|hey can you|can you|tell me about|what about|do you know about)\b"
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """Strips fillers, removes stutters, and normalizes text."""
        if not text:
            return ""

        cleaned = text

        # 1. Remove stutter repetitions (e.g. "what what did" -> "what did")
        cleaned = re.sub(r"\b(\w+)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)

        # 2. Remove filler words and conversational padding
        for pattern in cls.FILLER_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # 3. Collapse multiple spaces and trim
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # If stripping emptied the query, fallback to the original
        return cleaned if cleaned else text.strip()


# Self-test
if __name__ == "__main__":
    sample = "umm uh hey can you tell me like so basically what did sai build for greenops you know"
    print("Raw:    ", sample)
    print("Cleaned:", TranscriptCleaner.clean(sample))