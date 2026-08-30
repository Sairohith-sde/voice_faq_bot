import os
import sys
from typing import List, Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.cleaner import TranscriptCleaner
from core.rag import KnowledgeBase
from core.llm import VoiceReasoningEngine
from core.tts import TextToSpeechEngine

app = FastAPI(title="Sai Rohith's AI Voice FAQ Assistant")

# Setup directories & templates
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize AI Engines
kb = KnowledgeBase(data_path=os.path.join(BASE_DIR, "data", "portfolio_faq.json"))
llm = VoiceReasoningEngine()
tts = TextToSpeechEngine(output_dir=TEMP_AUDIO_DIR)


class TextQueryRequest(BaseModel):
    question: str
    voice: str = "en-US-ChristopherNeural"
    history: Optional[List[Dict[str, str]]] = []


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Renders the main interactive Voice FAQ UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask-text")
async def handle_text_query(data: TextQueryRequest):
    """
    Full Multi-Turn Voice Pipeline:
    1. Filter conversational fillers & stutters
    2. Retrieve accurate portfolio facts with Bi-gram RAG
    3. Reason with Gemini taking conversation history into account
    4. Synthesize natural neural speech
    """
    try:
        raw_question = data.question.strip()
        if not raw_question:
            return JSONResponse({"error": "Question cannot be empty"}, status_code=400)

        # Step 1: Clean Transcript & Strip Fillers
        cleaned_query = TranscriptCleaner.clean(raw_question)

        # Step 2: High-Precision RAG Retrieval
        context_chunks = kb.retrieve(cleaned_query, top_k=2)

        # Step 3: Multi-Turn Executive LLM Reasoning
        answer = llm.generate_spoken_response(
            question=raw_question,
            context_chunks=context_chunks,
            history=data.history
        )

        # Step 4: Neural TTS Audio Synthesis
        tts.voice = data.voice
        await tts.synthesize_async(answer, filename="response.mp3")

        return JSONResponse({
            "transcript": raw_question,
            "cleaned_query": cleaned_query,
            "answer": answer,
            "retrieved_context": context_chunks,
            "audio_url": "/api/audio"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/audio")
async def stream_audio_response():
    """Serves the generated response.mp3 audio file to the browser."""
    audio_file = os.path.join(TEMP_AUDIO_DIR, "response.mp3")
    if os.path.exists(audio_file):
        return FileResponse(audio_file, media_type="audio/mpeg")
    return JSONResponse({"error": "Audio not generated yet"}, status_code=404)


if __name__ == "__main__":
    print("\n🚀 Starting Multi-Turn WhisperFlow Voice Assistant on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)