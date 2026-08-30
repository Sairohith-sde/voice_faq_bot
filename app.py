import os
import sys

# Ensure current directory is always in Python's search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.rag import KnowledgeBase
from core.llm import VoiceReasoningEngine
from core.tts import TextToSpeechEngine
from core.stt import SpeechToTextEngine

# Initialize FastAPI App
app = FastAPI(title="Sai Rohith's AI Voice FAQ Assistant")

# Setup directories & templates
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize AI Engines
kb = KnowledgeBase(data_path=os.path.join(BASE_DIR, "data", "portfolio_faq.json"))
stt = SpeechToTextEngine()
llm = VoiceReasoningEngine()
tts = TextToSpeechEngine(output_dir=TEMP_AUDIO_DIR)


class TextQueryRequest(BaseModel):
    question: str
    voice: str = "en-US-ChristopherNeural"


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Renders the main interactive Voice FAQ UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask-voice")
async def handle_voice_query(
    audio_file: UploadFile = File(...),
    voice: str = Form("en-US-ChristopherNeural")
):
    """
    Complete Voice-In, Voice-Out Pipeline:
    1. Transcribe audio recording bytes with Gemini 3.6 Flash STT
    2. Retrieve relevant portfolio facts with RAG
    3. Generate natural, detailed spoken answer with Gemini 3.6 Flash LLM
    4. Synthesize spoken audio with Edge-TTS
    """
    try:
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            return JSONResponse({"error": "Empty audio recording received"}, status_code=400)

        # 1. Speech-to-Text (STT) with clean MIME type
        transcript = stt.transcribe(audio_bytes, mime_type=audio_file.content_type)
        if not transcript:
            return JSONResponse({"error": "Could not understand audio speech."}, status_code=400)

        # 2. RAG Retrieval (Get top 2 matching knowledge chunks)
        context_chunks = kb.retrieve(transcript, top_k=2)

        # 3. LLM Voice Reasoning
        answer = llm.generate_spoken_response(transcript, context_chunks)

        # 4. Text-to-Speech (TTS)
        tts.voice = voice
        await tts.synthesize_async(answer, filename="response.mp3")

        return JSONResponse({
            "transcript": transcript,
            "answer": answer,
            "retrieved_context": context_chunks,
            "audio_url": "/api/audio"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/ask-text")
async def handle_text_query(data: TextQueryRequest):
    """Text query route with voice audio output."""
    try:
        transcript = data.question.strip()
        context_chunks = kb.retrieve(transcript, top_k=2)
        answer = llm.generate_spoken_response(transcript, context_chunks)

        tts.voice = data.voice
        await tts.synthesize_async(answer, filename="response.mp3")

        return JSONResponse({
            "transcript": transcript,
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
    print(f"\n🚀 Starting Sai Rohith's AI Voice FAQ Assistant on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)