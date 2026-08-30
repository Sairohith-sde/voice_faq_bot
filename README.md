# 🎙️ VocalRohith AI — Interactive Voice Digital Twin

[![Live Demo](https://img.shields.io/badge/Live_Demo-vocalrohithai.vercel.app-0ea5e9?style=for-the-badge&logo=vercel)](https://vocalrohithai.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge)](LICENSE)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Google Gemini](https://img.shields.io/badge/LLM-Google_Gemini-4285F4?style=for-the-badge&logo=google)](https://aistudio.google.com)

An interactive, low-latency, voice-first AI assistant and conversational digital twin representing **Sai Rohith's** software engineering portfolio. Engineered with a **WhisperFlow-style audio pipeline**, **Bi-gram RAG retrieval**, **Google Gemini multi-model reasoning**, and **Microsoft Edge Neural Text-to-Speech**.

🌐 **Live Web Application:** [https://vocalrohithai.vercel.app/](https://vocalrohithai.vercel.app/)

---

## 🌟 Key Features

- 🎙️ **Voice-In, Voice-Out Pipeline:** Real-time speech recognition with browser hardware DSP active noise suppression & echo cancellation.
- 🧹 **WhisperFlow Disfluency Stripper:** Filters out conversational hesitation tokens (*"uh"*, *"um"*, *"like"*, *"so basically"*) for clean semantic querying.
- 🔍 **Bi-Gram RAG Engine:** High-precision TF-IDF vector space with cosine similarity indexing multi-word project titles, tech stacks, and case studies.
- 🧠 **Multi-Turn Executive AI Reasoning:** Powered by Google Gemini with multi-turn conversation memory and automatic model failover (`gemini-2.5-flash-lite` ➔ `gemini-3.5-flash-lite` ➔ `gemini-3.1-flash-lite`).
- 🌊 **Real-Time Audio Spectrum Visualizer:** Animated HTML5 Canvas waveform synchronized with microphone input (cyan) and AI audio playback (emerald).
- 🛑 **Barge-In Interruption:** Instant audio cutoff when the user starts speaking again.
- 🔔 **Synthesized Audio Earcons:** Zero-asset high-tech chimes for listening start and answer completion via the Web Audio API.
- 🔊 **Neural Text-to-Speech:** High-fidelity Microsoft Neural Voices (`en-US-ChristopherNeural`, `en-US-GuyNeural`, `en-IN-PrabhatNeural`, etc.).
- ⚡ **In-Memory Serverless Streaming:** Zero disk dependency with base64 audio payloads for sub-second responses on Vercel Serverless Functions.

---

## 🛠️ How AI Was Leveraged to Build This Project

This project was engineered using an **AI-augmented development workflow**, combining software system design with agentic AI pair programming to rapidly prototype, optimize, and deploy a production-grade voice application:

### 1. 📐 System Architecture & Pipeline Design
- **Multi-Modal Flow:** Leveraged AI to architect an end-to-end voice loop (Speech-to-Text ➔ WhisperFlow Cleaner ➔ Bi-Gram RAG ➔ Cascading LLM ➔ Neural TTS ➔ Canvas Equalizer).
- **State Machine & Concurrency:** Designed an asynchronous pipeline supporting instant **barge-in interruption**, real-time FFT audio visualizer synchronization, and multi-turn conversation memory.

### 2. 🗂️ Knowledge Base & Data Engineering
- **Automated Data Ingestion:** Utilized AI scripts to parse portfolio source files, hackathon project specs, and technical achievements into structured, high-density JSON knowledge chunks (`portfolio_faq.json`).
- **Bi-Gram Vector Indexing:** Engineered an n-gram TF-IDF vector retrieval engine with sublinear scaling to ensure 100% precision on technical terms (e.g., *"5-Agent loop"*, *"AES-256 Vault"*, *"Carbon Penalty"*).

### 3. 🎯 Prompt Engineering & Multi-Model Resilience
- **Executive Spoken Persona:** Iteratively engineered and tuned system prompts to produce clear, conversational 2–3 sentence responses formatted specifically for listening (eliminating bullet points, asterisks, and robotic phrasing).
- **Cascading Fallback Matrix:** Designed a multi-model failover pool (`gemini-2.5-flash-lite` ➔ `gemini-3.5-flash-lite` ➔ `gemini-3.1-flash-lite` ➔ `gemini-2.5-flash`) with automatic exception handling to maintain 99.9% uptime against API rate limits (`429 RESOURCE_EXHAUSTED`).

### 4. 🧹 Custom Audio Cleaning Algorithm (WhisperFlow Pipeline)
- **Speech Disfluency Stripper:** Co-developed regex-based conversational cleaning rules (`core/cleaner.py`) to detect and eliminate hesitation markers (*"uh"*, *"um"*, *"so basically"*) and stutter repetitions from raw browser speech transcripts before vector retrieval.

### 5. 🚀 Serverless Architecture & Cloud Deployment
- **In-Memory Audio Streaming:** Refactored disk-based TTS generation into an asynchronous in-memory base64 streaming architecture, making the entire FastAPI backend 100% stateless and serverless-ready for Vercel.
- **Rapid Debugging & Optimization:** Used AI assistance to diagnose cross-platform deployment bottlenecks (resolving Python 3.14 Starlette template issues and Windows UTF-8 BOM encoding for Vercel builds).

---

## 🏗️ System Architecture

```
User Voice Input (Web Speech API + Hardware DSP Noise Suppression)
                   │
                   ▼
Transcript Cleaner (Strips uh, um, stutters & conversational padding)
                   │
                   ▼
Bi-Gram RAG Engine (TF-IDF Vector Space with Cosine Similarity)
                   │
                   ▼
Gemini Reasoning Engine (Multi-turn Context + Multi-Model Failover)
                   │
                   ▼
Edge-TTS Neural Synthesizer (Executive Voice Delivery)
                   │
                   ▼
Spoken Audio Response + Animated Canvas Waveform Visualizer
```

---

## 📁 Project Structure

```
voice_faq_bot/
├── app.py                      # FastAPI Backend Server & Serverless Handler
├── requirements.txt            # Python Dependencies
├── vercel.json                 # Vercel Serverless Function Configuration
├── .env.example                # Example Environment Config
├── .gitignore                  # Git Ignore Rules (Protects .env & temp audio)
├── LICENSE                     # MIT Open Source License
├── README.md                   # Project Documentation
├── core/
│   ├── cleaner.py              # Transcript Normalizer & Filler Stripper
│   ├── rag.py                  # Bi-Gram TF-IDF Vector Retrieval Engine
│   ├── llm.py                  # Gemini Multi-Turn Reasoning with Failover
│   ├── tts.py                  # In-Memory Edge-TTS Neural Audio Synthesizer
│   └── stt.py                  # Multimodal Speech-to-Text Engine
├── data/
│   └── portfolio_faq.json      # Structured Knowledge Base (Projects, Skills, FAQs)
└── templates/
    └── index.html              # Glassmorphic UI, Canvas Waveform & Web Speech API
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Sairohith-sde/voice_faq_bot.git
cd voice_faq_bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Get a free API key from [Google AI Studio](https://aistudio.google.com/))*

### 4. Run the Application
```bash
python app.py
```
Open **http://127.0.0.1:8000** in your browser (Google Chrome or Microsoft Edge recommended).

---

## 👨‍💻 Author
**Sai Rohith (Pothuganti Sai Rohith)**
- 🌐 Portfolio: [sairohith1.vercel.app](https://sairohith1.vercel.app/)
- 🎙️ VocalRohith AI: [vocalrohithai.vercel.app](https://vocalrohithai.vercel.app/)
- 💻 GitHub: [@Sairohith-sde](https://github.com/Sairohith-sde)
- 📧 Email: rohithsai523@gmail.com

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
