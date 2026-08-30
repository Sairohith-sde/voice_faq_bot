# 🎙️ Sai Rohith's AI Voice FAQ Assistant

An interactive, low-latency, voice-first AI assistant representing **Sai Rohith's** software engineering portfolio. Engineered with a **WhisperFlow-style audio pipeline**, **Bi-gram RAG retrieval**, **Google Gemini multi-model reasoning**, and **Microsoft Edge Neural Text-to-Speech**.

---

## 🌟 Key Features

- 🎙️ **Voice-In, Voice-Out Pipeline:** Real-time speech recognition with browser hardware DSP active noise suppression & echo cancellation.
- 🧹 **WhisperFlow Disfluency Stripper:** Filters out conversational hesitation tokens (* uh*, *um*, *like*, *so basically*) for clean semantic querying.
- 🔍 **Bi-Gram RAG Engine:** High-precision TF-IDF vector space with cosine similarity indexing multi-word project titles, tech stacks, and case studies.
- 🧠 **Multi-Turn Executive AI Reasoning:** Powered by Google Gemini with multi-turn conversation memory and automatic model failover (gemini-2.5-flash-lite -> gemini-3.5-flash-lite -> gemini-3.1-flash-lite).
- 🌊 **Real-Time Audio Spectrum Visualizer:** Animated HTML5 Canvas waveform synchronized with microphone input (cyan) and AI audio playback (emerald).
- 🛑 **Barge-In Interruption:** Instant audio cutoff when the user starts speaking again.
- 🔔 **Synthesized Audio Earcons:** Zero-asset high-tech chimes for listening start and answer completion via the Web Audio API.
- 🔊 **Neural Text-to-Speech:** High-fidelity Microsoft Neural Voices (en-US-ChristopherNeural, en-US-GuyNeural, en-IN-PrabhatNeural, etc.).

---

## 🏗️ System Architecture

`
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
`

---

## 📁 Project Structure

`
voice_faq_bot/
├── app.py                      # FastAPI Backend Server & API Routes
├── requirements.txt            # Python Dependencies
├── .env.example                # Example Environment Config
├── .gitignore                  # Git Ignore Rules (Protects .env & temp audio)
├── LICENSE                     # MIT Open Source License
├── README.md                   # Project Documentation
├── core/
│   ├── cleaner.py              # Transcript Normalizer & Filler Stripper
│   ├── rag.py                  # Bi-Gram TF-IDF Vector Retrieval Engine
│   ├── llm.py                  # Gemini Multi-Turn Reasoning with Failover
│   ├── tts.py                  # Edge-TTS Neural Audio Synthesizer
│   └── stt.py                  # Multimodal Speech-to-Text Engine
├── data/
│   └── portfolio_faq.json      # Structured Knowledge Base (Projects, Skills, FAQs)
└── templates/
    └── index.html              # Glassmorphic UI, Canvas Waveform & Web Speech API
`

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
`ash
git clone https://github.com/Sairohith-sde/voice_faq_bot.git
cd voice_faq_bot
`

### 2. Install Dependencies
`ash
pip install -r requirements.txt
`

### 3. Set Up Environment Variables
Create a .env file in the root directory:
`env
GEMINI_API_KEY=your_gemini_api_key_here
`
*(Get a free API key from [Google AI Studio](https://aistudio.google.com/))*

### 4. Run the Application
`ash
python app.py
`
Open **http://127.0.0.1:8000** in your browser (Google Chrome or Microsoft Edge recommended).

---

## 👨‍💻 Author
**Sai Rohith (Pothuganti Sai Rohith)**
- 🌐 Portfolio: [sairohith1.vercel.app](https://sairohith1.vercel.app/)
- 💻 GitHub: [@Sairohith-sde](https://github.com/Sairohith-sde)
- 📧 Email: rohithsai523@gmail.com

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
