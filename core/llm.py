import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class VoiceReasoningEngine:
    """
    Executive AI Technical Reasoning Engine with Multi-Turn Memory and Automatic Model Failover.
    """
    MODEL_CASCADE = [
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash"
    ]

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
        
        self.client = genai.Client(api_key=self.api_key)

    def generate_spoken_response(
        self,
        question: str,
        context_chunks: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Synthesizes an articulate, professional, engineering-focused spoken explanation
        taking into account conversation history for seamless follow-up questions.
        """
        context_str = "\n\n".join(context_chunks)

        # Format conversation history
        history_str = ""
        if history and len(history) > 0:
            formatted = []
            for h in history[-4:]:  # Keep last 4 turns for context
                role = "User" if h.get("role") == "user" else "Assistant"
                formatted.append(f"{role}: {h.get('content')}")
            history_str = "PREVIOUS CONVERSATION HISTORY:\n" + "\n".join(formatted) + "\n\n"

        system_instruction = (
            "You are the executive AI Technical Assistant representing Sai Rohith's engineering portfolio.\n"
            "Deliver articulate, highly professional, and engineering-focused spoken answers tailored for technical recruiters, engineering leaders, and interviewers.\n\n"
            "PROFESSIONAL STYLE GUIDELINES:\n"
            "1. Tone: Confident, articulate, professional, and impactful (use strong engineering verbs like 'architected', 'engineered', 'deployed').\n"
            "2. Context Awareness: Resolve follow-up pronouns (like 'it', 'that project') using conversation history.\n"
            "3. Structure: 2 to 3 polished, cohesive sentences (around 45 to 65 words) designed for smooth spoken delivery.\n"
            "4. Persona: Speak in the third person about Sai Rohith.\n"
            "5. Presentation: Strictly avoid markdown formatting, asterisks, bullet points, and raw URLs."
        )

        user_prompt = f"""PORTFOLIO TECHNICAL CONTEXT:
========================================
{context_str}
========================================

{history_str}USER INQUIRY: {question}

PROFESSIONAL SPOKEN EXPLANATION:"""

        last_error = None

        # Try models in cascade sequence for guaranteed response
        for model_name in self.MODEL_CASCADE:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=350,
                )
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config
                )
                if response.text and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                last_error = e
                print(f"⚠️ Model {model_name} quota/error: {e}. Cascading to next model...")
                continue

        return f"I encountered an issue connecting to the AI models: {str(last_error)}"


# Self-test
if __name__ == "__main__":
    from core.rag import KnowledgeBase

    kb = KnowledgeBase()
    engine = VoiceReasoningEngine()
    
    q = "What did you build for GreenOps Code Auditor?"
    ctx = kb.retrieve(q, top_k=2)
    ans = engine.generate_spoken_response(q, ctx)
    print(f"\nQ: {q}\n\nExecutive Spoken Output:\n{ans}\n")