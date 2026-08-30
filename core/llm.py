import os
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class VoiceReasoningEngine:
    """
    Executive AI Technical Reasoning Engine for Sai Rohith's Portfolio Voice Assistant
    powered by Gemini 3.6 Flash.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-3.6-flash"

    def generate_spoken_response(self, question: str, context_chunks: List[str]) -> str:
        """
        Synthesizes an articulate, professional, engineering-focused spoken explanation.
        """
        context_str = "\n\n".join(context_chunks)

        system_instruction = (
            "You are the executive AI Technical Assistant representing Sai Rohith's engineering portfolio.\n"
            "Deliver articulate, highly professional, and engineering-focused spoken answers tailored for technical recruiters, engineering leaders, and interviewers.\n\n"
            "PROFESSIONAL STYLE GUIDELINES:\n"
            "1. Tone: Confident, articulate, professional, and impactful (use strong engineering verbs like 'architected', 'engineered', 'deployed').\n"
            "2. Focus: Highlight system architecture, engineering principles, tech stack choices, and measurable impact.\n"
            "3. Structure: 2 to 3 polished, cohesive sentences (around 45 to 65 words) designed for smooth spoken delivery.\n"
            "4. Persona: Speak in the third person about Sai Rohith.\n"
            "5. Presentation: Strictly avoid markdown formatting, asterisks, bullet points, and raw URLs."
        )

        user_prompt = f"""PORTFOLIO TECHNICAL CONTEXT:
========================================
{context_str}
========================================

USER INQUIRY: {question}

PROFESSIONAL SPOKEN EXPLANATION:"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=1500,
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"I encountered an issue connecting to Gemini: {str(e)}"


# Self-test
if __name__ == "__main__":
    from core.rag import KnowledgeBase

    kb = KnowledgeBase()
    engine = VoiceReasoningEngine()
    
    q = "Tell me about your Agentflow AI project"
    ctx = kb.retrieve(q, top_k=2)
    ans = engine.generate_spoken_response(q, ctx)
    print(f"\nQ: {q}\n\nExecutive Spoken Output (Gemini 3.6 Flash):\n{ans}\n")