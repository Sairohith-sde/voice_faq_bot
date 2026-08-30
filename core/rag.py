import json
import os
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KnowledgeBase:
    """
    High-Precision Bi-gram RAG Knowledge Base for Sai Rohith's Portfolio.
    """
    def __init__(self, data_path: str = "data/portfolio_faq.json"):
        self.data_path = data_path
        self.raw_data: Dict[str, Any] = {}
        self.chunks: List[Dict[str, str]] = []
        # Bi-gram vectorizer with sublinear term frequency scaling
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix = None
        self.load_and_index()

    def load_and_index(self):
        """Loads portfolio FAQ JSON data and converts sections into searchable knowledge chunks."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.chunks = []

        # 1. Chunk Personal / Profile Info
        if "personal" in self.raw_data:
            p = self.raw_data["personal"]
            profile_text = (
                f"Candidate Name: {p.get('name')} ({p.get('fullName')}). "
                f"Title: {p.get('title')}. Status: {p.get('status')}. "
                f"Location: {p.get('location')}. Email: {p.get('email')}. "
                f"GitHub: {p.get('github')}. LinkedIn: {p.get('linkedin')}. "
                f"Portfolio: {p.get('portfolio_url')}. Bio: {p.get('bio')}"
            )
            self.chunks.append({"category": "personal", "content": profile_text})

        # 2. Chunk Education
        if "education" in self.raw_data:
            e = self.raw_data["education"]
            courses = ", ".join(e.get("coursework", []))
            edu_text = (
                f"Education Background: Pursuing {e.get('degree')} at {e.get('institution')} ({e.get('duration')}), "
                f"located in {e.get('location')}. Key Coursework: {courses}."
            )
            self.chunks.append({"category": "education", "content": edu_text})

        # 3. Chunk Skills
        if "skills" in self.raw_data:
            s = self.raw_data["skills"]
            skills_text = (
                f"Technical Skills, Developer Stack and Proficiencies: "
                f"Programming Languages: {', '.join(s.get('languages', []))}. "
                f"AI, ML & LLMs: {', '.join(s.get('ai_and_ml', []))}. "
                f"Backend & APIs: {', '.join(s.get('backend_and_apis', []))}. "
                f"Databases & Tools: {', '.join(s.get('databases_and_tools', []))}."
            )
            self.chunks.append({"category": "skills", "content": skills_text})

        # 4. Chunk Flagship Case Study
        if "flagship_case_study" in self.raw_data:
            cs = self.raw_data["flagship_case_study"]
            cs_text = (
                f"Flagship Case Study / Research Project: {cs.get('title')} - {cs.get('tagline')}. "
                f"Validation Status: {cs.get('status')}. "
                f"Description: {cs.get('description')}. "
                f"Architecture Details: {cs.get('architecture')}."
            )
            self.chunks.append({"category": "case_study", "content": cs_text})

        # 5. Chunk Individual Projects
        if "projects" in self.raw_data:
            for proj in self.raw_data["projects"]:
                proj_text = (
                    f"Project: {proj.get('title')} ({proj.get('subtitle')}). "
                    f"Description: {proj.get('description')}. "
                    f"Tech Stack: {', '.join(proj.get('tech_stack', []))}. "
                    f"GitHub: {proj.get('github')}. Live Demo: {proj.get('live_demo')}. "
                    f"Key Highlights: {proj.get('highlights')}."
                )
                self.chunks.append({"category": "projects", "content": proj_text})

        # 6. Chunk Accolades & Achievements
        if "accolades" in self.raw_data:
            for acc in self.raw_data["accolades"]:
                acc_text = (
                    f"Accolade / Achievement: {acc.get('title')} by {acc.get('organization')}. "
                    f"Details: {acc.get('details')}."
                )
                self.chunks.append({"category": "accolades", "content": acc_text})

        # 7. Chunk FAQs
        if "faqs" in self.raw_data:
            for faq in self.raw_data["faqs"]:
                faq_text = f"FAQ - Question: {faq.get('question')} Answer: {faq.get('answer')}"
                self.chunks.append({"category": "faq", "content": faq_text})

        # Build Vector Space Matrix
        corpus = [chunk["content"] for chunk in self.chunks]
        if corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """
        Takes a query and returns the top_k most relevant knowledge chunks.
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [self.chunks[idx]["content"] for idx in top_indices if similarities[idx] > 0.0]
        if not results:
            results = [self.chunks[0]["content"]]

        return results


# Self-test
if __name__ == "__main__":
    kb = KnowledgeBase()
    print("Testing Bi-Gram RAG Retrieval:")
    for test_q in ["tell me about greenops auditor", "what is agentflow ai", "sign language pipeline"]:
        res = kb.retrieve(test_q, top_k=1)
        print(f"\nQ: {test_q}\nMatch: {res[0][:120]}...")