from __future__ import annotations
from typing import Dict, List, Optional

class RAGAnswerer:
    """
    Wrapper around FLAN-T5 to generate natural, detailed summaries
    that incorporate dates and sources without making lists.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_context_chars: int = 2000,
        max_new_tokens: int = 250,
        temperature: float = 0.3,       
        device: Optional[str] = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for RAGAnswerer. "
                "Install it via `pip install transformers sentencepiece`."
            ) from exc

        task = "text2text-generation"
        self.generator = pipeline(task, model=model_name, device=device)
        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, question: str, documents: List[Dict[str, Optional[str]]]) -> str:
        if not question:
            return "Please provide a question."
        if not documents:
            return "No supporting documents were provided."

        prompt = self._build_prompt(question, documents)
        
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=1.3,     # Slightly higher to absolutely kill loops
            temperature=self.temperature,
            do_sample=True,
        )
        return outputs[0]["generated_text"].strip()

    def _build_prompt(self, question: str, documents: List[Dict[str, Optional[str]]]) -> str:
        blocks = []
        for idx, doc in enumerate(documents, start=1):
            title = (doc.get("title") or "Untitled").strip()
            content = (doc.get("text") or doc.get("snippet") or "").strip()
            content = content[:800] 
            blocks.append(f"Article {idx}: {content}")

        context_text = "\n\n".join(blocks)
        context_text = context_text[:self.max_context_chars]

        # --- THE FIX: NATURAL SYNTHESIS PROMPT ---
        # We ask it to "Combine" the info. This is easier for the model than a list.
        prompt = (
            "Read the news articles below and write a single, detailed paragraph answering the question. "
            "You MUST mention specific dates and organization names if they are in the text.\n\n"
            f"--- ARTICLES ---\n{context_text}\n\n"
            f"Question: {question}\n"
            "Answer (include Who, When, and Where):"
        )
        return prompt