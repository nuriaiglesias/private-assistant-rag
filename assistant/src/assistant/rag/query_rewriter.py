from __future__ import annotations

"""LLM-based query rewriting to improve retrieval quality."""

from assistant.llm.base import ChatMessage, LLMClient


_SYSTEM_PROMPT = (
    "Eres un asistente especializado en reformulación de consultas para recuperación "
    "de información en documentación universitaria en español.\n"
    "Tu tarea es reescribir la pregunta del usuario para mejorar la recuperación de documentos "
    "relevantes de un corpus institucional.\n\n"
    "Reglas:\n"
    "- Mantén el mismo idioma (español).\n"
    "- Hazla más específica y orientada a la búsqueda documental.\n"
    "- Elimina marcadores conversacionales ('¿podría decirme...?', '¿sabes si...?').\n"
    "- Expande abreviaturas cuando sea útil.\n"
    "- Añade términos del dominio universitario si refuerzan la consulta.\n"
    "- Devuelve ÚNICAMENTE la consulta reformulada, sin explicaciones."
)


class QueryRewriter:
    """Rewrites a user query into a retrieval-optimised form using the LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def rewrite(self, question: str) -> str:
        """Return a retrieval-optimised rewrite of *question*.

        Falls back to the original question on any error.
        """
        user_prompt = f"Pregunta original: {question}\nConsulta reformulada:"
        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]
        try:
            response = self._llm.generate(messages, temperature=0.1, max_tokens=128)
            rewritten = response.content.strip()
            return rewritten if rewritten else question
        except Exception:
            return question
