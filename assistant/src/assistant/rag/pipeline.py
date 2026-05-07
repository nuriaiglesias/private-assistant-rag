from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from assistant.core.config import AppConfig
from assistant.llm.base import ChatMessage
from assistant.llm.factory import build_llm_client
from assistant.observability.phoenix import configure_tracer
from assistant.rag.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RagResponse:
	answer: str
	sources: List[RetrievedChunk]


class RagPipeline:
	def __init__(self, config: AppConfig) -> None:
		self._config = config
		self._retriever = Retriever(config)
		self._llm = build_llm_client(config)
		self._prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
		self._tracer = configure_tracer(config)

	def answer(self, question: str, top_k: int = 5) -> RagResponse:
		with self._tracer.start_as_current_span("rag.answer") as span:
			span.set_attribute("rag.question", question)
			span.set_attribute("rag.top_k", top_k)

			with self._tracer.start_as_current_span("rag.retrieve") as retrieve_span:
				sources = self._retriever.retrieve(question, top_k=top_k)
				retrieve_span.set_attribute("rag.sources", len(sources))
				retrieve_span.set_attribute(
					"rag.source_names",
					", ".join(chunk.source for chunk in sources),
				)
				retrieve_span.set_attribute(
					"rag.source_scores",
					", ".join(f"{chunk.score:.4f}" for chunk in sources),
				)

			context = self._format_context(sources)
			span.set_attribute("rag.context_chars", len(context))
			system_prompt = self._read_prompt("system_base.txt")
			user_prompt = self._read_prompt("rag_answer.txt").format(
				question=question,
				context=context,
			)

			messages = [
				ChatMessage(role="system", content=system_prompt),
				ChatMessage(role="user", content=user_prompt),
			]
			with self._tracer.start_as_current_span("rag.generate") as generate_span:
				generate_span.set_attribute("rag.prompt_chars", len(user_prompt))
				generate_span.set_attribute("rag.llm_provider", self._config.llm_provider)
				generate_span.set_attribute("rag.llm_model", self._config.llm_model)
				answer = self._llm.generate(
					messages,
					temperature=self._config.llm_temperature,
					max_tokens=self._config.llm_max_tokens,
				)
				generate_span.set_attribute("rag.answer_chars", len(answer))
				generate_span.set_attribute("rag.answer_preview", answer[:280])

		return RagResponse(answer=answer, sources=sources)

	def _read_prompt(self, filename: str) -> str:
		path = self._prompts_dir / filename
		return path.read_text(encoding="utf-8")

	def _format_context(self, sources: List[RetrievedChunk]) -> str:
		lines: List[str] = []
		for index, chunk in enumerate(sources, start=1):
			lines.append(
				f"[{index}] source={chunk.source} chunk={chunk.chunk_index}\n{chunk.text}"
			)
		return "\n\n".join(lines)
