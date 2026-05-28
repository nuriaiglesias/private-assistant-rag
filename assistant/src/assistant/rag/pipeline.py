from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import List

from assistant.core.config import AppConfig, load_config
from assistant.llm.base import ChatMessage
from assistant.llm.factory import build_llm_client
from assistant.observability.phoenix import configure_tracer
from assistant.orchestrator.orchestrator import RAGOrchestrator
from assistant.orchestrator.state import OrchestratorPlan, ToolCall
from assistant.rag.reranker import Reranker
from assistant.rag.retriever import RetrievedChunk, Retriever
from assistant.tools.email_tool import EmailDraft, EmailTool


@dataclass(frozen=True)
class RagResponse:
	answer: str
	sources: List[RetrievedChunk]
	plan: OrchestratorPlan | None = None
	tool_calls: List[ToolCall] | None = None
	refused: bool = False
	evidence_sufficient: bool = True
	email_draft: EmailDraft | None = None


class RagPipeline:
	def __init__(self, config: AppConfig) -> None:
		self._config = config
		self._retriever = Retriever(config)
		self._reranker = Reranker(config)
		self._llm = build_llm_client(config)
		self._prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
		self._tracer = configure_tracer(config)

	def answer(self, question: str, top_k: int = 5) -> RagResponse:
		return self.run_pipeline(
			question,
			top_k=top_k,
			use_reranking=False,
			use_orchestrator=False,
		)

	def run_pipeline(
		self,
		question: str,
		*,
		top_k: int = 5,
		use_reranking: bool = False,
		use_orchestrator: bool = False,
	) -> RagResponse:
		with self._tracer.start_as_current_span("rag.run") as span:
			start_time = time.perf_counter()
			span.set_attribute("rag.question", question)
			span.set_attribute("rag.top_k", top_k)
			span.set_attribute("rag.use_reranking", use_reranking)
			span.set_attribute("rag.use_orchestrator", use_orchestrator)

			if use_orchestrator:
				result = self._run_orchestrator(
					question,
					top_k=top_k,
					use_reranking=use_reranking,
				)
				answer = result.answer
				retrieved_sources = result.retrieved_chunks
				final_sources = result.final_chunks
				token_usage = result.token_usage
				plan = result.plan
				tool_calls = result.tool_calls
				refused = result.refused
				evidence_sufficient = result.evidence_sufficient
				email_draft = result.email_draft
			else:
				candidate_k = self._config.rag_candidate_k if use_reranking else top_k
				retrieved_sources = self.retrieve(
					question,
					top_k=candidate_k,
					use_hybrid=use_reranking,
				)
				final_sources = retrieved_sources
				if use_reranking:
					final_sources = self.rerank(question, retrieved_sources, top_k=top_k)
				if self._should_refuse(final_sources):
					answer = "No dispongo de informacion suficiente en la documentacion recuperada para responder con fiabilidad."
					token_usage = {"input_tokens": 0, "output_tokens": 0}
					final_sources = []
					refused = True
					evidence_sufficient = False
				else:
					answer, token_usage = self.generate_answer(question, final_sources)
					refused = False
					evidence_sufficient = True
				plan = None
				tool_calls = None
				email_draft = None

			latency_seconds = time.perf_counter() - start_time

			self.log_result(
				question,
				retrieved_sources,
				final_sources,
				answer,
				token_usage,
				latency_seconds,
				use_reranking=use_reranking,
				use_orchestrator=use_orchestrator,
				top_k=top_k,
			)

		return RagResponse(
			answer=answer,
			sources=final_sources,
			plan=plan,
			tool_calls=tool_calls,
			refused=refused,
			evidence_sufficient=evidence_sufficient,
			email_draft=email_draft,
		)

	def _should_refuse(self, sources: List[RetrievedChunk]) -> bool:
		min_score = self._config.rag_min_score
		if min_score <= 0:
			return False
		if not sources:
			return True

		max_score = max(chunk.score for chunk in sources)
		return max_score < min_score

	def retrieve(self, question: str, top_k: int = 5, use_hybrid: bool = False) -> List[RetrievedChunk]:
		with self._tracer.start_as_current_span("rag.retrieve") as retrieve_span:
			if use_hybrid:
				sources = self._retriever.retrieve_hybrid(
					question,
					top_k=top_k,
					candidate_k=top_k,
				)
			else:
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
		return sources

	def rerank(
		self,
		question: str,
		sources: List[RetrievedChunk],
		*,
		top_k: int = 5,
	) -> List[RetrievedChunk]:
		with self._tracer.start_as_current_span("rag.rerank") as rerank_span:
			reranked = self._reranker.rerank(question, sources, top_k=top_k)
			rerank_span.set_attribute("rag.reranked", len(reranked))
		return reranked

	def generate_answer(
		self,
		question: str,
		sources: List[RetrievedChunk],
		*,
		answer_style: str = "direct",
	) -> tuple[str, dict[str, int]]:
		context = self._format_context(sources)
		system_prompt = self._read_prompt("system_base.txt")
		prompt_name = "rag/rag_answer.txt"
		if answer_style == "comparison":
			prompt_name = "rag/rag_answer_comparison.txt"
		elif answer_style == "structured":
			prompt_name = "rag/rag_answer_multidoc.txt"
		user_prompt = self._read_prompt(prompt_name).format(
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
			response = self._llm.generate(
				messages,
				temperature=self._config.llm_temperature,
				max_tokens=self._config.llm_max_tokens,
			)
			answer = response.content
			generate_span.set_attribute("rag.answer_chars", len(answer))
			generate_span.set_attribute("rag.answer_preview", answer[:280])

		generate_span.set_attribute("rag.input_tokens", response.input_tokens)
		generate_span.set_attribute("rag.output_tokens", response.output_tokens)

		return answer, {
			"input_tokens": response.input_tokens,
			"output_tokens": response.output_tokens,
		}

	def log_result(
		self,
		question: str,
		retrieved_sources: List[RetrievedChunk],
		final_sources: List[RetrievedChunk],
		answer: str,
		token_usage: dict[str, int],
		latency_seconds: float,
		*,
		use_reranking: bool,
		use_orchestrator: bool,
		top_k: int,
	) -> None:
		with self._tracer.start_as_current_span("rag.result") as result_span:
			result_span.set_attribute("rag.question", question)
			result_span.set_attribute("rag.top_k", top_k)
			result_span.set_attribute("rag.use_reranking", use_reranking)
			result_span.set_attribute("rag.use_orchestrator", use_orchestrator)
			result_span.set_attribute("rag.sources", len(final_sources))
			result_span.set_attribute(
				"rag.source_names",
				", ".join(chunk.source for chunk in final_sources),
			)
			result_span.set_attribute("rag.answer_chars", len(answer))
			result_span.set_attribute(
				"rag.context_chars",
				len(self._format_context(final_sources)),
			)
			result_span.set_attribute("rag.latency_seconds", latency_seconds)
			result_span.set_attribute("rag.input_tokens", token_usage.get("input_tokens", 0))
			result_span.set_attribute("rag.output_tokens", token_usage.get("output_tokens", 0))

		variant = self._variant_name(use_reranking, use_orchestrator)
		sources = [chunk.source for chunk in final_sources]
		scores = [chunk.score for chunk in final_sources]
		result = {
			"question": question,
			"answer": answer,
			"use_reranking": use_reranking,
			"use_orchestrator": use_orchestrator,
			"variant": variant,
			"top_k": top_k,
			"sources": sources,
			"scores": scores,
			"latency_seconds": round(latency_seconds, 4),
			"input_tokens": token_usage.get("input_tokens", 0),
			"output_tokens": token_usage.get("output_tokens", 0),
			"retrieved_chunks": [
				self._chunk_to_dict(chunk) for chunk in retrieved_sources
			],
			"final_chunks": [self._chunk_to_dict(chunk) for chunk in final_sources],
		}
		self._append_result_log(result)

	def _run_orchestrator(
		self,
		question: str,
		*,
		top_k: int,
		use_reranking: bool,
	) -> "OrchestratorResult":
		from assistant.orchestrator.state import OrchestratorResult

		with self._tracer.start_as_current_span("rag.orchestrator") as span:
			steps = ["retrieve", "generate"]
			if use_reranking:
				steps.insert(1, "rerank")
			span.set_attribute("rag.orchestrator_steps", ", ".join(steps))

			candidate_k = self._config.rag_candidate_k if use_reranking else top_k
			orchestrator = RAGOrchestrator(
				retrieve=self.retrieve,
				rerank=self.rerank,
				generate_answer=lambda q, chunks, style: self.generate_answer(q, chunks, answer_style=style),
				min_score=self._config.rag_min_score,
				email_tool=EmailTool(),
			)
			return orchestrator.run(
				question,
				top_k=top_k,
				use_reranking=use_reranking,
				candidate_k=candidate_k,
			)

	def _read_prompt(self, filename: str) -> str:
		path = self._prompts_dir / filename
		return path.read_text(encoding="utf-8")

	def _format_context(self, sources: List[RetrievedChunk]) -> str:
		lines: List[str] = []
		for index, chunk in enumerate(sources, start=1):
			lines.append(f"[{index}]\n{chunk.text}")
		return "\n\n".join(lines)

	def _variant_name(self, use_reranking: bool, use_orchestrator: bool) -> str:
		if use_reranking and use_orchestrator:
			return "D_rag_reranking_orchestrator"
		if use_orchestrator:
			return "C_rag_orchestrator"
		if use_reranking:
			return "B_rag_reranking"
		return "A_baseline_rag"

	def _chunk_to_dict(self, chunk: RetrievedChunk) -> dict[str, object]:
		return {
			"text": chunk.text,
			"source": chunk.source,
			"chunk_index": chunk.chunk_index,
			"score": chunk.score,
		}


	def _append_result_log(self, result: dict[str, object]) -> None:
		log_path = self._results_log_path()
		log_path.parent.mkdir(parents=True, exist_ok=True)
		with log_path.open("a", encoding="utf-8") as file:
			file.write(json.dumps(result, ensure_ascii=True))
			file.write("\n")

	def _results_log_path(self) -> Path:
		root_dir = Path(__file__).resolve().parents[3]
		return root_dir / "data" / "processed" / "pipeline_runs.jsonl"


def run_pipeline(
	question: str,
	*,
	use_reranking: bool = False,
	use_orchestrator: bool = False,
	top_k: int = 5,
	config: AppConfig | None = None,
) -> RagResponse:
	active_config = config or load_config()
	pipeline = RagPipeline(active_config)
	return pipeline.run_pipeline(
		question,
		top_k=top_k,
		use_reranking=use_reranking,
		use_orchestrator=use_orchestrator,
	)
