from __future__ import annotations

"""Pipeline orchestration for retrieval-augmented generation and tracing."""

from dataclasses import dataclass
import json
import os
from datetime import datetime
from pathlib import Path
import time
from typing import List

from assistant.core.config import AppConfig, load_config
from assistant.core.paths import PIPELINE_RUNS_PATH
from assistant.llm.base import ChatMessage
from assistant.llm.factory import build_llm_client
from assistant.observability.phoenix import configure_tracer
from assistant.orchestrator.react_orchestrator import ReactOrchestrator
from assistant.orchestrator.state import OrchestratorPlan, ToolCall
from assistant.rag.query_rewriter import QueryRewriter
from assistant.rag.reranker import Reranker
from assistant.rag.retriever import RetrievedChunk, Retriever
from assistant.tools.email_tool import EmailDraft, EmailTool


@dataclass(frozen=True)
class RagResponse:
    """Structured response object returned by the RAG pipeline."""
    answer: str
    sources: List[RetrievedChunk]
    plan: OrchestratorPlan | None = None
    tool_calls: List[ToolCall] | None = None
    refused: bool = False
    evidence_sufficient: bool = True
    email_draft: EmailDraft | None = None
    token_usage: dict[str, int] | None = None
    latency_seconds: float | None = None
    variant: str | None = None
    retrieved_count: int = 0


class RagPipeline:
    """Coordinator for retrieval, reranking, generation, and observability."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._retriever = Retriever(config)
        self._reranker = Reranker(config)
        self._llm = build_llm_client(config)
        self._query_rewriter: QueryRewriter | None = None
        self._prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
        self._tracer = configure_tracer(config)
        self._pipeline_run_path = self._resolve_pipeline_run_path()

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
        use_lexical: bool = False,
        use_hybrid: bool = False,
        use_reranking: bool = False,
        use_orchestrator: bool = False,
        use_query_rewriting: bool = False,
    ) -> RagResponse:
        """Execute the RAG pipeline and return a structured response."""
        # Reranking always uses hybrid candidates.
        use_hybrid = use_hybrid or use_reranking

        with self._tracer.start_as_current_span("rag.run") as span:
            start_time = time.perf_counter()

            span.set_attribute("rag.question", question)
            span.set_attribute("rag.top_k", top_k)
            span.set_attribute("rag.use_lexical", use_lexical)
            span.set_attribute("rag.use_hybrid", use_hybrid)
            span.set_attribute("rag.use_reranking", use_reranking)
            span.set_attribute("rag.use_orchestrator", use_orchestrator)
            span.set_attribute("rag.use_query_rewriting", use_query_rewriting)

            if use_orchestrator:
                result = self._run_orchestrator(
                    question,
                    top_k=top_k,
                    use_reranking=use_reranking,
                )

                answer = result.answer
                retrieved_sources = result.retrieved_chunks
                final_sources = result.final_chunks
                token_usage = result.token_usage or {
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                plan = result.plan
                tool_calls = result.tool_calls
                refused = result.refused
                evidence_sufficient = result.evidence_sufficient
                email_draft = result.email_draft

            else:
                # Query rewriting: produce a retrieval-optimised version of the question.
                # The rewritten query is used only for retrieval; generation always uses
                # the original question so the answer stays relevant to what the user asked.
                retrieval_question = question
                if use_query_rewriting:
                    with self._tracer.start_as_current_span("rag.query_rewrite") as rw_span:
                        retrieval_question = self._get_query_rewriter().rewrite(question)
                        rw_span.set_attribute("rag.original_question", question)
                        rw_span.set_attribute("rag.rewritten_question", retrieval_question)

                candidate_k = self._config.rag_candidate_k if use_hybrid else top_k

                retrieved_sources = self.retrieve(
                    retrieval_question,
                    top_k=candidate_k,
                    use_lexical=use_lexical,
                    use_hybrid=use_hybrid,
                )

                final_sources = retrieved_sources

                if use_reranking:
                    final_sources = self.rerank(
                        retrieval_question,
                        retrieved_sources,
                        top_k=top_k,
                    )

                if self._should_refuse(final_sources):
                    answer = (
                        "No dispongo de información suficiente en la documentación "
                        "recuperada para responder con fiabilidad."
                    )
                    token_usage = {"input_tokens": 0, "output_tokens": 0}
                    final_sources = []
                    refused = True
                    evidence_sufficient = False
                else:
                    answer, token_usage = self.generate_answer(
                        question,
                        final_sources,
                    )
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
                use_lexical=use_lexical,
                use_hybrid=use_hybrid,
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
            token_usage=token_usage,
            latency_seconds=latency_seconds,
            variant=self._variant_name(
                use_lexical,
                use_hybrid,
                use_reranking,
                use_orchestrator,
            ),
            retrieved_count=len(retrieved_sources),
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        use_hybrid: bool = False,
        *,
        use_lexical: bool = False,
    ) -> List[RetrievedChunk]:
        with self._tracer.start_as_current_span("rag.retrieve") as retrieve_span:
            if use_lexical and not use_hybrid:
                retrieve_span.set_attribute("rag.retrieve_mode", "lexical")
                sources = self._retriever.retrieve_lexical(
                    question,
                    top_k=top_k,
                )

            elif use_hybrid:
                retrieve_span.set_attribute("rag.retrieve_mode", "hybrid")
                sources = self._retriever.retrieve_hybrid(
                    question,
                    top_k=top_k,
                    candidate_k=top_k,
                )

            else:
                retrieve_span.set_attribute("rag.retrieve_mode", "semantic")
                sources = self._retriever.retrieve(
                    question,
                    top_k=top_k,
                )

            retrieve_span.set_attribute("rag.question", question)
            retrieve_span.set_attribute("rag.sources", len(sources))
            retrieve_span.set_attribute(
                "rag.source_names",
                ", ".join(chunk.source for chunk in sources),
            )
            retrieve_span.set_attribute(
                "rag.source_scores",
                ", ".join(f"{chunk.score:.4f}" for chunk in sources),
            )
            for i, chunk in enumerate(sources):
                retrieve_span.set_attribute(f"rag.chunk_{i+1}.source", chunk.source)
                retrieve_span.set_attribute(f"rag.chunk_{i+1}.score", chunk.score)
                retrieve_span.set_attribute(f"rag.chunk_{i+1}.text", chunk.text[:300])

        return sources

    def rerank(
        self,
        question: str,
        sources: List[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        with self._tracer.start_as_current_span("rag.rerank") as rerank_span:
            rerank_span.set_attribute(
                "rag.reranker_model",
                self._config.reranker_model,
            )

            reranked = self._reranker.rerank(
                question,
                sources,
                top_k=top_k,
            )

            rerank_span.set_attribute("rag.reranked", len(reranked))
            rerank_span.set_attribute(
                "rag.rerank_scores",
                ", ".join(
                    f"{chunk.rerank_score:.4f}"
                    for chunk in reranked
                    if chunk.rerank_score is not None
                ),
            )

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
            generate_span.set_attribute("rag.question", question)
            generate_span.set_attribute("rag.context", context[:2000])
            generate_span.set_attribute("rag.num_sources", len(sources))

            response = self._llm.generate(
                messages,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )

            answer = response.content

            generate_span.set_attribute("rag.answer", answer)
            generate_span.set_attribute("rag.answer_chars", len(answer))
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
        use_lexical: bool,
        use_hybrid: bool,
        use_reranking: bool,
        use_orchestrator: bool,
        top_k: int,
    ) -> None:
        variant = self._variant_name(
            use_lexical,
            use_hybrid,
            use_reranking,
            use_orchestrator,
        )

        with self._tracer.start_as_current_span("rag.result") as result_span:
            result_span.set_attribute("rag.question", question)
            result_span.set_attribute("rag.top_k", top_k)
            result_span.set_attribute("rag.variant", variant)
            result_span.set_attribute("rag.use_lexical", use_lexical)
            result_span.set_attribute("rag.use_hybrid", use_hybrid)
            result_span.set_attribute("rag.use_reranking", use_reranking)
            result_span.set_attribute("rag.use_orchestrator", use_orchestrator)
            result_span.set_attribute("rag.question", question)
            result_span.set_attribute("rag.answer", answer)
            result_span.set_attribute("rag.sources", len(final_sources))
            result_span.set_attribute(
                "rag.source_names",
                ", ".join(chunk.source for chunk in final_sources),
            )
            result_span.set_attribute(
                "rag.source_scores",
                ", ".join(f"{chunk.score:.4f}" for chunk in final_sources),
            )
            result_span.set_attribute(
                "rag.source_texts",
                " || ".join(chunk.text[:200] for chunk in final_sources),
            )
            result_span.set_attribute("rag.answer_chars", len(answer))
            result_span.set_attribute(
                "rag.context_chars",
                len(self._format_context(final_sources)),
            )
            result_span.set_attribute("rag.latency_seconds", latency_seconds)
            result_span.set_attribute(
                "rag.input_tokens",
                token_usage.get("input_tokens", 0),
            )
            result_span.set_attribute(
                "rag.output_tokens",
                token_usage.get("output_tokens", 0),
            )

        result = {
            "question": question,
            "answer": answer,
            "variant": variant,
            "top_k": top_k,
            "use_lexical": use_lexical,
            "use_hybrid": use_hybrid,
            "use_reranking": use_reranking,
            "use_orchestrator": use_orchestrator,
            "sources": [chunk.source for chunk in final_sources],
            "scores": [chunk.score for chunk in final_sources],
            "rerank_scores": [
                chunk.rerank_score
                for chunk in final_sources
                if chunk.rerank_score is not None
            ],
            "latency_seconds": round(latency_seconds, 4),
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "retrieved_chunks": [
                self._chunk_to_dict(chunk)
                for chunk in retrieved_sources
            ],
            "final_chunks": [
                self._chunk_to_dict(chunk)
                for chunk in final_sources
            ],
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
            steps = ["reason", "retrieve", "generate"]
            if use_reranking:
                steps.insert(2, "rerank")
            span.set_attribute("rag.orchestrator_steps", ", ".join(steps))
            span.set_attribute("rag.orchestrator_type", "react")

            candidate_k = self._config.rag_candidate_k if use_reranking else top_k

            orchestrator = ReactOrchestrator(
                retrieve=self.retrieve,
                rerank=self.rerank,
                generate_answer=lambda question_text, chunks, style: (
                    self.generate_answer(question_text, chunks, answer_style=style)
                ),
                min_score=self._config.rag_min_score,
                llm_client=self._llm,
                email_tool=EmailTool(),
            )

            return orchestrator.run(
                question,
                top_k=top_k,
                use_reranking=use_reranking,
                candidate_k=candidate_k,
            )

    def _get_query_rewriter(self) -> QueryRewriter:
        if self._query_rewriter is None:
            self._query_rewriter = QueryRewriter(self._llm)
        return self._query_rewriter

    def _should_refuse(self, sources: List[RetrievedChunk]) -> bool:
        min_score = self._config.rag_min_score

        if min_score <= 0:
            return False

        if not sources:
            return True

        max_score = max(chunk.score for chunk in sources)
        return max_score < min_score

    def _read_prompt(self, filename: str) -> str:
        path = self._prompts_dir / filename
        return path.read_text(encoding="utf-8")

    def _format_context(self, sources: List[RetrievedChunk]) -> str:
        lines: List[str] = []

        for index, chunk in enumerate(sources, start=1):
            lines.append(
                "\n".join(
                    [
                        f"[{index}]",
                        f"Fuente: {chunk.source}",
                        f"Título: {chunk.title}",
                        f"URL: {chunk.url}",
                        f"Document ID: {chunk.document_id}",
                        f"Chunk ID: {chunk.chunk_id}",
                        f"Chunk index: {chunk.chunk_index}",
                        "",
                        chunk.text,
                    ]
                )
            )

        return "\n\n".join(lines)

    def _variant_name(
        self,
        use_lexical: bool,
        use_hybrid: bool,
        use_reranking: bool,
        use_orchestrator: bool,
    ) -> str:
        if use_orchestrator and use_reranking:
            return "D_orchestrator_hybrid_rerank"

        if use_orchestrator:
            return "C_orchestrator"

        if use_hybrid and use_reranking:
            return "B_hybrid_rerank"

        if use_hybrid:
            return "B_hybrid"

        if use_lexical:
            return "A_lexical"

        return "A_semantic"

    def _chunk_to_dict(self, chunk: RetrievedChunk) -> dict[str, object]:
        return {
            "text": chunk.text,
            "source": chunk.source,
            "title": chunk.title,
            "url": chunk.url,
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "score": chunk.score,
            "rerank_score": chunk.rerank_score,
        }

    def _append_result_log(self, result: dict[str, object]) -> None:
        log_path = self._pipeline_run_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False))
            file.write("\n")

    def _resolve_pipeline_run_path(self) -> Path:
        run_id = os.getenv("PIPELINE_RUN_ID")
        if run_id:
            filename = f"pipeline_runs_{run_id}.jsonl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_runs_{timestamp}.jsonl"

        return PIPELINE_RUNS_PATH.parent / filename


def run_pipeline(
    question: str,
    *,
    use_lexical: bool = False,
    use_hybrid: bool = False,
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
        use_lexical=use_lexical,
        use_hybrid=use_hybrid,
        use_reranking=use_reranking,
        use_orchestrator=use_orchestrator,
    )