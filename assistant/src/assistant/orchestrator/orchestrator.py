from __future__ import annotations

"""Core orchestrator logic for routing questions, retrieving evidence, and generating answers."""

from typing import Callable

from assistant.orchestrator.evidence import assess_evidence, build_context_summary
from assistant.orchestrator.planner import build_plan
from assistant.orchestrator.state import OrchestratorResult, ToolCall
from assistant.rag.retriever import RetrievedChunk
from assistant.tools.email_tool import EmailTool


RetrieveFn = Callable[[str, int, bool], list[RetrievedChunk]]
RerankFn = Callable[[str, list[RetrievedChunk], int], list[RetrievedChunk]]
GenerateFn = Callable[[str, list[RetrievedChunk], str], tuple[str, dict[str, int]]]


class RAGOrchestrator:
    """RAG orchestrator that executes retrieval, reranking and answer composition."""

    def __init__(
        self,
        retrieve: RetrieveFn,
        rerank: RerankFn,
        generate_answer: GenerateFn,
        min_score: float,
        email_tool: EmailTool | None = None,
    ) -> None:
        self._retrieve = retrieve
        self._rerank = rerank
        self._generate_answer = generate_answer
        self._min_score = min_score
        self._email_tool = email_tool or EmailTool()

    def run(
        self,
        question: str,
        *,
        top_k: int,
        use_reranking: bool,
        candidate_k: int,
    ) -> OrchestratorResult:
        """Run the orchestrator for a given question and return a structured result."""
        clarification_needed, clarification_question = self._needs_clarification(question)
        if clarification_needed:
            plan = build_plan(question)
            if plan.question_type == "single_doc" and plan.intent == "information":
                result = self._run_single_doc(
                    question=question,
                    plan=plan,
                    top_k=top_k,
                    use_reranking=use_reranking,
                    candidate_k=candidate_k,
                )
            elif plan.question_type == "multi_doc" and plan.intent == "information":
                result = self._run_multi_doc(
                    question=question,
                    plan=plan,
                    top_k=top_k,
                    use_reranking=use_reranking,
                    candidate_k=candidate_k,
                )
            else:
                return OrchestratorResult(
                    question=question,
                    plan=None,
                    retrieved_chunks=[],
                    final_chunks=[],
                    answer=clarification_question,
                    refused=False,
                    evidence_sufficient=False,
                    token_usage={"input_tokens": 0, "output_tokens": 0},
                    email_draft=None,
                    tool_calls=[],
                )
            respuesta = result.answer.strip()
            if not respuesta.endswith("\n"):
                respuesta += "\n"
            respuesta += f"\n{clarification_question}"
            return OrchestratorResult(
                question=question,
                plan=result.plan,
                retrieved_chunks=result.retrieved_chunks,
                final_chunks=result.final_chunks,
                answer=respuesta,
                refused=False,
                evidence_sufficient=result.evidence_sufficient,
                token_usage=result.token_usage,
                email_draft=result.email_draft,
                tool_calls=result.tool_calls,
            )

        plan = build_plan(question)
        if plan.question_type == "single_doc" and plan.intent == "information":
            return self._run_single_doc(
                question=question,
                plan=plan,
                top_k=top_k,
                use_reranking=use_reranking,
                candidate_k=candidate_k,
            )
        if plan.question_type == "multi_doc" and plan.intent == "information":
            return self._run_multi_doc(
                question=question,
                plan=plan,
                top_k=top_k,
                use_reranking=use_reranking,
                candidate_k=candidate_k,
            )

        retrieved_chunks: list[RetrievedChunk] = []
        tool_calls: list[ToolCall] = []
        fetch_k = candidate_k if use_reranking else top_k
        for subquery in plan.subqueries:
            chunks = self._retrieve(subquery, fetch_k, use_reranking)
            retrieved_chunks.extend(chunks)
            tool_calls.append(
                ToolCall(
                    tool_name="retriever.search",
                    tool_input={"query": subquery, "top_k": fetch_k, "hybrid": use_reranking},
                    tool_output_summary=f"{len(chunks)} chunks",
                )
            )

        retrieved_chunks = _deduplicate_chunks(retrieved_chunks)
        final_chunks = _select_final_chunks(
            question,
            retrieved_chunks,
            use_reranking=use_reranking,
            top_k=top_k,
            rerank=self._rerank,
            tool_calls=tool_calls,
        )

        evidence_sufficient = assess_evidence(
            question=question,
            plan=plan,
            chunks=final_chunks,
            min_score=self._min_score,
        )

        if plan.action == "draft_email":
            if not evidence_sufficient:
                return _refusal_result(
                    question=question,
                    plan=plan,
                    retrieved_chunks=retrieved_chunks,
                    final_chunks=final_chunks,
                    tool_calls=tool_calls,
                )

            context_summary = build_context_summary(final_chunks)
            draft = self._email_tool.draft_email(
                user_request=question,
                supporting_context=context_summary,
            )
            tool_calls.append(
                ToolCall(
                    tool_name="email.draft_email",
                    tool_input={"requires_confirmation": True},
                    tool_output_summary="draft created",
                )
            )
            answer = (
                "He preparado un borrador de correo con la evidencia recuperada. "
                "Revisalo antes de enviar."
            )
            return OrchestratorResult(
                question=question,
                plan=plan,
                retrieved_chunks=retrieved_chunks,
                final_chunks=final_chunks,
                answer=answer,
                refused=False,
                evidence_sufficient=True,
                token_usage={"input_tokens": 0, "output_tokens": 0},
                email_draft=draft,
                tool_calls=tool_calls,
            )

        if plan.action == "refuse" or not evidence_sufficient:
            return _refusal_result(
                question=question,
                plan=plan,
                retrieved_chunks=retrieved_chunks,
                final_chunks=final_chunks,
                tool_calls=tool_calls,
            )

        answer, token_usage = self._generate_answer(question, final_chunks, plan.answer_style)
        return OrchestratorResult(
            question=question,
            plan=plan,
            retrieved_chunks=retrieved_chunks,
            final_chunks=final_chunks,
            answer=answer,
            refused=False,
            evidence_sufficient=True,
            token_usage=token_usage,
            tool_calls=tool_calls,
        )

    def _needs_clarification(self, question: str) -> tuple[bool, str]:
        """Return whether the question needs a follow-up clarification."""
        q = question.strip().lower()
        if len(q) < 10:
            return True, "¿Podrías especificar con más detalle tu consulta?"

        generic_starts = [
            "cuanto cuesta",
            "precio",
            "informacion sobre",
            "quiero saber",
            "detalles de",
            "explica",
            "dime sobre",
            "ayuda con",
        ]
        for start in generic_starts:
            if q.startswith(start):
                return True, (
                    "¿Podrías concretar a qué te refieres exactamente? Por ejemplo, "
                    "¿a qué programa, trámite o tema concreto?"
                )

        if "matricula" in q or "matricularme" in q:
            if not ("grado" in q or "master" in q or "doctorado" in q):
                return True, (
                    "¿Te refieres a matrícula de grado, máster u otro tipo de estudio? "
                    "¿De qué programa concreto?"
                )

        if "beca" in q and not ("tipo" in q or "universidad" in q or "oficial" in q):
            return True, (
                "¿Quieres información sobre becas propias de la universidad, "
                "becas oficiales o ambas?"
            )

        return False, ""

    def _run_multi_doc(
        self,
        question: str,
        plan,
        *,
        top_k: int,
        use_reranking: bool,
        candidate_k: int,
    ) -> OrchestratorResult:
        """Retrieve and answer a multi-document question using subqueries."""
        tool_calls: list[ToolCall] = []
        subqueries = _build_subquery_list(question, plan.subqueries)
        per_query_k = max(1, top_k // max(1, len(subqueries)))
        per_query_k = max(per_query_k, 2) if len(subqueries) > 1 else top_k

        grouped_chunks: list[tuple[str, list[RetrievedChunk]]] = []
        for subquery in subqueries:
            fetch_k = candidate_k if use_reranking else per_query_k
            chunks = self._retrieve(subquery, fetch_k, use_reranking)
            tool_calls.append(
                ToolCall(
                    tool_name="retriever.search",
                    tool_input={"query": subquery, "top_k": fetch_k, "hybrid": use_reranking},
                    tool_output_summary=f"{len(chunks)} chunks",
                )
            )
            if use_reranking:
                chunks = self._rerank(subquery, chunks, top_k=per_query_k)
                tool_calls.append(
                    ToolCall(
                        tool_name="retriever.rerank",
                        tool_input={"top_k": per_query_k},
                        tool_output_summary=f"rerank top {per_query_k}",
                    )
                )
            grouped_chunks.append((subquery, _deduplicate_chunks(chunks)))

        flattened = _flatten_grouped_chunks(grouped_chunks)
        evidence_sufficient = bool(flattened)
        if not evidence_sufficient:
            return _refusal_result(
                question=question,
                plan=plan,
                retrieved_chunks=flattened,
                final_chunks=flattened,
                tool_calls=tool_calls,
            )

        answer, token_usage = self._generate_answer(question, flattened, plan.answer_style)
        return OrchestratorResult(
            question=question,
            plan=plan,
            retrieved_chunks=flattened,
            final_chunks=flattened,
            answer=answer,
            refused=False,
            evidence_sufficient=True,
            token_usage=token_usage,
            tool_calls=tool_calls,
        )

    def _run_single_doc(
        self,
        question: str,
        plan,
        *,
        top_k: int,
        use_reranking: bool,
        candidate_k: int,
    ) -> OrchestratorResult:
        """Retrieve and answer a single-document question."""
        tool_calls: list[ToolCall] = []
        fetch_k = candidate_k if use_reranking else top_k
        retrieved_chunks = self._retrieve(question, fetch_k, use_reranking)
        tool_calls.append(
            ToolCall(
                tool_name="retriever.search",
                tool_input={"query": question, "top_k": fetch_k, "hybrid": use_reranking},
                tool_output_summary=f"{len(retrieved_chunks)} chunks",
            )
        )

        final_chunks = retrieved_chunks
        if use_reranking:
            tool_calls.append(
                ToolCall(
                    tool_name="retriever.rerank",
                    tool_input={"top_k": top_k},
                    tool_output_summary=f"rerank top {top_k}",
                )
            )
            final_chunks = self._rerank(question, retrieved_chunks, top_k=top_k)
        else:
            final_chunks = sorted(retrieved_chunks, key=lambda item: item.score, reverse=True)[:top_k]

        if _should_refuse_single_doc(final_chunks, self._min_score):
            return _refusal_result(
                question=question,
                plan=plan,
                retrieved_chunks=retrieved_chunks,
                final_chunks=final_chunks,
                tool_calls=tool_calls,
            )

        answer, token_usage = self._generate_answer(question, final_chunks, "direct")
        return OrchestratorResult(
            question=question,
            plan=plan,
            retrieved_chunks=retrieved_chunks,
            final_chunks=final_chunks,
            answer=answer,
            refused=False,
            evidence_sufficient=True,
            token_usage=token_usage,
            tool_calls=tool_calls,
        )


def _select_final_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    use_reranking: bool,
    top_k: int,
    rerank: RerankFn,
    tool_calls: list[ToolCall],
) -> list[RetrievedChunk]:
    """Select the top final chunks, optionally reranking them first."""
    if not chunks:
        return []

    if use_reranking:
        tool_calls.append(
            ToolCall(
                tool_name="retriever.rerank",
                tool_input={"top_k": top_k},
                tool_output_summary=f"rerank top {top_k}",
            )
        )
        return rerank(question, chunks, top_k)

    sorted_chunks = sorted(chunks, key=lambda item: item.score, reverse=True)
    return sorted_chunks[:top_k]


def _deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove duplicate chunks keeping the highest scoring entry per source and index."""
    best: dict[tuple[str, int, str], RetrievedChunk] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.chunk_index, chunk.text)
        if key not in best or chunk.score > best[key].score:
            best[key] = chunk
    return list(best.values())


def _refusal_result(
    question: str,
    plan,
    retrieved_chunks: list[RetrievedChunk],
    final_chunks: list[RetrievedChunk],
    tool_calls: list[ToolCall],
) -> OrchestratorResult:
    """Generate a standardized refusal result for insufficient evidence."""
    answer = (
        "No dispongo de informacion suficiente en la documentacion recuperada "
        "para responder con fiabilidad."
    )
    return OrchestratorResult(
        question=question,
        plan=plan,
        retrieved_chunks=retrieved_chunks,
        final_chunks=[],
        answer=answer,
        refused=True,
        evidence_sufficient=False,
        token_usage={"input_tokens": 0, "output_tokens": 0},
        tool_calls=tool_calls,
    )


def _build_subquery_list(question: str, subqueries: list[str]) -> list[str]:
    """Ensure subqueries preserve the original question and avoid duplicates."""
    merged = [question]
    for subquery in subqueries:
        if subquery not in merged:
            merged.append(subquery)
    return merged


def _flatten_grouped_chunks(
    grouped: list[tuple[str, list[RetrievedChunk]]],
) -> list[RetrievedChunk]:
    """Flatten grouped subquery chunks into a single list with subquery annotations."""
    flattened: list[RetrievedChunk] = []
    for index, (subquery, chunks) in enumerate(grouped, start=1):
        prefix = f"Subconsulta {index}: {subquery}\n"
        for chunk in chunks:
            flattened.append(
                RetrievedChunk(
                    text=f"{prefix}{chunk.text}",
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                )
            )
    return flattened


def _should_refuse_single_doc(chunks: list[RetrievedChunk], min_score: float) -> bool:
    """Decide whether a single-document retrieval should be refused for low confidence."""
    if min_score <= 0:
        return False
    if not chunks:
        return True
    max_score = max(chunk.score for chunk in chunks)
    return max_score < min_score
