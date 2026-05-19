from __future__ import annotations

from assistant.orchestrator.router import normalize_text
from assistant.orchestrator.state import OrchestratorPlan
from assistant.rag.retriever import RetrievedChunk


def assess_evidence(
	question: str,
	plan: OrchestratorPlan,
	chunks: list[RetrievedChunk],
	min_score: float,
) -> bool:
	if not chunks:
		return False

	unique_sources = {chunk.source for chunk in chunks}
	max_score = max(chunk.score for chunk in chunks)

	if plan.question_type == "single_doc":
		return True

	risky_markers = [
		"exacto",
		"exacta",
		"tasa exacta",
		"porcentaje exacto",
		"ranking exacto",
		"plazas exactas",
		"numero exacto",
	]
	q = normalize_text(question)
	if any(marker in q for marker in risky_markers):
		if len(unique_sources) < 2:
			return False
		if min_score > 0 and max_score < min_score:
			return False

	if plan.question_type == "unanswerable_risk" and len(unique_sources) < 2:
		return False

	if min_score > 0 and max_score < min_score:
		return False

	return True


def build_context_summary(chunks: list[RetrievedChunk], max_chars: int = 1200) -> str:
	parts: list[str] = []
	for index, chunk in enumerate(chunks, start=1):
		parts.append(f"[{index}] {chunk.source}\n{chunk.text}")
	context = "\n\n".join(parts)
	return context[:max_chars]
