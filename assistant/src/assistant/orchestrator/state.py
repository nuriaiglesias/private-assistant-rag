from __future__ import annotations

"""Data models for orchestrator plans, tool calls, and run results."""

from dataclasses import dataclass, field
from typing import Any, Literal

from assistant.rag.retriever import RetrievedChunk
from assistant.tools.email_tool import EmailDraft

QuestionType = Literal[
    "single_doc",
    "multi_doc",
    "comparative",
    "unanswerable_risk",
]

IntentType = Literal[
    "information",
    "email_action",
]

ActionType = Literal[
    "answer_with_rag",
    "multi_step_rag",
    "comparative_rag",
    "refuse",
    "draft_email",
]


@dataclass(frozen=True)
class OrchestratorPlan:
    """Plan describing how the orchestrator should handle a question."""
    question_type: QuestionType
    intent: IntentType
    action: ActionType
    subqueries: list[str]
    answer_style: str
    requires_multiple_sources: bool = False
    refusal_check_required: bool = True
    requires_user_confirmation: bool = False


@dataclass
class ToolCall:
    """Record of a tool invocation performed during orchestrator execution."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_output_summary: str | None = None


@dataclass
class OrchestratorResult:
    """Structured result returned by the orchestrator pipeline."""
    question: str
    plan: OrchestratorPlan | None
    retrieved_chunks: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]
    answer: str
    refused: bool
    evidence_sufficient: bool
    token_usage: dict[str, int]
    email_draft: EmailDraft | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
