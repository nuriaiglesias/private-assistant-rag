from __future__ import annotations

"""Planner utilities that classify questions and build orchestrator plans."""

from assistant.orchestrator.router import (
    classify_question_type,
    detect_intent,
    normalize_text,
)
from assistant.orchestrator.state import OrchestratorPlan


def build_plan(question: str) -> OrchestratorPlan:
    """Create an orchestrator plan from a user question."""
    question_type = classify_question_type(question)
    intent = detect_intent(question)

    if question_type == "unanswerable_risk":
        return OrchestratorPlan(
            question_type=question_type,
            intent=intent,
            action="refuse",
            subqueries=[question],
            answer_style="refusal",
            requires_multiple_sources=False,
            refusal_check_required=True,
        )

    if intent == "email_action":
        return OrchestratorPlan(
            question_type=question_type,
            intent=intent,
            action="draft_email",
            subqueries=extract_email_information_needs(question),
            answer_style="email_draft",
            requires_multiple_sources=question_type in {"multi_doc", "comparative"},
            requires_user_confirmation=True,
        )

    if question_type == "single_doc":
        return OrchestratorPlan(
            question_type=question_type,
            intent=intent,
            action="answer_with_rag",
            subqueries=[question],
            answer_style="direct",
            requires_multiple_sources=False,
        )

    if question_type == "multi_doc":
        return OrchestratorPlan(
            question_type=question_type,
            intent=intent,
            action="multi_step_rag",
            subqueries=decompose_multi_doc_question(question),
            answer_style="structured",
            requires_multiple_sources=True,
        )

    if question_type == "comparative":
        return OrchestratorPlan(
            question_type=question_type,
            intent=intent,
            action="comparative_rag",
            subqueries=decompose_comparative_question(question),
            answer_style="comparison",
            requires_multiple_sources=True,
        )

    return OrchestratorPlan(
        question_type="single_doc",
        intent=intent,
        action="answer_with_rag",
        subqueries=[question],
        answer_style="direct",
        requires_multiple_sources=False,
    )


def decompose_multi_doc_question(question: str) -> list[str]:
    """Generate subqueries for a question that requires multiple document sources."""
    q = normalize_text(question)

    if "matricularme" in q and "requisitos" in q:
        return [
            "Como puedo matricularme en UNIR?",
            "Que requisitos de acceso se piden para estudiar en UNIR?",
        ]

    if "movilidad" in q and "centros de examenes" in q:
        return [
            "Que opciones de movilidad ofrece UNIR?",
            "Donde estan los centros de examenes de UNIR?",
        ]

    if "orientacion academica" in q and "servicio de solicitudes" in q:
        return [
            "Que servicios cubre la orientacion academica de UNIR?",
            "Para que sirve el servicio de solicitudes de UNIR?",
        ]

    return [question]


def decompose_comparative_question(question: str) -> list[str]:
    """Generate subqueries for comparative questions that involve more than one topic."""
    q = normalize_text(question)

    if "unir alumni" in q and "bolsa de empleo" in q:
        return [
            "Que es UNIR Alumni?",
            "Existe una bolsa de empleo para estudiantes en UNIR?",
        ]

    if "grados online" in q and "postgrados online" in q:
        return [
            "Que son los grados online de UNIR?",
            "Que son los postgrados online de UNIR?",
        ]

    if "facultad de educacion" in q and "facultad de derecho" in q:
        return [
            "Que oferta tiene la Facultad de Educacion de UNIR?",
            "Que oferta tiene la Facultad de Derecho de UNIR?",
        ]

    return [question]


def extract_email_information_needs(question: str) -> list[str]:
    """Extract the information needs from an email-related question for draft composition."""
    q = normalize_text(question)
    subqueries: list[str] = []

    if "beca" in q or "becas" in q:
        subqueries.append("Que becas y ayudas economicas ofrece UNIR?")

    if "requisitos" in q or "acceso" in q:
        subqueries.append("Que requisitos de acceso se piden para estudiar en UNIR?")

    if "matricula" in q or "matricularme" in q:
        subqueries.append("Como puedo matricularme en UNIR?")

    if not subqueries:
        subqueries.append(question)

    return subqueries
