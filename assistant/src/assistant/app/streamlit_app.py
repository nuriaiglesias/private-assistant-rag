from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import requests
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.rag.pipeline import RagPipeline


@st.cache_resource
def _get_pipeline() -> RagPipeline:
    config = load_config()
    return RagPipeline(config)


def _load_styles() -> str:
    style_path = Path(__file__).resolve().parent / "styles.css"
    return style_path.read_text(encoding="utf-8")


def _inject_styles() -> None:
    st.markdown(f"<style>{_load_styles()}</style>", unsafe_allow_html=True)


def _variant_to_flags(variant: str, orchestrator_rerank: bool) -> tuple[bool, bool, bool, bool]:
    if variant == "A_semantic_llm":
        # Lightweight path: BM25 lexical retrieval + LLM generation.
        return True, False, False, False
    if variant == "B_hybrid_llm":
        return False, True, False, False
    if variant == "C_hybrid_rerank_llm":
        return False, True, True, False
    return False, True, orchestrator_rerank, True


def _safe_get(url: str, timeout: float = 1.4) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code < 500
    except requests.RequestException:
        return False


def _service_status(config) -> list[tuple[str, bool, str]]:
    statuses: list[tuple[str, bool, str]] = []

    qdrant_ok = _safe_get(f"{config.qdrant_url}/collections")
    statuses.append(("Qdrant", qdrant_ok, f"{config.qdrant_url}/collections"))

    if config.llm_provider.lower() == "ollama":
        llm_ok = _safe_get(f"{config.llm_base_url}/api/tags")
        statuses.append(("Ollama", llm_ok, f"{config.llm_base_url}/api/tags"))
    else:
        llm_ok = _safe_get(config.llm_base_url)
        statuses.append((config.llm_provider.upper(), llm_ok, config.llm_base_url))

    phoenix_url = _get_phoenix_url(config)
    phoenix_ok = _safe_get(phoenix_url)
    statuses.append(("Phoenix UI", phoenix_ok, phoenix_url))

    return statuses


def _get_phoenix_url(config) -> str:
    return f"http://{config.phoenix_host}:{config.phoenix_ui_port}"


def _render_status_badge(name: str, is_ok: bool, url: str) -> None:
    dot_class = "dot-ok" if is_ok else "dot-bad"
    label = "up" if is_ok else "down"
    st.markdown(
        (
            f"<span class='status-dot {dot_class}'></span>"
            f"<b>{name}</b> <span class='tiny-note'>({label})</span><br/>"
            f"<span class='tiny-note'>{url}</span>"
        ),
        unsafe_allow_html=True,
    )


def _tool_calls_to_dict(tool_calls: list[Any] | None) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    converted: list[dict[str, Any]] = []
    for call in tool_calls:
        try:
            converted.append(asdict(call))
        except TypeError:
            converted.append({"value": str(call)})
    return converted


def _bootstrap_state() -> None:
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("latest_response", None)
    st.session_state.setdefault("history", [])


def _set_quick_prompt(prompt_text: str) -> None:
    st.session_state["question_input"] = prompt_text


def _save_run(question: str, response: Any) -> None:
    st.session_state["latest_response"] = response
    st.session_state["history"].insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer_preview": response.answer[:220],
            "variant": response.variant,
            "sources": len(response.sources),
            "latency_seconds": response.latency_seconds,
            "refused": response.refused,
        },
    )
    st.session_state["history"] = st.session_state["history"][:20]


def _render_hero() -> None:
    st.markdown(
        """
        <div class='hero-card'>
            <div class='hero-title'>RAG Assistant Showcase</div>
            <p class='hero-subtitle'>Interactive QA + observability-ready traces for a strong technical demo.</p>
            <p class='tiny-note'>Tip: use Variant D (orchestrator) for multi-step reasoning and tool visibility.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_controls(config) -> tuple[str, int, bool]:
    with st.sidebar:
        st.header("Demo Controls")
        variant = st.radio(
            "Pipeline variant",
            options=[
                "A_semantic_llm",
                "B_hybrid_llm",
                "C_hybrid_rerank_llm",
                "D_orchestrator",
            ],
            index=0,
            help="A: lexical lightweight | B: hybrid retrieve | C: hybrid + rerank | D: planner/orchestrator",
        )
        top_k = st.slider("Top K", min_value=1, max_value=12, value=4)
        orchestrator_rerank = st.checkbox(
            "Use reranking inside orchestrator",
            value=False,
            disabled=variant != "D_orchestrator",
        )
        if variant == "C_hybrid_rerank_llm" or (variant == "D_orchestrator" and orchestrator_rerank):
            st.warning("Reranking loads an additional model and may exceed memory on this machine.")

        st.markdown("---")
        st.subheader("Quick prompts")
        c1, c2 = st.columns(2)
        if c1.button("Becas", use_container_width=True):
            _set_quick_prompt("Que tipos de becas existen y cual es el proceso de solicitud?")
        if c2.button("Viajes", use_container_width=True):
            _set_quick_prompt("Resume la politica de viajes y el proceso de aprobacion.")
        c3, c4 = st.columns(2)
        if c3.button("Comparativa", use_container_width=True):
            _set_quick_prompt("Compara el procedimiento de becas con la politica de viajes.")
        if c4.button("Correo", use_container_width=True):
            _set_quick_prompt("Redacta un correo formal solicitando informacion sobre becas.")

        st.markdown("---")
        st.subheader("Service status")
        for service_name, is_ok, url in _service_status(config):
            _render_status_badge(service_name, is_ok, url)

        st.markdown("---")
        phoenix_url = _get_phoenix_url(config)
        if hasattr(st, "link_button"):
            st.link_button("Open Phoenix", phoenix_url, use_container_width=True)
        else:
            st.markdown(f"[Open Phoenix]({phoenix_url})")

    return variant, top_k, orchestrator_rerank


def _render_query_form() -> tuple[str, bool]:
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            key="question_input",
            height=120,
            placeholder="Ask something grounded in your indexed documents...",
        )
        submitted = st.form_submit_button("Run query", use_container_width=True)

    return question, submitted


def _render_latest_response(latest: Any) -> None:
    tokens = latest.token_usage or {"input_tokens": 0, "output_tokens": 0}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variant", latest.variant or "n/a")
    m2.metric("Latency (s)", f"{(latest.latency_seconds or 0.0):.2f}")
    m3.metric("Tokens", f"{tokens.get('input_tokens', 0)} in / {tokens.get('output_tokens', 0)} out")
    m4.metric("Sources", f"{len(latest.sources)} final / {latest.retrieved_count} retrieved")

    st.subheader("Answer")
    if latest.refused:
        st.warning("Model refusal triggered: insufficient evidence according to current threshold.")
    st.markdown(latest.answer)

    if latest.email_draft is not None:
        with st.expander("Email draft", expanded=False):
            st.text_input("Subject", value=latest.email_draft.subject, disabled=True)
            st.text_area("Body", value=latest.email_draft.body, height=200, disabled=True)
            st.caption(f"Draft status: {latest.email_draft.status}")

    st.subheader("Sources")
    if not latest.sources:
        st.info("No final sources were returned.")

    for index, chunk in enumerate(latest.sources, start=1):
        source_name = Path(chunk.source).name
        with st.expander(f"[{index}] {source_name} | score={chunk.score:.4f}", expanded=False):
            st.caption(f"chunk_index={chunk.chunk_index} | source={chunk.source}")
            st.write(chunk.text)


def _render_trace_panel() -> None:
    st.subheader("Trace detail")
    latest = st.session_state.get("latest_response")
    if latest is None:
        st.info("Run one query to see plan/tool calls and execution history.")
        return

    if latest.plan is not None:
        with st.expander("Orchestrator plan", expanded=True):
            st.json(asdict(latest.plan))

    tool_calls = _tool_calls_to_dict(latest.tool_calls)
    with st.expander("Tool calls", expanded=True):
        if tool_calls:
            st.json(tool_calls)
        else:
            st.caption("No tool calls for this run.")

    run_payload = {
        "variant": latest.variant,
        "latency_seconds": latest.latency_seconds,
        "token_usage": latest.token_usage,
        "refused": latest.refused,
        "evidence_sufficient": latest.evidence_sufficient,
        "sources": len(latest.sources),
    }
    with st.expander("Run summary JSON", expanded=False):
        st.json(run_payload)

    filename = f"rag_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    st.download_button(
        "Download latest run",
        data=json.dumps(run_payload, ensure_ascii=True, indent=2),
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Recent runs")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("No runs yet.")
        return

    for item in history[:8]:
        refused_label = "refused" if item["refused"] else "ok"
        st.markdown(
            f"**{item['timestamp']}**<br>{item['variant']} | {item['sources']} sources | {refused_label}",
            unsafe_allow_html=True,
        )
        st.caption(item["question"])
        st.caption(item["answer_preview"])
        st.markdown("---")


def main() -> None:
    st.set_page_config(page_title="Assistant RAG Demo", layout="wide")
    _bootstrap_state()
    _inject_styles()

    config = load_config()
    variant, top_k, orchestrator_rerank = _sidebar_controls(config)
    _render_hero()

    left_col, right_col = st.columns([2.1, 1.25], gap="large")

    with left_col:
        question, submitted = _render_query_form()
        if submitted:
            if not question.strip():
                st.warning("Please enter a question before running the pipeline.")
            else:
                use_lexical, use_hybrid, use_reranking, use_orchestrator = _variant_to_flags(
                    variant, orchestrator_rerank
                )
                pipeline = _get_pipeline()
                try:
                    with st.spinner("Running retrieval + generation..."):
                        response = pipeline.run_pipeline(
                            question.strip(),
                            top_k=top_k,
                            use_lexical=use_lexical,
                            use_hybrid=use_hybrid,
                            use_reranking=use_reranking,
                            use_orchestrator=use_orchestrator,
                        )
                    _save_run(question.strip(), response)
                except Exception as exc:  # pragma: no cover - defensive UX handling
                    message = str(exc)
                    if "Read timed out" in message:
                        st.error(
                            "Pipeline timeout while waiting for Ollama. "
                            "Use Variant A, keep Top K <= 4, and restart Ollama if needed."
                        )
                    else:
                        st.error(f"Pipeline failed: {exc}")

        latest = st.session_state.get("latest_response")
        if latest is not None:
            _render_latest_response(latest)

    with right_col:
        _render_trace_panel()


if __name__ == "__main__":
    main()
