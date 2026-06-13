from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config

# ── Constants ──────────────────────────────────────────────────────────────

VARIANTS: dict[str, tuple[bool, bool, bool, bool]] = {
    "D — Orquestador ReAct": (False, True, True, True),
    "C — Híbrido + Reranker": (False, True, True, False),
    "B — Híbrido (RRF)": (False, True, False, False),
    "A — Semántico": (False, False, False, False),
}

QUICK_PROMPTS = [
    ("Matrícula", "¿Cuáles son los plazos y requisitos de matrícula en UNIR?"),
    ("Becas", "¿Qué tipos de becas ofrece UNIR y cómo se solicitan?"),
    ("TFM", "¿Cuáles son los requisitos para entregar el TFM en el Máster de IA?"),
    ("Correo", "Redacta un correo para secretaría preguntando sobre los plazos de matrícula."),
]

USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🎓"

# ── Pipeline ───────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Iniciando pipeline RAG…")
def _get_pipeline():
    from assistant.rag.pipeline import RagPipeline
    config = load_config()
    return RagPipeline(config)


# ── Utilities ──────────────────────────────────────────────────────────────


def _safe_get(url: str, timeout: float = 1.5) -> bool:
    try:
        return requests.get(url, timeout=timeout).status_code < 500
    except Exception:
        return False


def _inject_css() -> None:
    css_path = Path(__file__).resolve().parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────


def _sidebar(config: Any) -> tuple[str, int]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sb-brand">
                <span class="sb-icon">🎓</span>
                <div>
                    <div class="sb-title">Asistente UNIR</div>
                    <div class="sb-sub">RAG · Híbrido · Orquestador ReAct</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Variante del pipeline</div>', unsafe_allow_html=True)
        variant = st.radio(
            "variant_radio",
            options=list(VARIANTS.keys()),
            index=1,  # default: C — Híbrido + Reranker (works well on laptop)
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-label">Fragmentos recuperados (k)</div>', unsafe_allow_html=True)
        top_k = st.slider(
            "top_k_slider", min_value=1, max_value=12, value=5, label_visibility="collapsed"
        )

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Ejemplos rápidos</div>', unsafe_allow_html=True)
        for label, prompt in QUICK_PROMPTS:
            if st.button(label, use_container_width=True, key=f"qp_{label}"):
                st.session_state["_pending"] = prompt

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Estado de servicios</div>', unsafe_allow_html=True)
        _service_status(config)

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        if st.button("🗑  Nueva conversación", use_container_width=True, key="clear_btn"):
            st.session_state["messages"] = []
            st.session_state["meta"] = {}
            st.rerun()

    return variant, top_k


def _service_status(config: Any) -> None:
    checks = []
    if hasattr(config, "qdrant_url") and config.qdrant_url.startswith("http"):
        checks.append(("Qdrant", f"{config.qdrant_url}/collections"))
    if hasattr(config, "llm_provider") and config.llm_provider.lower() == "ollama":
        checks.append(("Ollama", f"{config.llm_base_url}/api/tags"))
    if hasattr(config, "phoenix_enabled") and config.phoenix_enabled:
        phoenix_url = f"http://{config.phoenix_host}:{getattr(config, 'phoenix_ui_port', 6006)}"
        checks.append(("Phoenix", phoenix_url))

    for name, url in checks:
        ok = _safe_get(url)
        dot = "🟢" if ok else "🔴"
        st.markdown(
            f'<span>{dot}</span> <span style="font-size:0.84rem;color:#9ca3af">{name}</span>',
            unsafe_allow_html=True,
        )


# ── Response rendering ─────────────────────────────────────────────────────


def _render_meta(resp: Any, variant: str) -> None:
    tokens = resp.token_usage or {}
    latency = f"{(resp.latency_seconds or 0):.1f}s"
    sources = len(resp.sources)
    out_tok = tokens.get("output_tokens", 0)
    # Extract short label after " — "
    short = variant.split(" — ", 1)[-1] if " — " in variant else variant
    st.markdown(
        f"""
        <div class="meta-row">
            <span class="variant-badge">{short}</span>
            <span class="meta-chip">⏱ {latency}</span>
            <span class="meta-chip">📄 {sources} fuente(s)</span>
            <span class="meta-chip">🔤 {out_tok} tokens</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_email(resp: Any) -> None:
    d = resp.email_draft
    if d is None:
        return
    sent = d.status == "sent"
    icon = "✉️ Correo enviado" if sent else "📝 Borrador de correo"
    with st.expander(icon, expanded=True):
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**Asunto:** {d.subject}")
        if d.to:
            c2.markdown(f"**Para:** {d.to}")
        st.text_area(
            "Cuerpo del correo",
            value=d.body,
            height=160,
            disabled=True,
            key=f"email_body_{id(resp)}",
        )
        color = "#16a34a" if sent else "#d97706"
        st.markdown(
            f'<span style="color:{color};font-size:0.78rem;font-weight:500">● {d.status}</span>',
            unsafe_allow_html=True,
        )


def _render_sources(resp: Any) -> None:
    if not resp.sources:
        return
    label = f"📚 {len(resp.sources)} fragmento(s) recuperado(s)"
    with st.expander(label, expanded=False):
        for i, chunk in enumerate(resp.sources, 1):
            name = Path(chunk.source).name
            st.markdown(f"**[{i}] {name}** `score={chunk.score:.4f}`")
            st.caption(chunk.text[:280] + ("…" if len(chunk.text) > 280 else ""))
            if i < len(resp.sources):
                st.divider()


def _render_tools(resp: Any) -> None:
    if not getattr(resp, "tool_calls", None):
        return
    with st.expander(f"🔧 {len(resp.tool_calls)} llamada(s) a herramienta", expanded=False):
        for tc in resp.tool_calls:
            try:
                d = asdict(tc)
            except TypeError:
                d = {"value": str(tc)}
            name = d.get("tool_name", "?")
            summary = d.get("tool_output_summary") or ""
            st.markdown(f"**`{name}`** → {summary}")


def _render_response_extras(resp: Any, variant: str) -> None:
    _render_meta(resp, variant)
    _render_email(resp)
    _render_sources(resp)
    _render_tools(resp)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Asistente UNIR",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "meta" not in st.session_state:
        st.session_state["meta"] = {}

    config = load_config()
    variant, top_k = _sidebar(config)

    # Header
    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-header-title">Asistente Institucional UNIR</div>
            <div class="chat-header-sub">
                Consulta documentación universitaria · RAG con recuperación híbrida y orquestador ReAct
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    messages: list[dict] = st.session_state["messages"]
    meta: dict[int, Any] = st.session_state["meta"]

    # Welcome state
    if not messages:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-title">¿En qué puedo ayudarte?</div>
                <div class="welcome-body">
                    Pregúntame sobre matrícula, becas, normativas, el TFM o cualquier otra
                    consulta universitaria. Puedo buscar en la documentación de UNIR
                    y, si lo necesitas, redactar un correo formal.
                </div>
                <div class="welcome-examples">
                    <span class="eg-pill">Plazos de matrícula</span>
                    <span class="eg-pill">Tipos de becas</span>
                    <span class="eg-pill">Requisitos del TFM</span>
                    <span class="eg-pill">Redactar correo</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render conversation history
    for i, msg in enumerate(messages):
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i in meta:
                _render_response_extras(meta[i], meta[i].__dict__.get("variant", variant))

    # Consume pending quick prompt (set by sidebar buttons)
    pending: str | None = st.session_state.pop("_pending", None)

    # Chat input
    user_input: str | None = st.chat_input("Pregunta sobre UNIR: matrícula, becas, TFM, normativa…")
    question: str | None = pending or user_input

    if question:
        # Display user turn
        messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        # Generate assistant response
        flags = VARIANTS[variant]
        use_lexical, use_hybrid, use_reranking, use_orchestrator = flags
        pipeline = _get_pipeline()

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Consultando documentación…"):
                try:
                    resp = pipeline.run_pipeline(
                        question,
                        top_k=top_k,
                        use_lexical=use_lexical,
                        use_hybrid=use_hybrid,
                        use_reranking=use_reranking,
                        use_orchestrator=use_orchestrator,
                    )
                    answer = resp.answer
                    ok = True
                except Exception as exc:
                    timeout_hint = "Read timed out" in str(exc)
                    answer = (
                        "⚠️ Tiempo de espera agotado. Prueba con la variante A o reinicia Ollama."
                        if timeout_hint
                        else f"⚠️ Error al procesar la consulta: {exc}"
                    )
                    resp = None
                    ok = False

            st.markdown(answer)

            asst_idx = len(messages)
            messages.append({"role": "assistant", "content": answer})

            if ok and resp is not None:
                meta[asst_idx] = resp
                _render_response_extras(resp, variant)


if __name__ == "__main__":
    main()
