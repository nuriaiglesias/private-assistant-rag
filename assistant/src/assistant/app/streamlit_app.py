from __future__ import annotations

from pathlib import Path
import sys

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


def main() -> None:
    st.set_page_config(page_title="RAG Demo", layout="centered")
    st.title("RAG Demo")
    st.write("Ask a question and see the answer with sources.")

    question = st.text_input("Question", placeholder="Ask something about the documents...")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

    if st.button("Consultar"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        pipeline = _get_pipeline()
        response = pipeline.answer(question.strip(), top_k=top_k)

        st.subheader("Answer")
        st.write(response.answer)

        st.subheader("Sources")
        for index, chunk in enumerate(response.sources, start=1):
            label = f"[{index}] {Path(chunk.source).name} (score={chunk.score:.4f})"
            with st.expander(label):
                st.write(f"Chunk index: {chunk.chunk_index}")
                st.write(chunk.text)


if __name__ == "__main__":
    main()
