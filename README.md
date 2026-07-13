# Diseño de un asistente institucional agéntico
### Observabilidad, RAG y herramientas

**TFM · Master Universitario en Inteligencia Artificial · UNIR 2025–26**  
**Autor:** Nuria Iglesias Traviesa &nbsp;·&nbsp; **Tutor:** David Jiménez Cabello

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Vector%20store-Qdrant-DC143C?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-qwen2.5%3A7b-1976D2?style=flat-square)
![Phoenix](https://img.shields.io/badge/Observability-Phoenix-8E24AA?style=flat-square)

---

## Demo y recursos

| Recurso | Enlace |
|---------|--------|
| Vídeo de demostración | [Ver en Google Drive](https://drive.google.com/file/d/16hjMI20DtRpueGSbksihCKcH5a_2R5Pl/view?usp=sharing) |
| Memoria del TFM | `thesis/TFM.pdf` |
| Observabilidad (local) | http://localhost:6006 |

---

## Descripción

Prototipo de asistente conversacional institucional privado que combina **recuperación documental híbrida**, **orquestación agéntica con ReAct** y **accionamiento supervisado de herramientas externas**. Diseñado para funcionar íntegramente en local, con trazabilidad completa del pipeline mediante OpenTelemetry y Arize Phoenix.

**Capacidades principales:**

- Respuesta fundamentada sobre corpus institucional (26 documentos UNIR)
- Recuperación híbrida: densa (BGE-M3 + Qdrant) + léxica (BM25) con fusión RRF
- Reranking cross-encoder (BGE-Reranker-v2-M3) sobre los 20 candidatos iniciales
- Orquestador ReAct: razonamiento iterativo, abstención y uso de herramientas
- Envío supervisado de correo electrónico: borrador → revisión humana → envío
- Trazas completas del pipeline en Arize Phoenix (OpenTelemetry)

---

## Instalación rápida

> Todos los comandos se ejecutan desde `assistant/`

```bash
# 1. Entorno virtual
python3 -m venv .venv && source .venv/bin/activate

# 2. Dependencias
pip install -r requirements.txt
pip install -r requirements-llm.txt   # solo para generación de respuestas

# 3. Variables de entorno
cp .env.example .env
# Editar .env: LLM_PROVIDER, LLM_MODEL y, opcionalmente, PHOENIX_ENABLED

# 4. Infraestructura (Qdrant + Phoenix)
docker compose up -d

# 5. Ingestión del corpus
python3 scripts/ingest/ingest_documents.py

# 6. Modelo LLM local (Ollama)
ollama serve && ollama pull qwen2.5:7b

# 7. Interfaz de usuario
streamlit run src/assistant/app/streamlit_app.py

# — o prueba rápida desde CLI —
python3 scripts/query/ask.py "¿Qué requisitos de acceso pide UNIR?"
```

Trazas disponibles en **http://localhost:6006** (Phoenix).

---

## Variantes del estudio ablativo

Cuatro configuraciones progresivas evaluadas en la tesis:

| Variante | Retrieval | Reranker | Orquestador |
|----------|-----------|:--------:|:-----------:|
| **A** — Baseline denso | BGE-M3 + Qdrant | — | — |
| **B** — Híbrido RRF | Dense + BM25 (RRF) | — | — |
| **C** — Híbrido + Reranker | Dense + BM25 (RRF) | ✓ | — |
| **D** — Agéntico completo | Dense + BM25 (RRF) | ✓ | ✓ ReAct |

---

## Resultados principales (corpus UNIR, 71 preguntas)

| Variante | Fact-Coverage | BERTScore F1 | Hit@3 |
|----------|:-------------:|:------------:|:-----:|
| A — Baseline denso | 0.389 | 0.643 | 0.852 |
| B — Híbrido RRF | 0.496 | 0.641 | 0.902 |
| C — Híbrido + Reranker | **0.527** | **0.656** | **0.918** |
| D — Orquestador | 0.493 | 0.648 | 0.918 |

El orquestador (D) cede algo de cobertura factual léxica a cambio de mayor control del flujo: abstención del 60 % ante preguntas no respondibles (frente al 30–50 % de los pipelines deterministas) y detección autónoma de solicitudes de correo electrónico.

---

## Evaluación

```bash
# Métricas de recuperación (Hit@1, Hit@3, MRR)
python3 scripts/eval/evaluate_retrieval.py                          # variante A
python3 scripts/eval/evaluate_retrieval.py --use-hybrid             # variante B
python3 scripts/eval/evaluate_retrieval.py --use-reranking          # variante C

# Calidad de respuesta (Fact-Coverage, ROUGE-L, BERTScore)
python3 scripts/eval/evaluate_answers.py                            # variante A
python3 scripts/eval/evaluate_answers.py --use-reranking            # variante C
python3 scripts/eval/evaluate_answers.py --use-orchestrator         # variante D
python3 scripts/eval/evaluate_answers.py --use-reranking --use-orchestrator

# Evaluación funcional del orquestador (abstención, detección de correo)
python3 scripts/eval/evaluate_orchestrator.py --compare

# Validez externa — XQuAD-es (benchmark público)
python3 scripts/eval/evaluate_public_dataset.py --mode hybrid --limit 300
```

Los resultados se guardan en `assistant/results/` como ficheros JSONL con marca de tiempo.

---

## Estructura del repositorio

```
assistant/
├── corpus/
│   ├── processed_md/       26 documentos Markdown (el corpus)
│   └── raw/                URLs semilla y datos scrapeados
├── datasets/               Conjuntos de evaluación (71 Q UNIR, 6 Q correo)
├── results/                Resultados de experimentos (JSONL con timestamp)
├── scripts/
│   ├── ingest/             scrape_unir.py · ingest_documents.py · reset_vector_store.py
│   ├── query/              ask.py · search_documents.py
│   └── eval/               evaluate_retrieval.py · evaluate_answers.py
│                           evaluate_orchestrator.py · evaluate_public_dataset.py
├── src/assistant/
│   ├── core/               config, paths
│   ├── ingestion/          scraper, cleaner, document_loader, ingest
│   ├── rag/                chunker, vector_store, retriever, reranker, pipeline
│   ├── orchestrator/       router, rules, planner, evidence, orchestrator, state
│   ├── llm/                factory, ollama_provider, openai_provider
│   ├── evaluation/         retrieval, answers, orchestrator, public_dataset
│   ├── observability/      Phoenix / OpenTelemetry tracing
│   ├── tools/              email_tool, summary_tool, registry
│   └── app/                streamlit_app.py (interfaz web)
├── docker-compose.yml      Qdrant :6333 · Phoenix :6006
├── requirements.txt
└── requirements-llm.txt    Solo necesario para generación de respuestas
thesis/                     Fuentes LaTeX → TFM.pdf
planning/                   Backlogs de sprint, rúbrica de métricas, CSV de evaluación
deliveries/                 Archivos de entrega
```

---

## Stack tecnológico

| Capa | Tecnología |
|------|------------|
| Embedding | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Almacén vectorial | Qdrant |
| LLM local | `qwen2.5:7b` via Ollama |
| Observabilidad | Arize Phoenix + OpenTelemetry |
| Correo electrónico | Microsoft Graph API + OAuth 2.0 |
| Interfaz de usuario | Streamlit |
| Infraestructura | Docker Compose |
