# RAG-QA Demo

A live, deployable Streamlit app for a Retrieval-Augmented Generation (RAG) question-answering system built on SQuAD. This is the deployment companion to the full pipeline, evaluation, and error analysis notebook — see [rag-qa-squad](https://github.com/RuhelHaq/rag-qa-squad) for the complete build.

**[Live Demo](#)** *(https://rag-app-demo-pkw5ktqpqpre8zqhlat7ux.streamlit.app)*

## What it does

Ask a natural-language question. The app:
1. Embeds the question and retrieves the most relevant passages from a 2,067-passage SQuAD corpus using FAISS
2. Sends the retrieved passages + question to Claude, which generates an answer grounded strictly in that context
3. Displays the answer alongside the retrieved passages as supporting evidence

If the retrieved context doesn't contain the answer, the system responds "I don't know" rather than guessing — this was verified during evaluation, including on out-of-domain questions.

## Topics covered

The corpus is built from SQuAD's validation set, covering 48 Wikipedia articles across a range of subjects — historical events, scientific concepts, geography, institutions, and more:

1973 oil crisis · Amazon rainforest · American Broadcasting Company · Apollo program · Black Death · Chloroplast · Civil disobedience · Computational complexity theory · Construction · Ctenophora · Doctor Who · Economic inequality · European Union law · Force · French and Indian War · Fresno, California · Genghis Khan · Geology · Harvard University · Huguenot · Immune system · Imperialism · Intergovernmental Panel on Climate Change · Islamism · Jacksonville, Florida · Kenya · Martin Luther · Newcastle upon Tyne · Nikola Tesla · Normans · Oxygen · Packet switching · Pharmacy · Prime number · Private school · Rhine · Scottish Parliament · Sky (United Kingdom) · Southern California · Steam engine · Super Bowl 50 · Teacher · United Methodist Church · University of Chicago · Victoria (Australia) · Victoria and Albert Museum · Warsaw · Yuan dynasty

Questions outside these topics will correctly return "I don't know" rather than a hallucinated answer — this behavior was verified during evaluation.

## Tech stack

- **Streamlit** — web interface
- **fastembed** — lightweight, ONNX-based sentence embeddings (`all-MiniLM-L6-v2`), chosen over `sentence-transformers` for a much smaller memory footprint suited to free-tier hosting
- **FAISS** — vector similarity search
- **Anthropic Claude API** — grounded answer generation

## Repo contents

- `app.py` — Streamlit application
- `requirements.txt` — dependencies
- `passages.jsonl` — deduplicated SQuAD passage corpus
- `faiss.index` — prebuilt FAISS index over the passage embeddings

## Running locally

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with:
```toml
ANTHROPIC_API_KEY = "your-key-here"
```

Then run:
```bash
streamlit run app.py
```

## Deployment notes

This app runs on **Streamlit Community Cloud**. It was originally built with Gradio; deployment was attempted on Render and Hugging Face Spaces first:

- **Render (free tier)** — the ML dependency stack (embeddings + FAISS + web framework) didn't fit within the 512MB memory limit, and several Gradio dependencies had unresolved compatibility issues with Python 3.14 at the time of building.
- **Hugging Face Spaces (free tier)** — Gradio Spaces on free accounts require either a paid plan or ZeroGPU hardware, which itself requires an account older than 30 days.

Streamlit Cloud ran the same underlying pipeline (retrieval + generation logic unchanged) without these constraints, so the interface was rebuilt in Streamlit for deployment.

## Related

Full pipeline, evaluation (79% retrieval hit-rate@3, 47% EM, 65% F1), and error analysis: [rag-qa-squad](https://github.com/RuhelHaq/rag-qa-squad)
