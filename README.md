# myBOT

A minimal **RAG (Retrieval-Augmented Generation) chatbot** built with [Streamlit](https://streamlit.io) and [Ollama](https://ollama.com). It answers questions about Alenso (Alen Alex) by retrieving relevant facts from a local text file and grounding a local LLM's response in that context — no external APIs, no vector database server, everything runs on your machine.

## How it works

```
alenso.txt  ──chunk──►  embed (Ollama)  ──►  in-memory vector DB
                                                     │
user question ──embed (Ollama)──► cosine similarity ─┘
                                                     │
                                            top-N matching chunks
                                                     │
                                     prompt = system context + question
                                                     │
                                        Ollama chat model (streamed)
                                                     │
                                              answer in the UI
```

1. **Chunking** — [main.py](main.py) reads `alenso.txt` and splits it into paragraph-sized chunks (blank-line separated).
2. **Embedding** — each chunk is embedded via Ollama's embedding model and cached in memory for the session (`@st.cache_resource`), so this only happens once per app run.
3. **Retrieval** — when you ask a question, it's embedded the same way, then compared against every chunk using cosine similarity; the top 3 matches are selected.
4. **Generation** — the matched chunks are injected into a system prompt that instructs the model to answer *only* from that context, then the chat model streams its response token-by-token into the UI.
5. **Transparency** — the sidebar shows exactly which chunks were retrieved and their similarity scores, so you can see why the model answered the way it did.

## Prerequisites

- **Python 3.14** (see [.python-version](.python-version))
- **[Ollama](https://ollama.com/download)** installed and running locally
- The two models this app uses, pulled once via Ollama:

  ```bash
  ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
  ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
  ```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) as its package manager (see [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock)), but a plain [requirements.txt](requirements.txt) is also provided if you prefer pip.

**Option A — uv (recommended)**

```bash
uv sync
```

**Option B — pip**

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

## Running

Make sure Ollama is running in the background, then:

```bash
uv run streamlit run main.py
# or, with a plain venv:
streamlit run main.py
```

Streamlit will open the app in your browser (default: `http://localhost:8501`). On first launch it embeds every chunk in `alenso.txt`, so give it a few seconds before asking a question.

## Customizing the knowledge base

Edit [alenso.txt](alenso.txt) — it's the entire dataset the bot draws answers from. Separate distinct facts/topics with a blank line so each becomes its own embedded chunk. If the file is deleted, `main.py` regenerates a small fallback version automatically on next run.

To point the bot at a different persona or domain, swap out `alenso.txt`'s content and update the title/instructions in [main.py](main.py) accordingly.

## Project structure

| File | Purpose |
|---|---|
| [main.py](main.py) | Streamlit app: chunking, embedding, retrieval, and chat UI |
| [alenso.txt](alenso.txt) | Knowledge base the bot retrieves context from |
| [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) | uv project & locked dependencies |
| [requirements.txt](requirements.txt) | Plain pip dependency list |
| [.python-version](.python-version) | Pinned Python version (3.14) |
