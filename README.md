# Telecare RAG Chatbot

A retrieval-augmented generation (RAG) prototype that answers questions about medical and mobile alarm monitoring for Telecare Monitoring Representatives. It is built in a single Jupyter notebook with LangChain, hybrid retrieval (BM25 + dense vectors) and a Gradio UI, and can use a local LLM through Ollama or an OpenAI model.

This is a personal learning project. The source document is a general write-up on telecare monitoring hosted on my portfolio site.

## How it works

1. **Ingest:** scrapes the portfolio projects page (requests + BeautifulSoup), then uses Selenium to find and download the telecare PDF, and extracts its text with PyMuPDF.
2. **Chunk:** splits website and PDF text with `RecursiveCharacterTextSplitter` (1000 characters, 20 overlap). An LLM-based "agentic chunking" chain is also defined for comparison.
3. **Embed and retrieve:** embeds PDF pages with `sentence-transformers/all-MiniLM-L6-v2` into an in-memory `DocArrayInMemorySearch` store, and combines it with a `BM25Retriever` in an `EnsembleRetriever` (hybrid search).
4. **Generate:** passes the retrieved context and the question through a prompt to the selected LLM. If the answer is not in the documents, the model is instructed to say it does not know.
5. **UI:** a small Gradio app for asking questions.

## Project structure

```
telecare-rag-chatbot/
├── notebooks/
│   └── telecare_rag_chatbot.ipynb   # the full pipeline
├── .env.example                     # copy to .env and fill in
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11 (the version the notebook was developed with)
- Google Chrome (used by Selenium to find the PDF link)
- [Ollama](https://ollama.com) with a pulled model, for local use (default `llama3`), **or** an OpenAI API key
- Internet access on first run (downloads the PDF, the embedding model and the tiktoken vocabulary)

## Setup

```bash
git clone https://github.com/TitusDV/telecare-rag-chatbot.git
cd telecare-rag-chatbot

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create your `.env` file in the project root:

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Then edit `.env`:

- `MODEL=llama3` to run locally. Pull the model first with `ollama pull llama3`.
- `MODEL=gpt-4-turbo` (or another OpenAI model) and set `OPENAI_API_KEY` to use OpenAI.

## Run

```bash
jupyter notebook notebooks/telecare_rag_chatbot.ipynb
```

Run the cells top to bottom. The last cell launches the Gradio app.

## Known limitations

- Semantic chunking is listed as an approach but not implemented yet.
- The agentic chunking chain is defined but not yet used to build the vector store; retrieval uses the PDF pages.
- The vector store is in memory, so embeddings are rebuilt on every run.
- Ingestion scrapes my old Wix portfolio page (`titusdinkar.wixsite.com/titus/projects`) and the PDF hosted there, so it depends on that page staying online with its current layout; the two cells that display `pdf_chunks[4]` and `website_chunks[24]` assume at least that many chunks.
- LangChain packages are pinned below 1.0 because the notebook uses 0.3 import paths.
