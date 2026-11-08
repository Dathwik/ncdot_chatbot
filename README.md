# NCDOT Virtual Assistant (RAG chatbot)

LangChain + OpenAI + Chroma + FastAPI. Answers customer questions about permits,
recommends the right form, helps with login issues, and points to manuals — all
grounded in the documents you provide, with source citations.

## How it works

```
question ─► condense with chat history ─► classify intent (permits/forms/login/manuals/general)
        ─► retrieve from Chroma (filtered by category, falls back to all docs)
        ─► answer using only retrieved context + citations
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your OPENAI_API_KEY
```

## Add your documents

Put files in `data/docs/`. The **first sub-folder name is the category** and
improves retrieval routing:

```
data/docs/permits/   permit guides, fee schedules, requirements
data/docs/forms/     form catalog / "which form for what" guides, form instructions
data/docs/login/     login, password reset, account FAQs
data/docs/manuals/   user manuals, technical guides
```

Supported: `.pdf`, `.docx`, `.txt`, `.html`. For web pages, list URLs (one per line)
in `data/urls.txt`.

Tip: a short "form decision guide" document (e.g. *"If you are doing X, use form Y"*)
in `data/docs/forms/` dramatically improves form recommendations.

## Run

```bash
python -m app.ingest                 # build the vector store (re-run when docs change)
uvicorn app.main:app --reload        # open http://localhost:8000
```

API: `POST /chat` with `{"message": "...", "session_id": "abc"}` →
`{"answer": "...", "intent": "...", "sources": [...]}`.

## Notes / next steps

- Chat history is in memory per `session_id`; use Redis/DB for multi-instance deploys.
- Tune `chunk_size`, `TOP_K` and the prompts in `app/rag.py` using real customer questions.
- Add streaming, rate limiting, auth and request logging before public launch.
- Do not index documents containing customer PII.
- Evaluate with a set of real questions + expected sources before go-live.
