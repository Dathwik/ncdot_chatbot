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

## MVP showcase deployment

The frontend can be published on GitHub Pages; the live chatbot API runs
separately on Render. The repository includes a Render Blueprint in `render.yaml`
and a GitHub Actions Pages workflow in `.github/workflows/pages.yml`.

1. In Render, create a **Blueprint** from this repository. Add your
   `OPENAI_API_KEY` when prompted. The build ingests the checked-in documents
   and creates the Chroma database; this makes OpenAI embedding API calls.
2. After the Render web service is live, copy its public URL
   (for example, `https://ncdot-chatbot-api.onrender.com`).
3. In GitHub, open **Settings → Secrets and variables → Actions → Variables**
   and add `NCDOT_API_BASE_URL` with that Render URL (no trailing slash).
4. Open **Settings → Pages**, choose **GitHub Actions** as the source, then push
   to `main` or run the **Deploy GitHub Pages** workflow.
5. Visit `https://dathwik.github.io/ncdot_chatbot/`.

The repository is currently private. GitHub Pages publication from a private
repository requires a GitHub plan that supports it; otherwise, make the
repository public before enabling Pages. Making it public also exposes all
tracked files, including the PDFs in `data/docs/`. Never put your OpenAI key in
the repository or in a GitHub Actions variable; keep it in Render's environment.
The free Render service may sleep when idle, so the first request can take longer.

## Notes / next steps

- Chat history is in memory per `session_id`; use Redis/DB for multi-instance deploys.
- Tune `chunk_size`, `TOP_K` and the prompts in `app/rag.py` using real customer questions.
- Add streaming, rate limiting, auth and request logging before public launch.
- Do not index documents containing customer PII.
- Evaluate with a set of real questions + expected sources before go-live.
