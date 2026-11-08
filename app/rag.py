"""RAG pipeline: condense question -> classify intent -> retrieve -> answer."""
from __future__ import annotations

from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from app import config

Intent = Literal["permits", "forms", "login", "manuals", "general"]


class IntentResult(BaseModel):
    intent: Intent = Field(
        description=(
            "permits: questions about permit types, requirements, fees, status; "
            "forms: which form to fill out / how to complete a form; "
            "login: account, password, username, access or sign-in problems; "
            "manuals: asking for a manual, guide or how-to documentation; "
            "general: anything else"
        )
    )


CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message as a standalone question, using the "
            "chat history for context. Do not answer it. If it is already "
            "standalone, return it unchanged.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the NCDOT virtual assistant. You help customers with permits, \
choosing the right form, login/account problems, and finding the right manuals.

Rules:
- Answer ONLY from the numbered context below. Never invent fees, form numbers, \
deadlines, phone numbers, URLs or requirements.
- If the context does not contain the answer, say you could not find it in the \
available documents and suggest contacting {support_contact}.
- Cite the sources you used with their numbers, e.g. [1], [2].
- Detected topic: {intent}.
  * forms: if the request is unclear, ask one or two short clarifying questions \
(e.g. type of permit, project location, applicant type). Once clear, name the exact \
form(s) from the context and briefly say why, and what to do next.
  * login: give short numbered troubleshooting steps. NEVER ask for passwords, \
full SSNs, or payment details.
  * manuals: name the manual(s) and the relevant section/page when available.
  * permits: explain requirements and process; mention fees or timelines only if \
they appear in the context.
- You are not a lawyer; do not give legal advice. Keep answers clear and concise.

Context:
{context}""",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "(no relevant documents found)"
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        loc = f", page {page}" if page else ""
        parts.append(f"[{i}] ({src}{loc})\n{d.page_content}")
    return "\n\n".join(parts)


class NCDOTChatbot:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
        self.vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
            persist_directory=config.CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        self.intent_chain = self.llm.with_structured_output(IntentResult)
        self.answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        self.histories: dict[str, list[BaseMessage]] = {}

    # ---- steps -------------------------------------------------------
    def _classify(self, question: str) -> Intent:
        try:
            result = self.intent_chain.invoke(
                f"Classify this NCDOT customer message:\n{question}"
            )
            return result.intent
        except Exception:
            return "general"

    def _retrieve(self, question: str, intent: Intent) -> list[Document]:
        docs: list[Document] = []
        if intent in config.CATEGORIES:
            docs = self.vectorstore.similarity_search(
                question, k=config.TOP_K, filter={"category": intent}
            )
        # Fall back to the whole corpus when the category has too little content.
        if len(docs) < 2:
            docs = self.vectorstore.similarity_search(question, k=config.TOP_K)
        return docs

    # ---- public API --------------------------------------------------
    def ask(self, question: str, session_id: str = "default") -> dict:
        history = self.histories.setdefault(session_id, [])[-8:]

        standalone = (
            self.condense_chain.invoke({"history": history, "question": question})
            if history
            else question
        )
        intent = self._classify(standalone)
        docs = self._retrieve(standalone, intent)

        answer = self.answer_chain.invoke(
            {
                "context": _format_context(docs),
                "history": history,
                "question": question,
                "intent": intent,
                "support_contact": config.SUPPORT_CONTACT,
            }
        )

        full = self.histories[session_id]
        full.append(HumanMessage(content=question))
        full.append(AIMessage(content=answer))
        del full[:-20]  # cap memory per session

        sources = []
        seen = set()
        for i, d in enumerate(docs, 1):
            key = (d.metadata.get("source"), d.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "id": i,
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                    "category": d.metadata.get("category"),
                }
            )
        return {"answer": answer, "intent": intent, "sources": sources}

    def reset(self, session_id: str) -> None:
        self.histories.pop(session_id, None)
