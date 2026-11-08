"""Build (or rebuild) the Chroma vector store from the files in data/docs.

Usage:
    python -m app.ingest

Folder layout (the first sub-folder name becomes the document category):
    data/docs/permits/   permit guides, fee schedules, requirements
    data/docs/forms/     form instructions and form catalogs
    data/docs/login/     account / login / password-reset help
    data/docs/manuals/   user manuals and technical guides
    data/urls.txt        optional: one web page URL per line (ncdot.gov pages)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

LOADERS = {
    ".pdf": lambda p: PyPDFLoader(str(p)),
    ".docx": lambda p: Docx2txtLoader(str(p)),
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8", autodetect_encoding=True),
}
HTML_SUFFIXES = {".html", ".htm"}


def category_for(path: Path) -> str:
    rel = path.relative_to(config.DOCS_DIR)
    first = rel.parts[0] if len(rel.parts) > 1 else "general"
    return first.lower()


def load_files() -> list[Document]:
    docs: list[Document] = []
    if not config.DOCS_DIR.exists():
        print(f"Docs folder not found: {config.DOCS_DIR}")
        return docs

    for path in sorted(config.DOCS_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in LOADERS:
                loaded = LOADERS[suffix](path).load()
            elif suffix in HTML_SUFFIXES:
                from langchain_community.document_loaders import BSHTMLLoader

                loaded = BSHTMLLoader(str(path), open_encoding="utf-8").load()
            else:
                continue
        except Exception as exc:  # keep going if one file is bad
            print(f"  ! Skipped {path.name}: {exc}")
            continue

        category = category_for(path)
        for d in loaded:
            d.metadata["source"] = path.name
            d.metadata["category"] = category
            d.metadata["source_type"] = "file"
            # PyPDFLoader uses 0-based pages; make them human friendly
            if "page" in d.metadata and isinstance(d.metadata["page"], int):
                d.metadata["page"] = d.metadata["page"] + 1
        print(f"  + {path.relative_to(config.DOCS_DIR)} ({len(loaded)} section(s), category={category})")
        docs.extend(loaded)
    return docs


def load_urls() -> list[Document]:
    if not config.URLS_FILE.exists():
        return []
    urls = [
        line.strip()
        for line in config.URLS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    docs: list[Document] = []
    for url in urls:
        try:
            loaded = WebBaseLoader(url).load()
        except Exception as exc:
            print(f"  ! Skipped {url}: {exc}")
            continue
        for d in loaded:
            d.metadata["source"] = url
            d.metadata["category"] = "general"
            d.metadata["source_type"] = "web"
        print(f"  + {url}")
        docs.extend(loaded)
    return docs


def main() -> None:
    print("Loading documents...")
    docs = load_files() + load_urls()
    if not docs:
        print("No documents found. Add files under data/docs/ and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    # drop empty chunks (scanned pages with no text, blank pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"Split into {len(chunks)} chunks. Embedding...")

    # Rebuild from scratch so removed/changed documents don't linger.
    if Path(config.CHROMA_DIR).exists():
        shutil.rmtree(config.CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Done. Vector store saved to {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
