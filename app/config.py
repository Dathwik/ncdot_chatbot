import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = str((BASE_DIR / os.getenv("CHROMA_DIR", "./chroma_db")).resolve())
DOCS_DIR = (BASE_DIR / os.getenv("DOCS_DIR", "./data/docs")).resolve()
URLS_FILE = (BASE_DIR / "data" / "urls.txt").resolve()
TOP_K = int(os.getenv("TOP_K", "5"))
SUPPORT_CONTACT = os.getenv(
    "SUPPORT_CONTACT", "the NCDOT customer service line listed on ncdot.gov"
)

COLLECTION_NAME = "ncdot_docs"

# Sub-folders of DOCS_DIR double as document categories used for retrieval routing.
CATEGORIES = ["permits", "forms", "login", "manuals"]
