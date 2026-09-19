"""Runtime configuration, read from environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TRACEAI_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite keeps local runs zero-setup; docker-compose points this at PostgreSQL.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'traceai.db').as_posix()}")

# "insightface" (ArcFace) is the intended backend; "none" disables image scoring.
FACE_BACKEND = os.getenv("TRACEAI_FACE_BACKEND", "insightface")
TEXT_EMBED_MODEL = os.getenv("TRACEAI_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

API_URL = os.getenv("TRACEAI_API_URL", "http://localhost:8000")

DISCLAIMER = (
    "TraceAI is a research prototype for educational and research purposes only. "
    "It uses public research datasets and synthetic profiles, contains no real missing-person "
    "cases, and is an investigative support tool, not an identification system. "
    "Every lead needs human review."
)
