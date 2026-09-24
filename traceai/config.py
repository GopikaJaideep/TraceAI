"""Runtime configuration, read from environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TRACEAI_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
# The LFW cache is ~1.4 GB, so let it live somewhere shared (e.g. when evaluating in a temp data dir).
LFW_HOME = Path(os.getenv("TRACEAI_LFW_HOME", DATA_DIR / "lfw"))

# SQLite keeps local runs zero-setup; docker-compose points this at PostgreSQL.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'traceai.db').as_posix()}")

# "insightface" (ArcFace) is the intended backend; "none" disables image scoring.
FACE_BACKEND = os.getenv("TRACEAI_FACE_BACKEND", "insightface")
# buffalo_l (ArcFace R50, ~280 MB) is the accurate default; buffalo_sc (~15 MB) suits slow links.
FACE_MODEL = os.getenv("TRACEAI_FACE_MODEL", "buffalo_l")
TEXT_EMBED_MODEL = os.getenv("TRACEAI_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

API_URL = os.getenv("TRACEAI_API_URL", "http://localhost:8000")

DISCLAIMER = (
    "TraceAI is a research prototype for educational and research purposes only. "
    "It uses public research datasets and synthetic profiles, contains no real missing-person "
    "cases, and is an investigative support tool, not an identification system. "
    "Every lead needs human review."
)

# --- Access control -------------------------------------------------------------------------------
# Signs officer session tokens and keys the hash of tipster IPs. Set a long random value in any real
# deployment; without one a throwaway key is generated (sessions then end whenever the API restarts).
SECRET_KEY = os.getenv("TRACEAI_SECRET_KEY", "")
TOKEN_TTL_SECONDS = int(os.getenv("TRACEAI_TOKEN_TTL", str(4 * 3600)))
MIN_PASSWORD_LENGTH = 12
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15

# --- Public tip portal ----------------------------------------------------------------------------
TIP_MAX_TEXT = 2000
TIP_MAX_IMAGE_BYTES = 5 * 1024 * 1024
TIP_RATE_PER_10MIN = int(os.getenv("TRACEAI_TIP_RATE_10MIN", "5"))
TIP_RATE_PER_DAY = int(os.getenv("TRACEAI_TIP_RATE_DAY", "20"))
TIP_FLAG_SOURCE_VOLUME = 3  # flag the 3rd and later tip from one source within 24 h
LOGIN_RATE_PER_MIN = 10

# Public tips are deleted after this many days unless an officer marked a resulting lead useful.
TIP_RETENTION_DAYS = int(os.getenv("TRACEAI_TIP_RETENTION_DAYS", "90"))
