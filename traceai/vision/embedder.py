"""Face embeddings via InsightFace (ArcFace, 512-d). Similarity is cosine over L2-normalised vectors."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from traceai import config

log = logging.getLogger(__name__)


class FaceBackendUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _app():
    if config.FACE_BACKEND == "none":
        raise FaceBackendUnavailable("TRACEAI_FACE_BACKEND=none")
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise FaceBackendUnavailable(
            "insightface is not installed; image similarity is disabled (pip install insightface)"
        ) from exc
    app = FaceAnalysis(
        name=config.FACE_MODEL, allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app


def available() -> bool:
    try:
        _app()
        return True
    except Exception as exc:
        log.warning("Face backend unavailable: %s", exc)
        return False


def embed(image_bgr: np.ndarray) -> np.ndarray | None:
    """Embedding of the most prominent face, or None if no face is detected / backend is off."""
    if not available():
        return None
    faces = _app().get(image_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    vec = np.asarray(face.normed_embedding, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def embed_file(path: str) -> np.ndarray | None:
    import cv2

    img = cv2.imread(path)
    return None if img is None else embed(img)


def cosine(a: list[float] | np.ndarray | None, b: list[float] | np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
