"""Safe handling of uploaded photos: verify they really are images, strip metadata, bound the size."""
from __future__ import annotations

import io
import uuid

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from traceai import config

UPLOAD_DIR = config.DATA_DIR / "uploads"
MAX_DIMENSION = 1600
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs


def store_image(raw: bytes) -> str:
    """Validate `raw` as a JPEG/PNG, re-encode it (dropping EXIF, which often carries GPS), and save it
    under a random name. Returns the file path. Raises 400 for anything else, whatever the filename."""
    if len(raw) > config.TIP_MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is too large (limit 5 MB)")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format not in {"JPEG", "PNG"}:
                raise HTTPException(400, "Image must be a JPEG or PNG")
            img = img.convert("RGB")
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
            img.save(path, "JPEG", quality=88)  # no exif= argument: metadata is not carried over
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "File is not a valid image")
    return str(path)
