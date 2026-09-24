"""Append-only, hash-chained audit log.

Each row's hash covers its own content and the previous row's hash, so editing or deleting a past row
breaks every hash after it and `verify_chain` reports where. That makes tampering *detectable*; it
does not make it impossible for someone with database admin rights, so production should also ship the
log to write-once storage.
"""
from __future__ import annotations

import hashlib
import json

from fastapi import Request
from sqlalchemy.orm import Session

from traceai.api.security import client_ip
from traceai.db import AuditLog, User, utcnow

GENESIS = "0" * 64


def _digest(prev: str, row: AuditLog) -> str:
    body = json.dumps(
        [row.ts.isoformat(), row.user_id, row.username, row.action, row.object_type, row.object_id,
         row.ip, row.detail],
        sort_keys=True, default=str,
    )
    return hashlib.sha256((prev + body).encode()).hexdigest()


def record(
    session: Session, action: str, *, user: User | None = None, username: str = "",
    object_type: str = "", object_id: int | None = None, request: Request | None = None,
    detail: dict | None = None,
) -> AuditLog:
    last = session.query(AuditLog).order_by(AuditLog.id.desc()).first()
    prev = last.hash if last else GENESIS
    row = AuditLog(
        ts=utcnow(), user_id=user.id if user else None, username=user.username if user else username,
        action=action, object_type=object_type, object_id=object_id,
        ip=client_ip(request) if request else "", detail=detail or {}, prev_hash=prev,
    )
    row.hash = _digest(prev, row)
    session.add(row)
    session.commit()
    return row


def verify_chain(session: Session) -> tuple[bool, int | None]:
    """(True, None) if intact, else (False, id of the first row that does not verify)."""
    prev = GENESIS
    for row in session.query(AuditLog).order_by(AuditLog.id):
        if row.prev_hash != prev or row.hash != _digest(prev, row):
            return False, row.id
        prev = row.hash
    return True, None
