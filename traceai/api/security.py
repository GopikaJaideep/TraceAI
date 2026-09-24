"""Authentication for law-enforcement users: password hashing, signed session tokens, lockout, roles.

Real deployments should put this behind the agency's identity provider (SSO with MFA). This module
is the minimum needed so that no officer route is ever reachable without a valid, unexpired token.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from datetime import timedelta

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from traceai import config, db
from traceai.db import CaseAccess, MissingPerson, User, utcnow

log = logging.getLogger(__name__)

_SCRYPT = {"n": 2**14, "r": 8, "p": 1}

_secret = config.SECRET_KEY
if not _secret:
    _secret = secrets.token_urlsafe(48)
    log.warning("TRACEAI_SECRET_KEY is not set; using a throwaway key (sessions reset on restart).")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)

    def b64(b: bytes) -> str:
        return base64.b64encode(b).decode()

    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def check_password_policy(password: str) -> None:
    if len(password) < config.MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters")


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_secret, salt="traceai-officer-session")


def issue_token(user: User) -> str:
    return _serializer().dumps({"uid": user.id})


def read_token(token: str) -> int | None:
    try:
        return int(_serializer().loads(token, max_age=config.TOKEN_TTL_SECONDS)["uid"])
    except (BadSignature, SignatureExpired, KeyError, ValueError, TypeError):
        return None


def source_hash(ip: str) -> str:
    """Keyed hash of the client IP, so repeat tipsters can be spotted without storing the address."""
    return hmac.new(_secret.encode(), ip.encode(), hashlib.sha256).hexdigest()[:32]


def client_ip(request: Request) -> str:
    # Deliberately ignores X-Forwarded-For: it is client-controlled unless a trusted proxy sets it.
    return request.client.host if request.client else "unknown"


def authenticate(session: Session, username: str, password: str) -> User | None:
    """Verify credentials, applying lockout. Returns None on any failure (callers give one message)."""
    user = session.query(User).filter(User.username == username).one_or_none()
    if user is None:
        hash_password(password)  # keep timing similar whether or not the user exists
        return None
    now = utcnow()
    if not user.active or (user.locked_until and user.locked_until > now):
        return None
    if verify_password(password, user.password_hash):
        user.failed_logins = 0
        user.locked_until = None
        session.commit()
        return user
    user.failed_logins += 1
    if user.failed_logins >= config.MAX_FAILED_LOGINS:
        user.locked_until = now + timedelta(minutes=config.LOCKOUT_MINUTES)
        user.failed_logins = 0
    session.commit()
    return None


_bearer = HTTPBearer(auto_error=False)


def get_db():
    session = db.session()
    try:
        yield session
    finally:
        session.close()


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer), session: Session = Depends(get_db)
) -> User:
    unauthorised = HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    if creds is None:
        raise unauthorised
    uid = read_token(creds.credentials)
    user = session.get(User, uid) if uid is not None else None
    if user is None or not user.active:
        raise unauthorised
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Administrator role required")
    return user


def case_for_user(session: Session, user: User, person_id: int) -> MissingPerson:
    """Need-to-know gate. Unknown and not-granted are indistinguishable (both 404), and holding the
    admin role grants no case access: admins manage accounts and audit; they don't browse cases."""
    person = session.get(MissingPerson, person_id)
    granted = (
        person is not None
        and session.query(CaseAccess).filter_by(user_id=user.id, person_id=person_id).first() is not None
    )
    if not granted:
        raise HTTPException(404, "Case not found")
    return person
