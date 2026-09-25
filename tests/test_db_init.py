"""Schema creation must be safe when several services start at once (see db.init_db)."""
from sqlalchemy import event, inspect

from traceai import db
from traceai.db import Base


def _tables():
    return set(inspect(db.engine()).get_table_names())


def test_init_db_creates_tables_and_is_repeatable(env):
    Base.metadata.drop_all(db.engine())
    assert "users" not in _tables()
    db.init_db()
    db.init_db()  # a second service starting later must not fail
    assert {"users", "missing_persons", "sightings", "leads", "audit_log", "case_access"} <= _tables()


def test_postgres_takes_an_advisory_lock_before_creating_tables(env, monkeypatch):
    calls = []
    eng = db.engine()
    Base.metadata.drop_all(eng)

    # SQLite has no advisory locks, so give it a stand-in function that records the call and the moment
    # it happens relative to table creation.
    def on_connect(dbapi_conn, _record):
        dbapi_conn.create_function("pg_advisory_xact_lock", 1, lambda lock_id: calls.append(("lock", lock_id)) or 0)

    event.listen(eng, "connect", on_connect)
    eng.dispose()  # drop pooled connections so the listener applies to the next one
    monkeypatch.setattr(eng.dialect, "name", "postgresql")
    try:
        event.listen(eng, "before_cursor_execute", lambda *a: calls.append(("sql", a[2].strip()[:12])))
        db.init_db()
    finally:
        event.remove(eng, "connect", on_connect)
        eng.dispose()

    assert calls[0] == ("lock", db._SCHEMA_LOCK_ID) or calls[1] == ("lock", db._SCHEMA_LOCK_ID)
    first_create = next(i for i, c in enumerate(calls) if c[0] == "sql" and c[1].startswith("CREATE TABLE"))
    lock_at = next(i for i, c in enumerate(calls) if c[0] == "lock")
    assert lock_at < first_create, "the lock must be taken before any table is created"


def test_sqlite_does_not_try_to_take_a_lock(env):
    eng = db.engine()
    statements = []
    event.listen(eng, "before_cursor_execute", lambda *a: statements.append(a[2]))
    Base.metadata.drop_all(eng)
    db.init_db()
    assert not any("advisory" in s for s in statements)
