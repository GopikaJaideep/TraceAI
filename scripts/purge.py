"""Delete expired public tips: python -m scripts.purge [--days N]. Schedule this daily (cron / task)."""
import argparse

from traceai import config, db
from traceai.cases import purge_expired_tips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=config.TIP_RETENTION_DAYS)
    args = ap.parse_args()
    db.init_db()
    session = db.session()
    print(f"Removed {purge_expired_tips(session, args.days)} public tips older than {args.days} days.")


if __name__ == "__main__":
    main()
