"""Prepare a photo-free copy of the demo database for README screenshots.

The seeded photos are LFW faces, which are real people, so they must not appear in published images.
This copies the database and repoints every photo at a neutral placeholder avatar.

    python -m scripts.make_screenshot_data [--out data_shots]
"""
import argparse
import shutil
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw

from traceai import config

PALETTE = [(196, 214, 210), (214, 205, 226), (226, 214, 196), (226, 200, 206), (200, 212, 232), (210, 224, 196)]


def avatar(path: Path, colour: tuple[int, int, int]) -> None:
    img = Image.new("RGB", (250, 250), colour)
    d = ImageDraw.Draw(img)
    ink = tuple(max(0, c - 70) for c in colour)
    d.ellipse((85, 45, 165, 125), fill=ink)  # head
    d.ellipse((45, 140, 205, 330), fill=ink)  # shoulders
    img.save(path, "JPEG", quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_shots")
    out = Path(ap.parse_args().out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    avatars = []
    for i, colour in enumerate(PALETTE):
        p = (out / "img" / f"avatar_{i}.jpg").resolve()
        avatar(p, colour)
        avatars.append(str(p))

    src = config.DATA_DIR / "traceai.db"
    dst = out / "traceai.db"
    shutil.copyfile(src, dst)
    con = sqlite3.connect(dst)
    for table, col in (("missing_persons", "photo_path"), ("sightings", "image_path")):
        rows = con.execute(f"select id from {table} where {col} is not null order by id").fetchall()
        for n, (row_id,) in enumerate(rows):
            con.execute(f"update {table} set {col}=? where id=?", (avatars[n % len(avatars)], row_id))
    con.commit()
    con.close()
    print(f"Wrote {dst} with placeholder photos in {out / 'img'}")


if __name__ == "__main__":
    main()
