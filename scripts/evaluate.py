"""Reproducible ranking evaluation on synthetic data: python -m scripts.evaluate [--seeds 5]

For each synthetic case, every sighting in the run is scored against it and ranked. A sighting is a
positive only if it was generated as a true sighting of that case's person (a *different* LFW photo of
the same identity, degraded to mimic another camera); everything else (decoys with other faces and other
outfits, other cases' sightings, impossible journeys) is a negative.

Settings:
  standard    decoys wear a different outfit or hedge their wording, so text alone can separate them.
  hard        decoys share the outfit, wording, places and times of true sightings; only the face differs.
              (Every decoy carries a photo but one true sighting per person does not, which favours fusion.)
  photo-rate  like hard, but every sighting keeps its photo with probability p, whatever its label. Run with
              several p to see how much the result depends on how many tips carry a photo.

The numbers are for comparing configurations against each other, not a forecast of real-world accuracy.
See the caveats stored in docs/eval.json and shown in the README.
"""
import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path

import numpy as np

# Work in a temp data dir so the evaluation never overwrites the demo's images or database, but reuse
# the (large) LFW cache that already exists.
_real_data = Path(os.getenv("TRACEAI_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
os.environ.setdefault("TRACEAI_LFW_HOME", str(_real_data / "lfw"))
os.environ["TRACEAI_DATA_DIR"] = tempfile.mkdtemp(prefix="traceai-eval-")

from sklearn.metrics import roc_auc_score  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from traceai import config, pipeline, scoring  # noqa: E402
from traceai.db import Base, Lead  # noqa: E402
from traceai.synthetic import build_dataset, save_spec_photo  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "eval.json"
K = 3  # every case has exactly 3 true sightings, so precision@3 equals recall@3

CONFIGS = {
    "random": "Random ordering (expected value)",
    "proximity": "Proximity to last-known place only",
    "context": "Text, place, time and credibility (no face)",
    "face": "Face similarity only",
    "full": "TraceAI: all signals fused",
}

CAVEATS = [
    "Synthetic data: the report templates and the extraction rules were written together, so the text "
    "side is optimistic, especially in the standard setting.",
    "Faces are frontal LFW photos of public figures; 'other camera' is simulated with brightness, blur "
    "and downscaling, not real CCTV.",
    "Only 30 cases per setting (5 seeds x 6 identities): differences of a few points are noise.",
    "Standard and hard settings: only 2 of each person's 3 true sightings carry a photo, so face-only "
    "ranking cannot exceed precision@3 of about 0.67.",
    "Hard setting: every decoy carries a photo but one of each person's three true sightings does not, "
    "which favours fusion, because mismatched faces push decoys down while the photo-less true sighting "
    "is not penalised. The photo-rate settings remove that bias: whether a tip has a photo no longer "
    "depends on its label.",
    "Compare rows with each other; do not read any row as real-world accuracy.",
]


def _is_hard_zero(c: dict) -> bool:
    """Impossible journeys and out-of-order timelines are forced to 0 by scoring, with proximity 0."""
    return c.get("proximity") == 0.0


def _score(name: str, lead: Lead) -> float:
    c = lead.components
    if name == "proximity":
        return c.get("proximity") or 0.0
    if name == "face":
        return c.get("image") or 0.0
    if name == "context":
        return 0.0 if _is_hard_zero(c) else scoring.combine({**c, "image": None})
    return lead.score  # "full"


def _rank_metrics(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> dict:
    order = np.argsort(-(scores + rng.random(len(scores)) * 1e-9))  # random tie-breaking
    ranked = labels[order]
    first = int(np.argmax(ranked)) + 1
    return {
        "p@3": float(ranked[:K].sum() / K), "hit@1": float(ranked[0]),
        "mrr": 1.0 / first, "auroc": float(roc_auc_score(labels, scores)) if len(set(scores)) > 1 else 0.5,
    }


def run(seeds: int, persons: int, hard: bool, photo_rate: float | None = None) -> dict:
    rng = np.random.default_rng(0)
    per_config = {name: [] for name in CONFIGS}
    n_cases = 0
    with_photo = {"true": [0, 0], "decoy": [0, 0]}  # [have a photo, total]
    for seed in range(1, seeds + 1):
        print(f"seed {seed}/{seeds}: building data and scoring ...", flush=True)
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        db = sessionmaker(engine)()
        profiles, specs = build_dataset(n_persons=persons, seed=seed, hard=hard, photo_rate=photo_rate)
        selected = {s["photo_identity"] for s in specs if s["kind"] == "true"}
        clash = selected & {s["photo_identity"] for s in specs if s["kind"] == "decoy"}
        assert not clash, f"decoy photos reuse selected identities {clash}: the ground truth would be wrong"
        for s in specs:
            if s["kind"] in with_photo:
                with_photo[s["kind"]][0] += s["photo"] is not None
                with_photo[s["kind"]][1] += 1
        db.add_all(profiles)
        db.commit()
        truth = {}
        for i, spec in enumerate(specs):
            sighting, _ = pipeline.ingest_sighting(
                db, text=spec["text"], seen_at=spec["seen_at"], image_path=save_spec_photo(spec, i),
                reported_at=spec["seen_at"], source="public",
            )
            truth[sighting.id] = None if spec["truth"] is None else profiles[spec["truth"]].id
        for person in profiles:
            leads = db.query(Lead).filter(Lead.person_id == person.id).all()
            labels = np.array([truth[l.sighting_id] == person.id for l in leads], dtype=float)
            n_cases += 1
            for name in CONFIGS:
                if name == "random":
                    # Keep every draw as its own outcome, so the reported spread is per ranking (like the
                    # other rows) and not the much smaller spread of averages of 300 draws.
                    per_config[name].extend(_rank_metrics(labels, rng.random(len(labels)), rng) for _ in range(300))
                else:
                    per_config[name].append(_rank_metrics(labels, np.array([_score(name, l) for l in leads]), rng))
        db.close()
    metrics = {
        name: {
            m: {"mean": round(statistics.fmean(r[m] for r in rows), 3),
                "sd": round(statistics.pstdev(r[m] for r in rows), 3)}
            for m in rows[0]
        }
        for name, rows in per_config.items()
    }
    share = {k: round(v[0] / v[1], 3) for k, v in with_photo.items()}
    print(f"  tips with a photo: true {share['true']:.0%}, decoy {share['decoy']:.0%}", flush=True)
    return {
        "cases": n_cases, "sightings_ranked_per_case": 7 * persons, "true_per_case": K, "metrics": metrics,
        "photo_share": share, "photo_rate": photo_rate,
    }


def _title_and_note(key: str, rate: float | None) -> tuple[str, str]:
    if key == "standard":
        return "Standard setting", "Decoys have a different outfit or hedged wording, so text alone can separate them."
    if key == "hard":
        return "Hard setting", ("Decoys share the outfit, confident wording, places and times of true sightings; "
                                "only the face differs.")
    pct = round(rate * 100)
    return (f"Hard setting, {pct}% of tips carry a photo",
            f"Look-alike decoys as in the hard setting, but every tip, true or decoy, keeps its photo with "
            f"probability {rate:.0%}, independent of its label.")


def _print_table(key: str, setting: dict, seeds: int, persons: int) -> None:
    per_case = setting["sightings_ranked_per_case"]
    print()
    print(f"Setting: {key} | face model: {config.FACE_MODEL} | {setting['cases']} cases | {per_case} sightings "
          f"ranked per case ({K} true) | random p@3 = {K / per_case:.2f}")
    print()
    print("| Ranking method | Precision@3 | Top-1 correct | MRR | AUROC |")
    print("|---|---|---|---|---|")
    for name, label in CONFIGS.items():
        m = setting["metrics"][name]
        print(f"| {label} | {m['p@3']['mean']:.2f} ± {m['p@3']['sd']:.2f} | {m['hit@1']['mean']:.2f} | "
              f"{m['mrr']['mean']:.2f} | {m['auroc']['mean']:.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--persons", type=int, default=6)
    ap.add_argument("--setting", choices=["standard", "hard", "photo-rate"], default="standard")
    ap.add_argument("--photo-rates", type=float, nargs="+", default=[0.2, 0.5, 1.0],
                    help="for --setting photo-rate: the probabilities that a tip keeps its photo")
    args = ap.parse_args()

    if args.setting == "photo-rate":
        jobs = [(f"photo_rate_{round(r * 100)}", True, r) for r in args.photo_rates]
    else:
        jobs = [(args.setting, args.setting == "hard", None)]

    # Results for every setting live in one file; running some settings keeps the others' numbers.
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    settings = previous.get("settings", {})
    titles = previous.get("setting_titles", {})
    notes = previous.get("setting_notes", {})
    for key, hard, rate in jobs:
        settings[key] = run(args.seeds, args.persons, hard=hard, photo_rate=rate)
        titles[key], notes[key] = _title_and_note(key, rate)
        _print_table(key, settings[key], args.seeds, args.persons)

    result = {
        "face_model": config.FACE_MODEL, "seeds": args.seeds, "labels": CONFIGS, "settings": settings,
        "setting_titles": titles, "setting_notes": notes, "caveats": CAVEATS,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
