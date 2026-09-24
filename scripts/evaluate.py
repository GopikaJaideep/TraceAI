"""Reproducible ranking evaluation on synthetic data: python -m scripts.evaluate [--seeds 5]

For each synthetic case, every sighting in the run is scored against it and ranked. A sighting is a
positive only if it was generated as a true sighting of that case's person (a *different* LFW photo of
the same identity, degraded to mimic another camera); everything else (decoys with other faces and other
outfits, other cases' sightings, impossible journeys) is a negative.

The numbers are for comparing configurations against each other, not a forecast of real-world accuracy.
See the caveats printed at the end and in the README.
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


def run(seeds: int, persons: int, hard: bool) -> tuple[dict, int, int]:
    rng = np.random.default_rng(0)
    per_config = {name: [] for name in CONFIGS}
    n_cases = n_sightings = 0
    for seed in range(1, seeds + 1):
        print(f"seed {seed}/{seeds}: building data and scoring ...", flush=True)
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        db = sessionmaker(engine)()
        profiles, specs = build_dataset(n_persons=persons, seed=seed, hard=hard)
        db.add_all(profiles)
        db.commit()
        truth = {}
        for i, spec in enumerate(specs):
            sighting, _ = pipeline.ingest_sighting(
                db, text=spec["text"], seen_at=spec["seen_at"], image_path=save_spec_photo(spec, i),
                reported_at=spec["seen_at"], source="public",
            )
            truth[sighting.id] = None if spec["truth"] is None else profiles[spec["truth"]].id
        n_sightings += len(specs)
        for person in profiles:
            leads = db.query(Lead).filter(Lead.person_id == person.id).all()
            labels = np.array([truth[l.sighting_id] == person.id for l in leads], dtype=float)
            n_cases += 1
            for name in CONFIGS:
                if name == "random":
                    runs = [_rank_metrics(labels, rng.random(len(labels)), rng) for _ in range(300)]
                    per_config[name].append({k: statistics.fmean(r[k] for r in runs) for k in runs[0]})
                else:
                    per_config[name].append(_rank_metrics(labels, np.array([_score(name, l) for l in leads]), rng))
        db.close()
    summary = {}
    for name, rows in per_config.items():
        summary[name] = {
            m: {"mean": round(statistics.fmean(r[m] for r in rows), 3),
                "sd": round(statistics.pstdev(r[m] for r in rows), 3)}
            for m in rows[0]
        }
    return summary, n_cases, n_sightings // persons * persons  # sightings scored per case set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--persons", type=int, default=6)
    ap.add_argument("--setting", choices=["standard", "hard"], default="standard",
                    help="hard: decoys match the outfit and area, so only the face separates them")
    args = ap.parse_args()

    summary, n_cases, _ = run(args.seeds, args.persons, hard=args.setting == "hard")
    per_case = 7 * args.persons  # 3 true + 3 decoys + 1 impossible per person
    setting = {
        "cases": n_cases, "sightings_ranked_per_case": per_case, "true_per_case": K, "metrics": summary,
    }

    # Results for both settings live in one file; running one setting keeps the other's numbers.
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    settings = previous.get("settings", {})
    if "metrics" in previous:  # older flat file from the standard setting
        settings.setdefault("standard", {k: previous[k] for k in ("cases", "sightings_ranked_per_case", "true_per_case", "metrics")})
    settings[args.setting] = setting
    result = {
        "face_model": config.FACE_MODEL, "seeds": args.seeds, "labels": CONFIGS, "settings": settings,
        "setting_notes": {
            "standard": "Decoys have a different outfit or hedged wording, so text alone can separate them.",
            "hard": "Decoys match the outfit, area and confident wording of true sightings; only the face differs.",
        },
        "caveats": [
            "Synthetic data: the report templates and the extraction rules were written together, so the text "
            "side is optimistic, especially in the standard setting.",
            "Faces are frontal LFW photos of public figures; 'other camera' is simulated with brightness, blur "
            "and downscaling, not real CCTV.",
            "Only 30 cases per setting (5 seeds x 6 identities): differences of a few points are noise.",
            "Only 2 of each person's 3 true sightings carry a photo, so face-only ranking cannot exceed "
            "precision@3 of about 0.67.",
            "Compare rows with each other; do not read any row as real-world accuracy.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print()
    print(f"Setting: {args.setting} | face model: {config.FACE_MODEL} | {n_cases} cases | {per_case} sightings "
          f"ranked per case ({K} true) | random p@3 = {K / per_case:.2f}")
    print()
    print("| Ranking method | Precision@3 | Top-1 correct | MRR | AUROC |")
    print("|---|---|---|---|---|")
    for name, label in CONFIGS.items():
        m = summary[name]
        print(f"| {label} | {m['p@3']['mean']:.2f} ± {m['p@3']['sd']:.2f} | {m['hit@1']['mean']:.2f} | "
              f"{m['mrr']['mean']:.2f} | {m['auroc']['mean']:.2f} |")
    print()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
