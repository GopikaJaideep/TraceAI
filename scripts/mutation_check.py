"""Prove the tests bite: break each safeguard on purpose and require the suite to fail.

    python -m scripts.mutation_check

Each entry replaces one snippet of source, runs the tests, restores the file, and reports whether a
test caught the change. A surviving mutant means a safeguard is not really tested, and the script exits
non-zero. Files are always restored, even on Ctrl-C.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (name, file, original snippet, broken replacement)
MUTANTS = [
    ("need-to-know check disabled", "traceai/api/security.py", "    if not granted:", "    if False:"),
    ("family-violence attestation skipped", "traceai/cases.py", "    if not family_violence_screened:", "    if False:"),
    ("honeypot ignored", "traceai/api/public_app.py", "    if website:", "    if False:"),
    ("tip rate limit removed", "traceai/api/public_app.py", "    SHORT_WINDOW.check(source)", "    pass"),
    ("closing a case keeps the face embedding", "traceai/cases.py", "    person.embedding = None", "    pass"),
    ("upload format check removed", "traceai/api/images.py", 'if img.format not in {"JPEG", "PNG"}:', "if False:"),
    ("audit chain verification weakened", "traceai/api/audit.py", "row.hash != _digest(prev, row)", "False"),
    ("closed cases still receive leads", "traceai/pipeline.py", 'MissingPerson.status == "open"', "MissingPerson.id > 0"),
    ("predating sightings not rejected", "traceai/scoring.py", "        if hours < 0:", "        if False:"),
    ("login lockout removed", "traceai/api/security.py", "    if user.failed_logins >= config.MAX_FAILED_LOGINS:", "    if False:"),
]


def tests_pass() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:warnings", "tests"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return proc.returncode == 0


def main() -> int:
    print("Baseline: unmodified code must pass ...", flush=True)
    if not tests_pass():
        print("The unmodified test suite already fails; fix that first.")
        return 2

    survivors = []
    for name, rel, old, new in MUTANTS:
        path = ROOT / rel
        original = path.read_bytes()
        text = original.decode("utf-8")
        if old not in text:
            print(f"  ?  {name}: snippet not found in {rel} (update scripts/mutation_check.py)")
            survivors.append(name)
            continue
        try:
            path.write_bytes(text.replace(old, new, 1).encode("utf-8"))
            caught = not tests_pass()
        finally:
            path.write_bytes(original)
        print(f"  {'caught  ' if caught else 'SURVIVED'} {name}", flush=True)
        if not caught:
            survivors.append(name)

    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} broken safeguards were caught by the tests.")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
