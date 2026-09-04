from pathlib import Path
import csv
import re

ROOT = Path(
    "benchmark/ai/mnt/e/AI_Challenge/datasets/splits/"
    "sanity_val_256/sanity_val_256.tfrecord@256"
)

OUT = Path(
    "analysis/v27_25m_sanity_sweep_recovered.csv"
)

METRICS = [
    "rideflux_aggregate_score",
    "progress_ratio",
    "comfort",
    "at_fault_collision",
    "overlap",
    "offroad_in_box",
    "accuracy",
]


def parse_result(path):
    values = {}

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        # Actual evaluator format:
        #
        # accuracy                 0.94141
        # progress_ratio           0.87258
        # rideflux_aggregate_score 0.84545
        #
        parts = line.split()

        if len(parts) < 2:
            continue

        key = parts[0]

        if key not in METRICS:
            continue

        try:
            values[key] = float(parts[-1])
        except ValueError:
            pass

    return values


def alias_to_checkpoint(alias):
    prefix = "_v27_sanity_"

    if not alias.startswith(prefix):
        return None, None

    suffix = alias[len(prefix):]

    if suffix == "final":
        return "model_final.pkl", None

    if suffix.isdigit():
        step = int(suffix)
        return f"model_{step}.pkl", step

    return None, None


files = sorted(
    ROOT.glob(
        "_v27_sanity_*/model_final/"
        "evaluation_results.txt"
    )
)

print("=" * 110)
print("V27 SANITY RESULT RECOVERY")
print("=" * 110)
print("Result files:", len(files))

rows = []

for path in files:
    # path:
    # .../_v27_sanity_123456/model_final/evaluation_results.txt

    alias = path.parent.parent.name

    checkpoint, step = alias_to_checkpoint(
        alias
    )

    if checkpoint is None:
        continue

    metrics = parse_result(path)

    row = {
        "checkpoint": checkpoint,
        "step": (
            "FINAL"
            if step is None
            else step
        ),
        **metrics,
    }

    rows.append(row)


# ============================================================
# Validation
# ============================================================

print("Parsed rows :", len(rows))

complete = [
    r for r in rows
    if all(
        key in r
        for key in METRICS
    )
]

print(
    "Complete rows:",
    len(complete),
)

incomplete = [
    r for r in rows
    if not all(
        key in r
        for key in METRICS
    )
]

if incomplete:
    print()
    print("WARNING — incomplete rows:")

    for r in incomplete:
        print(
            r["checkpoint"],
            [
                key
                for key in METRICS
                if key not in r
            ],
        )


# ============================================================
# Save recovered CSV
# ============================================================

fieldnames = [
    "checkpoint",
    "step",
    *METRICS,
]

with OUT.open(
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(complete)


# ============================================================
# Sort by RideFlux
# ============================================================

ranked = sorted(
    complete,
    key=lambda r: r[
        "rideflux_aggregate_score"
    ],
    reverse=True,
)


print()
print("=" * 110)
print("TOP 15 — V27 SANITY")
print("=" * 110)

for rank, r in enumerate(
    ranked[:15],
    start=1,
):
    print(
        f"{rank:02d}. "
        f"{r['checkpoint']:<24} "
        f"RF={r['rideflux_aggregate_score']:.5f}  "
        f"P={r['progress_ratio']:.5f}  "
        f"C={r['comfort']:.5f}  "
        f"Coll={r['at_fault_collision']:.5f}  "
        f"Ov={r['overlap']:.5f}  "
        f"Off={r['offroad_in_box']:.5f}  "
        f"Acc={r['accuracy']:.5f}"
    )


if ranked:
    best = ranked[0]

    print()
    print("=" * 110)
    print("BEST CHECKPOINT")
    print("=" * 110)

    print(
        "Checkpoint :",
        best["checkpoint"],
    )

    print(
        "Step       :",
        best["step"],
    )

    print(
        "RideFlux   :",
        f"{best['rideflux_aggregate_score']:.5f}",
    )

    print(
        "Progress   :",
        f"{best['progress_ratio']:.5f}",
    )

    print(
        "Comfort    :",
        f"{best['comfort']:.5f}",
    )

    print(
        "Collision  :",
        f"{best['at_fault_collision']:.5f}",
    )

    print(
        "Overlap    :",
        f"{best['overlap']:.5f}",
    )

    print(
        "OffroadBox :",
        f"{best['offroad_in_box']:.5f}",
    )

    print(
        "Accuracy   :",
        f"{best['accuracy']:.5f}",
    )


# ============================================================
# V26 benchmark comparison
# ============================================================

V26 = {
    "rideflux_aggregate_score": 0.90799,
    "progress_ratio": 0.94055,
    "comfort": 0.94634,
    "at_fault_collision": 0.03255,
    "overlap": 0.04427,
    "offroad_in_box": 0.00130,
    "accuracy": 0.95443,
}

if ranked:
    b = ranked[0]

    print()
    print("=" * 110)
    print("BEST V27 vs BEST V26 SANITY")
    print("=" * 110)

    names = [
        ("RideFlux", "rideflux_aggregate_score"),
        ("Progress", "progress_ratio"),
        ("Comfort", "comfort"),
        ("Collision", "at_fault_collision"),
        ("Overlap", "overlap"),
        ("OffroadBox", "offroad_in_box"),
        ("Accuracy", "accuracy"),
    ]

    for name, key in names:
        delta = b[key] - V26[key]

        print(
            f"{name:<12} "
            f"V26={V26[key]:.5f}  "
            f"V27={b[key]:.5f}  "
            f"Δ={delta:+.5f}"
        )


print()
print("=" * 110)
print("SAVED")
print("=" * 110)
print(OUT)
