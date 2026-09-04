from pathlib import Path
import csv
import os
import re
import shutil
import subprocess
import time

RUN_NAME = "safety_v27_geometry_seed1_25m"

DATASET = (
    "/mnt/e/AI_Challenge/datasets/splits/"
    "sanity_val_256/sanity_val_256.tfrecord@256"
)

RUN_DIR = Path("runs") / RUN_NAME
MODEL_DIR = RUN_DIR / "model"

OUT_CSV = Path("analysis/v27_25m_sanity_sweep.csv")
LOG_PATH = Path("analysis/v27_25m_sanity_sweep.log")

ALIAS_PREFIX = "_v27_sanity_"

METRICS = [
    "rideflux_aggregate_score",
    "progress_ratio",
    "comfort",
    "at_fault_collision",
    "overlap",
    "offroad_in_box",
    "accuracy",
]


def checkpoint_step(path: Path):
    if path.name == "model_final.pkl":
        return 10**18

    m = re.search(r"model_(\d+)\.pkl", path.name)

    if not m:
        return -1

    return int(m.group(1))


def read_result(path: Path):
    values = {}

    for line in path.read_text().splitlines():
        line = line.strip()

        for key in METRICS:
            if line.startswith(key):
                parts = re.split(r"[:=]", line, maxsplit=1)

                if len(parts) == 2:
                    try:
                        values[key] = float(parts[1].strip())
                    except ValueError:
                        pass

    return values


def remove_alias(alias_dir: Path):
    if alias_dir.is_symlink():
        alias_dir.unlink()
    elif alias_dir.exists():
        shutil.rmtree(alias_dir)


checkpoints = sorted(
    list(MODEL_DIR.glob("model_*.pkl"))
    + [MODEL_DIR / "model_final.pkl"],
    key=checkpoint_step,
)

checkpoints = [
    p for p in checkpoints
    if p.exists()
]

if not checkpoints:
    raise RuntimeError(
        f"No checkpoints found in {MODEL_DIR}"
    )

print("=" * 100)
print("V27 25M SANITY CHECKPOINT SWEEP")
print("=" * 100)
print("Run       :", RUN_NAME)
print("Dataset   :", DATASET)
print("Candidates:", len(checkpoints))
print()

LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

rows = []

with LOG_PATH.open("w") as log_file:

    for i, ckpt in enumerate(checkpoints, start=1):

        stem = ckpt.stem

        if stem == "model_final":
            alias_name = f"{ALIAS_PREFIX}final"
        else:
            step = checkpoint_step(ckpt)
            alias_name = f"{ALIAS_PREFIX}{step}"

        alias_dir = Path("runs") / alias_name
        alias_model_dir = alias_dir / "model"

        remove_alias(alias_dir)

        alias_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        alias_model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Reuse exact training config.
        os.symlink(
            RUN_DIR.resolve() / ".hydra",
            alias_dir / ".hydra",
            target_is_directory=True,
        )

        # Evaluator always prefers model_final.pkl,
        # so point it to the checkpoint being tested.
        os.symlink(
            ckpt.resolve(),
            alias_model_dir / "model_final.pkl",
        )

        print(
            f"[{i:03d}/{len(checkpoints):03d}] "
            f"{ckpt.name}",
            flush=True,
        )

        start = time.time()

        cmd = [
            "uv",
            "run",
            "python",
            "vmax/scripts/evaluate/evaluate.py",
            "--waymo_dataset=true",
            f"--path_dataset={DATASET}",
            "--sdc_actor=ai",
            f"--path_model={alias_name}",
            "--batch_size=64",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        log_file.write(
            "\n"
            + "=" * 100
            + f"\n{ckpt.name}\n"
            + "=" * 100
            + "\n"
        )
        log_file.flush()

        result = subprocess.run(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        elapsed = time.time() - start

        if result.returncode != 0:
            print(
                f"    ERROR returncode={result.returncode}",
                flush=True,
            )

            rows.append(
                {
                    "checkpoint": ckpt.name,
                    "step": (
                        checkpoint_step(ckpt)
                        if ckpt.name != "model_final.pkl"
                        else "FINAL"
                    ),
                    "status": "ERROR",
                }
            )

            continue

        # Locate this alias's evaluation result.
        matches = list(
            Path("benchmark").glob(
                f"**/{alias_name}/model_final/"
                "evaluation_results.txt"
            )
        )

        if not matches:
            print(
                "    ERROR: evaluation_results.txt not found",
                flush=True,
            )

            rows.append(
                {
                    "checkpoint": ckpt.name,
                    "step": (
                        checkpoint_step(ckpt)
                        if ckpt.name != "model_final.pkl"
                        else "FINAL"
                    ),
                    "status": "NO_RESULT",
                }
            )

            continue

        result_path = max(
            matches,
            key=lambda p: p.stat().st_mtime,
        )

        metrics = read_result(
            result_path
        )

        row = {
            "checkpoint": ckpt.name,
            "step": (
                checkpoint_step(ckpt)
                if ckpt.name != "model_final.pkl"
                else "FINAL"
            ),
            "status": "OK",
            **metrics,
            "eval_seconds": elapsed,
        }

        rows.append(row)

        rf = metrics.get(
            "rideflux_aggregate_score",
            float("nan"),
        )

        prog = metrics.get(
            "progress_ratio",
            float("nan"),
        )

        comfort = metrics.get(
            "comfort",
            float("nan"),
        )

        acc = metrics.get(
            "accuracy",
            float("nan"),
        )

        print(
            f"    RF={rf:.5f} "
            f"P={prog:.5f} "
            f"C={comfort:.5f} "
            f"Acc={acc:.5f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )


# ============================================================
# Save raw CSV
# ============================================================

fieldnames = [
    "checkpoint",
    "step",
    "status",
    *METRICS,
    "eval_seconds",
]

with OUT_CSV.open(
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)


# ============================================================
# Summary
# ============================================================

valid = [
    r for r in rows
    if (
        r.get("status") == "OK"
        and "rideflux_aggregate_score" in r
    )
]

valid = sorted(
    valid,
    key=lambda r: r[
        "rideflux_aggregate_score"
    ],
    reverse=True,
)

print()
print("=" * 100)
print("TOP 15 — V27 SANITY")
print("=" * 100)

for rank, row in enumerate(
    valid[:15],
    start=1,
):
    print(
        f"{rank:02d}. "
        f"{row['checkpoint']:<24} "
        f"RF={row.get('rideflux_aggregate_score', float('nan')):.5f} "
        f"P={row.get('progress_ratio', float('nan')):.5f} "
        f"C={row.get('comfort', float('nan')):.5f} "
        f"Coll={row.get('at_fault_collision', float('nan')):.5f} "
        f"Ov={row.get('overlap', float('nan')):.5f} "
        f"Off={row.get('offroad_in_box', float('nan')):.5f} "
        f"Acc={row.get('accuracy', float('nan')):.5f}"
    )

if valid:
    best = valid[0]

    print()
    print("=" * 100)
    print("BEST CHECKPOINT")
    print("=" * 100)

    for key, value in best.items():
        print(
            f"{key:<28}: {value}"
        )

print()
print("=" * 100)
print("SAVED")
print("=" * 100)
print(OUT_CSV)
print(LOG_PATH)
