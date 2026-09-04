from pathlib import Path
import sys

SRC = Path(
    "analysis/v28_selected_failure_action_compare.py"
)

text = SRC.read_text()

MARKER = (
    "# ============================================================\n"
    "# V27 REPLAY"
)

if MARKER not in text:
    raise RuntimeError(
        "Could not find V27 REPLAY marker in source script."
    )

# 기존 스크립트에서
# imports / config / helper / replay_model 정의까지만 실행.
prefix = text.split(
    MARKER,
    1,
)[0]

ns = {
    "__name__": "_v28_selected_worker_base",
}

exec(
    compile(
        prefix,
        str(SRC),
        "exec",
    ),
    ns,
)


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python v28_selected_failure_action_worker.py V27|V28"
    )

tag = sys.argv[1].upper()


if tag == "V27":

    model = ns["MODEL_V27"]
    expected_eval = ns["v27_eval"]

    out = Path(
        "analysis/"
        "v28_selected_failure_action_v27.csv"
    )

elif tag == "V28":

    model = ns["MODEL_V28"]
    expected_eval = ns["v28_eval"]

    out = Path(
        "analysis/"
        "v28_selected_failure_action_v28.csv"
    )

else:

    raise SystemExit(
        f"Unknown tag: {tag}"
    )


rows, mismatches, reference_failures = (
    ns["replay_model"](
        model,
        tag,
        expected_eval,
    )
)


print()
print("=" * 100)
print(f"{tag} FINAL CONSISTENCY")
print("=" * 100)

print(
    "Rows:",
    len(rows),
)

print(
    "Rollout mismatches:",
    len(mismatches),
)

print(
    "Reference failures:",
    len(reference_failures),
)


rows.to_csv(
    out,
    index=False,
)


if (
    mismatches
    or reference_failures
):

    print()
    print(
        "WARNING: inconsistent scenarios were excluded "
        "from the saved replay rows."
    )

    print(
        "Mismatches:",
        mismatches,
    )

    print(
        "Reference failures:",
        reference_failures,
    )

print()
print(
    "Saved:",
    out,
)

