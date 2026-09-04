from pathlib import Path
import ast
import gc
import runpy

import jax
import numpy as np
import pandas as pd

from vmax.scripts.evaluate import utils
from vmax.simulator import datasets, make_data_generator


# ============================================================
# CONFIG
# ============================================================

DATASET = (
    "/mnt/e/AI_Challenge/datasets/splits/"
    "secondary_val_1024/secondary_val_1024.tfrecord@1024"
)

ROOT = Path(
    "benchmark/ai/mnt/e/AI_Challenge/datasets/splits/"
    "secondary_val_1024/"
    "secondary_val_1024.tfrecord@1024"
)

MODEL_V27 = "v27_secondary_frozen_21506560"
MODEL_V28 = "v28sec_19970560"

V27_EVAL = (
    ROOT
    / MODEL_V27
    / "model_final"
    / "evaluation_episodes.csv"
)

V28_EVAL = (
    ROOT
    / MODEL_V28
    / "model_final"
    / "evaluation_episodes.csv"
)

SOURCE_ANALYZER = Path(
    "analysis/analyze_v28_action_response.py"
)

OUT_RAW = Path(
    "analysis/v28_selected_failure_action_raw.csv"
)

OUT_COMPARE = Path(
    "analysis/v28_selected_failure_action_compare.csv"
)

MAX_OBJECTS = 64
NUM_CLOSEST_OBJECTS = 16

BATCH_SIZE = 64
DT = 0.1

LEADS = [
    3.0,
    2.0,
    1.0,
    0.5,
]


# ============================================================
# SELECTED SCENARIOS
#
# V28 NEW FAILURES
# ============================================================

GROUP_MAP = {
    # Early / mid catastrophic collision
    24:   "EARLY_COLLISION",
    2101: "EARLY_COLLISION",
    2139: "EARLY_COLLISION",
    2922: "EARLY_COLLISION",

    # Progress mostly preserved but new collision
    1124: "LATE_COLLISION",
    1237: "LATE_COLLISION",

    # Progress 1.0 but new offroad
    1424: "LATE_OFFROAD",
    2747: "LATE_OFFROAD",
}

SELECTED = sorted(
    GROUP_MAP.keys()
)


# ============================================================
# REUSE VERIFIED FUNCTIONS WITHOUT EXECUTING OLD MAIN
#
# We only import function definitions from the old analyzer.
# Its old V26/V27 main analysis is NOT executed.
# ============================================================

source = SOURCE_ANALYZER.read_text()

tree = ast.parse(
    source,
    filename=str(SOURCE_ANALYZER),
)

reuse_body = []

for node in tree.body:

    if isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        reuse_body.append(
            node
        )


reuse_module = ast.Module(
    body=reuse_body,
    type_ignores=[],
)

reuse_ns = {
    "__name__": "_reused_v28_analyzer",
}

exec(
    compile(
        reuse_module,
        str(SOURCE_ANALYZER),
        "exec",
    ),
    reuse_ns,
)


# Globals used by reused functions.
reuse_ns["DT"] = DT
reuse_ns["MAX_OBJECTS"] = MAX_OBJECTS
reuse_ns["NUM_CLOSEST_OBJECTS"] = NUM_CLOSEST_OBJECTS
reuse_ns["BATCH_SIZE"] = BATCH_SIZE


interaction_snapshot = reuse_ns[
    "interaction_snapshot"
]

rollout = reuse_ns[
    "rollout"
]


# ============================================================
# VERIFIED BATCH ITEM EXTRACTION
# ============================================================

rankmod = runpy.run_path(
    "analysis/diagnose_v26_collision_rank.py"
)

take_batch_item = rankmod[
    "take_batch_item"
]


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

if not V27_EVAL.exists():
    raise FileNotFoundError(
        V27_EVAL
    )

if not V28_EVAL.exists():
    raise FileNotFoundError(
        V28_EVAL
    )


v27_eval = (
    pd.read_csv(
        V27_EVAL
    )
    .set_index(
        "scenario_index"
    )
)

v28_eval = (
    pd.read_csv(
        V28_EVAL
    )
    .set_index(
        "scenario_index"
    )
)


for idx in SELECTED:

    if idx not in v27_eval.index:
        raise RuntimeError(
            f"Scenario {idx} missing from V27 eval"
        )

    if idx not in v28_eval.index:
        raise RuntimeError(
            f"Scenario {idx} missing from V28 eval"
        )


# ============================================================
# IMPORTANT:
#
# Every V27/V28 comparison uses V28 failure episode length
# as the common absolute reference time.
# ============================================================

REFERENCE_STEP = {
    idx: int(
        v28_eval.loc[
            idx,
            "episode_length",
        ]
    )
    for idx in SELECTED
}


print()
print("=" * 110)
print("V27 ↔ V28 SELECTED FAILURE ACTION COMPARISON")
print("=" * 110)

print(
    "V27:",
    MODEL_V27,
)

print(
    "V28:",
    MODEL_V28,
)

print(
    "Selected:",
    SELECTED,
)

print(
    "Leads:",
    LEADS,
)

print()

for idx in SELECTED:

    print(
        f"{idx:4d}  "
        f"{GROUP_MAP[idx]:<18} "
        f"V27_len="
        f"{int(v27_eval.loc[idx, 'episode_length']):2d}  "
        f"V28_len="
        f"{int(v28_eval.loc[idx, 'episode_length']):2d}  "
        f"reference="
        f"{REFERENCE_STEP[idx]:2d}"
    )


# ============================================================
# REPLAY ONE MODEL
# ============================================================

def replay_model(
    model_name,
    model_tag,
    expected_eval,
):

    print()
    print("=" * 110)
    print(
        f"SETUP {model_tag}: {model_name}"
    )
    print("=" * 110)

    env, step_fn, _, _ = (
        utils.setup_evaluation(
            policy_type="ai",
            path_model=model_name,
            source_dir="runs",
            path_dataset=DATASET,
            eval_name=(
                f"analysis/"
                f"v28_selected_{model_tag.lower()}_tmp"
            ),
            max_num_objects=MAX_OBJECTS,
            noisy_init=False,
            sdc_paths_from_data=False,
        )
    )

    reset_fn = jax.jit(
        env.reset
    )

    step_fn = jax.jit(
        step_fn
    )

    feature_extractor = (
        env.get_wrapper_attr(
            "features_extractor"
        )
    )

    data_generator = make_data_generator(
        path=datasets.get_dataset(
            DATASET
        ),
        max_num_objects=MAX_OBJECTS,
        include_sdc_paths=False,
        batch_dims=(
            BATCH_SIZE,
            1,
        ),
        seed=0,
        repeat=1,
    )

    base_key = (
        jax.random.PRNGKey(0)
    )

    rows = []

    done_selected = set()

    mismatches = []

    insufficient_reference = []

    global_start = 0


    for batch in data_generator:

        batch_end = (
            global_start
            + BATCH_SIZE
        )

        wanted = [
            idx
            for idx in SELECTED
            if (
                global_start
                <= idx
                < batch_end
                and idx
                not in done_selected
            )
        ]


        for scenario_index in wanted:

            local_idx = (
                scenario_index
                - global_start
            )

            scenario = take_batch_item(
                batch,
                local_idx,
            )

            print(
                f"[{model_tag}] "
                f"{GROUP_MAP[scenario_index]:<18} "
                f"scenario {scenario_index}",
                flush=True,
            )


            history, action_history = rollout(
                scenario,
                scenario_index,
                reset_fn,
                step_fn,
                base_key,
            )


            actual_steps = (
                len(history)
                - 1
            )

            expected_steps = int(
                expected_eval.loc[
                    scenario_index,
                    "episode_length",
                ]
            )


            # ------------------------------------------------
            # Strong consistency guard.
            #
            # If replay under current source does not reproduce
            # the stored evaluation episode length, do NOT use
            # that scenario for interpretation.
            # ------------------------------------------------

            if actual_steps != expected_steps:

                mismatches.append(
                    (
                        scenario_index,
                        expected_steps,
                        actual_steps,
                    )
                )

                print(
                    "  !! ROLLOUT MISMATCH:",
                    "expected",
                    expected_steps,
                    "actual",
                    actual_steps,
                )

                done_selected.add(
                    scenario_index
                )

                continue


            ref = int(
                REFERENCE_STEP[
                    scenario_index
                ]
            )


            # V27 should normally survive at least until
            # the V28 failure reference.
            if actual_steps < ref:

                insufficient_reference.append(
                    (
                        scenario_index,
                        ref,
                        actual_steps,
                    )
                )

                print(
                    "  !! CANNOT REACH COMMON REFERENCE:",
                    "ref",
                    ref,
                    "actual",
                    actual_steps,
                )

                done_selected.add(
                    scenario_index
                )

                continue


            for lead in LEADS:

                lead_steps = int(
                    round(
                        lead / DT
                    )
                )

                frame = (
                    ref
                    - lead_steps
                )

                if frame < 0:
                    continue

                if frame >= len(history):
                    continue

                if frame >= len(action_history):
                    continue


                feat = interaction_snapshot(
                    history[
                        frame
                    ],
                    feature_extractor,
                )

                if feat is None:
                    continue


                action = np.asarray(
                    action_history[
                        frame
                    ],
                    dtype=float,
                ).reshape(-1)


                if action.size < 2:
                    raise RuntimeError(
                        "Unexpected action shape: "
                        f"{np.asarray(action_history[frame]).shape}"
                    )


                action_accel = float(
                    action[0]
                )

                action_steer = float(
                    action[1]
                )


                if frame > 0:

                    previous = np.asarray(
                        action_history[
                            frame - 1
                        ],
                        dtype=float,
                    ).reshape(-1)

                    delta_accel = float(
                        action_accel
                        - previous[0]
                    )

                    delta_steer = float(
                        action_steer
                        - previous[1]
                    )

                else:

                    delta_accel = np.nan
                    delta_steer = np.nan


                row = {
                    "model":
                        model_tag,

                    "scenario_index":
                        scenario_index,

                    "group":
                        GROUP_MAP[
                            scenario_index
                        ],

                    "reference_step":
                        ref,

                    "reference_seconds":
                        ref * DT,

                    "lead_s":
                        lead,

                    "frame":
                        frame,

                    "time_from_start_s":
                        frame * DT,

                    "expected_episode_length":
                        expected_steps,

                    "actual_episode_length":
                        actual_steps,

                    "action_accel":
                        action_accel,

                    "action_steer":
                        action_steer,

                    "abs_steer":
                        abs(
                            action_steer
                        ),

                    "brake_strength":
                        max(
                            -action_accel,
                            0.0,
                        ),

                    "throttle_strength":
                        max(
                            action_accel,
                            0.0,
                        ),

                    "delta_accel":
                        delta_accel,

                    "delta_steer":
                        delta_steer,

                    "abs_delta_accel":
                        abs(
                            delta_accel
                        )
                        if np.isfinite(
                            delta_accel
                        )
                        else np.nan,

                    "abs_delta_steer":
                        abs(
                            delta_steer
                        )
                        if np.isfinite(
                            delta_steer
                        )
                        else np.nan,

                    **feat,
                }

                rows.append(
                    row
                )


            done_selected.add(
                scenario_index
            )


        global_start = (
            batch_end
        )

        if (
            len(done_selected)
            == len(SELECTED)
        ):
            break


    result = pd.DataFrame(
        rows
    )


    print()
    print(
        f"{model_tag} processed:",
        len(done_selected),
    )

    print(
        f"{model_tag} rollout mismatches:",
        len(mismatches),
    )

    for item in mismatches:
        print(
            " ",
            item,
        )

    print(
        f"{model_tag} reference failures:",
        len(insufficient_reference),
    )

    for item in insufficient_reference:
        print(
            " ",
            item,
        )


    del feature_extractor
    del reset_fn
    del step_fn
    del env

    gc.collect()
    jax.clear_caches()


    return (
        result,
        mismatches,
        insufficient_reference,
    )


# ============================================================
# V27 REPLAY
# ============================================================

v27_rows, v27_mismatch, v27_ref_fail = (
    replay_model(
        MODEL_V27,
        "V27",
        v27_eval,
    )
)


# ============================================================
# V28 REPLAY
# ============================================================

v28_rows, v28_mismatch, v28_ref_fail = (
    replay_model(
        MODEL_V28,
        "V28",
        v28_eval,
    )
)


# ============================================================
# CONSISTENCY GATE
# ============================================================

print()
print("=" * 110)
print("FINAL ROLLOUT CONSISTENCY")
print("=" * 110)

print(
    "V27 mismatch:",
    len(v27_mismatch),
)

print(
    "V28 mismatch:",
    len(v28_mismatch),
)

print(
    "V27 reference failure:",
    len(v27_ref_fail),
)

print(
    "V28 reference failure:",
    len(v28_ref_fail),
)


if (
    v27_mismatch
    or v28_mismatch
    or v27_ref_fail
    or v28_ref_fail
):

    print()
    print(
        "STOP: replay is not fully consistent with "
        "stored evaluation. Do not interpret action deltas yet."
    )

    raise RuntimeError(
        "Replay consistency check failed."
    )


# ============================================================
# SAVE RAW
# ============================================================

raw = pd.concat(
    [
        v27_rows,
        v28_rows,
    ],
    ignore_index=True,
)

raw.to_csv(
    OUT_RAW,
    index=False,
)


# ============================================================
# PAIRED MERGE
# ============================================================

KEYS = [
    "scenario_index",
    "group",
    "reference_step",
    "lead_s",
    "frame",
]

v27_merge = (
    v27_rows
    .drop(
        columns=[
            "model",
        ],
        errors="ignore",
    )
)

v28_merge = (
    v28_rows
    .drop(
        columns=[
            "model",
        ],
        errors="ignore",
    )
)


compare = v27_merge.merge(
    v28_merge,
    on=KEYS,
    suffixes=(
        "_v27",
        "_v28",
    ),
    validate="one_to_one",
)


# ============================================================
# DELTAS
# ============================================================

DELTA_FIELDS = [
    "action_accel",
    "action_steer",
    "abs_steer",

    "delta_accel",
    "delta_steer",

    "abs_delta_accel",
    "abs_delta_steer",

    "brake_strength",
    "throttle_strength",

    "risk_static_clearance",
    "risk_predicted_clearance",

    "risk_future_tcpa",
    "risk_tcpa_clipped",

    "risk_closing_speed",
    "risk_approaching",

    "second_predicted_clearance",
    "top2_mean_predicted_clearance",
]


for field in DELTA_FIELDS:

    old = (
        f"{field}_v27"
    )

    new = (
        f"{field}_v28"
    )

    if (
        old in compare.columns
        and new in compare.columns
    ):

        compare[
            f"delta_{field}"
        ] = (
            compare[new]
            - compare[old]
        )


compare.to_csv(
    OUT_COMPARE,
    index=False,
)


# ============================================================
# COMPACT OUTPUT
# ============================================================

print()
print("=" * 150)
print("SELECTED 8 SCENARIO COMPARISON")
print("=" * 150)


preferred = [
    "scenario_index",
    "group",
    "reference_step",
    "lead_s",

    "action_accel_v27",
    "action_accel_v28",
    "delta_action_accel",

    "action_steer_v27",
    "action_steer_v28",
    "delta_action_steer",

    "risk_predicted_clearance_v27",
    "risk_predicted_clearance_v28",
    "delta_risk_predicted_clearance",

    "risk_static_clearance_v27",
    "risk_static_clearance_v28",
    "delta_risk_static_clearance",

    "risk_future_tcpa_v27",
    "risk_future_tcpa_v28",
    "delta_risk_future_tcpa",
]

show_cols = [
    col
    for col in preferred
    if col in compare.columns
]


print(
    compare[
        show_cols
    ]
    .sort_values(
        [
            "scenario_index",
            "lead_s",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .to_string(
        index=False
    )
)


# ============================================================
# PER-SCENARIO ACTION DIVERGENCE SUMMARY
# ============================================================

print()
print("=" * 110)
print("PER-SCENARIO ACTION DIVERGENCE")
print("=" * 110)


for idx in SELECTED:

    x = compare[
        compare[
            "scenario_index"
        ] == idx
    ].copy()

    if len(x) == 0:

        print(
            idx,
            "NO DATA",
        )

        continue


    max_accel = (
        x[
            "delta_action_accel"
        ]
        .abs()
        .max()
        if "delta_action_accel"
        in x.columns
        else np.nan
    )

    max_steer = (
        x[
            "delta_action_steer"
        ]
        .abs()
        .max()
        if "delta_action_steer"
        in x.columns
        else np.nan
    )


    print(
        f"{idx:4d} "
        f"{GROUP_MAP[idx]:<18} "
        f"max|Δaccel|="
        f"{max_accel:8.4f}  "
        f"max|Δsteer|="
        f"{max_steer:8.4f}"
    )


# ============================================================
# AVAILABLE RISK FIELDS
# ============================================================

print()
print("=" * 110)
print("AVAILABLE RISK / OBJECT FIELDS")
print("=" * 110)

for col in compare.columns:

    if (
        "risk_" in col
        or "object" in col
        or "tcpa" in col
        or "clearance" in col
    ):
        print(
            col
        )


print()
print("=" * 110)
print("SAVED")
print("=" * 110)

print(
    OUT_RAW
)

print(
    OUT_COMPARE
)

