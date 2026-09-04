from pathlib import Path
import runpy

import jax
import numpy as np
import pandas as pd

from vmax.scripts.evaluate import utils
from vmax.simulator import datasets, make_data_generator


# ============================================================
# REUSE VERIFIED DIAGNOSTIC HELPERS
# ============================================================

rankmod = runpy.run_path(
    "analysis/diagnose_v26_collision_rank.py"
)

take_batch_item = rankmod["take_batch_item"]

DATASET = (
    "/mnt/e/AI_Challenge/datasets/splits/"
    "secondary_val_1024/secondary_val_1024.tfrecord@1024"
)

MODEL = "v27_secondary_frozen_21506560"

MAX_OBJECTS = 64
BATCH_SIZE = 64
DT = 0.1

LEADS = [
    3.0,
    2.0,
    1.0,
    0.5,
]

ROOT = Path(
    "benchmark/ai/mnt/e/AI_Challenge/datasets/splits/"
    "secondary_val_1024/"
    "secondary_val_1024.tfrecord@1024"
)

V26_EVAL = (
    ROOT
    / "v26_secondary_frozen_24322560"
    / "model_final"
    / "evaluation_episodes.csv"
)

V27_EVAL = (
    ROOT
    / "v27_secondary_frozen_21506560"
    / "model_final"
    / "evaluation_episodes.csv"
)

PAIR_PATH = Path(
    "analysis/v27_zeroout_transition_detail.csv"
)

OUT_PATH = Path(
    "analysis/v28_road_margin_calibration.csv"
)


# ============================================================
# ARRAY HELPERS
# ============================================================

def strip_single_batch(x):
    a = np.asarray(x)

    # Evaluation rollout uses batch=1.
    # Remove only leading singleton batch dimensions.
    while (
        a.ndim >= 2
        and a.shape[0] == 1
    ):
        a = a[0]

    return a


def safe_min(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.min(x)
    )


# ============================================================
# ROAD MARGIN SNAPSHOT
# ============================================================

def road_snapshot(
    state,
    feature_extractor,
):
    """
    Direction-aware road-boundary diagnostic.

    The V27 policy receives roadgraph xy + dir_xy.
    Ego is at the origin and its heading is the local +x axis.

    Instead of simple Euclidean nearest-point distance, derive
    lateral margin only from road-edge points whose local tangent
    is approximately aligned with the ego heading.

    Two angle tolerances are tested:
      strict: <= 30 deg
      loose : <= 45 deg

    Local longitudinal window:
      -5 m <= x <= +15 m
    """

    # --------------------------------------------------------
    # Evaluation state has batch_dims=(1,).
    # Remove only that evaluation batch dimension.
    # --------------------------------------------------------

    if state.batch_dims:
        if tuple(state.batch_dims) != (1,):
            raise RuntimeError(
                f"Unexpected state.batch_dims: "
                f"{state.batch_dims}"
            )

        state = jax.tree.map(
            lambda x: x[0],
            state,
        )

    sdc_obs = (
        feature_extractor
        ._get_sdc_observation(
            state
        )
    )

    roadgraph = (
        feature_extractor
        ._reduce_and_filter_roadgraph_points(
            sdc_obs.roadgraph_static_points
        )
    )


    # --------------------------------------------------------
    # Roadgraph arrays
    # --------------------------------------------------------

    xy = np.asarray(
        roadgraph.xy,
        dtype=float,
    ).reshape(-1, 2)

    dir_xy = np.asarray(
        roadgraph.dir_xy,
        dtype=float,
    ).reshape(-1, 2)

    valid = np.asarray(
        roadgraph.valid
    ).astype(bool).reshape(-1)

    types = np.asarray(
        roadgraph.types
    ).reshape(-1)


    # --------------------------------------------------------
    # Ego size
    # --------------------------------------------------------

    widths = np.asarray(
        sdc_obs.trajectory.width,
        dtype=float,
    )

    widths = np.squeeze(
        widths
    )

    if widths.ndim == 2:
        ego_width = float(
            widths[0, -1]
        )

    elif widths.ndim == 1:
        ego_width = float(
            widths[0]
        )

    else:
        raise RuntimeError(
            f"Unexpected width shape: "
            f"{widths.shape}"
        )

    ego_half_width = (
        0.5 * ego_width
    )


    # --------------------------------------------------------
    # Basic validity
    # --------------------------------------------------------

    dir_norm = np.linalg.norm(
        dir_xy,
        axis=1,
    )

    finite = (
        np.isfinite(xy).all(axis=1)
        & np.isfinite(dir_xy).all(axis=1)
    )

    edge_type = (
        (types == 15)
        | (types == 16)
    )

    base_mask = (
        valid
        & finite
        & edge_type
    )


    # --------------------------------------------------------
    # Original simple distance
    # --------------------------------------------------------

    center_dist = np.linalg.norm(
        xy,
        axis=1,
    )

    simple_candidates = (
        center_dist[
            base_mask
        ]
    )

    nearest_any = (
        float(
            np.min(
                simple_candidates
            )
        )
        if len(
            simple_candidates
        )
        else np.nan
    )

    road_clearance = (
        nearest_any
        - ego_half_width
        if np.isfinite(
            nearest_any
        )
        else np.nan
    )


    # --------------------------------------------------------
    # Direction alignment
    #
    # Ego heading = +x in SDC-local frame.
    #
    # abs(dir_x / ||dir||) allows map tangents stored in either
    # direction (+x or -x).
    # --------------------------------------------------------

    safe_norm = np.where(
        dir_norm > 1e-6,
        dir_norm,
        1.0,
    )

    parallel_score = np.abs(
        dir_xy[:, 0]
        / safe_norm
    )

    cos30 = np.cos(
        np.deg2rad(30.0)
    )

    cos45 = np.cos(
        np.deg2rad(45.0)
    )


    # --------------------------------------------------------
    # Local window around ego.
    #
    # Ignore road-edge segments far behind/ahead because their
    # Euclidean proximity can describe a different branch of an
    # intersection instead of the ego's current corridor.
    # --------------------------------------------------------

    local_window = (
        (xy[:, 0] >= -5.0)
        & (xy[:, 0] <= 15.0)
    )


    # --------------------------------------------------------
    # Distance from ego to tangent line.
    #
    # For line:
    #   q + t*d
    #
    # distance(origin, line)
    #   = |cross(d, q)| / ||d||
    #
    # This is more meaningful than ||q|| when the road edge
    # point is longitudinally offset from ego.
    # --------------------------------------------------------

    tangent_distance = np.abs(
        (
            dir_xy[:, 0]
            * xy[:, 1]
        )
        - (
            dir_xy[:, 1]
            * xy[:, 0]
        )
    ) / safe_norm

    point_lateral_distance = np.abs(
        xy[:, 1]
    )


    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def calc_directional(
        cos_threshold,
    ):
        directional = (
            base_mask
            & local_window
            & (dir_norm > 1e-6)
            & (
                parallel_score
                >= cos_threshold
            )
        )

        left = (
            directional
            & (xy[:, 1] > 0.0)
        )

        right = (
            directional
            & (xy[:, 1] < 0.0)
        )


        def min_or_nan(
            values,
            mask,
        ):
            x = values[
                mask
            ]

            x = x[
                np.isfinite(
                    x
                )
            ]

            if len(x) == 0:
                return np.nan

            return float(
                np.min(x)
            )


        # Point-based lateral distance.
        left_point = min_or_nan(
            point_lateral_distance,
            left,
        )

        right_point = min_or_nan(
            point_lateral_distance,
            right,
        )


        # Tangent-line perpendicular distance.
        left_line = min_or_nan(
            tangent_distance,
            left,
        )

        right_line = min_or_nan(
            tangent_distance,
            right,
        )


        def min_pair(a, b):
            values = [
                x
                for x in [a, b]
                if np.isfinite(x)
            ]

            if not values:
                return np.nan

            return float(
                min(values)
            )


        min_point = min_pair(
            left_point,
            right_point,
        )

        min_line = min_pair(
            left_line,
            right_line,
        )


        def clearance(x):
            if not np.isfinite(x):
                return np.nan

            return float(
                x
                - ego_half_width
            )


        return {
            "candidate_count":
                int(
                    directional.sum()
                ),

            "left_point_distance":
                left_point,

            "right_point_distance":
                right_point,

            "min_point_distance":
                min_point,

            "point_clearance":
                clearance(
                    min_point
                ),

            "left_line_distance":
                left_line,

            "right_line_distance":
                right_line,

            "min_line_distance":
                min_line,

            "line_clearance":
                clearance(
                    min_line
                ),
        }


    strict = calc_directional(
        cos30
    )

    loose = calc_directional(
        cos45
    )


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "ego_width":
            ego_width,

        "nearest_any":
            nearest_any,

        "road_clearance":
            road_clearance,

        # Backward-compatible fields expected by old summary.
        "nearest_boundary":
            nearest_any,

        "boundary_clearance":
            road_clearance,

        "nearest_median":
            nearest_any,

        "median_clearance":
            road_clearance,

        "left_distance":
            loose[
                "left_point_distance"
            ],

        "right_distance":
            loose[
                "right_point_distance"
            ],

        "min_side_distance":
            loose[
                "min_point_distance"
            ],

        "min_side_clearance":
            loose[
                "point_clearance"
            ],

        "num_edge_points":
            int(
                base_mask.sum()
            ),

        # ----------------------------------------------------
        # NEW: strict <=30 deg
        # ----------------------------------------------------

        "dir30_count":
            strict[
                "candidate_count"
            ],

        "dir30_left_point":
            strict[
                "left_point_distance"
            ],

        "dir30_right_point":
            strict[
                "right_point_distance"
            ],

        "dir30_point_clearance":
            strict[
                "point_clearance"
            ],

        "dir30_left_line":
            strict[
                "left_line_distance"
            ],

        "dir30_right_line":
            strict[
                "right_line_distance"
            ],

        "dir30_line_clearance":
            strict[
                "line_clearance"
            ],

        # ----------------------------------------------------
        # NEW: loose <=45 deg
        # ----------------------------------------------------

        "dir45_count":
            loose[
                "candidate_count"
            ],

        "dir45_left_point":
            loose[
                "left_point_distance"
            ],

        "dir45_right_point":
            loose[
                "right_point_distance"
            ],

        "dir45_point_clearance":
            loose[
                "point_clearance"
            ],

        "dir45_left_line":
            loose[
                "left_line_distance"
            ],

        "dir45_right_line":
            loose[
                "right_line_distance"
            ],

        "dir45_line_clearance":
            loose[
                "line_clearance"
            ],
    }


# ============================================================
# ROLLOUT
# ============================================================

def rollout(
    scenario,
    scenario_index,
    reset_fn,
    step_fn,
    base_key,
):
    rng_key = jax.random.fold_in(
        base_key,
        scenario_index,
    )

    rng_key, reset_key = (
        jax.random.split(
            rng_key
        )
    )

    reset_key = jax.random.split(
        reset_key,
        1,
    )

    transition = reset_fn(
        scenario,
        reset_key,
    )

    history = [
        transition.state
    ]

    while not bool(
        np.asarray(
            transition.done
        ).reshape(-1)[0]
    ):
        rng_key, step_key = (
            jax.random.split(
                rng_key
            )
        )

        step_key = jax.random.split(
            step_key,
            1,
        )

        transition, _ = step_fn(
            transition,
            key=step_key,
        )

        history.append(
            transition.state
        )

    return history


# ============================================================
# AUC
# ============================================================

def auc_pairwise(
    positive,
    negative,
    lower_is_risk=True,
):
    p = np.asarray(
        positive,
        dtype=float,
    )

    n = np.asarray(
        negative,
        dtype=float,
    )

    p = p[
        np.isfinite(p)
    ]

    n = n[
        np.isfinite(n)
    ]

    if (
        len(p) == 0
        or len(n) == 0
    ):
        return np.nan

    if lower_is_risk:
        wins = (
            p[:, None]
            < n[None, :]
        )

        ties = (
            p[:, None]
            == n[None, :]
        )

    else:
        wins = (
            p[:, None]
            > n[None, :]
        )

        ties = (
            p[:, None]
            == n[None, :]
        )

    return float(
        wins.mean()
        + 0.5 * ties.mean()
    )


# ============================================================
# LOAD GROUPS
# ============================================================

v26 = (
    pd.read_csv(
        V26_EVAL
    )
    .set_index(
        "scenario_index"
    )
)

v27 = (
    pd.read_csv(
        V27_EVAL
    )
    .set_index(
        "scenario_index"
    )
)

pair = pd.read_csv(
    PAIR_PATH
)


new_offroad = sorted(
    pair.loc[
        pair["new_offroad"],
        "scenario_index",
    ]
    .astype(int)
    .tolist()
)

fixed_offroad = sorted(
    pair.loc[
        pair["fixed_offroad"],
        "scenario_index",
    ]
    .astype(int)
    .tolist()
)


# ============================================================
# SAFE CONTROLS
# ============================================================

safe_mask = (
    (pair["v27_rf"] >= 0.95)
    & (pair["v27_accuracy"] > 0.5)
    & (pair["v27_overlap"] <= 0.5)
    & (pair["v27_offroad"] <= 0.5)
    & (pair["v27_collision"] <= 0.5)
)

safe_candidates = (
    pair.loc[
        safe_mask,
        "scenario_index",
    ]
    .astype(int)
    .sort_values()
    .to_numpy()
)

NUM_SAFE = min(
    30,
    len(
        safe_candidates
    ),
)

if NUM_SAFE == 0:
    raise RuntimeError(
        "No safe controls found."
    )

safe_pick_idx = np.linspace(
    0,
    len(safe_candidates) - 1,
    NUM_SAFE,
    dtype=int,
)

safe_controls = (
    safe_candidates[
        safe_pick_idx
    ]
    .tolist()
)


# ============================================================
# REFERENCE TIME
# ============================================================

reference_step = {}
group_map = {}


# NEW_OFFROAD:
# V27's actual offroad termination time.
for idx in new_offroad:
    reference_step[idx] = int(
        v27.loc[
            idx,
            "episode_length",
        ]
    )

    group_map[idx] = (
        "NEW_OFFROAD"
    )


# FIXED_OFFROAD:
# Analyze V27 at the absolute time
# where V26 used to terminate offroad.
for idx in fixed_offroad:
    reference_step[idx] = int(
        v26.loc[
            idx,
            "episode_length",
        ]
    )

    group_map[idx] = (
        "FIXED_OFFROAD"
    )


# SAFE:
# Match the absolute-time distribution of NEW_OFFROAD.
new_event_steps = [
    int(
        v27.loc[
            idx,
            "episode_length",
        ]
    )
    for idx in new_offroad
]

for i, idx in enumerate(
    safe_controls
):
    reference_step[idx] = (
        new_event_steps[
            i
            % len(
                new_event_steps
            )
        ]
    )

    group_map[idx] = (
        "SAFE"
    )


selected = sorted(
    set(
        new_offroad
        + fixed_offroad
        + safe_controls
    )
)


print("=" * 110)
print("V28 ROAD-MARGIN CALIBRATION")
print("=" * 110)

print(
    "NEW_OFFROAD :",
    len(new_offroad),
)

print(
    "FIXED_OFFROAD:",
    len(fixed_offroad),
)

print(
    "SAFE         :",
    len(safe_controls),
)

print(
    "Total replay :",
    len(selected),
)

print(
    "Leads        :",
    LEADS,
)

print()


# ============================================================
# SETUP V27 FROZEN POLICY
# ============================================================

env, step_fn, _, _ = (
    utils.setup_evaluation(
        policy_type="ai",
        path_model=MODEL,
        source_dir="runs",
        path_dataset=DATASET,
        eval_name=(
            "analysis/"
            "v28_road_margin_tmp"
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

global_start = 0

mismatches = []


# ============================================================
# REPLAY SELECTED SCENARIOS
# ============================================================

for batch in data_generator:

    batch_end = (
        global_start
        + BATCH_SIZE
    )

    wanted = [
        idx
        for idx in selected
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
            f"[{len(done_selected)+1:02d}/"
            f"{len(selected):02d}] "
            f"{group_map[scenario_index]:<13} "
            f"scenario {scenario_index}",
            flush=True,
        )

        history = rollout(
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
            v27.loc[
                scenario_index,
                "episode_length",
            ]
        )

        if (
            actual_steps
            != expected_steps
        ):
            mismatches.append(
                (
                    scenario_index,
                    expected_steps,
                    actual_steps,
                )
            )

            print(
                "  WARNING rollout mismatch:",
                expected_steps,
                actual_steps,
            )

            done_selected.add(
                scenario_index
            )

            continue


        ref = int(
            reference_step[
                scenario_index
            ]
        )

        # Fixed-offroad/safe reference
        # must not exceed the V27 rollout.
        ref = min(
            ref,
            actual_steps,
        )


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

            if (
                frame < 0
                or frame >= len(
                    history
                )
            ):
                continue

            feat = road_snapshot(
                history[frame],
                feature_extractor,
            )

            rows.append({
                "scenario_index":
                    scenario_index,

                "group":
                    group_map[
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

                **feat,
            })


        done_selected.add(
            scenario_index
        )


    global_start = batch_end

    if (
        len(done_selected)
        == len(selected)
    ):
        break


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    rows
)

df.to_csv(
    OUT_PATH,
    index=False,
)


print()
print("=" * 110)
print("ROLLOUT CONSISTENCY")
print("=" * 110)

print(
    "Processed :",
    len(done_selected),
)

print(
    "Mismatches:",
    len(mismatches),
)

for x in mismatches:
    print(
        " ",
        x,
    )


# ============================================================
# SUMMARY
# ============================================================

METRICS = [
    "road_clearance",
    "dir30_point_clearance",
    "dir30_line_clearance",
    "dir45_point_clearance",
    "dir45_line_clearance",
]


for lead in LEADS:

    x = df[
        df["lead_s"]
        == lead
    ]

    new = x[
        x["group"]
        == "NEW_OFFROAD"
    ]

    fixed = x[
        x["group"]
        == "FIXED_OFFROAD"
    ]

    safe = x[
        x["group"]
        == "SAFE"
    ]

    print()
    print("=" * 110)
    print(
        f"LEAD = {lead:.1f}s"
    )
    print("=" * 110)

    print(
        "Rows:",
        f"NEW={len(new)}",
        f"FIXED={len(fixed)}",
        f"SAFE={len(safe)}",
    )

    print()

    print(
        f"{'Metric':<24}"
        f"{'NEW med':>12}"
        f"{'FIXED med':>12}"
        f"{'SAFE med':>12}"
        f"{'AUC NEW/SAFE':>16}"
    )

    print("-" * 80)

    for metric in METRICS:

        auc = auc_pairwise(
            new[metric],
            safe[metric],
            lower_is_risk=True,
        )

        print(
            f"{metric:<24}"
            f"{new[metric].median():>12.3f}"
            f"{fixed[metric].median():>12.3f}"
            f"{safe[metric].median():>12.3f}"
            f"{auc:>16.3f}"
        )


# ============================================================
# THRESHOLD VIEW
# ============================================================

print()
print("=" * 110)
print("ROAD CLEARANCE THRESHOLDS")
print("=" * 110)

for lead in [
    2.0,
    1.0,
    0.5,
]:

    x = df[
        df["lead_s"]
        == lead
    ]

    new = x[
        x["group"]
        == "NEW_OFFROAD"
    ]

    safe = x[
        x["group"]
        == "SAFE"
    ]

    print()
    print(
        f"Lead {lead:.1f}s"
    )

    for threshold in [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ]:

        new_rate = (
            new[
                "dir45_line_clearance"
            ]
            <= threshold
        ).mean()

        safe_rate = (
            safe[
                "dir45_line_clearance"
            ]
            <= threshold
        ).mean()

        print(
            f"  clearance <= {threshold:3.1f}m"
            f"   NEW={100*new_rate:6.1f}%"
            f"   SAFE={100*safe_rate:6.1f}%"
        )


# ============================================================
# PER-SCENE NEW OFFROAD
# ============================================================

print()
print("=" * 110)
print("NEW OFFROAD DETAIL")
print("=" * 110)

detail = (
    df[
        df["group"]
        == "NEW_OFFROAD"
    ]
    [
        [
            "scenario_index",
            "reference_seconds",
            "lead_s",
            "road_clearance",
            "dir30_point_clearance",
            "dir30_line_clearance",
            "dir45_point_clearance",
            "dir45_line_clearance",
            "dir45_left_line",
            "dir45_right_line",
            "dir45_count",
        ]
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
)

print(
    detail.to_string(
        index=False
    )
)


print()
print("=" * 110)
print("SAVED")
print("=" * 110)
print(
    OUT_PATH
)
