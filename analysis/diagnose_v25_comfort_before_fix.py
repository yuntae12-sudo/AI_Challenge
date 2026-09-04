from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from waymax import datatypes

from vmax.scripts.evaluate import utils as eval_utils
from vmax.simulator import (
    constants,
    datasets,
    make_data_generator,
    operations,
)
from vmax.simulator.metrics import comfort as comfort_metric


# ============================================================
# CONFIG
# ============================================================

PAIRED_CSV = Path(
    "analysis/v17_vs_v25_secondary_paired.csv"
)

DATASET = (
    "/mnt/e/AI_Challenge/datasets/splits/"
    "secondary_val_1024/"
    "secondary_val_1024.tfrecord@1024"
)

MODEL_ALIAS = "sec1024_v25_21250560"

MAX_NUM_OBJECTS = 64
SEED = 0


# ============================================================
# LOAD PAIRED RESULT
# ============================================================

df = pd.read_csv(PAIRED_CSV)

safe_v25 = (
    (df["overlap_v25"] == 0)
    & (df["offroad_in_box_v25"] == 0)
    & (df["at_fault_collision_v25"] == 0)
)


# ------------------------------------------------------------
# A.
# Progress improved but Comfort degraded.
#
# This is the main V25 trade-off group.
# ------------------------------------------------------------

tradeoff_bad = (
    df[
        (df["d_Progress"] > 0.05)
        & (df["d_Comfort"] < -0.05)
        & safe_v25
    ]
    .sort_values(
        ["d_Comfort", "d_Progress"],
        ascending=[True, False],
    )
    .head(80)
)


# ------------------------------------------------------------
# B.
# Progress + Comfort both improved.
#
# Control group.
# ------------------------------------------------------------

good_both = (
    df[
        (df["d_Progress"] > 0.05)
        & (df["d_Comfort"] > 0.05)
        & safe_v25
    ]
    .sort_values(
        "d_RideFlux",
        ascending=False,
    )
    .head(80)
)


# ------------------------------------------------------------
# C.
# V25 is safe and high-progress, but Comfort is poor.
#
# Pure comfort problem.
# ------------------------------------------------------------

low_comfort_safe = (
    df[
        safe_v25
        & (df["progress_ratio_v25"] >= 0.90)
        & (df["comfort_v25"] < 0.70)
    ]
    .sort_values(
        "comfort_v25",
        ascending=True,
    )
    .head(80)
)


# ------------------------------------------------------------
# D.
# V25 has low progress without safety failure.
#
# Pure progress problem.
# ------------------------------------------------------------

low_progress_safe = (
    df[
        safe_v25
        & (df["progress_ratio_v25"] < 0.70)
    ]
    .sort_values(
        "progress_ratio_v25",
        ascending=True,
    )
    .head(80)
)


# ------------------------------------------------------------
# E.
# V17 safe -> V25 overlap.
#
# New failure introduced by V25.
# ------------------------------------------------------------

new_overlap = (
    df[
        (df["overlap_v17"] == 0)
        & (df["overlap_v25"] > 0)
    ]
    .sort_values(
        "rideflux_aggregate_score_v25",
        ascending=True,
    )
)


groups = {
    "TRADEOFF_BAD": set(
        tradeoff_bad["scenario_index"].astype(int)
    ),
    "GOOD_BOTH": set(
        good_both["scenario_index"].astype(int)
    ),
    "LOW_COMFORT_SAFE": set(
        low_comfort_safe["scenario_index"].astype(int)
    ),
    "LOW_PROGRESS_SAFE": set(
        low_progress_safe["scenario_index"].astype(int)
    ),
    "NEW_OVERLAP": set(
        new_overlap["scenario_index"].astype(int)
    ),
}

selected = set()

for ids in groups.values():
    selected |= ids


print("=" * 90)
print("V25 DIAGNOSTIC SCENARIO SELECTION")
print("=" * 90)

for name, ids in groups.items():
    print(
        f"{name:20s}: {len(ids):4d}"
    )

print("-" * 90)
print(
    "Unique scenarios      :",
    len(selected),
)


# ============================================================
# SAVE GROUP MAP
# ============================================================

group_rows = []

for idx in sorted(selected):
    labels = [
        name
        for name, ids in groups.items()
        if idx in ids
    ]

    group_rows.append(
        {
            "scenario_index": idx,
            "groups": "|".join(labels),
        }
    )

group_df = pd.DataFrame(group_rows)

group_df.to_csv(
    "analysis/v25_diagnostic_groups.csv",
    index=False,
)


# ============================================================
# SETUP MODEL / ENV
# ============================================================

data_generator = make_data_generator(
    path=datasets.get_dataset(DATASET),
    max_num_objects=MAX_NUM_OBJECTS,
    include_sdc_paths=False,
    batch_dims=(1,),
    seed=SEED,
    repeat=1,
)

env, step_fn, _, _ = eval_utils.setup_evaluation(
    policy_type="ai",
    path_model=MODEL_ALIAS,
    source_dir="runs",
    path_dataset=DATASET,
    eval_name="analysis/v25_trace_eval",
    max_num_objects=MAX_NUM_OBJECTS,
    noisy_init=False,
    sdc_paths_from_data=False,
)

reset_fn = jax.jit(env.reset)
step_fn = jax.jit(step_fn)


# ============================================================
# HELPERS
# ============================================================

def unbatch(tree):
    return jax.tree.map(
        lambda x:
            x[0]
            if getattr(x, "ndim", 0) > 0
            else x,
        tree,
    )


def scalar(x):
    return float(
        np.asarray(x)
        .reshape(-1)[0]
    )


def get_comfort_components(
    batched_state,
):
    state = unbatch(
        batched_state
    )

    past_traj = datatypes.dynamic_slice(
        state.sim_trajectory,
        state.timestep - 9,
        10,
        -1,
    )

    sdc_index = operations.get_index(
        state.object_metadata.is_sdc
    )

    sdc_traj = jax.tree.map(
        lambda x: x[sdc_index],
        past_traj,
    )

    valid = bool(
        np.asarray(
            jnp.all(
                sdc_traj.valid
            )
        )
    )

    if not valid:
        return {
            "comfort_valid": 0,
            "lat_accel_max": np.nan,
            "long_accel_min": np.nan,
            "long_accel_max": np.nan,
            "long_jerk_max": np.nan,
            "yaw_rate_max": np.nan,
            "yaw_accel_max": np.nan,
            "viol_lat_accel": np.nan,
            "viol_long_brake": np.nan,
            "viol_long_accel": np.nan,
            "viol_long_jerk": np.nan,
            "viol_yaw_rate": np.nan,
            "viol_yaw_accel": np.nan,
            "x": np.nan,
            "y": np.nan,
            "yaw": np.nan,
            "speed": np.nan,
        }

    dt = constants.TIME_DELTA

    lat_accel = np.asarray(
        comfort_metric
        ._compute_lateral_acceleration(
            sdc_traj,
            dt,
        )
    )

    long_accel = np.asarray(
        comfort_metric
        ._compute_longitudinal_acceleration(
            sdc_traj,
            dt,
        )
    )

    long_jerk = np.asarray(
        comfort_metric
        ._compute_longitudinal_jerk(
            sdc_traj,
            dt,
        )
    )

    yaw_rate = np.asarray(
        comfort_metric
        ._compute_ego_yaw_rate(
            sdc_traj,
            dt,
        )
    )

    yaw_accel = np.asarray(
        comfort_metric
        ._compute_ego_yaw_acceleration(
            sdc_traj,
            dt,
        )
    )

    lat_max = float(
        np.max(
            np.abs(lat_accel)
        )
    )

    long_min = float(
        np.min(long_accel)
    )

    long_max = float(
        np.max(long_accel)
    )

    jerk_max = float(
        np.max(
            np.abs(long_jerk)
        )
    )

    yaw_rate_max = float(
        np.max(
            np.abs(yaw_rate)
        )
    )

    yaw_accel_max = float(
        np.max(
            np.abs(yaw_accel)
        )
    )

    vel_x = float(
        np.asarray(
            sdc_traj.vel_x[-1]
        )
    )

    vel_y = float(
        np.asarray(
            sdc_traj.vel_y[-1]
        )
    )

    speed = (
        vel_x**2
        + vel_y**2
    ) ** 0.5

    return {
        "comfort_valid": 1,

        "lat_accel_max":
            lat_max,

        "long_accel_min":
            long_min,

        "long_accel_max":
            long_max,

        "long_jerk_max":
            jerk_max,

        "yaw_rate_max":
            yaw_rate_max,

        "yaw_accel_max":
            yaw_accel_max,

        # Exact ComfortMetric thresholds
        "viol_lat_accel":
            int(lat_max > 2.0),

        "viol_long_brake":
            int(long_min < -4.05),

        "viol_long_accel":
            int(long_max > 2.40),

        "viol_long_jerk":
            int(jerk_max > 8.3),

        "viol_yaw_rate":
            int(yaw_rate_max > 0.95),

        "viol_yaw_accel":
            int(yaw_accel_max > 2.2),

        "x":
            float(
                np.asarray(
                    sdc_traj.x[-1]
                )
            ),

        "y":
            float(
                np.asarray(
                    sdc_traj.y[-1]
                )
            ),

        "yaw":
            float(
                np.asarray(
                    sdc_traj.yaw[-1]
                )
            ),

        "speed":
            speed,
    }


# ============================================================
# RUN SELECTED SCENARIOS
# ============================================================

rows = []

rng = jax.random.PRNGKey(
    SEED
)

done_count = 0

for scenario_index, scenario in enumerate(
    data_generator
):
    if scenario_index not in selected:
        continue

    labels = "|".join(
        name
        for name, ids in groups.items()
        if scenario_index in ids
    )

    rng, reset_key = jax.random.split(
        rng
    )

    reset_key = jax.random.split(
        reset_key,
        1,
    )

    transition = reset_fn(
        scenario,
        reset_key,
    )

    previous_action = None
    local_step = 0

    while not bool(
        np.asarray(
            transition.done
        ).reshape(-1)[0]
    ):
        rng, step_key = jax.random.split(
            rng
        )

        step_key = jax.random.split(
            step_key,
            1,
        )

        transition, rl_transition = step_fn(
            transition,
            key=step_key,
        )

        action = (
            np.asarray(
                rl_transition.action
            )
            .reshape(-1, 2)[0]
            .astype(float)
        )

        if previous_action is None:
            da0 = np.nan
            da1 = np.nan
        else:
            da0 = abs(
                action[0]
                - previous_action[0]
            )

            da1 = abs(
                action[1]
                - previous_action[1]
            )

        previous_action = action.copy()

        comp = get_comfort_components(
            transition.state
        )

        comfort_value = np.nan

        if (
            "comfort"
            in transition.metrics
        ):
            comfort_value = scalar(
                transition.metrics[
                    "comfort"
                ]
            )

        row = {
            "scenario_index":
                scenario_index,

            "groups":
                labels,

            "step":
                local_step,

            # Keep generic names:
            # exact semantics of each normalized action
            # are not assumed here.
            "action_0":
                action[0],

            "action_1":
                action[1],

            "abs_delta_action_0":
                da0,

            "abs_delta_action_1":
                da1,

            "comfort_step":
                comfort_value,

            **comp,
        }

        rows.append(row)

        local_step += 1

    done_count += 1

    if (
        done_count % 20 == 0
        or done_count == len(selected)
    ):
        print(
            f"[TRACE] "
            f"{done_count}/"
            f"{len(selected)} "
            f"scenarios"
        )


# ============================================================
# SAVE TRACE
# ============================================================

trace = pd.DataFrame(rows)

TRACE_OUT = Path(
    "analysis/"
    "v25_diagnostic_trace.csv"
)

trace.to_csv(
    TRACE_OUT,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

violation_cols = [
    "viol_lat_accel",
    "viol_long_brake",
    "viol_long_accel",
    "viol_long_jerk",
    "viol_yaw_rate",
    "viol_yaw_accel",
]


print()
print("=" * 110)
print("V25 COMFORT / ACTION DIAGNOSTIC")
print("=" * 110)


for group_name, ids in groups.items():

    g = trace[
        trace["scenario_index"]
        .isin(ids)
    ].copy()

    valid = g[
        g["comfort_valid"] == 1
    ].copy()

    if len(valid) == 0:
        continue

    print()
    print("=" * 110)
    print(group_name)
    print("=" * 110)

    print(
        "Scenarios                 :",
        g["scenario_index"]
        .nunique(),
    )

    print(
        "Valid comfort windows     :",
        len(valid),
    )

    print(
        "Mean step comfort         :",
        f"{valid['comfort_step'].mean():.4f}",
    )

    print(
        "Mean |Δ action_0|         :",
        f"{valid['abs_delta_action_0'].mean():.5f}",
    )

    print(
        "Mean |Δ action_1|         :",
        f"{valid['abs_delta_action_1'].mean():.5f}",
    )

    print(
        "P95  |Δ action_0|         :",
        f"{valid['abs_delta_action_0'].quantile(.95):.5f}",
    )

    print(
        "P95  |Δ action_1|         :",
        f"{valid['abs_delta_action_1'].quantile(.95):.5f}",
    )

    print()
    print(
        "Comfort threshold violation "
        "(% of valid windows)"
    )

    for col in violation_cols:
        rate = (
            valid[col].mean()
            * 100
        )

        print(
            f"  {col:22s}: "
            f"{rate:6.2f}%"
        )

    print()
    print(
        "Scenarios with >=1 violation"
    )

    per_scenario = (
        valid
        .groupby(
            "scenario_index"
        )[violation_cols]
        .max()
    )

    for col in violation_cols:
        rate = (
            per_scenario[col]
            .mean()
            * 100
        )

        print(
            f"  {col:22s}: "
            f"{rate:6.2f}%"
        )


# ============================================================
# GLOBAL RANKING OF COMFORT CAUSES
# ============================================================

valid_all = trace[
    trace["comfort_valid"] == 1
]

print()
print("=" * 110)
print("GLOBAL COMFORT VIOLATION RANKING")
print("=" * 110)

ranking = []

for col in violation_cols:
    ranking.append(
        (
            col,
            valid_all[col].mean(),
        )
    )

ranking.sort(
    key=lambda x: x[1],
    reverse=True,
)

for rank, (name, rate) in enumerate(
    ranking,
    start=1,
):
    print(
        f"{rank}. "
        f"{name:22s} "
        f"{rate*100:6.2f}%"
    )


print()
print("=" * 110)
print("SAVED")
print("=" * 110)
print(
    "analysis/v25_diagnostic_groups.csv"
)
print(
    "analysis/v25_diagnostic_trace.csv"
)
