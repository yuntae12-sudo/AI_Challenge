from pathlib import Path
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

MODEL = "v27_secondary_frozen_21506560"

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
    "analysis/v28_action_response.csv"
)

WINDOW_OUT_PATH = Path(
    "analysis/v28_steer_threshold_sweep.csv"
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

NUM_SAFE = 60


# ============================================================
# REUSE VERIFIED BATCH EXTRACTION
# ============================================================

rankmod = runpy.run_path(
    "analysis/diagnose_v26_collision_rank.py"
)

take_batch_item = rankmod[
    "take_batch_item"
]


# ============================================================
# HELPERS
# ============================================================

def as_bool(series):
    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
        .fillna(False)
        .astype(bool)
    )


def auc_pairwise(
    positive,
    negative,
    higher_is_risk=False,
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

    if higher_is_risk:
        wins = (
            p[:, None]
            > n[None, :]
        )
    else:
        wins = (
            p[:, None]
            < n[None, :]
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
# INTERACTION SNAPSHOT
# ============================================================

def interaction_snapshot(
    state,
    feature_extractor,
):
    """
    Reproduce V27 interaction geometry from current causal state.

    V27:
      static_clearance
      predicted_clearance

    Candidate V28:
      tcpa_clipped
      future_tcpa
      approaching

    future_tcpa:
      tcpa_raw > 0 -> clip(tcpa_raw, 0, 5)
      tcpa_raw <= 0 -> 5

    This avoids treating a diverging object as
    "immediate TCPA = 0".
    """

    # Evaluation state is vmap(batch=1).
    if state.batch_dims:
        if tuple(state.batch_dims) != (1,):
            raise RuntimeError(
                f"Unexpected batch_dims: "
                f"{state.batch_dims}"
            )

        state = jax.tree.map(
            lambda x: x[0],
            state,
        )


    # Same SDC-local representation used by policy.
    sdc_obs = (
        feature_extractor
        ._get_sdc_observation(
            state
        )
    )


    # --------------------------------------------------------
    # Current state
    # --------------------------------------------------------

    xy = np.asarray(
        sdc_obs.trajectory.xy[
            :,
            -1,
            :,
        ],
        dtype=float,
    )

    vel = np.asarray(
        sdc_obs.trajectory.vel_xy[
            :,
            -1,
            :,
        ],
        dtype=float,
    )

    length = np.asarray(
        sdc_obs.trajectory.length[
            :,
            -1,
        ],
        dtype=float,
    )

    width = np.asarray(
        sdc_obs.trajectory.width[
            :,
            -1,
        ],
        dtype=float,
    )

    valid = np.asarray(
        sdc_obs.trajectory.valid[
            :,
            -1,
        ]
    ).astype(bool)


    # --------------------------------------------------------
    # Find ego.
    #
    # In SDC-local coordinates ego is distance ~0.
    # --------------------------------------------------------

    dist_from_origin = np.linalg.norm(
        xy,
        axis=-1,
    )

    valid_dist = np.where(
        valid,
        dist_from_origin,
        np.inf,
    )

    ego = int(
        np.argmin(
            valid_dist
        )
    )


    # --------------------------------------------------------
    # Reproduce V27 closest-object selection:
    # ego + 16 closest objects.
    # --------------------------------------------------------

    order = np.argsort(
        valid_dist
    )

    selected = [
        int(i)
        for i in order
        if valid[i]
    ][
        : NUM_CLOSEST_OBJECTS + 1
    ]

    objects = [
        i
        for i in selected
        if i != ego
    ][
        :NUM_CLOSEST_OBJECTS
    ]

    if len(objects) == 0:
        return None


    # --------------------------------------------------------
    # Ego geometry
    # --------------------------------------------------------

    ego_xy = xy[ego]
    ego_vel = vel[ego]

    ego_radius = (
        0.5
        * np.sqrt(
            length[ego] ** 2
            + width[ego] ** 2
        )
    )


    rows = []

    for obj in objects:

        r = (
            xy[obj]
            - ego_xy
        )

        v = (
            vel[obj]
            - ego_vel
        )

        center_distance = float(
            np.linalg.norm(r)
        )

        object_radius = (
            0.5
            * np.sqrt(
                length[obj] ** 2
                + width[obj] ** 2
            )
        )

        radius_sum = (
            ego_radius
            + object_radius
        )


        # ----------------------------------------------------
        # V27 static clearance
        # ----------------------------------------------------

        static_clearance = float(
            center_distance
            - radius_sum
        )


        # ----------------------------------------------------
        # Same V27 TCPA formula
        # ----------------------------------------------------

        relative_speed_sq = float(
            np.dot(
                v,
                v,
            )
        )

        position_velocity_dot = float(
            np.dot(
                r,
                v,
            )
        )

        if relative_speed_sq > 1e-6:
            tcpa_raw = (
                -position_velocity_dot
                / relative_speed_sq
            )
        else:
            tcpa_raw = 0.0


        tcpa_clipped = float(
            np.clip(
                tcpa_raw,
                0.0,
                5.0,
            )
        )


        # ----------------------------------------------------
        # Proposed time representation.
        #
        # Negative raw TCPA means closest approach was in the
        # past / object is diverging.
        #
        # Encode that as max horizon rather than zero.
        # ----------------------------------------------------

        if tcpa_raw > 0.0:
            future_tcpa = float(
                min(
                    tcpa_raw,
                    5.0,
                )
            )
        else:
            future_tcpa = 5.0


        approaching = bool(
            tcpa_raw > 0.0
            and tcpa_raw <= 5.0
        )


        # ----------------------------------------------------
        # V27 predicted clearance
        # ----------------------------------------------------

        closest_relative_xy = (
            r
            + tcpa_clipped
            * v
        )

        dcpa = float(
            np.linalg.norm(
                closest_relative_xy
            )
        )

        predicted_clearance = float(
            dcpa
            - radius_sum
        )


        # Radial closing speed, diagnostic only.
        if center_distance > 1e-6:
            closing_speed = float(
                -np.dot(
                    r / center_distance,
                    v,
                )
            )
        else:
            closing_speed = 0.0


        rows.append({
            "object_index":
                obj,

            "distance":
                center_distance,

            "static_clearance":
                static_clearance,

            "predicted_clearance":
                predicted_clearance,

            "dcpa":
                dcpa,

            "tcpa_raw":
                float(tcpa_raw),

            "tcpa_clipped":
                tcpa_clipped,

            "future_tcpa":
                future_tcpa,

            "approaching":
                int(approaching),

            "closing_speed":
                closing_speed,
        })


    obj_df = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Object V27 considers geometrically most dangerous:
    # minimum predicted clearance.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Order objects by V27 predicted clearance.
    #
    # first  = object V27 already considers most dangerous
    # second = second-most-dangerous interaction candidate
    # --------------------------------------------------------

    ordered = (
        obj_df
        .sort_values(
            "predicted_clearance",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    risk = ordered.iloc[0]

    if len(ordered) >= 2:
        second = ordered.iloc[1]
    else:
        second = ordered.iloc[0]

    first_pred = float(
        risk[
            "predicted_clearance"
        ]
    )

    second_pred = float(
        second[
            "predicted_clearance"
        ]
    )

    top2_mean_pred = float(
        0.5
        * (
            first_pred
            + second_pred
        )
    )

    second_minus_first_gap = float(
        second_pred
        - first_pred
    )


    # --------------------------------------------------------
    # Additional conflict counts.
    # --------------------------------------------------------

    conflict0 = obj_df[
        obj_df[
            "predicted_clearance"
        ] <= 0.0
    ]

    conflict1 = obj_df[
        obj_df[
            "predicted_clearance"
        ] <= 1.0
    ]

    conflict2 = obj_df[
        obj_df[
            "predicted_clearance"
        ] <= 2.0
    ]


    def min_future_tcpa(x):
        if len(x) == 0:
            return 5.0

        return float(
            x[
                "future_tcpa"
            ].min()
        )


    return {
        "risk_object_index":
            int(
                risk[
                    "object_index"
                ]
            ),

        "risk_distance":
            float(
                risk[
                    "distance"
                ]
            ),

        "risk_static_clearance":
            float(
                risk[
                    "static_clearance"
                ]
            ),

        "risk_predicted_clearance":
            float(
                risk[
                    "predicted_clearance"
                ]
            ),

        "risk_dcpa":
            float(
                risk[
                    "dcpa"
                ]
            ),

        "risk_tcpa_raw":
            float(
                risk[
                    "tcpa_raw"
                ]
            ),

        "risk_tcpa_clipped":
            float(
                risk[
                    "tcpa_clipped"
                ]
            ),

        "risk_future_tcpa":
            float(
                risk[
                    "future_tcpa"
                ]
            ),

        "risk_approaching":
            int(
                risk[
                    "approaching"
                ]
            ),

        "risk_closing_speed":
            float(
                risk[
                    "closing_speed"
                ]
            ),

        # ----------------------------------------------------
        # SECOND-RISK interaction geometry
        # ----------------------------------------------------

        "second_object_index":
            int(
                second[
                    "object_index"
                ]
            ),

        "second_distance":
            float(
                second[
                    "distance"
                ]
            ),

        "second_static_clearance":
            float(
                second[
                    "static_clearance"
                ]
            ),

        "second_predicted_clearance":
            float(
                second[
                    "predicted_clearance"
                ]
            ),

        "second_dcpa":
            float(
                second[
                    "dcpa"
                ]
            ),

        "second_future_tcpa":
            float(
                second[
                    "future_tcpa"
                ]
            ),

        "second_approaching":
            int(
                second[
                    "approaching"
                ]
            ),

        "top2_mean_predicted_clearance":
            top2_mean_pred,

        "second_minus_first_pred_gap":
            second_minus_first_gap,

        "num_pred_conflict_0":
            int(
                len(
                    conflict0
                )
            ),

        "num_pred_conflict_1":
            int(
                len(
                    conflict1
                )
            ),

        "num_pred_conflict_2":
            int(
                len(
                    conflict2
                )
            ),

        "min_future_tcpa_conflict_0":
            min_future_tcpa(
                conflict0
            ),

        "min_future_tcpa_conflict_1":
            min_future_tcpa(
                conflict1
            ),

        "min_future_tcpa_conflict_2":
            min_future_tcpa(
                conflict2
            ),
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

    # action_history[t] is the action chosen from history[t]
    # to produce history[t + 1].
    action_history = []

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

        transition, rl_transition = step_fn(
            transition,
            key=step_key,
        )

        action_history.append(
            rl_transition.action
        )

        history.append(
            transition.state
        )

    return history, action_history


# ============================================================
# LOAD RESULTS / GROUPS
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

pair["new_collision"] = as_bool(
    pair["new_collision"]
)

pair["fixed_collision"] = as_bool(
    pair["fixed_collision"]
)


new_collision = sorted(
    pair.loc[
        pair[
            "new_collision"
        ],
        "scenario_index",
    ]
    .astype(int)
    .tolist()
)

fixed_collision = sorted(
    pair.loc[
        pair[
            "fixed_collision"
        ],
        "scenario_index",
    ]
    .astype(int)
    .tolist()
)


# ============================================================
# SAFE CONTROL POOL
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

num_safe = min(
    NUM_SAFE,
    len(
        safe_candidates
    ),
)

safe_pick = np.linspace(
    0,
    len(safe_candidates) - 1,
    num_safe,
    dtype=int,
)

safe_controls = (
    safe_candidates[
        safe_pick
    ]
    .tolist()
)


# ============================================================
# REFERENCE TIMES
# ============================================================

reference_step = {}
group_map = {}


# New collision:
# V27 actual collision termination.
for idx in new_collision:

    reference_step[idx] = int(
        v27.loc[
            idx,
            "episode_length",
        ]
    )

    group_map[idx] = (
        "NEW_COLLISION"
    )


# Fixed collision:
# inspect V27 at the time V26 used to collide.
for idx in fixed_collision:

    reference_step[idx] = int(
        v26.loc[
            idx,
            "episode_length",
        ]
    )

    group_map[idx] = (
        "FIXED_COLLISION"
    )


# Safe:
# match absolute event-time distribution of NEW collision.
new_event_steps = [
    int(
        v27.loc[
            idx,
            "episode_length",
        ]
    )
    for idx in new_collision
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
        new_collision
        + fixed_collision
        + safe_controls
    )
)


print("=" * 110)
print("V28 TCPA CALIBRATION")
print("=" * 110)

print(
    "NEW_COLLISION  :",
    len(new_collision),
)

print(
    "FIXED_COLLISION:",
    len(fixed_collision),
)

print(
    "SAFE           :",
    len(safe_controls),
)

print(
    "Total replay   :",
    len(selected),
)

print(
    "Leads          :",
    LEADS,
)


# ============================================================
# SETUP FROZEN V27
# ============================================================

env, step_fn, _, _ = (
    utils.setup_evaluation(
        policy_type="ai",
        path_model=MODEL,
        source_dir="runs",
        path_dataset=DATASET,
        eval_name=(
            "analysis/"
            "v28_tcpa_tmp"
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


STEER_THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
]


def threshold_tag(threshold):
    return (
        f"{threshold:.3f}"
        .replace(".", "")
    )


# ============================================================
# ACTION WINDOW FEATURES
# ============================================================

def action_window_features(
    action_history,
    ref,
    window_s,
):
    """
    Analyze all actions during the final `window_s` seconds
    before the reference event.

    action_history[t] produces state[t + 1].
    Therefore actions [ref-window, ref) are the controls
    immediately preceding the reference state.
    """

    window_steps = int(
        round(
            window_s / DT
        )
    )

    # Require a full window for fair comparison.
    if ref < window_steps:
        return None

    end = min(
        ref,
        len(action_history),
    )

    start = (
        end
        - window_steps
    )

    if start < 0 or end <= start:
        return None


    actions = []

    for i in range(
        start,
        end,
    ):

        a = np.asarray(
            action_history[i],
            dtype=float,
        ).reshape(-1)

        if a.size < 2:
            return None

        actions.append(
            a[:2]
        )


    a = np.asarray(
        actions,
        dtype=float,
    )

    accel = a[:, 0]
    steer = a[:, 1]

    brake = np.maximum(
        -accel,
        0.0,
    )

    throttle = np.maximum(
        accel,
        0.0,
    )


    if len(a) >= 2:

        d_accel = np.diff(
            accel
        )

        d_steer = np.diff(
            steer
        )

    else:

        d_accel = np.asarray(
            [],
            dtype=float,
        )

        d_steer = np.asarray(
            [],
            dtype=float,
        )


    # --------------------------------------------------------
    # Steering persistence
    #
    # 1.0:
    #   steering keeps the same direction.
    #
    # near 0:
    #   positive / negative steering cancel each other.
    # --------------------------------------------------------

    abs_steer_sum = float(
        np.sum(
            np.abs(
                steer
            )
        )
    )

    steer_persistence = float(
        abs(
            np.sum(
                steer
            )
        )
        / (
            abs_steer_sum
            + 1e-8
        )
    )


    # --------------------------------------------------------
    # Count meaningful steering-direction flips.
    #
    # Ignore tiny commands close to zero.
    # --------------------------------------------------------

    STEER_DEADBAND = 0.01

    meaningful = steer[
        np.abs(
            steer
        )
        >= STEER_DEADBAND
    ]

    if len(meaningful) >= 2:

        signs = np.sign(
            meaningful
        )

        steer_sign_flips = int(
            np.sum(
                signs[1:]
                != signs[:-1]
            )
        )

    else:

        steer_sign_flips = 0


    steer_total_variation = float(
        np.sum(
            np.abs(
                d_steer
            )
        )
    ) if len(d_steer) else 0.0


    accel_total_variation = float(
        np.sum(
            np.abs(
                d_accel
            )
        )
    ) if len(d_accel) else 0.0


    # High variation relative to total steering effort
    # indicates unstable / oscillatory control.
    steer_variation_ratio = float(
        steer_total_variation
        / (
            abs_steer_sum
            + 1e-8
        )
    )


    return {
        "window_s":
            float(
                window_s
            ),

        "window_steps":
            int(
                len(a)
            ),

        "mean_accel":
            float(
                np.mean(
                    accel
                )
            ),

        "min_accel":
            float(
                np.min(
                    accel
                )
            ),

        "mean_brake":
            float(
                np.mean(
                    brake
                )
            ),

        "max_brake":
            float(
                np.max(
                    brake
                )
            ),

        "brake_impulse":
            float(
                np.sum(
                    brake
                )
                * DT
            ),

        "throttle_impulse":
            float(
                np.sum(
                    throttle
                )
                * DT
            ),

        "mean_abs_steer":
            float(
                np.mean(
                    np.abs(
                        steer
                    )
                )
            ),

        "max_abs_steer":
            float(
                np.max(
                    np.abs(
                        steer
                    )
                )
            ),

        "steer_effort":
            float(
                abs_steer_sum
                * DT
            ),

        "steer_persistence":
            steer_persistence,

        "steer_sign_flips":
            steer_sign_flips,

        "steer_total_variation":
            steer_total_variation,

        "steer_variation_ratio":
            steer_variation_ratio,

        "accel_total_variation":
            accel_total_variation,

        "mean_abs_delta_steer":
            float(
                np.mean(
                    np.abs(
                        d_steer
                    )
                )
            )
            if len(d_steer)
            else 0.0,

        "max_abs_delta_steer":
            float(
                np.max(
                    np.abs(
                        d_steer
                    )
                )
            )
            if len(d_steer)
            else 0.0,

        "mean_abs_delta_accel":
            float(
                np.mean(
                    np.abs(
                        d_accel
                    )
                )
            )
            if len(d_accel)
            else 0.0,

        "max_abs_delta_accel":
            float(
                np.max(
                    np.abs(
                        d_accel
                    )
                )
            )
            if len(d_accel)
            else 0.0,
    }


# ============================================================
# STEERING THRESHOLD FEATURES
# ============================================================

def steering_threshold_features(
    action_history,
    ref,
    window_s,
):
    """
    Compute the steering component of the V26/V27 temporal
    consistency penalty over an entire pre-event window.

    For threshold T:

      excess = relu(|delta_steer| - T)

      penalty_proxy = mean(excess^2)

    `penalty_proxy` is the steering-only part before multiplying
    by lambda=5.
    """

    window_steps = int(
        round(
            window_s / DT
        )
    )

    if ref < window_steps:
        return None

    end = min(
        ref,
        len(action_history),
    )

    start = (
        end
        - window_steps
    )

    if (
        start < 0
        or end - start < 2
    ):
        return None


    steer = []

    for i in range(
        start,
        end,
    ):

        action = np.asarray(
            action_history[i],
            dtype=float,
        ).reshape(-1)

        if action.size < 2:
            return None

        steer.append(
            float(
                action[1]
            )
        )


    steer = np.asarray(
        steer,
        dtype=float,
    )

    abs_delta = np.abs(
        np.diff(
            steer
        )
    )


    result = {
        "num_steer_transitions":
            int(
                len(abs_delta)
            ),

        "window_max_abs_delta_steer":
            float(
                np.max(
                    abs_delta
                )
            ),

        "window_mean_abs_delta_steer":
            float(
                np.mean(
                    abs_delta
                )
            ),
    }


    for threshold in STEER_THRESHOLDS:

        tag = threshold_tag(
            threshold
        )

        excess = np.maximum(
            abs_delta
            - threshold,
            0.0,
        )

        active = (
            abs_delta
            > threshold
        )

        result[
            f"act_{tag}"
        ] = float(
            np.mean(
                active
            )
        )

        result[
            f"excess_{tag}"
        ] = float(
            np.mean(
                excess
            )
        )

        result[
            f"sq_{tag}"
        ] = float(
            np.mean(
                excess ** 2
            )
        )

        result[
            f"max_excess_{tag}"
        ] = float(
            np.max(
                excess
            )
        )


    return result


# ============================================================
# REPLAY
# ============================================================

rows = []

# Full pre-event action-window diagnostics.
window_rows = []

done_selected = set()

mismatches = []

global_start = 0


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
            f"[{len(done_selected)+1:03d}/"
            f"{len(selected):03d}] "
            f"{group_map[scenario_index]:<16} "
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
            v27.loc[
                scenario_index,
                "episode_length",
            ]
        )

        if actual_steps != expected_steps:

            mismatches.append(
                (
                    scenario_index,
                    expected_steps,
                    actual_steps,
                )
            )

            print(
                "  WARNING mismatch:",
                expected_steps,
                actual_steps,
            )

            done_selected.add(
                scenario_index
            )

            continue


        ref = min(
            int(
                reference_step[
                    scenario_index
                ]
            ),
            actual_steps,
        )


        # ----------------------------------------------------
        # Full action windows ending at the reference event.
        # ----------------------------------------------------

        for window_s in LEADS:

            wf = action_window_features(
                action_history,
                ref,
                window_s,
            )

            tf = steering_threshold_features(
                action_history,
                ref,
                window_s,
            )

            if (
                wf is None
                or tf is None
            ):
                continue

            window_rows.append({
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

                **wf,
                **tf,
            })


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
                or frame >= len(history)
            ):
                continue

            feat = interaction_snapshot(
                history[
                    frame
                ],
                feature_extractor,
            )

            if feat is None:
                continue

            # ------------------------------------------------
            # Policy action at this exact pre-event frame.
            #
            # action[0] = normalized acceleration
            # action[1] = normalized steering/curvature
            # ------------------------------------------------

            if frame >= len(action_history):
                continue

            action = np.asarray(
                action_history[frame],
                dtype=float,
            ).reshape(-1)

            if action.size < 2:
                raise RuntimeError(
                    f"Unexpected action shape: "
                    f"{np.asarray(action_history[frame]).shape}"
                )

            action_accel = float(
                action[0]
            )

            action_steer = float(
                action[1]
            )

            brake_strength = float(
                max(
                    -action_accel,
                    0.0,
                )
            )

            throttle_strength = float(
                max(
                    action_accel,
                    0.0,
                )
            )

            abs_steer = float(
                abs(
                    action_steer
                )
            )


            # ------------------------------------------------
            # Action change from previous control cycle.
            #
            # V26/V27 temporal regularization thresholds:
            #   accel = 0.05
            #   steer = 0.15
            # ------------------------------------------------

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

                abs_delta_accel = float(
                    abs(
                        delta_accel
                    )
                )

                abs_delta_steer = float(
                    abs(
                        delta_steer
                    )
                )

            else:

                delta_accel = np.nan
                delta_steer = np.nan
                abs_delta_accel = np.nan
                abs_delta_steer = np.nan


            accel_change_excess = (
                max(
                    abs_delta_accel - 0.05,
                    0.0,
                )
                if np.isfinite(
                    abs_delta_accel
                )
                else np.nan
            )

            steer_change_excess = (
                max(
                    abs_delta_steer - 0.15,
                    0.0,
                )
                if np.isfinite(
                    abs_delta_steer
                )
                else np.nan
            )

            smooth_excess = (
                accel_change_excess
                + steer_change_excess
                if (
                    np.isfinite(
                        accel_change_excess
                    )
                    and np.isfinite(
                        steer_change_excess
                    )
                )
                else np.nan
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

                "action_accel":
                    action_accel,

                "action_steer":
                    action_steer,

                "brake_strength":
                    brake_strength,

                "throttle_strength":
                    throttle_strength,

                "abs_steer":
                    abs_steer,

                "delta_accel":
                    delta_accel,

                "delta_steer":
                    delta_steer,

                "abs_delta_accel":
                    abs_delta_accel,

                "abs_delta_steer":
                    abs_delta_steer,

                "accel_change_excess":
                    accel_change_excess,

                "steer_change_excess":
                    steer_change_excess,

                "smooth_excess":
                    smooth_excess,

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


df = pd.DataFrame(
    rows
)

df.to_csv(
    OUT_PATH,
    index=False,
)

window_df = pd.DataFrame(
    window_rows
)

window_df.to_csv(
    WINDOW_OUT_PATH,
    index=False,
)


# ============================================================
# CONSISTENCY
# ============================================================

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

for m in mismatches:
    print(
        " ",
        m,
    )


# ============================================================
# STANDARD SUMMARY
# ============================================================

METRICS = [
    (
        "risk_predicted_clearance",
        False,
    ),
    (
        "risk_static_clearance",
        False,
    ),
    (
        "risk_future_tcpa",
        False,
    ),
    (
        "risk_tcpa_clipped",
        False,
    ),
    (
        "risk_approaching",
        True,
    ),
    (
        "risk_closing_speed",
        True,
    ),
    (
        "second_predicted_clearance",
        False,
    ),
    (
        "top2_mean_predicted_clearance",
        False,
    ),
    (
        "second_minus_first_pred_gap",
        False,
    ),
    (
        "second_future_tcpa",
        False,
    ),
    (
        "num_pred_conflict_0",
        True,
    ),
    (
        "num_pred_conflict_1",
        True,
    ),
]


for lead in LEADS:

    x = df[
        df[
            "lead_s"
        ] == lead
    ]

    new = x[
        x["group"]
        == "NEW_COLLISION"
    ]

    fixed = x[
        x["group"]
        == "FIXED_COLLISION"
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
        f"{'Metric':<28}"
        f"{'NEW med':>12}"
        f"{'FIXED med':>12}"
        f"{'SAFE med':>12}"
        f"{'AUC NEW/SAFE':>16}"
    )

    print(
        "-" * 84
    )

    for metric, higher in METRICS:

        auc = auc_pairwise(
            new[metric],
            safe[metric],
            higher_is_risk=higher,
        )

        print(
            f"{metric:<28}"
            f"{new[metric].median():>12.3f}"
            f"{fixed[metric].median():>12.3f}"
            f"{safe[metric].median():>12.3f}"
            f"{auc:>16.3f}"
        )


# ============================================================
# HARD MATCHING
#
# Does TCPA add information after controlling for
# V27 predicted clearance?
# ============================================================

def hard_match(
    positive,
    negative,
    use_static=False,
):
    """
    Match each NEW collision scene to the SAFE scene with
    closest existing V27 geometry.

    Match A:
      predicted_clearance

    Match B:
      predicted_clearance + static_clearance
    """

    if (
        len(positive) == 0
        or len(negative) == 0
    ):
        return pd.DataFrame()

    neg = (
        negative
        .reset_index(
            drop=True
        )
        .copy()
    )

    matched = []

    # Scales prevent one feature dominating 2D distance.
    pred_scale = max(
        float(
            neg[
                "risk_predicted_clearance"
            ].std()
        ),
        1e-6,
    )

    static_scale = max(
        float(
            neg[
                "risk_static_clearance"
            ].std()
        ),
        1e-6,
    )


    for _, p in positive.iterrows():

        d_pred = (
            (
                neg[
                    "risk_predicted_clearance"
                ]
                - p[
                    "risk_predicted_clearance"
                ]
            )
            / pred_scale
        ) ** 2


        distance = d_pred


        if use_static:

            d_static = (
                (
                    neg[
                        "risk_static_clearance"
                    ]
                    - p[
                        "risk_static_clearance"
                    ]
                )
                / static_scale
            ) ** 2

            distance = (
                distance
                + d_static
            )


        best_idx = int(
            distance.idxmin()
        )

        n = neg.loc[
            best_idx
        ]


        matched.append({
            "positive_scenario":
                int(
                    p[
                        "scenario_index"
                    ]
                ),

            "control_scenario":
                int(
                    n[
                        "scenario_index"
                    ]
                ),

            "pos_pred":
                float(
                    p[
                        "risk_predicted_clearance"
                    ]
                ),

            "ctl_pred":
                float(
                    n[
                        "risk_predicted_clearance"
                    ]
                ),

            "pred_gap":
                float(
                    abs(
                        p[
                            "risk_predicted_clearance"
                        ]
                        - n[
                            "risk_predicted_clearance"
                        ]
                    )
                ),

            "pos_static":
                float(
                    p[
                        "risk_static_clearance"
                    ]
                ),

            "ctl_static":
                float(
                    n[
                        "risk_static_clearance"
                    ]
                ),

            "pos_future_tcpa":
                float(
                    p[
                        "risk_future_tcpa"
                    ]
                ),

            "ctl_future_tcpa":
                float(
                    n[
                        "risk_future_tcpa"
                    ]
                ),

            "pos_approaching":
                float(
                    p[
                        "risk_approaching"
                    ]
                ),

            "ctl_approaching":
                float(
                    n[
                        "risk_approaching"
                    ]
                ),
        })


    return pd.DataFrame(
        matched
    )


print()
print("=" * 110)
print("HARD-CONTROL TCPA TEST")
print("=" * 110)


for lead in LEADS:

    x = df[
        df[
            "lead_s"
        ] == lead
    ]

    new = x[
        x["group"]
        == "NEW_COLLISION"
    ]

    safe = x[
        x["group"]
        == "SAFE"
    ]


    print()
    print(
        f"----- LEAD {lead:.1f}s -----"
    )


    for label, use_static in [
        (
            "MATCH predicted_clearance",
            False,
        ),
        (
            "MATCH predicted + static",
            True,
        ),
    ]:

        m = hard_match(
            new,
            safe,
            use_static=use_static,
        )

        if len(m) == 0:
            continue


        auc_tcpa = auc_pairwise(
            m[
                "pos_future_tcpa"
            ],
            m[
                "ctl_future_tcpa"
            ],
            higher_is_risk=False,
        )


        print()
        print(label)

        print(
            "  pairs            :",
            len(m),
        )

        print(
            "  mean |pred gap|  :",
            f"{m['pred_gap'].mean():.3f}",
        )

        print(
            "  NEW TCPA median  :",
            f"{m['pos_future_tcpa'].median():.3f}",
        )

        print(
            "  SAFE TCPA median :",
            f"{m['ctl_future_tcpa'].median():.3f}",
        )

        print(
            "  TCPA AUC         :",
            f"{auc_tcpa:.3f}",
        )

        print(
            "  NEW approaching  :",
            f"{100*m['pos_approaching'].mean():.1f}%",
        )

        print(
            "  SAFE approaching :",
            f"{100*m['ctl_approaching'].mean():.1f}%",
        )


# ============================================================
# SECOND-RISK HARD-CONTROL TEST
#
# Core question:
# After matching the geometry of V27's most dangerous object,
# does geometry of the second-most-dangerous object explain
# NEW collision vs SAFE / FIXED collision?
# ============================================================

def match_second_risk(
    positive,
    negative,
    keys,
):
    if (
        len(positive) == 0
        or len(negative) == 0
    ):
        return pd.DataFrame()

    neg = (
        negative
        .reset_index(
            drop=True
        )
        .copy()
    )

    scales = {}

    for key in keys:
        scales[key] = max(
            float(
                neg[key].std()
            ),
            1e-6,
        )

    rows = []

    for _, pos in positive.iterrows():

        distance = np.zeros(
            len(neg),
            dtype=float,
        )

        for key in keys:

            distance += (
                (
                    neg[key]
                    - pos[key]
                )
                / scales[key]
            ) ** 2


        best = int(
            np.argmin(
                distance
            )
        )

        ctl = neg.iloc[
            best
        ]


        rows.append({
            "pos_first_pred":
                pos[
                    "risk_predicted_clearance"
                ],

            "ctl_first_pred":
                ctl[
                    "risk_predicted_clearance"
                ],

            "first_pred_gap":
                abs(
                    pos[
                        "risk_predicted_clearance"
                    ]
                    - ctl[
                        "risk_predicted_clearance"
                    ]
                ),

            "pos_second_pred":
                pos[
                    "second_predicted_clearance"
                ],

            "ctl_second_pred":
                ctl[
                    "second_predicted_clearance"
                ],

            "pos_top2":
                pos[
                    "top2_mean_predicted_clearance"
                ],

            "ctl_top2":
                ctl[
                    "top2_mean_predicted_clearance"
                ],

            "pos_second_gap":
                pos[
                    "second_minus_first_pred_gap"
                ],

            "ctl_second_gap":
                ctl[
                    "second_minus_first_pred_gap"
                ],

            "pos_second_tcpa":
                pos[
                    "second_future_tcpa"
                ],

            "ctl_second_tcpa":
                ctl[
                    "second_future_tcpa"
                ],
        })


    return pd.DataFrame(
        rows
    )


print()
print("=" * 110)
print("SECOND-RISK HARD-CONTROL TEST")
print("=" * 110)


for lead in LEADS:

    x = df[
        df[
            "lead_s"
        ] == lead
    ]

    new = x[
        x["group"]
        == "NEW_COLLISION"
    ]

    safe = x[
        x["group"]
        == "SAFE"
    ]

    fixed = x[
        x["group"]
        == "FIXED_COLLISION"
    ]


    print()
    print(
        f"----- LEAD {lead:.1f}s -----"
    )


    for label, keys in [

        (
            "MATCH first predicted",
            [
                "risk_predicted_clearance",
            ],
        ),

        (
            "MATCH first predicted + static",
            [
                "risk_predicted_clearance",
                "risk_static_clearance",
            ],
        ),

    ]:

        print()
        print(label)


        # ----------------------------------------------------
        # NEW vs SAFE
        # ----------------------------------------------------

        ms = match_second_risk(
            new,
            safe,
            keys,
        )


        # ----------------------------------------------------
        # NEW vs FIXED
        # ----------------------------------------------------

        mf = match_second_risk(
            new,
            fixed,
            keys,
        )


        if (
            len(ms) == 0
            or len(mf) == 0
        ):
            continue


        auc_second_safe = auc_pairwise(
            ms[
                "pos_second_pred"
            ],
            ms[
                "ctl_second_pred"
            ],
            higher_is_risk=False,
        )

        auc_second_fixed = auc_pairwise(
            mf[
                "pos_second_pred"
            ],
            mf[
                "ctl_second_pred"
            ],
            higher_is_risk=False,
        )


        auc_top2_safe = auc_pairwise(
            ms[
                "pos_top2"
            ],
            ms[
                "ctl_top2"
            ],
            higher_is_risk=False,
        )

        auc_top2_fixed = auc_pairwise(
            mf[
                "pos_top2"
            ],
            mf[
                "ctl_top2"
            ],
            higher_is_risk=False,
        )


        auc_gap_safe = auc_pairwise(
            ms[
                "pos_second_gap"
            ],
            ms[
                "ctl_second_gap"
            ],
            higher_is_risk=False,
        )

        auc_gap_fixed = auc_pairwise(
            mf[
                "pos_second_gap"
            ],
            mf[
                "ctl_second_gap"
            ],
            higher_is_risk=False,
        )


        auc_tcpa_safe = auc_pairwise(
            ms[
                "pos_second_tcpa"
            ],
            ms[
                "ctl_second_tcpa"
            ],
            higher_is_risk=False,
        )

        auc_tcpa_fixed = auc_pairwise(
            mf[
                "pos_second_tcpa"
            ],
            mf[
                "ctl_second_tcpa"
            ],
            higher_is_risk=False,
        )


        print(
            "  mean |first pred gap| SAFE :",
            f"{ms['first_pred_gap'].mean():.3f}",
        )

        print(
            "  mean |first pred gap| FIXED:",
            f"{mf['first_pred_gap'].mean():.3f}",
        )

        print()

        print(
            "  second_pred AUC NEW/SAFE :",
            f"{auc_second_safe:.3f}",
        )

        print(
            "  second_pred AUC NEW/FIXED:",
            f"{auc_second_fixed:.3f}",
        )

        print(
            "  top2_mean   AUC NEW/SAFE :",
            f"{auc_top2_safe:.3f}",
        )

        print(
            "  top2_mean   AUC NEW/FIXED:",
            f"{auc_top2_fixed:.3f}",
        )

        print(
            "  second_gap  AUC NEW/SAFE :",
            f"{auc_gap_safe:.3f}",
        )

        print(
            "  second_gap  AUC NEW/FIXED:",
            f"{auc_gap_fixed:.3f}",
        )

        print(
            "  second_TCPA AUC NEW/SAFE :",
            f"{auc_tcpa_safe:.3f}",
        )

        print(
            "  second_TCPA AUC NEW/FIXED:",
            f"{auc_tcpa_fixed:.3f}",
        )

        print()

        print(
            "  NEW second_pred median   :",
            f"{ms['pos_second_pred'].median():.3f}",
        )

        print(
            "  SAFE second_pred median  :",
            f"{ms['ctl_second_pred'].median():.3f}",
        )

        print(
            "  FIXED second_pred median :",
            f"{mf['ctl_second_pred'].median():.3f}",
        )


# ============================================================
# ACTION RESPONSE ANALYSIS
# ============================================================

ACTION_METRICS = [
    "action_accel",
    "brake_strength",
    "throttle_strength",
    "abs_steer",
    "abs_delta_accel",
    "abs_delta_steer",
    "accel_change_excess",
    "steer_change_excess",
    "smooth_excess",
]


def separability_auc(
    pos,
    neg,
):
    """
    Direction-free separability.

    0.50 = same distribution
    1.00 = perfectly separated
    """

    auc = auc_pairwise(
        pos,
        neg,
        higher_is_risk=True,
    )

    if not np.isfinite(auc):
        return np.nan

    return float(
        max(
            auc,
            1.0 - auc,
        )
    )


def match_action_controls(
    positive,
    negative,
    keys,
):
    if (
        len(positive) == 0
        or len(negative) == 0
    ):
        return pd.DataFrame()

    neg = (
        negative
        .reset_index(
            drop=True
        )
        .copy()
    )

    scales = {}

    for key in keys:

        scales[key] = max(
            float(
                neg[
                    key
                ].std()
            ),
            1e-6,
        )


    rows = []

    for _, pos in positive.iterrows():

        distance = np.zeros(
            len(neg),
            dtype=float,
        )

        for key in keys:

            distance += (
                (
                    neg[key]
                    - pos[key]
                )
                / scales[key]
            ) ** 2


        best = int(
            np.argmin(
                distance
            )
        )

        ctl = neg.iloc[
            best
        ]


        row = {
            "positive_scenario":
                int(
                    pos[
                        "scenario_index"
                    ]
                ),

            "control_scenario":
                int(
                    ctl[
                        "scenario_index"
                    ]
                ),

            "first_pred_gap":
                abs(
                    pos[
                        "risk_predicted_clearance"
                    ]
                    - ctl[
                        "risk_predicted_clearance"
                    ]
                ),

            "static_gap":
                abs(
                    pos[
                        "risk_static_clearance"
                    ]
                    - ctl[
                        "risk_static_clearance"
                    ]
                ),

            "second_pred_gap":
                abs(
                    pos[
                        "second_predicted_clearance"
                    ]
                    - ctl[
                        "second_predicted_clearance"
                    ]
                ),
        }


        for metric in ACTION_METRICS:

            row[
                "pos_" + metric
            ] = pos[
                metric
            ]

            row[
                "ctl_" + metric
            ] = ctl[
                metric
            ]


        rows.append(
            row
        )


    return pd.DataFrame(
        rows
    )


print()
print("=" * 110)
print("ACTION RESPONSE ANALYSIS")
print("=" * 110)


for lead in LEADS:

    x = df[
        df[
            "lead_s"
        ] == lead
    ]

    new = x[
        x["group"]
        == "NEW_COLLISION"
    ]

    fixed = x[
        x["group"]
        == "FIXED_COLLISION"
    ]

    safe = x[
        x["group"]
        == "SAFE"
    ]


    print()
    print("=" * 110)
    print(
        f"LEAD {lead:.1f}s"
    )
    print("=" * 110)

    print(
        "Rows:",
        f"NEW={len(new)}",
        f"FIXED={len(fixed)}",
        f"SAFE={len(safe)}",
    )


    # --------------------------------------------------------
    # Raw behavior
    # --------------------------------------------------------

    print()
    print("RAW ACTION MEDIANS")

    print(
        f"{'Metric':<24}"
        f"{'NEW':>12}"
        f"{'FIXED':>12}"
        f"{'SAFE':>12}"
        f"{'Sep NEW/FIXED':>17}"
    )

    print("-" * 80)


    for metric in ACTION_METRICS:

        sep = separability_auc(
            new[
                metric
            ],
            fixed[
                metric
            ],
        )

        print(
            f"{metric:<24}"
            f"{new[metric].median():>12.3f}"
            f"{fixed[metric].median():>12.3f}"
            f"{safe[metric].median():>12.3f}"
            f"{sep:>17.3f}"
        )


    # --------------------------------------------------------
    # Geometry-matched NEW vs FIXED.
    # --------------------------------------------------------

    for label, keys in [

        (
            "MATCH first predicted + static",
            [
                "risk_predicted_clearance",
                "risk_static_clearance",
            ],
        ),

        (
            "MATCH first + static + second",
            [
                "risk_predicted_clearance",
                "risk_static_clearance",
                "second_predicted_clearance",
            ],
        ),

    ]:

        m = match_action_controls(
            new,
            fixed,
            keys,
        )

        print()
        print(label)

        if len(m) == 0:
            print(
                "  no pairs"
            )
            continue


        print(
            "  pairs                   :",
            len(m),
        )

        print(
            "  mean first_pred gap     :",
            f"{m['first_pred_gap'].mean():.3f}",
        )

        print(
            "  mean static gap         :",
            f"{m['static_gap'].mean():.3f}",
        )

        print(
            "  mean second_pred gap    :",
            f"{m['second_pred_gap'].mean():.3f}",
        )

        print()

        print(
            f"{'Metric':<24}"
            f"{'NEW med':>12}"
            f"{'FIXED med':>12}"
            f"{'NEW-FIXED':>14}"
            f"{'Separability':>15}"
        )

        print("-" * 80)


        for metric in ACTION_METRICS:

            pos = m[
                "pos_" + metric
            ]

            ctl = m[
                "ctl_" + metric
            ]

            diff = (
                pos
                - ctl
            )

            sep = separability_auc(
                pos,
                ctl,
            )

            print(
                f"{metric:<24}"
                f"{pos.median():>12.3f}"
                f"{ctl.median():>12.3f}"
                f"{diff.mean():>14.3f}"
                f"{sep:>15.3f}"
            )


        # Explicit interpretation helpers.
        accel_diff = (
            m[
                "pos_action_accel"
            ]
            - m[
                "ctl_action_accel"
            ]
        )

        brake_diff = (
            m[
                "pos_brake_strength"
            ]
            - m[
                "ctl_brake_strength"
            ]
        )

        steer_diff = (
            m[
                "pos_abs_steer"
            ]
            - m[
                "ctl_abs_steer"
            ]
        )

        delta_steer_diff = (
            m[
                "pos_abs_delta_steer"
            ]
            - m[
                "ctl_abs_delta_steer"
            ]
        )


        print()

        print(
            "  P(NEW more accel / less brake):",
            f"{100*(accel_diff > 0).mean():.1f}%",
        )

        print(
            "  P(NEW stronger braking)       :",
            f"{100*(brake_diff > 0).mean():.1f}%",
        )

        print(
            "  P(NEW larger |steer|)         :",
            f"{100*(steer_diff > 0).mean():.1f}%",
        )

        print(
            "  P(NEW larger steering change) :",
            f"{100*(delta_steer_diff > 0).mean():.1f}%",
        )


# ============================================================
# ACTION PERSISTENCE / OSCILLATION ANALYSIS
# ============================================================

WINDOW_METRICS = [
    "mean_brake",
    "max_brake",
    "brake_impulse",
    "mean_abs_steer",
    "max_abs_steer",
    "steer_effort",
    "steer_persistence",
    "steer_sign_flips",
    "steer_total_variation",
    "steer_variation_ratio",
    "mean_abs_delta_steer",
    "max_abs_delta_steer",
]


print()
print("=" * 110)
print("ACTION PERSISTENCE / OSCILLATION")
print("=" * 110)


for window_s in LEADS:

    w = window_df[
        window_df[
            "window_s"
        ] == window_s
    ].copy()


    # Geometry at the start of this same pre-event window.
    geom = (
        df[
            df[
                "lead_s"
            ] == window_s
        ]
        [
            [
                "scenario_index",
                "group",
                "risk_predicted_clearance",
                "risk_static_clearance",
                "second_predicted_clearance",
            ]
        ]
    )


    w = w.merge(
        geom,
        on=[
            "scenario_index",
            "group",
        ],
        how="inner",
    )


    new = w[
        w[
            "group"
        ] == "NEW_COLLISION"
    ]

    fixed = w[
        w[
            "group"
        ] == "FIXED_COLLISION"
    ]


    print()
    print("=" * 110)
    print(
        f"WINDOW = {window_s:.1f}s"
    )
    print("=" * 110)

    print(
        "Rows:",
        f"NEW={len(new)}",
        f"FIXED={len(fixed)}",
    )

    print()

    print(
        f"{'Metric':<28}"
        f"{'NEW med':>12}"
        f"{'FIXED med':>12}"
        f"{'Separability':>15}"
    )

    print("-" * 70)


    for metric in WINDOW_METRICS:

        sep = separability_auc(
            new[
                metric
            ],
            fixed[
                metric
            ],
        )

        print(
            f"{metric:<28}"
            f"{new[metric].median():>12.3f}"
            f"{fixed[metric].median():>12.3f}"
            f"{sep:>15.3f}"
        )


    # --------------------------------------------------------
    # Geometry-matched window comparison.
    # --------------------------------------------------------

    for label, keys in [

        (
            "MATCH first predicted + static",
            [
                "risk_predicted_clearance",
                "risk_static_clearance",
            ],
        ),

        (
            "MATCH first + static + second",
            [
                "risk_predicted_clearance",
                "risk_static_clearance",
                "second_predicted_clearance",
            ],
        ),

    ]:

        if (
            len(new) == 0
            or len(fixed) == 0
        ):
            continue


        neg = (
            fixed
            .reset_index(
                drop=True
            )
            .copy()
        )


        scales = {}

        for key in keys:

            scales[key] = max(
                float(
                    neg[
                        key
                    ].std()
                ),
                1e-6,
            )


        matched = []

        for _, pos in new.iterrows():

            distance = np.zeros(
                len(neg),
                dtype=float,
            )

            for key in keys:

                distance += (
                    (
                        neg[
                            key
                        ]
                        - pos[
                            key
                        ]
                    )
                    / scales[
                        key
                    ]
                ) ** 2


            best = int(
                np.argmin(
                    distance
                )
            )

            ctl = neg.iloc[
                best
            ]


            row = {}

            for metric in WINDOW_METRICS:

                row[
                    "pos_" + metric
                ] = pos[
                    metric
                ]

                row[
                    "ctl_" + metric
                ] = ctl[
                    metric
                ]

            matched.append(
                row
            )


        m = pd.DataFrame(
            matched
        )


        print()
        print(label)

        print(
            f"{'Metric':<28}"
            f"{'NEW med':>12}"
            f"{'FIXED med':>12}"
            f"{'Separability':>15}"
        )

        print("-" * 70)


        for metric in WINDOW_METRICS:

            pos = m[
                "pos_" + metric
            ]

            ctl = m[
                "ctl_" + metric
            ]

            sep = separability_auc(
                pos,
                ctl,
            )

            print(
                f"{metric:<28}"
                f"{pos.median():>12.3f}"
                f"{ctl.median():>12.3f}"
                f"{sep:>15.3f}"
            )


print()
print("=" * 110)
print("WINDOW CSV SAVED")
print("=" * 110)

print(
    WINDOW_OUT_PATH
)


# ============================================================
# STEERING THRESHOLD SWEEP
# ============================================================

print()
print("=" * 118)
print("STEERING TEMPORAL-CONSISTENCY THRESHOLD SWEEP")
print("=" * 118)

print()
print(
    "Interpretation:"
)
print(
    "  act      = fraction of steering transitions above threshold"
)
print(
    "  sq       = mean ReLU(|dsteer|-threshold)^2"
)
print(
    "  lambda5  = sq * 5, approximate steering regularization contribution"
)


for window_s in [
    3.0,
    2.0,
    1.0,
    0.5,
]:

    w = window_df[
        window_df[
            "window_s"
        ] == window_s
    ].copy()

    new = w[
        w[
            "group"
        ] == "NEW_COLLISION"
    ]

    fixed = w[
        w[
            "group"
        ] == "FIXED_COLLISION"
    ]

    safe = w[
        w[
            "group"
        ] == "SAFE"
    ]


    print()
    print("=" * 118)
    print(
        f"WINDOW = {window_s:.1f}s"
    )
    print("=" * 118)

    print(
        "Rows:",
        f"NEW={len(new)}",
        f"FIXED={len(fixed)}",
        f"SAFE={len(safe)}",
    )

    print()

    print(
        f"{'Thr':>6}"
        f"{'NEW act':>11}"
        f"{'FIX act':>11}"
        f"{'SAFE act':>11}"
        f"{'Sep N/F':>10}"
        f"{'Sep N/S':>10}"
        f"{'NEW sq':>12}"
        f"{'FIX sq':>12}"
        f"{'SAFE sq':>12}"
        f"{'Sq Sep N/F':>12}"
    )

    print(
        "-" * 118
    )


    for threshold in STEER_THRESHOLDS:

        tag = threshold_tag(
            threshold
        )

        act_col = (
            f"act_{tag}"
        )

        sq_col = (
            f"sq_{tag}"
        )


        act_sep_nf = separability_auc(
            new[
                act_col
            ],
            fixed[
                act_col
            ],
        )

        act_sep_ns = separability_auc(
            new[
                act_col
            ],
            safe[
                act_col
            ],
        )

        sq_sep_nf = separability_auc(
            new[
                sq_col
            ],
            fixed[
                sq_col
            ],
        )


        print(
            f"{threshold:>6.3f}"
            f"{new[act_col].median():>11.3f}"
            f"{fixed[act_col].median():>11.3f}"
            f"{safe[act_col].median():>11.3f}"
            f"{act_sep_nf:>10.3f}"
            f"{act_sep_ns:>10.3f}"
            f"{new[sq_col].median():>12.6f}"
            f"{fixed[sq_col].median():>12.6f}"
            f"{safe[sq_col].median():>12.6f}"
            f"{sq_sep_nf:>12.3f}"
        )


# ============================================================
# Candidate ranking
# ============================================================

print()
print("=" * 118)
print("CANDIDATE THRESHOLD RANKING")
print("=" * 118)

ranking_rows = []


for threshold in STEER_THRESHOLDS:

    tag = threshold_tag(
        threshold
    )

    act_col = (
        f"act_{tag}"
    )

    sq_col = (
        f"sq_{tag}"
    )


    # Prioritize the 2-second window because it provides enough
    # time for an evasive response while remaining close to the
    # failure event.
    w2 = window_df[
        window_df[
            "window_s"
        ] == 2.0
    ]

    n2 = w2[
        w2[
            "group"
        ] == "NEW_COLLISION"
    ]

    f2 = w2[
        w2[
            "group"
        ] == "FIXED_COLLISION"
    ]

    s2 = w2[
        w2[
            "group"
        ] == "SAFE"
    ]


    # 3-second window used as an early-warning consistency check.
    w3 = window_df[
        window_df[
            "window_s"
        ] == 3.0
    ]

    n3 = w3[
        w3[
            "group"
        ] == "NEW_COLLISION"
    ]

    f3 = w3[
        w3[
            "group"
        ] == "FIXED_COLLISION"
    ]

    s3 = w3[
        w3[
            "group"
        ] == "SAFE"
    ]


    sep_nf_2 = separability_auc(
        n2[
            act_col
        ],
        f2[
            act_col
        ],
    )

    sep_ns_2 = separability_auc(
        n2[
            act_col
        ],
        s2[
            act_col
        ],
    )

    sep_nf_3 = separability_auc(
        n3[
            act_col
        ],
        f3[
            act_col
        ],
    )


    new_act_2 = float(
        n2[
            act_col
        ].median()
    )

    fixed_act_2 = float(
        f2[
            act_col
        ].median()
    )

    safe_act_2 = float(
        s2[
            act_col
        ].median()
    )


    # Simple diagnostic ranking only.
    #
    # Higher:
    #   - good NEW/FIXED separation
    #   - good NEW/SAFE separation
    #   - consistent signal at 3 s
    #
    # Lower:
    #   - penalty activates on too much SAFE behavior
    score = (
        0.40
        * sep_nf_2

        + 0.25
        * sep_ns_2

        + 0.20
        * sep_nf_3

        + 0.15
        * (
            1.0
            - safe_act_2
        )
    )


    ranking_rows.append({
        "threshold":
            threshold,

        "score":
            score,

        "new_act_2s":
            new_act_2,

        "fixed_act_2s":
            fixed_act_2,

        "safe_act_2s":
            safe_act_2,

        "sep_nf_2s":
            sep_nf_2,

        "sep_ns_2s":
            sep_ns_2,

        "sep_nf_3s":
            sep_nf_3,

        "new_sq_2s":
            float(
                n2[
                    sq_col
                ].median()
            ),

        "fixed_sq_2s":
            float(
                f2[
                    sq_col
                ].median()
            ),

        "safe_sq_2s":
            float(
                s2[
                    sq_col
                ].median()
            ),
    })


ranking = (
    pd.DataFrame(
        ranking_rows
    )
    .sort_values(
        "score",
        ascending=False,
    )
)


print()
print(
    ranking.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


ranking.to_csv(
    "analysis/v28_steer_threshold_ranking.csv",
    index=False,
)


print()
print("=" * 118)
print("FILES SAVED")
print("=" * 118)

print(
    WINDOW_OUT_PATH
)

print(
    "analysis/v28_steer_threshold_ranking.csv"
)


# ============================================================
# TCPA THRESHOLDS
# ============================================================

print()
print("=" * 110)
print("FUTURE TCPA THRESHOLDS")
print("=" * 110)


for lead in LEADS:

    x = df[
        df[
            "lead_s"
        ] == lead
    ]

    new = x[
        x["group"]
        == "NEW_COLLISION"
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
        0.5,
        1.0,
        2.0,
        3.0,
    ]:

        new_rate = (
            new[
                "risk_future_tcpa"
            ]
            <= threshold
        ).mean()

        safe_rate = (
            safe[
                "risk_future_tcpa"
            ]
            <= threshold
        ).mean()

        print(
            f"  TCPA <= {threshold:3.1f}s"
            f"   NEW={100*new_rate:6.1f}%"
            f"   SAFE={100*safe_rate:6.1f}%"
        )


print()
print("=" * 110)
print("SAVED")
print("=" * 110)

print(
    OUT_PATH
)
