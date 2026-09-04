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
    "analysis/v28_tcpa_calibration.csv"
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

    risk_idx = (
        obj_df[
            "predicted_clearance"
        ]
        .idxmin()
    )

    risk = obj_df.loc[
        risk_idx
    ]


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


# ============================================================
# REPLAY
# ============================================================

rows = []

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


df = pd.DataFrame(
    rows
)

df.to_csv(
    OUT_PATH,
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
