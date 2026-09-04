from pathlib import Path
import runpy

import jax
import numpy as np
import pandas as pd

from vmax.scripts.evaluate import utils
from vmax.simulator import datasets, make_data_generator


# ============================================================
# REUSE VERIFIED V26 DIAGNOSTIC HELPERS
# ============================================================

rankmod = runpy.run_path(
    "analysis/diagnose_v26_collision_rank.py"
)

take_batch_item = rankmod["take_batch_item"]
strip_batch = rankmod["strip_batch"]

DATASET = rankmod["DATASET"]
MODEL = rankmod["MODEL"]

DT = 0.1
TOP_K = 16
MAX_OBJECTS = 64
BATCH_SIZE = 64

LEADS = [2.0, 1.0]

DETAIL_PATH = Path(
    "analysis/v26_collision_nearest_rank_detail.csv"
)

EVAL_PATH = Path(
    "benchmark/ai/mnt/e/AI_Challenge/datasets/splits/"
    "secondary_val_1024/secondary_val_1024.tfrecord@1024/"
    "v26_secondary_frozen_24322560/model_final/"
    "evaluation_episodes.csv"
)

OUT_PATH = Path(
    "analysis/v27_interaction_calibration.csv"
)


# ============================================================
# HELPERS
# ============================================================

def wrap_angle(angle):
    return np.arctan2(
        np.sin(angle),
        np.cos(angle),
    )


def snapshot(state):
    traj = state.sim_trajectory

    t = int(
        np.asarray(state.timestep)
        .reshape(-1)[0]
    )

    x = strip_batch(traj.x)
    y = strip_batch(traj.y)

    speed = strip_batch(traj.speed)

    yaw = strip_batch(traj.yaw)
    length = strip_batch(traj.length)
    width = strip_batch(traj.width)
    valid = strip_batch(traj.valid).astype(bool)

    is_sdc = strip_batch(
        state.object_metadata.is_sdc
    ).astype(bool)

    ego = int(
        np.flatnonzero(is_sdc)[0]
    )

    px = x[:, t]
    py = y[:, t]

    spd = speed[:, t]
    ang = yaw[:, t]

    lng = length[:, t]
    wid = width[:, t]

    v = valid[:, t]

    dx = px - px[ego]
    dy = py - py[ego]

    dist = np.sqrt(
        dx * dx + dy * dy
    )

    valid_other = (
        v
        & (np.arange(len(v)) != ego)
    )

    masked = np.where(
        valid_other,
        dist,
        np.inf,
    )

    order = np.argsort(masked)

    ranks = np.full(
        len(v),
        np.nan,
        dtype=float,
    )

    rank = 1

    for idx in order:
        if not np.isfinite(masked[idx]):
            break

        ranks[idx] = rank
        rank += 1

    return {
        "timestep": t,
        "ego": ego,
        "x": px,
        "y": py,
        "speed": spd,
        "yaw": ang,
        "length": lng,
        "width": wid,
        "valid": v,
        "distance": dist,
        "rank": ranks,
    }


def interaction_features(
    snap,
    obj_idx,
):
    ego = snap["ego"]

    rx = (
        snap["x"][obj_idx]
        - snap["x"][ego]
    )

    ry = (
        snap["y"][obj_idx]
        - snap["y"][ego]
    )

    r = np.array(
        [rx, ry],
        dtype=float,
    )

    distance = float(
        np.linalg.norm(r)
    )

    ego_yaw = float(
        snap["yaw"][ego]
    )

    obj_yaw = float(
        snap["yaw"][obj_idx]
    )

    ego_speed = float(
        snap["speed"][ego]
    )

    obj_speed = float(
        snap["speed"][obj_idx]
    )

    ego_vel = ego_speed * np.array(
        [
            np.cos(ego_yaw),
            np.sin(ego_yaw),
        ]
    )

    obj_vel = obj_speed * np.array(
        [
            np.cos(obj_yaw),
            np.sin(obj_yaw),
        ]
    )

    rel_vel = (
        obj_vel
        - ego_vel
    )

    rel_speed = float(
        np.linalg.norm(rel_vel)
    )

    if distance > 1e-6:
        closing_speed = float(
            -np.dot(
                r,
                rel_vel,
            )
            / distance
        )
    else:
        closing_speed = 0.0

    relative_heading = abs(
        wrap_angle(
            obj_yaw
            - ego_yaw
        )
    )

    vv = float(
        np.dot(
            rel_vel,
            rel_vel,
        )
    )

    if vv > 1e-6:
        tcpa_raw = float(
            -np.dot(
                r,
                rel_vel,
            )
            / vv
        )
    else:
        tcpa_raw = np.inf

    tcpa = float(
        np.clip(
            tcpa_raw,
            0.0,
            5.0,
        )
    )

    future_r = (
        r
        + tcpa * rel_vel
    )

    dcpa = float(
        np.linalg.norm(
            future_r
        )
    )

    ego_radius = 0.5 * np.sqrt(
        snap["length"][ego] ** 2
        + snap["width"][ego] ** 2
    )

    obj_radius = 0.5 * np.sqrt(
        snap["length"][obj_idx] ** 2
        + snap["width"][obj_idx] ** 2
    )

    predicted_clearance = float(
        dcpa
        - ego_radius
        - obj_radius
    )

    approaching = bool(
        closing_speed > 0.0
        and np.isfinite(tcpa_raw)
        and 0.0 <= tcpa_raw <= 5.0
    )

    if approaching:
        interaction_risk = (
            max(
                closing_speed,
                0.0,
            )
            /
            (
                max(
                    predicted_clearance,
                    0.0,
                )
                + 1.0
            )
            /
            (
                1.0
                + tcpa
            )
        )
    else:
        interaction_risk = 0.0

    return {
        "distance":
            distance,

        "ego_speed":
            ego_speed,

        "object_speed":
            obj_speed,

        "relative_speed":
            rel_speed,

        "closing_speed":
            closing_speed,

        "relative_heading_rad":
            float(relative_heading),

        "relative_heading_deg":
            float(
                np.degrees(
                    relative_heading
                )
            ),

        "tcpa_raw":
            tcpa_raw,

        "tcpa_clipped":
            tcpa,

        "dcpa":
            dcpa,

        "predicted_clearance":
            predicted_clearance,

        "approaching":
            int(approaching),

        "interaction_risk":
            float(interaction_risk),
    }


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
        snapshot(
            transition.state
        )
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
            snapshot(
                transition.state
            )
        )

    return history


def auc_pairwise(
    positive,
    negative,
    higher_is_risk=True,
):
    p = np.asarray(
        positive,
        dtype=float,
    )

    n = np.asarray(
        negative,
        dtype=float,
    )

    p = p[np.isfinite(p)]
    n = n[np.isfinite(n)]

    if len(p) == 0 or len(n) == 0:
        return np.nan

    if higher_is_risk:
        wins = (
            p[:, None]
            > n[None, :]
        )

        ties = (
            p[:, None]
            == n[None, :]
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


def summarize(
    df,
    lead,
):
    x = df[
        df["lead_s"] == lead
    ]

    pos = x[
        x["is_collider"] == 1
    ]

    neg = x[
        x["is_collider"] == 0
    ]

    print()
    print("=" * 110)
    print(
        f"LEAD = {lead:.1f} s"
    )
    print("=" * 110)

    print(
        "Collider rows:",
        len(pos),
    )

    print(
        "Control rows :",
        len(neg),
    )

    configs = [
        (
            "distance",
            False,
        ),
        (
            "closing_speed",
            True,
        ),
        (
            "dcpa",
            False,
        ),
        (
            "predicted_clearance",
            False,
        ),
        (
            "interaction_risk",
            True,
        ),
    ]

    print()
    print(
        f"{'Metric':<24}"
        f"{'Collider med':>14}"
        f"{'Control med':>14}"
        f"{'Collider mean':>15}"
        f"{'Control mean':>14}"
        f"{'AUC':>9}"
    )

    print("-" * 90)

    for metric, higher in configs:
        auc = auc_pairwise(
            pos[metric],
            neg[metric],
            higher_is_risk=higher,
        )

        print(
            f"{metric:<24}"
            f"{pos[metric].median():>14.3f}"
            f"{neg[metric].median():>14.3f}"
            f"{pos[metric].mean():>15.3f}"
            f"{neg[metric].mean():>14.3f}"
            f"{auc:>9.3f}"
        )

    print()
    print(
        "Collider approaching:",
        f"{100*pos['approaching'].mean():.1f}%"
    )

    print(
        "Control approaching :",
        f"{100*neg['approaching'].mean():.1f}%"
    )

    for threshold in [
        0.0,
        1.0,
        2.0,
        3.0,
    ]:
        pos_rate = (
            pos["predicted_clearance"]
            <= threshold
        ).mean()

        neg_rate = (
            neg["predicted_clearance"]
            <= threshold
        ).mean()

        print(
            f"Clearance <= {threshold:.1f} m  "
            f"Collider={100*pos_rate:6.1f}%  "
            f"Control={100*neg_rate:6.1f}%"
        )


# ============================================================
# LOAD COLLIDER LIST
# ============================================================

detail = pd.read_csv(
    DETAIL_PATH
)

eval_df = (
    pd.read_csv(
        EVAL_PATH
    )
    .set_index(
        "scenario_index"
    )
)

selected = sorted(
    detail[
        "scenario_index"
    ]
    .astype(int)
    .unique()
    .tolist()
)

collider_map = {}

for scenario_index, group in (
    detail.groupby(
        "scenario_index"
    )
):
    collider_map[
        int(scenario_index)
    ] = (
        group[
            "collider_object_index"
        ]
        .astype(int)
        .unique()
        .tolist()
    )


print("=" * 110)
print("V27 INTERACTION FEATURE CALIBRATION")
print("=" * 110)

print(
    "Selected scenarios:",
    len(selected),
)

print(
    "Collider rows     :",
    len(detail),
)

print(
    "Leads             :",
    LEADS,
)

print()


# ============================================================
# SETUP V26 FROZEN EVALUATION
# ============================================================

env, step_fn, _, _ = (
    utils.setup_evaluation(
        policy_type="ai",
        path_model=MODEL,
        source_dir="runs",
        path_dataset=DATASET,
        eval_name=(
            "analysis/"
            "v27_interaction_tmp"
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

base_key = jax.random.PRNGKey(0)

rows = []

done_selected = set()

global_start = 0

rollout_mismatch = []


# ============================================================
# ROLLOUT
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
            f"scenario {scenario_index}"
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
            eval_df.loc[
                scenario_index,
                "episode_length",
            ]
        )

        if actual_steps != expected_steps:
            rollout_mismatch.append(
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

        colliders = collider_map[
            scenario_index
        ]

        collision_frame = (
            actual_steps
        )

        for lead in LEADS:
            lead_steps = int(
                round(
                    lead / DT
                )
            )

            frame = (
                collision_frame
                - lead_steps
            )

            if frame < 0:
                continue

            snap = history[
                frame
            ]

            valid_colliders = [
                idx
                for idx in colliders
                if (
                    idx < len(
                        snap["valid"]
                    )
                    and snap["valid"][idx]
                )
            ]

            if not valid_colliders:
                continue

            # Positive collider rows
            for obj_idx in (
                valid_colliders
            ):
                feat = interaction_features(
                    snap,
                    obj_idx,
                )

                rows.append(
                    {
                        "scenario_index":
                            scenario_index,

                        "lead_s":
                            lead,

                        "object_index":
                            obj_idx,

                        "object_rank":
                            snap[
                                "rank"
                            ][obj_idx],

                        "is_collider":
                            1,

                        **feat,
                    }
                )

            # Matched control objects:
            # same frame, currently Top-16,
            # excluding ALL known colliders.
            control_ids = []

            for obj_idx in range(
                len(
                    snap["valid"]
                )
            ):
                if obj_idx in colliders:
                    continue

                rank = snap[
                    "rank"
                ][obj_idx]

                if (
                    snap["valid"][obj_idx]
                    and np.isfinite(rank)
                    and rank <= TOP_K
                ):
                    control_ids.append(
                        obj_idx
                    )

            for obj_idx in control_ids:
                feat = interaction_features(
                    snap,
                    obj_idx,
                )

                rows.append(
                    {
                        "scenario_index":
                            scenario_index,

                        "lead_s":
                            lead,

                        "object_index":
                            obj_idx,

                        "object_rank":
                            snap[
                                "rank"
                            ][obj_idx],

                        "is_collider":
                            0,

                        **feat,
                    }
                )

        done_selected.add(
            scenario_index
        )

    global_start = batch_end

    if len(done_selected) == len(
        selected
    ):
        break


# ============================================================
# RESULT
# ============================================================

result = pd.DataFrame(
    rows
)

print()
print("=" * 110)
print("ROLLOUT CONSISTENCY")
print("=" * 110)

print(
    "Processed:",
    len(done_selected),
)

print(
    "Mismatches:",
    len(rollout_mismatch),
)

for item in rollout_mismatch:
    print(
        " ",
        item,
    )

if result.empty:
    raise RuntimeError(
        "No calibration rows produced."
    )

result.to_csv(
    OUT_PATH,
    index=False,
)

for lead in LEADS:
    summarize(
        result,
        lead,
    )


# ============================================================
# MATCHED COLLIDER RANK AMONG TOP-16 BY RISK
# ============================================================

print()
print("=" * 110)
print("COLLIDER RISK RANK WITHIN SAME SCENE")
print("=" * 110)

for lead in LEADS:
    x = result[
        result["lead_s"] == lead
    ]

    scene_ranks = []

    for scenario_index, group in (
        x.groupby(
            "scenario_index"
        )
    ):
        if not (
            group[
                "is_collider"
            ] == 1
        ).any():
            continue

        g = group.sort_values(
            "interaction_risk",
            ascending=False,
        ).reset_index(
            drop=True
        )

        collider_positions = (
            np.flatnonzero(
                g[
                    "is_collider"
                ].to_numpy()
                == 1
            )
            + 1
        )

        scene_ranks.extend(
            collider_positions.tolist()
        )

    if not scene_ranks:
        continue

    ranks = np.asarray(
        scene_ranks
    )

    print()
    print(
        f"Lead {lead:.1f}s:"
    )

    print(
        "  collider median risk rank:",
        float(
            np.median(
                ranks
            )
        ),
    )

    print(
        "  risk rank <= 1:",
        f"{100*np.mean(ranks <= 1):.1f}%"
    )

    print(
        "  risk rank <= 3:",
        f"{100*np.mean(ranks <= 3):.1f}%"
    )

    print(
        "  risk rank <= 5:",
        f"{100*np.mean(ranks <= 5):.1f}%"
    )


print()
print("=" * 110)
print("SAVED")
print("=" * 110)

print(
    OUT_PATH
)
