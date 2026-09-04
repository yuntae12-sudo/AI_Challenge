from pathlib import Path
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from waymax.utils import geometry

from vmax.scripts.evaluate import utils
from vmax.simulator import datasets, make_data_generator


DATASET = (
    "/mnt/e/AI_Challenge/datasets/splits/"
    "secondary_val_1024/secondary_val_1024.tfrecord@1024"
)

MODEL = "v26_secondary_frozen_24322560"

HIGH_ZERO_CSV = Path(
    "analysis/v26_high_value_zero_scenes.csv"
)

OUT_DETAIL = Path(
    "analysis/v26_collision_nearest_rank_detail.csv"
)

OUT_SCENARIO = Path(
    "analysis/v26_collision_nearest_rank_scenario.csv"
)

DT = 0.1
TOP_K = 16
MAX_OBJECTS = 64
BATCH_SIZE = 64


def scalar(x):
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def take_batch_item(tree, idx):
    return jax.tree_util.tree_map(
        lambda x: x[idx],
        tree,
    )


def strip_batch(x):
    x = np.asarray(x)

    # Single-scenario env generally keeps leading batch dim = 1.
    if x.ndim >= 1 and x.shape[0] == 1:
        x = x[0]

    return x


def current_snapshot(state):
    """Current distances/ranks for every object."""

    traj = state.sim_trajectory

    t = int(
        np.asarray(state.timestep)
        .reshape(-1)[0]
    )

    x = strip_batch(traj.x)
    y = strip_batch(traj.y)
    valid = strip_batch(traj.valid)

    is_sdc = strip_batch(
        state.object_metadata.is_sdc
    ).astype(bool)

    ego = int(np.flatnonzero(is_sdc)[0])

    px = x[:, t]
    py = y[:, t]
    v = valid[:, t].astype(bool)

    dx = px - px[ego]
    dy = py - py[ego]

    dist = np.sqrt(
        dx * dx + dy * dy
    )

    # Rank among OTHER valid objects.
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
        "ego_index": ego,
        "distance": dist,
        "rank": ranks,
        "valid": v,
    }


def colliding_objects(state):
    """Return actual OBB-overlapping object indices."""

    traj = state.sim_trajectory

    t = int(
        np.asarray(state.timestep)
        .reshape(-1)[0]
    )

    x = strip_batch(traj.x)
    y = strip_batch(traj.y)
    length = strip_batch(traj.length)
    width = strip_batch(traj.width)
    yaw = strip_batch(traj.yaw)
    valid = strip_batch(traj.valid)

    is_sdc = strip_batch(
        state.object_metadata.is_sdc
    ).astype(bool)

    ego = int(np.flatnonzero(is_sdc)[0])

    traj_5dof = np.stack(
        [
            x[:, t],
            y[:, t],
            length[:, t],
            width[:, t],
            yaw[:, t],
        ],
        axis=-1,
    )

    overlaps = np.asarray(
        geometry.compute_pairwise_overlaps(
            jnp.asarray(traj_5dof)
        )
    )

    collision = (
        overlaps[ego].astype(bool)
        & valid[:, t].astype(bool)
    )

    collision[ego] = False

    return np.flatnonzero(collision)


def value_at_lead(
    history,
    collision_idx,
    object_idx,
    lead_sec,
    field,
):
    steps = int(
        round(lead_sec / DT)
    )

    idx = collision_idx - steps

    if idx < 0:
        return np.nan

    return float(
        history[idx][field][object_idx]
    )


def continuous_top16_lead(
    history,
    collision_idx,
    object_idx,
):
    count = 0

    # Walk backwards from just BEFORE collision.
    for idx in range(
        collision_idx - 1,
        -1,
        -1,
    ):
        rank = history[idx]["rank"][object_idx]

        if (
            np.isfinite(rank)
            and rank <= TOP_K
        ):
            count += 1
        else:
            break

    return count * DT


def first_top16_lead(
    history,
    collision_idx,
    object_idx,
):
    first = None

    for idx in range(
        0,
        collision_idx + 1,
    ):
        rank = history[idx]["rank"][object_idx]

        if (
            np.isfinite(rank)
            and rank <= TOP_K
        ):
            first = idx
            break

    if first is None:
        return np.nan

    return (
        collision_idx - first
    ) * DT


def top16_fraction(
    history,
    collision_idx,
    object_idx,
    seconds,
):
    steps = int(
        round(seconds / DT)
    )

    start = max(
        0,
        collision_idx - steps,
    )

    # Exclude collision frame itself.
    samples = history[
        start:collision_idx
    ]

    if not samples:
        return np.nan

    hits = []

    for sample in samples:
        rank = sample["rank"][object_idx]

        hits.append(
            np.isfinite(rank)
            and rank <= TOP_K
        )

    return float(
        np.mean(hits)
    )


def diagnose_one(
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

    rng_key, reset_key = jax.random.split(
        rng_key
    )

    reset_key = jax.random.split(
        reset_key,
        1,
    )

    env_transition = reset_fn(
        scenario,
        reset_key,
    )

    history = [
        current_snapshot(
            env_transition.state
        )
    ]

    collision_found = False
    collision_indices = np.array(
        [],
        dtype=int,
    )

    at_fault_value = 0.0
    overlap_value = 0.0

    while not bool(
        np.asarray(
            env_transition.done
        ).reshape(-1)[0]
    ):
        rng_key, step_key = jax.random.split(
            rng_key
        )

        step_key = jax.random.split(
            step_key,
            1,
        )

        env_transition, _ = step_fn(
            env_transition,
            key=step_key,
        )

        history.append(
            current_snapshot(
                env_transition.state
            )
        )

        overlap_value = scalar(
            env_transition.metrics[
                "overlap"
            ]
        )

        at_fault_value = scalar(
            env_transition.metrics[
                "at_fault_collision"
            ]
        )

        if overlap_value > 0:
            collision_indices = (
                colliding_objects(
                    env_transition.state
                )
            )

            collision_found = True
            break

    if not collision_found:
        return [], {
            "scenario_index": scenario_index,
            "collision_found": False,
            "num_colliders": 0,
        }

    collision_frame = len(history) - 1

    rows = []

    for object_idx in collision_indices:
        row = {
            "scenario_index":
                scenario_index,

            "collider_object_index":
                int(object_idx),

            "collision_timestep":
                history[
                    collision_frame
                ]["timestep"],

            "episode_time_to_collision_s":
                collision_frame * DT,

            "metric_overlap":
                overlap_value,

            "metric_at_fault":
                at_fault_value,

            "rank_m3_0s":
                value_at_lead(
                    history,
                    collision_frame,
                    object_idx,
                    3.0,
                    "rank",
                ),

            "rank_m2_0s":
                value_at_lead(
                    history,
                    collision_frame,
                    object_idx,
                    2.0,
                    "rank",
                ),

            "rank_m1_0s":
                value_at_lead(
                    history,
                    collision_frame,
                    object_idx,
                    1.0,
                    "rank",
                ),

            "rank_m0_5s":
                value_at_lead(
                    history,
                    collision_frame,
                    object_idx,
                    0.5,
                    "rank",
                ),

            "dist_m3_0s":
                value_at_lead(
                    history,
                    collision_frame,
                    object_idx,
                    3.0,
                    "distance",
                ),

            "dist_m2_0s":
                value_at_lead(
                    history,
                    collision_frame,
                    object_idx,
                    2.0,
                    "distance",
                ),

            "dist_m1_0s":
                value_at_lead(
                    history,
                    collision_frame,
                    object_idx,
                    1.0,
                    "distance",
                ),

            "dist_m0_5s":
                value_at_lead(
                    history,
                    collision_frame,
                    object_idx,
                    0.5,
                    "distance",
                ),

            "top16_fraction_last3s":
                top16_fraction(
                    history,
                    collision_frame,
                    object_idx,
                    3.0,
                ),

            "top16_fraction_last2s":
                top16_fraction(
                    history,
                    collision_frame,
                    object_idx,
                    2.0,
                ),

            "top16_fraction_last1s":
                top16_fraction(
                    history,
                    collision_frame,
                    object_idx,
                    1.0,
                ),

            "continuous_top16_lead_s":
                continuous_top16_lead(
                    history,
                    collision_frame,
                    object_idx,
                ),

            "first_top16_lead_s":
                first_top16_lead(
                    history,
                    collision_frame,
                    object_idx,
                ),
        }

        rows.append(row)

    scenario_summary = {
        "scenario_index":
            scenario_index,

        "collision_found":
            True,

        "num_colliders":
            len(collision_indices),

        "metric_overlap":
            overlap_value,

        "metric_at_fault":
            at_fault_value,
    }

    return rows, scenario_summary


def print_lead_summary(df, col):
    available = df[col].notna()

    if not available.any():
        print(
            f"{col:<14}: no samples"
        )
        return

    x = df.loc[
        available,
        col,
    ]

    top16 = (
        x <= TOP_K
    ).mean()

    print(
        f"{col:<14}: "
        f"Top16 {100*top16:6.2f}% "
        f"({int((x <= TOP_K).sum())}/{len(x)}), "
        f"median rank={x.median():.1f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all selected collision scenes",
    )

    args = parser.parse_args()

    failure_df = pd.read_csv(
        HIGH_ZERO_CSV
    )

    eval_df = pd.read_csv(
        "benchmark/ai/mnt/e/AI_Challenge/datasets/splits/"
        "secondary_val_1024/secondary_val_1024.tfrecord@1024/"
        "v26_secondary_frozen_24322560/model_final/evaluation_episodes.csv"
    ).set_index("scenario_index")

    collision_ids = (
        failure_df[
            failure_df["FailureType"]
            == "OVERLAP+AT_FAULT"
        ]["scenario_index"]
        .astype(int)
        .tolist()
    )

    # Only collisions with >= 2.0 s of policy response time.
    selected = (
        eval_df.loc[collision_ids]
        .query("episode_length >= 20")
        .index
        .astype(int)
        .sort_values()
        .tolist()
    )

    if args.limit > 0:
        selected = selected[
            : args.limit
        ]

    selected_set = set(selected)

    print("=" * 100)
    print("V26 COLLIDING-OBJECT NEAREST-RANK DIAGNOSTIC")
    print("=" * 100)
    print("Selected scenarios :", len(selected))
    print("Object Top-K       :", TOP_K)
    print("Dataset batch      :", BATCH_SIZE)
    print()

    env, step_fn, _, _ = (
        utils.setup_evaluation(
            policy_type="ai",
            path_model=MODEL,
            source_dir="runs",
            path_dataset=DATASET,
            eval_name=(
                "analysis/"
                "v26_collision_rank_tmp"
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

        # IMPORTANT:
        # same ordering as the frozen
        # secondary evaluation.
        batch_dims=(
            BATCH_SIZE,
            1,
        ),

        seed=0,
        repeat=1,
    )

    all_rows = []
    scenario_rows = []

    done_selected = set()

    base_key = jax.random.PRNGKey(0)

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

        if wanted:
            for scenario_index in wanted:
                local_idx = (
                    scenario_index
                    - global_start
                )

                one_scenario = (
                    take_batch_item(
                        batch,
                        local_idx,
                    )
                )

                print(
                    f"[{len(done_selected)+1:02d}/"
                    f"{len(selected):02d}] "
                    f"scenario "
                    f"{scenario_index}"
                )

                rows, summary = (
                    diagnose_one(
                        one_scenario,
                        scenario_index,
                        reset_fn,
                        step_fn,
                        base_key,
                    )
                )

                all_rows.extend(rows)
                scenario_rows.append(
                    summary
                )

                done_selected.add(
                    scenario_index
                )

        global_start = batch_end

        if len(done_selected) == len(
            selected_set
        ):
            break

    detail = pd.DataFrame(
        all_rows
    )

    scenarios = pd.DataFrame(
        scenario_rows
    )

    print()
    print("=" * 100)
    print("DIAGNOSTIC RESULT")
    print("=" * 100)

    print(
        "Requested scenarios :",
        len(selected),
    )

    print(
        "Processed scenarios :",
        len(done_selected),
    )

    if len(scenarios):
        print(
            "Collision found    :",
            int(
                scenarios[
                    "collision_found"
                ].sum()
            ),
        )

    print(
        "Collider rows       :",
        len(detail),
    )

    if detail.empty:
        print()
        print(
            "No collider detail produced."
        )
        return

    print()
    print("=" * 100)
    print("TOP-16 VISIBILITY BEFORE COLLISION")
    print("=" * 100)

    print_lead_summary(
        detail,
        "rank_m3_0s",
    )

    print_lead_summary(
        detail,
        "rank_m2_0s",
    )

    print_lead_summary(
        detail,
        "rank_m1_0s",
    )

    print_lead_summary(
        detail,
        "rank_m0_5s",
    )

    print()
    print(
        "Continuous Top16 >= 1.0 s :",
        f"{100 * (detail['continuous_top16_lead_s'] >= 1.0).mean():.2f}%"
    )

    print(
        "Continuous Top16 >= 2.0 s :",
        f"{100 * (detail['continuous_top16_lead_s'] >= 2.0).mean():.2f}%"
    )

    print(
        "Continuous Top16 >= 3.0 s :",
        f"{100 * (detail['continuous_top16_lead_s'] >= 3.0).mean():.2f}%"
    )

    print()
    print(
        "Mean Top16 fraction last 1 s :",
        f"{detail['top16_fraction_last1s'].mean():.3f}"
    )

    print(
        "Mean Top16 fraction last 2 s :",
        f"{detail['top16_fraction_last2s'].mean():.3f}"
    )

    print(
        "Mean Top16 fraction last 3 s :",
        f"{detail['top16_fraction_last3s'].mean():.3f}"
    )

    print()
    print("=" * 100)
    print("PER-COLLIDER SUMMARY")
    print("=" * 100)

    show_cols = [
        "scenario_index",
        "collider_object_index",
        "episode_time_to_collision_s",
        "rank_m3_0s",
        "rank_m2_0s",
        "rank_m1_0s",
        "rank_m0_5s",
        "dist_m2_0s",
        "dist_m1_0s",
        "continuous_top16_lead_s",
    ]

    print(
        detail[
            show_cols
        ].to_string(
            index=False
        )
    )

    detail.to_csv(
        OUT_DETAIL,
        index=False,
    )

    scenarios.to_csv(
        OUT_SCENARIO,
        index=False,
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)
    print(OUT_DETAIL)
    print(OUT_SCENARIO)


if __name__ == "__main__":
    main()
