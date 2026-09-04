# Copyright 2025 Valeo.

"""Reward functions for the simulator."""

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import metrics as waymax_metrics
from waymax.env.planning_agent_environment import PlanningAgentEnvironment
from waymax.utils import geometry as waymax_geometry

from vmax.simulator import metrics, operations
from vmax.simulator.wrappers.base import Wrapper


class RewardLinearWrapper(Wrapper):
    """Wraps the environment to compute rewards using a linear combination of functions."""

    def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
        """Initialize the reward linear wrapper.

        Args:
            env: The environment to wrap.
            reward_config: Configuration for reward computation.

        """
        super().__init__(env)
        self._reward_config = reward_config

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Combine rewards provided from the reward functions.
        The reward is computed as a linear combination of the individual rewards,
        weighted by their respective coefficients.

        Args:
            state: Current simulator state.

        Returns:
            Combined reward value.
        """
        reward = 0.0

        for reward_name, reward_weigth in self._reward_config.items():
            reward_fn = _get_reward_fn(reward_name)
            reward += reward_fn(state) * reward_weigth

        return jnp.array(reward, dtype=jnp.float32)


class RewardCustomWrapper(Wrapper):
    """Wraps the environment to compute rewards using a custom function."""

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Compute a custom reward.

        This is a placeholder function, use it as you wish o7

        Args:
            state: Current simulator state.
            action: Action taken.

        Returns:
            Reward value.

        """
        # Implement your custom reward logic here
        return jnp.array(0.0)


def _get_reward_fn(reward_name: str) -> callable:
    """Retrieve a reward function by its name.

    Args:
        reward_name: Name identifier of the reward function.

    Returns:
        Callable reward function.

    """
    reward_dict = {
        "log_div_clip": _compute_log_divergence_clip_reward,
        "log_div": _compute_log_divergence_reward,
        "overlap": _compute_overlap_reward,
        "proximity": _compute_proximity_reward,
        "offroad": _compute_offroad_reward,
        "offroad_in_box": _compute_offroad_in_box_reward,
        "off_route": _compute_off_route_reward,
        "below_ttc": _compute_below_ttc_reward,
        "below_ttc_all": _compute_below_ttc_all_reward,
        "red_light": _compute_red_light_reward,
        "comfort": _compute_comfort_reward,
        "overspeed": _compute_overspeed_limit_reward,
        "driving_direction": _compute_driving_direction_reward,
        "lane_deviation": _compute_deviate_lane_reward,
        "sdc_progression": _compute_making_progress_reward,
        "expert_progression": _compute_expert_progress_reward,
    }

    if reward_name not in reward_dict:
        raise ValueError(f"Reward function {reward_name} not implemented.")

    return reward_dict[reward_name]


# Penalty based rewards


def _compute_overlap_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing overlaps between the SDC and other objects.

    Args:
        state: Current simulator state.

    Returns:
        True if overlap detected, False otherwise.

    """
    overlap = waymax_metrics.OverlapMetric().compute(state).value

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_overlap = jax.tree.map(lambda x: x[sdc_idx], overlap)

    return sdc_overlap == 1.0


def _compute_proximity_reward(
    state: datatypes.SimulatorState,
    longitudinal_margin: float = 1.0,
    lateral_margin: float = 0.5,
) -> bool:
    """Penalize objects entering an expanded safety envelope around the SDC."""
    current_traj = datatypes.dynamic_slice(
        state.sim_trajectory,
        state.timestep,
        1,
        -1,
    )

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)

    traj_5dof = current_traj.stack_fields(
        ["x", "y", "length", "width", "yaw"]
    ).squeeze()

    # Expand only the SDC bounding box. Length and width are full dimensions,
    # so adding a margin on both sides requires twice the desired margin.
    expanded_traj = traj_5dof.at[sdc_idx, 2].add(2.0 * longitudinal_margin)
    expanded_traj = expanded_traj.at[sdc_idx, 3].add(2.0 * lateral_margin)

    pairwise_overlap = waymax_geometry.compute_pairwise_overlaps(expanded_traj)

    valid = current_traj.valid.squeeze()

    proximity = pairwise_overlap[sdc_idx]
    proximity = jnp.logical_and(proximity, valid)

    # Ignore overlap with itself.
    proximity = proximity.at[sdc_idx].set(False)

    return jnp.any(proximity)


def _compute_offroad_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing when the SDC drives off the road.

    Args:
        state: Current simulator state.

    Returns:
        True if off-road detected, False otherwise.

    """
    offroad = waymax_metrics.OffroadMetric().compute(state).value

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_offroad = jax.tree.map(lambda x: x[sdc_idx], offroad)

    return sdc_offroad == 1.0


def _compute_offroad_in_box_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing offroad via the direction-free in-box test.

    Mirrors :func:`_compute_offroad_reward` but uses :class:`OffroadInBoxMetric`
    (a road-edge point inside the SDC 2D footprint) instead of Waymax's
    ``OffroadMetric``. Intended for datasets where road-edge ``dir_xyz`` / ``ids``
    are unreliable (e.g. rideflux). ``compute`` already returns the SDC-scoped
    scalar, so no per-object indexing is needed. The footprint margin is hardcoded
    (the default of ``is_sdc_offroad_in_box``), shared with the metric and
    termination — not configurable here. Assign a negative weight.

    Args:
        state: Current simulator state.

    Returns:
        True if off-road detected, False otherwise.

    """
    offroad = metrics.OffroadInBoxMetric().compute(state).value

    return offroad == 1.0


def _compute_red_light_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing red light violations.

    Args:
        state: Current simulator state.

    Returns:
        True if red light violation detected, False otherwise.

    """
    has_runned_red_light = metrics.RunRedLightMetric().compute(state).value

    return has_runned_red_light


def _compute_overspeed_limit_reward(
    state: datatypes.SimulatorState, threshold: float = 2.23
) -> bool:
    """Compute a reward based on speed-limit adherence.

    Returns True if the SDC's speed exceeds the road's speed limit by more than
    2.23 m/s (approximately 5 mph).

    Args:
        state: Current simulator state.

    Returns:
        True if speed limit is exceeded by more than 2.23 m/s, False otherwise.

    """
    speed_limit = metrics.infer_speed_limit_from_simulator_state(state)
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    ego_speed = state.current_sim_trajectory.speed[sdc_idx].squeeze()

    return ego_speed > speed_limit + threshold


def _compute_below_ttc_reward(state: datatypes.SimulatorState, threshold: float = 1.5) -> bool:
    """Compute a reward based on time-to-collision with other objects.

    The time-to-collision (TTC) is a measure of how long it would take for the SDC to
    collide with another object if both continue on their current trajectories.

    Args:
        state: Current simulator state.
        threshold: Minimum safe time-to-collision in seconds. Default is 1.5s.

    Returns:
        True if TTC is below threshold (unsafe), False otherwise (safe).

    """
    ttc = metrics.TimeToCollisionMetric().compute(state).value

    return ttc < threshold


def _compute_below_ttc_all_reward(
    state: datatypes.SimulatorState,
    threshold: float = 1.5,
) -> bool:
    """Penalize predicted collisions with any valid surrounding object.

    Unlike _compute_below_ttc_reward(), this reward does not filter objects
    using the ego-front 'is_ahead' condition. It reuses the same OBB-based
    constant-velocity collision prediction from TimeToCollisionMetric.
    """
    current_traj = datatypes.dynamic_slice(
        state.sim_trajectory,
        state.timestep,
        1,
        -1,
    )

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)

    ttc_metric = metrics.TimeToCollisionMetric()

    ttc, _ = ttc_metric._compute_ttc(
        current_traj,
        sdc_idx,
        dt=0.1,
        time_horizon=5.0,
    )

    min_ttc_all = jnp.min(ttc)

    return min_ttc_all < threshold


def _compute_off_route_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing deviation from the planned route.

    Returns True if the SDC has deviated from its planned route. A deviation occurs
    when the SDC's position has a non-zero distance to the nearest point on the route.

    Args:
        state: Current simulator state.

    Returns:
        True if off-route detected, False otherwise.
    """
    # Waymax offroute's score: 0 if you're on-route, distance to route if you're offroute
    off_route_score = metrics.OffRouteMetric().compute(state).value

    return off_route_score > 0


def _compute_driving_direction_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward for driving direction compliance.

    This reward is based on the driving direction compliance metric, which evaluates
    whether the SDC is following the intended driving direction.

    Args:
        state: Current simulator state.

    Returns:
        True if driving direction is compliant, False otherwise.

    """
    driving_direction_metric = metrics.DrivingDirectionComplianceMetric().compute(state).value

    return driving_direction_metric > 0


def _compute_deviate_lane_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing lane deviations.

    This reward is based on the lane deviation metric, which evaluates how much the SDC
    deviates from its intended lane.

    Args:
        state: Current simulator state.

    Returns:
        True if lane deviation is detected, False otherwise.

    """
    on_multiple_lanes_metric = metrics.OnMultipleLanesMetric().compute(state).value

    return on_multiple_lanes_metric > 0


# Reward based rewards


def _compute_log_divergence_clip_reward(
    state: datatypes.SimulatorState, threshold: float = 0.3
) -> bool:
    """Compute reward based on whether log divergence exceeds a threshold.

    The log divergence measures how much the SDC's trajectory deviates from the expected path.

    Args:
        state: Current simulator state.
        threshold: Maximum acceptable divergence (in log space) before considering it a deviation.
                  Default is 0.3, higher values are more permissive.

    Returns:
        True if divergence exceeds threshold, False otherwise.
    """
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    return log_divergence < threshold


def _compute_making_progress_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward promoting forward progress along the route.

    Compares the current progression metric with the previous timestep to determine
    if the SDC is making forward progress along its intended route.

    Args:
        state: Current simulator state.

    Returns:
        True if the SDC made forward progress, False otherwise.
    """
    current_progression = waymax_metrics.ProgressionMetric().compute(state).value
    n_state = state.replace(timestep=state.timestep - 1)
    previous_progression = waymax_metrics.ProgressionMetric().compute(n_state).value

    return current_progression > previous_progression


# Expert-path progression: roadgraph-independent alternative to the sdc_paths-based
# ProgressionMetric above. Wired to the "expert_progression" reward key in _get_reward_fn.

_EXPERT_PROGRESS_MAX_STEP_M = 3.0  # clip per-step arc delta (m): bounds projection jumps
_EXPERT_MIN_PATH_M = 0.5  # expert barely moved → emit no progress signal


def _expert_path_arclength(state: datatypes.SimulatorState) -> tuple[jax.Array, jax.Array]:
    """Project the SDC's current position onto its own expert (logged) trajectory.

    Returns ``(arc, total)``: ``arc`` is the cumulative arc-length (m) of the nearest
    expert point to the SDC's current xy, ``total`` is the expert path's full length
    (m). The reference is the logged ego trajectory itself, so this is fully
    independent of the roadgraph / sdc_paths.
    """
    sdc_index = operations.get_index(state.object_metadata.is_sdc)
    sim = jax.tree.map(lambda x: x[sdc_index], state.sim_trajectory)
    expert = jax.tree.map(lambda x: x[sdc_index], state.log_trajectory)

    # Cumulative arc-length along the expert path: cum[i] = distance from point 0 to i.
    seg = jnp.linalg.norm(jnp.diff(expert.xy, axis=0), axis=-1)
    cum = jnp.concatenate([jnp.zeros((1,), dtype=seg.dtype), jnp.cumsum(seg)])

    # Arc-length of the nearest valid expert point to the current sim position.
    sdc_xy = sim.xy[state.timestep]
    dist = jnp.where(expert.valid, jnp.linalg.norm(sdc_xy - expert.xy, axis=-1), jnp.inf)
    return cum[jnp.argmin(dist)], cum[-1]


def _compute_expert_progress_reward(state: datatypes.SimulatorState) -> float:
    """Dense forward-progress reward measured along the expert (logged) path.

    Drop-in replacement for the sdc_paths-based ProgressionMetric: projects the SDC's
    current and previous positions onto the logged ego trajectory and rewards the gain
    in arc-length, normalized by the expert path's total length. Stays valid when
    sdc_paths are inaccurate (e.g. rideflux) since it never touches the roadgraph.

    The per-step delta is clipped to bound spurious jumps from the nearest-point
    projection (self-intersections / U-turns); the episode return then telescopes to
    the ``progress_ratio`` evaluation metric.
    """
    curr_arc, total = _expert_path_arclength(state)
    prev_arc, _ = _expert_path_arclength(state.replace(timestep=state.timestep - 1))

    delta = jnp.clip(
        curr_arc - prev_arc, -_EXPERT_PROGRESS_MAX_STEP_M, _EXPERT_PROGRESS_MAX_STEP_M
    )
    return jnp.where(total < _EXPERT_MIN_PATH_M, 0.0, delta / total)


# Linar reward functions


def _compute_log_divergence_reward(state: datatypes.SimulatorState) -> float:
    """Compute reward based on log divergence.

    The log divergence measures how much the SDC's trajectory deviates from the expected path.

    Args:
        state: Current simulator state.

    Returns:
        Log divergence value.

    """
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    return log_divergence


def _compute_comfort_reward(state: datatypes.SimulatorState) -> float:
    """Compute a comfort-related reward.

    This reward is based on the comfort metric, which evaluates the smoothness of the SDC's
    trajectory. A higher comfort metric indicates a smoother and more comfortable ride.

    Args:
        state: Current simulator state.

    Returns:
        Comfort metric value.

    """
    comfort_metric_reward_2 = metrics.ComfortMetric().compute_reward(state).value

    return comfort_metric_reward_2
