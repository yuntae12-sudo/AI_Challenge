# Copyright 2025 Valeo.

"""Base class for feature extractors."""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib as mpl
from waymax import datatypes
from waymax.utils import geometry

from vmax.simulator import features, operations, overrides
from vmax.simulator.features import extractor

FEATURE_MAP = {
    "waypoints": ("xy",),
    "velocity": ("vel_xy",),
    "speed": ("speed",),
    "yaw": ("yaw",),
    "size": ("length", "width"),
    "valid": ("valid",),
    "direction": ("dir_xy",),
    "types": ("types",),
    "state": ("state",),
    "object_types": ("object_types",),
    "static_clearance": ("static_clearance",),
    "predicted_clearance": ("predicted_clearance",),
}


TYPE_MAP = {
    "lane_center": [
        datatypes.MapElementIds.LANE_SURFACE_STREET,
        datatypes.MapElementIds.LANE_FREEWAY,
    ],
    "road_edge": [
        datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
        datatypes.MapElementIds.ROAD_EDGE_MEDIAN,
    ],
    "road_line": [
        datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE,
        datatypes.MapElementIds.ROAD_LINE_SOLID_SINGLE_WHITE,
        datatypes.MapElementIds.ROAD_LINE_SOLID_DOUBLE_WHITE,
        datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_YELLOW,
        datatypes.MapElementIds.ROAD_LINE_BROKEN_DOUBLE_YELLOW,
        datatypes.MapElementIds.ROAD_LINE_SOLID_SINGLE_YELLOW,
        datatypes.MapElementIds.ROAD_LINE_SOLID_DOUBLE_YELLOW,
        datatypes.MapElementIds.ROAD_LINE_PASSING_DOUBLE_YELLOW,
    ],
    "traffic_signs": [
        datatypes.MapElementIds.STOP_SIGN,
        datatypes.MapElementIds.CROSSWALK,
        datatypes.MapElementIds.SPEED_BUMP,
    ],
    "bike_lane": [datatypes.MapElementIds.LANE_BIKE_LANE],
    "unknown": [
        datatypes.MapElementIds.LANE_UNDEFINED,
        datatypes.MapElementIds.UNKNOWN,
        datatypes.MapElementIds.ROAD_EDGE_UNKNOWN,
    ],
}


class VecFeaturesExtractor(extractor.AbstractFeaturesExtractor):
    """Base class for feature extractors."""

    def __init__(
        self,
        obs_past_num_steps: int | None = None,
        objects_config: dict[str, Any] | None = None,
        roadgraphs_config: dict[str, Any] | None = None,
        traffic_lights_config: dict[str, Any] | None = None,
        path_target_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the base features extractor.

        Args:
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration for object features.
            roadgraphs_config: Configuration for roadgraph features.
                Can include 'element_types' to filter roadgraph elements.
                Example values:
                - For lane features: [datatypes.MapElementIds.LANE_SURFACE_STREET]
                - For road features: [datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
                                     datatypes.MapElementIds.ROAD_EDGE_MEDIAN]
                - For meta-types: ["lane_center", "road_edge"]
            traffic_lights_config: Configuration for traffic light features.
            path_target_config: Configuration for path target features.
        """
        self._obs_past_num_steps = obs_past_num_steps or 1

        # Set default configs if None
        self._objects_config = objects_config or {"features": []}
        self._roadgraphs_config = roadgraphs_config or {"features": []}
        self._traffic_light_config = traffic_lights_config or {"features": []}
        self._path_target_config = path_target_config or {"features": []}

        # Extract roadgraph element types from config and process them
        self._roadgraph_element_types = None
        raw_element_types = self._roadgraphs_config.get("element_types", None)
        if raw_element_types is not None:
            self._roadgraph_element_types = []
            for element_type in raw_element_types:
                if isinstance(element_type, int):
                    # Direct integer type
                    self._roadgraph_element_types.append(element_type)
                elif isinstance(element_type, str) and element_type in TYPE_MAP:
                    # String meta-type that needs to be converted using TYPE_MAP
                    self._roadgraph_element_types.extend(TYPE_MAP[element_type])
                else:
                    print(f"Warning: Unknown element type '{element_type}'")

        # Extract parameters with defaults
        # Objects
        self._num_closest_objects = self._objects_config.get("num_closest_objects", 8)

        # Roadgraph points
        self._meters_box = self._roadgraphs_config.get("meters_box")
        self._roadgraph_top_k = self._roadgraphs_config.get("roadgraph_top_k", 1000)
        self._roadgraph_interval = self._roadgraphs_config.get("interval", 1)
        self._max_meters = self._roadgraphs_config.get("max_meters", 50)

        if self._meters_box is None:
            self._roadgraph_top_k_prefilter = max(self._roadgraph_top_k, 2000)
        else:
            self._roadgraph_top_k_prefilter = (
                self._meters_box["front"] + self._meters_box["back"]
            ) * (self._meters_box["left"] + self._meters_box["right"])

        # Traffic lights
        self._num_closest_traffic_lights = self._traffic_light_config.get(
            "num_closest_traffic_lights", 16
        )

        # Path target
        self._num_target_path_points = self._path_target_config.get("num_points", 10)

        self._dict_mapping = {
            "types": extractor.utils.RG_MAPPING,  # roadgraph
            "state": extractor.utils.TL_MAPPING,  # trafficlight
            "object_types": extractor.utils.OBJECT_MAPPING,  # objects
        }

        # Build feature keys
        self._object_features_key = self._extract_feature_keys(self._objects_config["features"])
        self._roadgraph_features_key = self._extract_feature_keys(
            self._roadgraphs_config["features"]
        )
        self._traffic_lights_features_key = self._extract_feature_keys(
            self._traffic_light_config["features"]
        )
        self._path_target_features_key = self._extract_feature_keys(
            self._path_target_config["features"]
        )

    def _extract_feature_keys(self, feature_names: list[str]) -> list[str]:
        """Extract feature keys from feature names using FEATURE_MAP.

        Args:
            feature_names: List of feature names to extract.

        Returns:
            List of actual feature keys.
        """
        result = []
        for key in feature_names:
            if key not in FEATURE_MAP:
                # Log warning or raise error for unknown feature names
                print(f"Warning: Unknown feature name '{key}' not found in FEATURE_MAP")
                continue
            result.extend(FEATURE_MAP[key])
        return result

    @property
    def obs_past_num_steps(self) -> int:
        """Return the number of past steps considered for observation."""
        return self._obs_past_num_steps

    def _get_sdc_observation(
        self, state: datatypes.SimulatorState, key: jax.Array = None
    ) -> datatypes.Observation:
        """Retrieve the SDC observation from the simulator state.

        Args:
            state: The simulator state.
            key: Random key for noise and masking.

        Returns:
            The observation corresponding to the SDC.
        """
        sdc_observation = overrides.sdc_observation_from_state(
            state,
            self._obs_past_num_steps,
            self._roadgraph_top_k_prefilter,
            self._meters_box,
        )

        sdc_observation = jax.tree.map(lambda x: x[0], sdc_observation)

        if key is not None:
            gaussian_key, masking_key = jax.random.split(key, 2)
            sdc_observation = features.apply_obstruction(sdc_observation)
            sdc_observation = features.apply_gaussian_noise(sdc_observation, gaussian_key)
            sdc_observation = features.apply_random_masking(sdc_observation, masking_key)

        return sdc_observation

    def get_features_size(self, feature_keys: str) -> int:
        """Calculate the total feature size for given feature keys.

        Args:
            feature_keys: The key or keys representing the features.

        Returns:
            The sum of feature sizes.

        """
        return sum([extractor.get_feature_size(key, self._dict_mapping) for key in feature_keys])

    def extract_features(
        self,
        state: datatypes.SimulatorState,
        key: jax.Array = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Extract features from the simulator state.

        Args:
            state: The simulator state.
            key: Random key for noise sampling.

        Returns:
            A tuple containing:
              - SDC object features.
              - Other objects features.
              - Roadgraph features.
              - Traffic lights features.
              - Path target features.

        """
        sdc_observation = self._get_sdc_observation(state, key)

        objects_features = self._build_objects_features(sdc_observation)
        roadgraphs_features = self._build_roadgraph_features(sdc_observation)
        traffic_lights_features = self._build_traffic_lights_features(sdc_observation)
        path_target_features = self._build_expert_target_features(sdc_observation, state)

        # (num_agents + 1, obs_past_num_steps, num_trajectories_features)
        stack_object_features = objects_features.stack_fields()
        # (obs_past_num_steps, num_trajectories_features), assuming the SDC is always the first object
        sdc_object_features = stack_object_features[0, :, :]
        # (num_closest_agents, obs_past_num_steps, num_trajectories_features)
        other_objects_features = stack_object_features[1:, :, :]

        return (
            sdc_object_features,
            other_objects_features,
            roadgraphs_features.stack_fields(),
            traffic_lights_features.stack_fields(),
            path_target_features.data,
        )

    def unflatten_features(
        self, vectorized_obs: jax.Array
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """Unflatten a vectorized observation into features and masks.

        Args:
            vectorized_obs: The vectorized observation.

        Returns:
            A tuple that contains:
              - A tuple of features.
              - A tuple of masks.

        """
        batch_dims = vectorized_obs.shape[-3:-1]
        flatten_size = vectorized_obs.shape[-1]
        unflatten_size = 0

        object_features_size = self.get_features_size(self._object_features_key)
        roadgraph_features_size = self.get_features_size(self._roadgraph_features_key)
        traffic_lights_features_size = self.get_features_size(self._traffic_lights_features_key)
        path_target_feature_size = self.get_features_size(self._path_target_features_key)

        sdc_object_size = 1 * self._obs_past_num_steps * object_features_size
        sdc_object_features = vectorized_obs[
            ..., unflatten_size : unflatten_size + sdc_object_size
        ]
        sdc_object_features = sdc_object_features.reshape(
            *batch_dims,
            1,
            self._obs_past_num_steps,
            object_features_size,
        )
        unflatten_size += sdc_object_size

        other_objects_size = (
            self._num_closest_objects * self._obs_past_num_steps * object_features_size
        )
        other_objects_features = vectorized_obs[
            ..., unflatten_size : unflatten_size + other_objects_size
        ]
        other_objects_features = other_objects_features.reshape(
            *batch_dims,
            self._num_closest_objects,
            self._obs_past_num_steps,
            object_features_size,
        )
        unflatten_size += other_objects_size

        roadgraph_size = self._roadgraph_top_k * roadgraph_features_size
        roadgraphs_features = vectorized_obs[..., unflatten_size : unflatten_size + roadgraph_size]
        roadgraphs_features = roadgraphs_features.reshape(
            *batch_dims,
            self._roadgraph_top_k,
            roadgraph_features_size,
        )
        unflatten_size += roadgraph_size

        traffic_lights_size = (
            self._num_closest_traffic_lights
            * self._obs_past_num_steps
            * traffic_lights_features_size
        )
        traffic_lights_features = vectorized_obs[
            ..., unflatten_size : unflatten_size + traffic_lights_size
        ]
        traffic_lights_features = traffic_lights_features.reshape(
            *batch_dims,
            self._num_closest_traffic_lights,
            self._obs_past_num_steps,
            traffic_lights_features_size,
        )
        unflatten_size += traffic_lights_size

        path_target_size = self._num_target_path_points * path_target_feature_size
        path_target_features = vectorized_obs[
            ..., unflatten_size : unflatten_size + path_target_size
        ]
        path_target_features = path_target_features.reshape(
            *batch_dims,
            self._num_target_path_points,
            path_target_feature_size,
        )
        unflatten_size += path_target_size

        assert (
            flatten_size == unflatten_size
        ), f"Unflatten size {unflatten_size} does not match {flatten_size}"

        features = (
            sdc_object_features[..., :-1],
            other_objects_features[..., :-1],
            roadgraphs_features[..., :-1],
            traffic_lights_features[..., :-1],
            path_target_features,
        )
        masks = (
            sdc_object_features[..., -1].astype(bool),
            other_objects_features[..., -1].astype(bool),
            roadgraphs_features[..., -1].astype(bool),
            traffic_lights_features[..., -1].astype(bool),
        )

        return features, masks

    def plot_features(self, state: datatypes.SimulatorState, ax: mpl.axes.Axes) -> None:
        """Plot all extracted features on the provided axes.

        Args:
            state: The simulator state.
            ax: The matplotlib axes to plot on.

        """
        sdc_observation = self._get_sdc_observation(state)

        objects_features = self._build_objects_features(sdc_observation)
        roadgraphs_features = self._build_roadgraph_features(sdc_observation)
        traffic_lights_features = self._build_traffic_lights_features(sdc_observation)
        path_target_features = self._build_expert_target_features(sdc_observation, state)

        # 1. Plot objects trajectories and bbox
        objects_features.plot(ax)

        # 2. Plot roadgraph points
        roadgraphs_features.plot(ax)

        # 3. Plot traffic lights
        traffic_lights_features.plot(ax)

        # 4. Plot path target
        path_target_features.plot(ax)

    def _build_objects_features(
        self,
        sdc_obs: datatypes.Observation,
    ) -> features.ObjectFeatures:
        """Create features for dynamic objects.

        Objects are still selected strictly by current Euclidean distance.
        V27 additionally exposes two geometry-aware interaction signals:

        - static_clearance:
          current center distance minus ego/object bounding-circle radii.

        - predicted_clearance:
          constant-velocity closest-approach distance minus the same radii.

        Negative clearance indicates geometric conflict under the
        corresponding circle approximation.
        """
        object_features = features.ObjectFeatures(
            field_names=self._object_features_key
        )

        if not self._object_features_key:
            return object_features

        # ----------------------------------------------------
        # Keep V26 object-selection behavior unchanged.
        # ----------------------------------------------------

        distances_ego_objects = jnp.linalg.norm(
            sdc_obs.trajectory.xy[:, -1, :],
            axis=-1,
        )

        distances_ego_valid_objects = jnp.where(
            sdc_obs.trajectory.valid[:, -1],
            distances_ego_objects,
            jnp.inf,
        )

        closest_object_idxs = operations.get_index(
            -distances_ego_valid_objects,
            k=self._num_closest_objects + 1,
            squeeze=False,
        )

        # ----------------------------------------------------
        # Geometry-aware features for selected objects.
        #
        # Shape:
        #   xy / vel: (objects, history, 2)
        #   scalar:   (objects, history)
        #
        # closest_object_idxs[0] is the SDC, consistent with
        # the existing downstream assumption.
        # ----------------------------------------------------

        selected_xy = (
            sdc_obs.trajectory.xy[
                closest_object_idxs
            ]
        )

        selected_vel = (
            sdc_obs.trajectory.vel_xy[
                closest_object_idxs
            ]
        )

        selected_length = (
            sdc_obs.trajectory.length[
                closest_object_idxs
            ]
        )

        selected_width = (
            sdc_obs.trajectory.width[
                closest_object_idxs
            ]
        )

        selected_valid = (
            sdc_obs.trajectory.valid[
                closest_object_idxs
            ]
        )

        # Waymax scalar trajectory fields can be (..., T)
        # or (..., T, 1); canonicalize to (..., T).
        if selected_length.ndim == selected_xy.ndim:
            selected_length = selected_length[..., 0]

        if selected_width.ndim == selected_xy.ndim:
            selected_width = selected_width[..., 0]

        if selected_valid.ndim == selected_xy.ndim:
            selected_valid = selected_valid[..., 0]

        ego_xy = selected_xy[0:1]
        ego_vel = selected_vel[0:1]

        relative_xy = (
            selected_xy
            - ego_xy
        )

        relative_vel = (
            selected_vel
            - ego_vel
        )

        center_distance = jnp.linalg.norm(
            relative_xy,
            axis=-1,
        )

        object_radius = (
            0.5
            * jnp.sqrt(
                selected_length**2
                + selected_width**2
            )
        )

        ego_radius = object_radius[0:1]

        static_clearance = (
            center_distance
            - ego_radius
            - object_radius
        )

        # ----------------------------------------------------
        # Constant-velocity closest approach.
        # Horizon = 5 seconds.
        #
        # t* = -(r dot v) / ||v||^2
        # ----------------------------------------------------

        relative_speed_sq = jnp.sum(
            relative_vel**2,
            axis=-1,
        )

        position_velocity_dot = jnp.sum(
            relative_xy
            * relative_vel,
            axis=-1,
        )

        tcpa_raw = jnp.where(
            relative_speed_sq > 1e-6,
            -position_velocity_dot
            / relative_speed_sq,
            0.0,
        )

        tcpa = jnp.clip(
            tcpa_raw,
            0.0,
            5.0,
        )

        closest_relative_xy = (
            relative_xy
            + tcpa[..., None]
            * relative_vel
        )

        dcpa = jnp.linalg.norm(
            closest_relative_xy,
            axis=-1,
        )

        predicted_clearance = (
            dcpa
            - ego_radius
            - object_radius
        )

        pair_valid = (
            selected_valid
            & selected_valid[0:1]
        )

        static_clearance = jnp.where(
            pair_valid,
            static_clearance,
            0.0,
        )

        predicted_clearance = jnp.where(
            pair_valid,
            predicted_clearance,
            0.0,
        )

        custom_features = {
            "static_clearance":
                static_clearance[..., None],
            "predicted_clearance":
                predicted_clearance[..., None],
        }

        # ----------------------------------------------------
        # Build final object tensor.
        # ----------------------------------------------------

        object_features = features.ObjectFeatures(
            field_names=self._object_features_key
        )

        for key in self._object_features_key:
            if key in custom_features:
                feature = custom_features[key]

            else:
                feature = (
                    getattr(
                        sdc_obs.metadata,
                        key,
                    )
                    if key == "object_types"
                    else getattr(
                        sdc_obs.trajectory,
                        key,
                    )
                )

                feature = feature[
                    closest_object_idxs
                ]

            feature = extractor.normalize_by_feature(
                feature,
                key,
                self._max_meters,
                self._dict_mapping,
            )

            if feature.ndim == 2:
                feature = jnp.expand_dims(
                    feature,
                    axis=-1,
                )

            setattr(
                object_features,
                key,
                feature,
            )

        return object_features


    def _build_roadgraph_features(
        self, sdc_obs: datatypes.Observation
    ) -> features.RoadgraphFeatures:
        """Create roadgraph features from the SDC observation.

        Args:
            sdc_obs: The SDC observation.

        Returns:
            An instance of RoadgraphFeatures.

        """
        roadgraph_features = features.RoadgraphFeatures(field_names=self._roadgraph_features_key)

        if len(self._roadgraph_features_key) == 0:
            return roadgraph_features

        roadgraph_points = self._reduce_and_filter_roadgraph_points(
            sdc_obs.roadgraph_static_points
        )

        for key in self._roadgraph_features_key:
            feature = getattr(roadgraph_points, key)
            feature = extractor.normalize_by_feature(
                feature, key, self._max_meters, self._dict_mapping
            )

            if feature.ndim == 1:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(roadgraph_features, key, feature)

        return roadgraph_features

    def _build_traffic_lights_features(
        self, sdc_obs: datatypes.Observation
    ) -> features.TrafficLightFeatures:
        """Create traffic light features from the SDC observation.

        Args:
            sdc_obs: The SDC observation.

        Returns:
            An instance of TrafficLightFeatures.

        """
        traffic_light_features = features.TrafficLightFeatures(
            field_names=self._traffic_lights_features_key
        )

        if len(self._traffic_lights_features_key) == 0:
            return traffic_light_features

        # (num_agents,)
        distances_traffic_lights = jnp.linalg.norm(sdc_obs.traffic_lights.xy[:, -1], axis=-1)
        # (num_agents,)
        distances_traffic_lights_valid = jnp.where(
            sdc_obs.traffic_lights.valid[:, -1],
            distances_traffic_lights,
            jnp.inf,
        )
        # (num_closest_agents + 1,)
        closest_tl_idxs = operations.get_index(
            -distances_traffic_lights_valid,
            k=self._num_closest_traffic_lights,
            squeeze=False,
        )

        for key in self._traffic_lights_features_key:
            feature = getattr(sdc_obs.traffic_lights, key)[closest_tl_idxs]
            feature = extractor.normalize_by_feature(
                feature, key, self._max_meters, self._dict_mapping
            )

            if feature.ndim == 2:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(traffic_light_features, key, feature)

        return traffic_light_features

    def _build_expert_target_features(
        self,
        sdc_obs: datatypes.Observation,
        state: datatypes.SimulatorState,
    ) -> features.PathTargetFeatures:
        """Build hybrid near-term + long-term route guidance.

        Uses four fixed near-term expert trajectory points and
        one final goal point. This retains local route geometry
        while avoiding the widely spaced targets used in V24.
        """
        if not self._path_target_features_key:
            return features.PathTargetFeatures()

        sdc_index = operations.get_index(
            state.object_metadata.is_sdc
        )

        sdc_log = jax.tree.map(
            lambda x: x[sdc_index],
            state.log_trajectory,
        )

        steps = jnp.arange(
            sdc_log.valid.shape[-1]
        )

        last_valid_idx = jnp.max(
            jnp.where(
                sdc_log.valid,
                steps,
                0,
            )
        )

        current_idx = jnp.minimum(
            state.timestep,
            last_valid_idx,
        )

        # Near-term offsets in trajectory steps.
        #
        # For a 10 Hz trajectory these correspond approximately
        # to 0.5 s, 1.0 s, 2.0 s and 4.0 s.
        near_offsets = jnp.array(
            [5, 10, 20, 40],
            dtype=jnp.int32,
        )

        near_idx = current_idx + near_offsets

        near_idx = jnp.minimum(
            near_idx,
            last_valid_idx,
        )

        # Final point preserves long-term route intent.
        future_idx = jnp.concatenate(
            [
                near_idx,
                jnp.asarray(
                    [last_valid_idx],
                    dtype=jnp.int32,
                ),
            ],
            axis=0,
        )

        target_global = sdc_log.xy[
            future_idx
        ]

        target_local = geometry.transform_points(
            sdc_obs.pose2d.matrix,
            target_global,
        )

        target_local = extractor.normalize_path(
            target_local,
            self._max_meters,
        )

        return features.PathTargetFeatures(
            xy=target_local
        )

    def _reduce_and_filter_roadgraph_points(
        self, roadgraph: datatypes.RoadgraphPoints
    ) -> datatypes.RoadgraphPoints:
        """Reduce and filter roadgraph points.

        Args:
            roadgraph: The roadgraph points.

        Returns:
            A reduced and filtered roadgraph points.

        """
        # Apply custom filter
        valid_filter = self._filter_roadgraph_points(roadgraph)
        roadgraph.valid = roadgraph.valid & valid_filter

        # Calculate distances
        xy = roadgraph.xy
        dist = jnp.linalg.norm(xy, axis=-1)
        dist = jnp.where(roadgraph.valid, dist, jnp.inf)

        # Create interval mask more efficiently
        idx_to_keep = jnp.arange(0, len(dist), self._roadgraph_interval)
        mask = jnp.zeros_like(dist, dtype=bool).at[idx_to_keep].set(True)

        # Apply mask to distances
        masked_dist = jnp.where(mask, dist, jnp.inf)

        # Get top-k indices - we're trying to find the k closest points
        _, idx = jax.lax.top_k(-masked_dist, min(self._roadgraph_top_k, masked_dist.size))

        return jax.tree.map(lambda x: x[idx], roadgraph)

    def _filter_roadgraph_points(self, roadgraph: datatypes.RoadgraphPoints) -> jax.Array:
        """Filter roadgraph points based on element types specified in configuration.

        Args:
            roadgraph: The roadgraph points.

        Returns:
            A boolean array indicating the valid points based on element types.

        """
        if self._roadgraph_element_types is None:
            # No filtering if no types specified
            return jnp.ones_like(roadgraph.valid, dtype=bool)

        # Initialize with zeros (all False)
        result = jnp.zeros_like(roadgraph.valid, dtype=bool)

        # Add each type with logical OR
        for element_type in self._roadgraph_element_types:
            result = result | (roadgraph.types == element_type)

        return result
