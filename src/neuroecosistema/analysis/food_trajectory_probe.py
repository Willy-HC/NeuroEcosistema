"""Step-by-step diagnostics for frozen prey policies."""

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import neuroecosistema.simulation.environment as simulation_environment
from neuroecosistema.agents import neural_controller
from neuroecosistema.analysis.food_orientation_probe import (
    sha256_file,
)
from neuroecosistema.analysis.population_condition_probe import (
    CONDITIONS,
    MODEL_IDS,
    _attach_networks,
    _create_environment,
    _create_initial_layout,
    _load_models,
    _validate_max_steps,
    _validate_seeds,
    default_model_paths,
)
from neuroecosistema.train import load_configs

DEFAULT_SEEDS = (1001, 1002, 1003)
MAX_STEPS = 500
PREY_COUNT = 10
OUTPUT_FILENAMES = (
    "food_trajectory.csv",
    "episode_summary.csv",
    "manifest.json",
)

PERCEPTION_COLUMNS = (
    "plant_dx",
    "plant_dy",
    "plant_distance",
    "predator_dx",
    "predator_dy",
    "predator_distance",
    "energy",
    "wall_distance",
)

TRACE_COLUMNS = (
    "episode_id",
    "model_id",
    "model_sha256",
    "seed",
    "agent_index",
    "step",
    "x_before",
    "y_before",
    "energy_before",
    "target_plant_id",
    "target_plant_x",
    "target_plant_y",
    "target_distance_before",
    "target_distance_after_move",
    "target_distance_change",
    "distance_change_status",
    "distance_change_na_reason",
    "ideal_vector_dx",
    "ideal_vector_dy",
    "ideal_unit_dx",
    "ideal_unit_dy",
    *(f"perception_{name}" for name in PERCEPTION_COLUMNS),
    "network_output_0",
    "network_output_1",
    "effective_direction_dx",
    "effective_direction_dy",
    "angular_error_deg",
    "angular_error_na_reason",
    "wall_escape_active_before",
    "wall_escape_active_after",
    "wall_avoidance_applied",
    "x_after_move",
    "y_after_move",
    "energy_after_move",
    "movement_dx",
    "movement_dy",
    "distance_traveled_step",
    "previous_target_plant_id",
    "target_changed",
    "target_switch",
    "target_change_cause",
    "plant_consumed",
    "consumed_plant_id",
    "consumed_current_target",
    "consumption_by_agent_index",
    "target_plant_present_after_interaction",
    "energy_after_interaction",
    "food_eaten_total",
    "plants_count_before_step",
)

SUMMARY_COLUMNS = (
    "episode_id",
    "model_id",
    "model_sha256",
    "seed",
    "prey_agents_initial",
    "predator_agents_initial",
    "steps_requested",
    "steps_executed",
    "prey_agents_alive",
    "mean_angular_error_deg",
    "median_angular_error_deg",
    "angular_error_steps",
    "angular_error_le_30_percent",
    "angular_error_ge_90_percent",
    "distance_change_steps",
    "approaching_steps",
    "approaching_percent",
    "receding_steps",
    "receding_percent",
    "unchanged_distance_steps",
    "target_changes",
    "target_switches",
    "target_acquisitions",
    "target_losses",
    "target_changes_consumed_by_self",
    "target_changes_consumed_by_other",
    "target_switches_nearest_changed",
    "target_changes_previous_plant_missing",
    "total_distance_traveled",
    "plants_consumed_total",
    "mean_plants_consumed_per_prey",
    "median_plants_consumed_per_prey",
    "min_plants_consumed_per_prey",
    "max_plants_consumed_per_prey",
    "distance_per_plant_consumed",
    "distance_per_plant_na_reason",
    "consumption_intervals",
    "mean_steps_between_consumptions",
    "median_steps_between_consumptions",
    "consumption_interval_na_reason",
    "wall_avoidance_steps",
    "wall_avoidance_frequency_percent",
    "diagnostic_steps",
)

DISTANCE_EPSILON = 1e-9
VECTOR_EPSILON = 1e-12


class PlantRegistry:
    """Assign stable episode-local IDs while retaining plant identities."""

    def __init__(self):
        self._plant_ids = {}
        self._plants = {}

    def ensure(self, plants):
        for plant in plants:
            identity = id(plant)
            if identity not in self._plant_ids:
                self._plant_ids[identity] = len(self._plant_ids) + 1
                self._plants[identity] = plant

    def plant_id(self, plant):
        if plant is None:
            return None
        self.ensure((plant,))
        return self._plant_ids[id(plant)]

    def active_ids(self, plants):
        self.ensure(plants)
        return {self._plant_ids[id(plant)] for plant in plants}


def calculate_angular_error(ideal_vector, direction_vector):
    """Return absolute angular error in degrees, or None for a null vector."""

    ideal_x, ideal_y = ideal_vector
    direction_x, direction_y = direction_vector
    ideal_magnitude = math.hypot(ideal_x, ideal_y)
    direction_magnitude = math.hypot(direction_x, direction_y)
    if ideal_magnitude <= VECTOR_EPSILON:
        return None
    if direction_magnitude <= VECTOR_EPSILON:
        return None

    cross = ideal_x * direction_y - ideal_y * direction_x
    dot = ideal_x * direction_x + ideal_y * direction_y
    return math.degrees(math.atan2(abs(cross), dot))


def classify_distance_change(distance_before, distance_after):
    """Classify movement relative to the same target plant."""

    if distance_before is None or distance_after is None:
        return "not_applicable"
    change = distance_after - distance_before
    if change < -DISTANCE_EPSILON:
        return "approaching"
    if change > DISTANCE_EPSILON:
        return "receding"
    return "unchanged"


def classify_target_change(
    previous_target_id,
    current_target_id,
    step,
    removed_targets,
    active_target_ids,
):
    """Describe why the selected plant differs from the prior step."""

    if previous_target_id == current_target_id:
        return "no_target" if current_target_id is None else "unchanged"
    if previous_target_id is None:
        if current_target_id is None:
            return "no_target"
        return "initial_acquisition" if step == 1 else "acquired"

    removal_reason = removed_targets.get(previous_target_id)
    if removal_reason is not None:
        return removal_reason
    if previous_target_id not in active_target_ids:
        return "previous_plant_missing"
    if current_target_id is None:
        return "target_lost"
    return "nearest_changed"


def match_consumed_plants(
    prey_agents,
    plants_before,
    plants_after,
    food_eaten_before,
    agent_indices,
    plant_registry,
):
    """Associate actual food-count increments with removed colliding plants."""

    after_ids = {id(plant) for plant in plants_after}
    removed_plants = [
        plant for plant in plants_before if id(plant) not in after_ids
    ]
    unmatched_removed = {id(plant) for plant in removed_plants}
    consumed_by_agent = {}
    consumer_by_plant = {}

    for prey in prey_agents:
        increase = prey.food_eaten - food_eaten_before[id(prey)]
        if increase == 0:
            continue
        if increase != 1:
            raise RuntimeError(
                "Se esperaba como máximo una planta consumida por presa y paso."
            )

        consumed_plant = next(
            (
                plant
                for plant in plants_before
                if id(plant) in unmatched_removed
                and math.dist(
                    (prey.x, prey.y),
                    (plant.x, plant.y),
                )
                <= prey.radius + plant.radius
            ),
            None,
        )
        if consumed_plant is None:
            raise RuntimeError(
                "No se pudo asociar el consumo con la planta eliminada."
            )

        plant_id = plant_registry.plant_id(consumed_plant)
        unmatched_removed.remove(id(consumed_plant))
        consumed_by_agent[id(prey)] = plant_id
        consumer_by_plant[plant_id] = agent_indices[id(prey)]

    if unmatched_removed:
        raise RuntimeError(
            "Se detectaron plantas eliminadas sin evento de consumo asociado."
        )

    return consumed_by_agent, consumer_by_plant


def _unit_vector(vector):
    x, y = vector
    magnitude = math.hypot(x, y)
    if magnitude <= VECTOR_EPSILON:
        return None
    return x / magnitude, y / magnitude


def _capture_neural_decision(
    agent,
    network,
    environment,
    original_calculate_direction,
    context,
):
    """Capture values from the actual controller invocation without replaying it."""

    original_perceive = neural_controller.perceive_prey
    original_wall_avoidance = neural_controller.apply_wall_avoidance
    original_forward = network.forward
    had_forward_override = "forward" in network.__dict__

    def capture_perception(*args, **kwargs):
        values = original_perceive(*args, **kwargs)
        context["perception"] = np.asarray(values).copy()
        return values

    def capture_forward(inputs):
        outputs = original_forward(inputs)
        context["network_inputs"] = np.asarray(inputs).copy()
        context["network_outputs"] = np.asarray(outputs).copy()
        return outputs

    def capture_wall_avoidance(
        wall_agent,
        dx,
        dy,
        width,
        height,
        *args,
        **kwargs,
    ):
        active_before = wall_agent.wall_escape_active
        direction = original_wall_avoidance(
            wall_agent,
            dx,
            dy,
            width,
            height,
            *args,
            **kwargs,
        )
        context["wall_escape_active_before"] = active_before
        context["wall_escape_active_after"] = wall_agent.wall_escape_active
        context["wall_avoidance_applied"] = wall_agent.wall_escape_active
        return direction

    neural_controller.perceive_prey = capture_perception
    neural_controller.apply_wall_avoidance = capture_wall_avoidance
    network.forward = capture_forward
    try:
        direction = original_calculate_direction(agent, network, environment)
    finally:
        neural_controller.perceive_prey = original_perceive
        neural_controller.apply_wall_avoidance = original_wall_avoidance
        if had_forward_override:
            network.forward = original_forward
        else:
            del network.forward

    if "perception" not in context or "network_outputs" not in context:
        raise RuntimeError("No se capturaron percepción y salida neuronales.")
    if not np.array_equal(context["perception"], context["network_inputs"]):
        raise RuntimeError("Las entradas registradas no coinciden con la red.")

    context["effective_direction"] = tuple(map(float, direction))
    return direction


def _trace_episode(
    condition,
    seed,
    model_id,
    model,
    simulation_config,
    max_steps,
    layout,
):
    environment = _create_environment(
        simulation_config,
        simulation_config["energy"],
        condition,
        seed,
        layout,
    )
    prey_networks, network_clones = _attach_networks(
        environment.prey,
        model,
    )
    environment.prey_networks = prey_networks

    original_calculate_direction = (
        simulation_environment.calculate_neural_direction
    )
    agent_indices = {
        id(agent): index for index, agent in enumerate(environment.prey)
    }
    plant_registry = PlantRegistry()
    plant_registry.ensure(environment.plants)
    previous_targets = {}
    removed_targets = {}
    trace_rows = []
    episode_id = f"foodtraj_seed{seed}_prey{model_id}"
    steps_executed = 0

    for step in range(1, max_steps + 1):
        if not environment.prey:
            break
        plant_registry.ensure(environment.plants)
        step_context = {}
        active_target_ids = plant_registry.active_ids(environment.plants)
        plants_count_before_step = len(environment.plants)

        def traced_calculate_direction(
            agent,
            network,
            current_environment,
            step_number=step,
            current_target_ids=active_target_ids,
            current_plants_count=plants_count_before_step,
            current_step_context=step_context,
        ):
            plant = current_environment.nearest_plant(
                agent,
                current_environment.plants,
            )
            target_id = plant_registry.plant_id(plant)
            previous_target_id = previous_targets.get(id(agent))
            target_change_cause = classify_target_change(
                previous_target_id,
                target_id,
                step_number,
                removed_targets,
                current_target_ids,
            )
            context = {
                "episode_id": episode_id,
                "model_id": model_id,
                "model_sha256": model["sha256"],
                "seed": seed,
                "agent_index": agent_indices[id(agent)],
                "step": step_number,
                "agent": agent,
                "target_plant": plant,
                "target_plant_id": target_id,
                "previous_target_plant_id": previous_target_id,
                "target_changed": previous_target_id != target_id,
                "target_switch": (
                    previous_target_id is not None
                    and target_id is not None
                    and previous_target_id != target_id
                ),
                "target_change_cause": target_change_cause,
                "x_before": float(agent.x),
                "y_before": float(agent.y),
                "energy_before": float(agent.energy),
                "food_eaten_before": agent.food_eaten,
                "plants_count_before_step": current_plants_count,
                "wall_escape_active_before": agent.wall_escape_active,
            }
            if plant is None:
                context.update(
                    {
                        "target_plant_x": None,
                        "target_plant_y": None,
                        "target_distance_before": None,
                        "ideal_vector": None,
                    }
                )
            else:
                ideal_vector = (
                    float(plant.x - agent.x),
                    float(plant.y - agent.y),
                )
                context.update(
                    {
                        "target_plant_x": float(plant.x),
                        "target_plant_y": float(plant.y),
                        "target_distance_before": math.hypot(*ideal_vector),
                        "ideal_vector": ideal_vector,
                    }
                )

            previous_targets[id(agent)] = target_id
            direction = _capture_neural_decision(
                agent,
                network,
                current_environment,
                original_calculate_direction,
                context,
            )
            current_step_context[id(agent)] = context
            return direction

        simulation_environment.calculate_neural_direction = (
            traced_calculate_direction
        )
        try:
            environment.update_agents(
                movement_cost=simulation_config["energy"]["movement_cost"],
                prey_strategy="neural",
                predator_strategy="neural",
            )
        finally:
            simulation_environment.calculate_neural_direction = (
                original_calculate_direction
            )

        interaction_prey = list(environment.prey)
        food_eaten_before_interactions = {
            id(agent): agent.food_eaten for agent in interaction_prey
        }
        plants_before_interactions = list(environment.plants)
        for context in step_context.values():
            agent = context["agent"]
            context["x_after_move"] = float(agent.x)
            context["y_after_move"] = float(agent.y)
            context["energy_after_move"] = float(agent.energy)
            context["movement_dx"] = agent.x - context["x_before"]
            context["movement_dy"] = agent.y - context["y_before"]
            context["distance_traveled_step"] = math.hypot(
                context["movement_dx"],
                context["movement_dy"],
            )
            plant = context["target_plant"]
            if plant is None:
                context["target_distance_after_move"] = None
                context["distance_change_status"] = "not_applicable"
                context["distance_change_na_reason"] = "no_plant"
                context["target_distance_change"] = None
                context["angular_error_deg"] = None
                context["angular_error_na_reason"] = "no_plant"
                context["ideal_unit_vector"] = None
            else:
                distance_after = math.dist(
                    (agent.x, agent.y),
                    (plant.x, plant.y),
                )
                context["target_distance_after_move"] = distance_after
                context["target_distance_change"] = (
                    distance_after - context["target_distance_before"]
                )
                context["distance_change_status"] = (
                    classify_distance_change(
                        context["target_distance_before"],
                        distance_after,
                    )
                )
                context["distance_change_na_reason"] = None
                context["ideal_unit_vector"] = _unit_vector(
                    context["ideal_vector"]
                )
                if context["ideal_unit_vector"] is None:
                    context["angular_error_deg"] = None
                    context["angular_error_na_reason"] = (
                        "zero_target_vector"
                    )
                else:
                    angular_error = calculate_angular_error(
                        context["ideal_vector"],
                        context["effective_direction"],
                    )
                    context["angular_error_deg"] = angular_error
                    context["angular_error_na_reason"] = (
                        None if angular_error is not None
                        else "zero_effective_direction"
                    )

        environment.resolve_interactions(
            capture_reward=simulation_config["energy"]["capture_reward"],
        )
        consumed_by_agent, consumer_by_plant = match_consumed_plants(
            interaction_prey,
            plants_before_interactions,
            environment.plants,
            food_eaten_before_interactions,
            agent_indices,
            plant_registry,
        )
        for plant_id, consumer_index in consumer_by_plant.items():
            consumer_reason = None
            for agent_id, consumed_id in consumed_by_agent.items():
                if consumed_id == plant_id:
                    consumer_reason = (
                        "consumed_by_self"
                        if agent_indices[agent_id] == consumer_index
                        else "consumed_by_other"
                    )
                    break
            removed_targets[plant_id] = consumer_reason

        environment.update_food()
        plant_registry.ensure(environment.plants)
        consumed_by_agent_id = {
            agent_id: plant_id
            for agent_id, plant_id in consumed_by_agent.items()
        }
        for agent_id, context in step_context.items():
            agent = context["agent"]
            consumed_plant_id = consumed_by_agent_id.get(agent_id)
            target_id = context["target_plant_id"]
            ideal_vector = context["ideal_vector"]
            ideal_unit_vector = context["ideal_unit_vector"]
            network_outputs = context["network_outputs"]
            perception = context["perception"]
            effective_direction = context["effective_direction"]
            angular_error_reason = context["angular_error_na_reason"]

            row = {
                "episode_id": episode_id,
                "model_id": model_id,
                "model_sha256": model["sha256"],
                "seed": seed,
                "agent_index": context["agent_index"],
                "step": step,
                "x_before": context["x_before"],
                "y_before": context["y_before"],
                "energy_before": context["energy_before"],
                "target_plant_id": target_id,
                "target_plant_x": context["target_plant_x"],
                "target_plant_y": context["target_plant_y"],
                "target_distance_before": context[
                    "target_distance_before"
                ],
                "target_distance_after_move": context[
                    "target_distance_after_move"
                ],
                "target_distance_change": context[
                    "target_distance_change"
                ],
                "distance_change_status": context[
                    "distance_change_status"
                ],
                "distance_change_na_reason": context[
                    "distance_change_na_reason"
                ],
                "ideal_vector_dx": (
                    ideal_vector[0] if ideal_vector is not None else None
                ),
                "ideal_vector_dy": (
                    ideal_vector[1] if ideal_vector is not None else None
                ),
                "ideal_unit_dx": (
                    ideal_unit_vector[0]
                    if ideal_unit_vector is not None
                    else None
                ),
                "ideal_unit_dy": (
                    ideal_unit_vector[1]
                    if ideal_unit_vector is not None
                    else None
                ),
                "network_output_0": float(network_outputs[0]),
                "network_output_1": float(network_outputs[1]),
                "effective_direction_dx": effective_direction[0],
                "effective_direction_dy": effective_direction[1],
                "angular_error_deg": context["angular_error_deg"],
                "angular_error_na_reason": angular_error_reason,
                "wall_escape_active_before": context[
                    "wall_escape_active_before"
                ],
                "wall_escape_active_after": context[
                    "wall_escape_active_after"
                ],
                "wall_avoidance_applied": context[
                    "wall_avoidance_applied"
                ],
                "x_after_move": context["x_after_move"],
                "y_after_move": context["y_after_move"],
                "energy_after_move": context["energy_after_move"],
                "movement_dx": context["movement_dx"],
                "movement_dy": context["movement_dy"],
                "distance_traveled_step": context[
                    "distance_traveled_step"
                ],
                "previous_target_plant_id": context[
                    "previous_target_plant_id"
                ],
                "target_changed": context["target_changed"],
                "target_switch": context["target_switch"],
                "target_change_cause": context["target_change_cause"],
                "plant_consumed": consumed_plant_id is not None,
                "consumed_plant_id": consumed_plant_id,
                "consumed_current_target": (
                    consumed_plant_id == target_id
                    if consumed_plant_id is not None and target_id is not None
                    else False
                ),
                "consumption_by_agent_index": (
                    agent_indices[agent_id]
                    if consumed_plant_id is not None
                    else None
                ),
                "target_plant_present_after_interaction": (
                    any(
                        plant is context["target_plant"]
                        for plant in environment.plants
                    )
                    if context["target_plant"] is not None
                    else None
                ),
                "energy_after_interaction": float(agent.energy),
                "food_eaten_total": agent.food_eaten,
                "plants_count_before_step": context[
                    "plants_count_before_step"
                ],
            }
            row.update(
                {
                    f"perception_{name}": float(perception[index])
                    for index, name in enumerate(PERCEPTION_COLUMNS)
                }
            )
            trace_rows.append(row)

        steps_executed += 1

    for network in network_clones:
        if not np.array_equal(network.get_genome(), model["genome"]):
            raise RuntimeError("Una red clonada cambió durante el diagnóstico.")

    summary = _summarize_episode(
        trace_rows,
        episode_id,
        model_id,
        model,
        seed,
        max_steps,
        steps_executed,
        len(environment.prey),
    )
    return trace_rows, summary, environment


def _percentage(numerator, denominator):
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _summarize_episode(
    rows,
    episode_id,
    model_id,
    model,
    seed,
    max_steps,
    steps_executed,
    prey_alive,
):
    angular_errors = [
        float(row["angular_error_deg"])
        for row in rows
        if row["angular_error_deg"] is not None
    ]
    approach_count = sum(
        row["distance_change_status"] == "approaching" for row in rows
    )
    receding_count = sum(
        row["distance_change_status"] == "receding" for row in rows
    )
    unchanged_count = sum(
        row["distance_change_status"] == "unchanged" for row in rows
    )
    distance_change_steps = (
        approach_count + receding_count + unchanged_count
    )
    target_switches = [
        row for row in rows if row["target_switch"]
    ]
    target_changes = [
        row for row in rows if row["target_changed"]
    ]
    consumption_rows = [
        row for row in rows if row["plant_consumed"]
    ]
    agents = sorted({int(row["agent_index"]) for row in rows})
    food_per_agent = {
        agent_index: max(
            (
                int(row["food_eaten_total"])
                for row in rows
                if int(row["agent_index"]) == agent_index
            ),
            default=0,
        )
        for agent_index in agents
    }
    consumption_steps_by_agent = defaultdict(list)
    for row in consumption_rows:
        consumption_steps_by_agent[int(row["agent_index"])].append(
            int(row["step"])
        )
    consumption_intervals = [
        current - previous
        for steps in consumption_steps_by_agent.values()
        for previous, current in itertools.pairwise(steps)
    ]
    total_distance = sum(float(row["distance_traveled_step"]) for row in rows)
    plants_consumed = len(consumption_rows)
    wall_avoidance_steps = sum(
        bool(row["wall_avoidance_applied"]) for row in rows
    )

    def average(values):
        return statistics.fmean(values) if values else None

    def median(values):
        return statistics.median(values) if values else None

    summary = {
        "episode_id": episode_id,
        "model_id": model_id,
        "model_sha256": model["sha256"],
        "seed": seed,
        "prey_agents_initial": PREY_COUNT,
        "predator_agents_initial": 0,
        "steps_requested": max_steps,
        "steps_executed": steps_executed,
        "prey_agents_alive": prey_alive,
        "mean_angular_error_deg": average(angular_errors),
        "median_angular_error_deg": median(angular_errors),
        "angular_error_steps": len(angular_errors),
        "angular_error_le_30_percent": _percentage(
            sum(error <= 30.0 for error in angular_errors),
            len(angular_errors),
        ),
        "angular_error_ge_90_percent": _percentage(
            sum(error >= 90.0 for error in angular_errors),
            len(angular_errors),
        ),
        "distance_change_steps": distance_change_steps,
        "approaching_steps": approach_count,
        "approaching_percent": _percentage(
            approach_count,
            distance_change_steps,
        ),
        "receding_steps": receding_count,
        "receding_percent": _percentage(
            receding_count,
            distance_change_steps,
        ),
        "unchanged_distance_steps": unchanged_count,
        "target_changes": len(target_changes),
        "target_switches": len(target_switches),
        "target_acquisitions": sum(
            row["target_change_cause"] in {
                "initial_acquisition",
                "acquired",
            }
            for row in target_changes
        ),
        "target_losses": sum(
            row["target_change_cause"] in {
                "consumed_by_self",
                "consumed_by_other",
                "previous_plant_missing",
                "target_lost",
            }
            for row in target_changes
        ),
        "target_changes_consumed_by_self": sum(
            row["target_change_cause"] == "consumed_by_self"
            for row in target_changes
        ),
        "target_changes_consumed_by_other": sum(
            row["target_change_cause"] == "consumed_by_other"
            for row in target_changes
        ),
        "target_switches_nearest_changed": sum(
            row["target_switch"]
            and row["target_change_cause"] == "nearest_changed"
            for row in rows
        ),
        "target_changes_previous_plant_missing": sum(
            row["target_change_cause"] == "previous_plant_missing"
            for row in target_changes
        ),
        "total_distance_traveled": total_distance,
        "plants_consumed_total": plants_consumed,
        "mean_plants_consumed_per_prey": average(
            list(food_per_agent.values())
        ),
        "median_plants_consumed_per_prey": median(
            list(food_per_agent.values())
        ),
        "min_plants_consumed_per_prey": (
            min(food_per_agent.values()) if food_per_agent else None
        ),
        "max_plants_consumed_per_prey": (
            max(food_per_agent.values()) if food_per_agent else None
        ),
        "distance_per_plant_consumed": (
            total_distance / plants_consumed if plants_consumed else None
        ),
        "distance_per_plant_na_reason": (
            None if plants_consumed else "no_plant_consumed"
        ),
        "consumption_intervals": len(consumption_intervals),
        "mean_steps_between_consumptions": average(consumption_intervals),
        "median_steps_between_consumptions": median(consumption_intervals),
        "consumption_interval_na_reason": (
            None if consumption_intervals else "fewer_than_two_consumptions"
        ),
        "wall_avoidance_steps": wall_avoidance_steps,
        "wall_avoidance_frequency_percent": _percentage(
            wall_avoidance_steps,
            len(rows),
        ),
        "diagnostic_steps": len(rows),
    }
    return summary


def _validate_output_dir(output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"El destino no es un directorio: {output_dir}")
    conflicts = [
        output_dir / filename
        for filename in OUTPUT_FILENAMES
        if (output_dir / filename).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "No se sobrescribirán diagnósticos existentes:\n"
            + "\n".join(str(path) for path in conflicts)
        )
    return output_dir


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_food_trajectory_probe(
    output_dir,
    seeds=DEFAULT_SEEDS,
    max_steps=MAX_STEPS,
    model_paths=None,
):
    """Evaluate frozen prey models and write step-level diagnostics."""

    seeds = _validate_seeds(seeds)
    max_steps = _validate_max_steps(max_steps)
    output_dir = _validate_output_dir(output_dir)
    simulation_config, neural_config, _ = load_configs()
    simulation_config["simulation"]["max_steps"] = max_steps
    model_paths = default_model_paths() if model_paths is None else model_paths
    loaded_models = _load_models(model_paths, neural_config)
    prey_models = {
        model_id: loaded_models[model_id]["prey"] for model_id in MODEL_IDS
    }
    no_predator_condition = next(
        condition
        for condition in CONDITIONS
        if condition.prey_count == PREY_COUNT
        and condition.predator_count == 0
    )
    width = simulation_config["simulation"]["width"]
    height = simulation_config["simulation"]["height"]
    initial_food = simulation_config["environment"]["initial_food"]
    layouts = {
        seed: _create_initial_layout(
            seed,
            width,
            height,
            initial_food,
        )
        for seed in seeds
    }

    trace_rows = []
    summary_rows = []
    for seed in seeds:
        for model_id in MODEL_IDS:
            rows, summary, _ = _trace_episode(
                condition=no_predator_condition,
                seed=seed,
                model_id=model_id,
                model=prey_models[model_id],
                simulation_config=simulation_config,
                max_steps=max_steps,
                layout=layouts[seed],
            )
            trace_rows.extend(rows)
            summary_rows.append(summary)

    for model_id, model in prey_models.items():
        if not np.array_equal(
            model["network"].get_genome(),
            model["genome"],
        ):
            raise RuntimeError(f"El genoma de presa {model_id} fue modificado.")
        if sha256_file(model["path"]) != model["sha256"]:
            raise RuntimeError(f"El archivo de modelo {model['path']} cambió.")

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "food_trajectory.csv"
    summary_path = output_dir / "episode_summary.csv"
    manifest_path = output_dir / "manifest.json"
    _write_csv(trace_path, TRACE_COLUMNS, trace_rows)
    _write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)

    manifest = {
        "tool": "food_trajectory_probe",
        "seeds": list(seeds),
        "training_seeds": [42, 123, 456],
        "max_steps": max_steps,
        "effective_config": {
            "simulation": {
                "width": width,
                "height": height,
                "max_steps": max_steps,
            },
            "environment": {
                "initial_food": initial_food,
                "max_food": initial_food,
                "food_respawn": simulation_config["environment"].get(
                    "food_respawn",
                    0,
                ),
                "food_energy": simulation_config["energy"]["food_reward"],
            },
            "energy": simulation_config["energy"],
            "neural_network": neural_config,
        },
        "condition": {
            "condition_id": no_predator_condition.condition_id,
            "prey_agents": PREY_COUNT,
            "predator_agents": 0,
        },
        "models": {
            model_id: {
                "species": "prey",
                "path": str(model["path"]),
                "sha256": model["sha256"],
                "parameter_count": int(model["genome"].size),
            }
            for model_id, model in prey_models.items()
        },
        "comparability": {
            "initial_layout_shared_between_models": True,
            "shared_layout_fields": [
                "initial plants",
                "first 10 prey positions",
            ],
            "layout_generation": (
                "Seeded explicit layout independent of Environment RNG."
            ),
            "later_plant_respawns_can_diverge": True,
            "policy_population": "10 clones of one frozen prey genome",
            "instrumentation": (
                "Temporarily wraps the real neural controller, perception, "
                "network forward pass, and wall-avoidance function; calls "
                "each original operation exactly once and restores wrappers "
                "in finally blocks."
            ),
            "additional_random_calls": 0,
            "update_order": [
                "Environment.update_agents",
                "Environment.resolve_interactions",
                "Environment.update_food",
            ],
        },
        "measurement_definitions": {
            "target_selection": (
                "Environment.nearest_plant at the prey's pre-movement "
                "position."
            ),
            "target_distance_after": (
                "Distance after movement to the exact same plant object "
                "selected before movement."
            ),
            "angular_error": (
                "Absolute angle between the ideal vector to that plant and "
                "the effective direction returned by the real controller."
            ),
            "movement_percentages": (
                "Denominator includes only steps with a target and an "
                "applicable before/after distance."
            ),
            "consumption_intervals": (
                "Step gaps between consecutive consumption events for the "
                "same prey; pooled within the episode."
            ),
            "not_applicable": (
                "CSV numeric cells are empty and reason fields identify no "
                "plant, zero vectors, no consumption, or fewer than two "
                "consumption events."
            ),
        },
        "initial_layouts": {
            str(seed): {
                "sha256": layout["sha256"],
                "plants": layout["plants"],
                "prey_positions": layout["prey_positions"][:PREY_COUNT],
            }
            for seed, layout in layouts.items()
        },
        "episode_count": len(summary_rows),
        "trajectory_rows": len(trace_rows),
        "summary_rows": len(summary_rows),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)

    return {
        "trajectory": trace_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "episode_count": len(summary_rows),
        "trajectory_rows": len(trace_rows),
        "summary_rows": len(summary_rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Registra el movimiento paso a paso de políticas de presa "
            "congeladas."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio explícito para los CSV y el manifiesto.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Semillas no negativas; por defecto 1001, 1002 y 1003.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Pasos por episodio (por defecto: 500).",
    )
    args = parser.parse_args()

    outputs = run_food_trajectory_probe(
        output_dir=args.output_dir,
        seeds=args.seeds,
        max_steps=args.max_steps,
    )
    print(
        f"Episodios: {outputs['episode_count']}; "
        f"filas temporales: {outputs['trajectory_rows']}; "
        f"resúmenes: {outputs['summary_rows']}"
    )
    for name in ("trajectory", "summary", "manifest"):
        print(f"{name}: {outputs[name]}")


if __name__ == "__main__":
    main()
