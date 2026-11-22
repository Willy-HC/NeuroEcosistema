
"""Herramientas para diagnosticar oscilaciones durante persecuciones."""

import math

import csv
from collections import deque
from pathlib import Path

from contextlib import contextmanager

import neuroecosistema.agents.neural_controller as controller_module
import neuroecosistema.simulation.environment as environment_module

from neuroecosistema.analysis.population_condition_probe import (
    PopulationCondition,
    _attach_networks,
    _create_environment,
    _create_initial_layout,
    _load_models,
)
from neuroecosistema.train import load_configs

def calculate_turn_angle(previous_direction, current_direction):
    """Calcula el giro absoluto entre dos vectores en grados."""
    px, py = previous_direction
    cx, cy = current_direction

    previous_magnitude = math.hypot(px, py)
    current_magnitude = math.hypot(cx, cy)

    if previous_magnitude <= 1e-8 or current_magnitude <= 1e-8:
        return None

    cosine = (
        px * cx + py * cy
    ) / (previous_magnitude * current_magnitude)

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def calculate_path_efficiency(positions):
    """Calcula desplazamiento neto dividido por distancia recorrida."""
    if len(positions) < 2:
        return None

    total_distance = sum(
        math.dist(start, end)
        for start, end in zip(positions, positions[1:])
    )

    if total_distance <= 1e-8:
        return None

    net_distance = math.dist(positions[0], positions[-1])

    return net_distance / total_distance


def detect_oscillation(
    positions,
    angle_threshold=150.0,
    min_reversals=3,
    max_efficiency=0.25,
):
    """Detecta posibles oscilaciones en una ventana de posiciones."""
    if len(positions) < 3:
        return False

    directions = [
        (end[0] - start[0], end[1] - start[1])
        for start, end in zip(positions, positions[1:])
    ]

    reversals = 0

    for previous, current in zip(directions, directions[1:]):
        angle = calculate_turn_angle(previous, current)

        if angle is not None and angle >= angle_threshold:
            reversals += 1

    efficiency = calculate_path_efficiency(positions)

    return (
        efficiency is not None
        and reversals >= min_reversals
        and efficiency <= max_efficiency
    )

def _nearest_distance(agent, opponents):
    """Distancia al rival vivo más cercano."""
    if not opponents:
        return None

    return min(
        math.dist((agent.x, agent.y), (other.x, other.y))
        for other in opponents
    )


def run_pursuit_probe(
    output_dir,
    model_paths,
    seed=42,
    max_steps=200,
    prey_count=10,
    predator_count=2,
    food_count=30,
    window_size=10,
):
    """Ejecuta una persecución y registra las trayectorias."""
    if max_steps < 1 or window_size < 3:
        raise ValueError("Parámetros de pasos o ventana inválidos.")
    if prey_count < 1 or predator_count < 1:
        raise ValueError("Se requieren presas y depredadores.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation_config, neural_config, _ = load_configs()
    simulation_config["simulation"]["max_steps"] = max_steps
    simulation_config["environment"]["initial_food"] = food_count
    simulation_config["environment"]["max_food"] = food_count

    models = _load_models(model_paths, neural_config)
    condition = PopulationCondition(
        "pursuit",
        prey_count,
        predator_count,
    )

    simulation = simulation_config["simulation"]
    layout = _create_initial_layout(
        seed,
        simulation["width"],
        simulation["height"],
        food_count,
    )

    rows = []
    neural_records = []

    for model_id in ("5G", "30G"):
        environment = _create_environment(
            simulation_config,
            simulation_config["energy"],
            condition,
            seed,
            layout,
        )

        environment.prey_networks, _ = _attach_networks(
            environment.prey,
            models[model_id]["prey"],
        )
        environment.predator_networks, _ = _attach_networks(
            environment.predators,
            models[model_id]["predator"],
        )

        agents = [
            ("prey", index, agent)
            for index, agent in enumerate(environment.prey)
        ] + [
            ("predator", index, agent)
            for index, agent in enumerate(environment.predators)
        ]

        histories = {
            id(agent): deque(maxlen=window_size)
            for _, _, agent in agents
        }
        previous_directions = {}

        for _, _, agent in agents:
            histories[id(agent)].append((agent.x, agent.y))

        for step in range(max_steps):
            active_prey = list(environment.prey)
            active_predators = list(environment.predators)

            before = {}
            for species, index, agent in agents:
                if agent not in (
                    active_prey if species == "prey"
                    else active_predators
                ):
                    continue

                opponents = (
                    active_predators if species == "prey"
                    else active_prey
                )
                before[id(agent)] = {
                    "position": (agent.x, agent.y),
                    "nearest_distance": _nearest_distance(
                        agent, opponents
                    ),
                }

            with record_neural_decisions(
                neural_records,
                step,
                model_id,
            ):
                environment.update_agents(
                    movement_cost=simulation_config["energy"][
                        "movement_cost"
                    ],
                    strategy="neural",
                )

            step_rows = []
            for species, index, agent in agents:
                if id(agent) not in before:
                    continue

                agent_id = id(agent)
                start = before[agent_id]["position"]
                end = (agent.x, agent.y)

                movement = (
                    end[0] - start[0],
                    end[1] - start[1],
                )
                previous = previous_directions.get(agent_id)
                turn_angle = (
                    calculate_turn_angle(previous, movement)
                    if previous is not None
                    else None
                )

                if math.hypot(*movement) > 1e-8:
                    previous_directions[agent_id] = movement

                histories[agent_id].append(end)

                opponents = (
                    active_predators if species == "prey"
                    else active_prey
                )

                step_rows.append({
                    "model": model_id,
                    "seed": seed,
                    "step": step,
                    "species": species,
                    "agent_id": f"{species}_{index}",
                    "x_before": start[0],
                    "y_before": start[1],
                    "x_after": end[0],
                    "y_after": end[1],
                    "distance_before": before[agent_id][
                        "nearest_distance"
                    ],
                    "distance_after": _nearest_distance(
                        agent, opponents
                    ),
                    "turn_angle": turn_angle,
                    "wall_escape_active": getattr(
                        agent, "wall_escape_active", False
                    ),
                    "oscillation": detect_oscillation(
                        list(histories[agent_id])
                    ),
                    "captured": False,
                })

            environment.resolve_interactions(
                capture_reward=simulation_config["energy"][
                    "capture_reward"
                ]
            )

            surviving_prey = {
                id(agent) for agent in environment.prey
            }

            for row in step_rows:
                if row["species"] == "prey":
                    matching_agent = next(
                        agent
                        for species, index, agent in agents
                        if species == "prey"
                        and f"prey_{index}" == row["agent_id"]
                    )
                    row["captured"] = (
                        id(matching_agent) not in surviving_prey
                    )

            rows.extend(step_rows)
            environment.update_food()

            if not environment.prey:
                break

    output_path = output_dir / "agent_trajectories.csv"

    if rows:
        with output_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(rows[0])
            )
            writer.writeheader()
            writer.writerows(rows)

    neural_output_path = output_dir / "neural_decisions.csv"

    if neural_records:
        with neural_output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(neural_records[0]),
            )
            writer.writeheader()
            writer.writerows(neural_records)

    return output_path


@contextmanager
def record_neural_decisions(records, step, model_id):
    """Observa decisiones neuronales sin cambiar el controlador original.

    Reemplaza temporalmente la referencia que Environment utiliza
    para calcular direcciones. Restaura siempre la función original.
    """
    original_controller = environment_module.calculate_neural_direction
    original_wall_avoidance = controller_module.apply_wall_avoidance

    def observed_controller(agent, network, environment):
        observation = {
            "model": model_id,
            "step": step,
            "agent_object_id": id(agent),
            "species": (
                "prey"
                if any(candidate is agent for candidate in environment.prey)
                else "predator"
            ),
            "raw_dx": None,
            "raw_dy": None,
            "normalized_dx": None,
            "normalized_dy": None,
            "final_dx": None,
            "final_dy": None,
            "wall_escape_before": bool(
                getattr(agent, "wall_escape_active", False)
            ),
            "wall_escape_after": None,
            "wall_direction_changed": None,
        }

        original_forward = network.forward

        def observed_forward(inputs):
            outputs = original_forward(inputs)

            observation["raw_dx"] = float(outputs[0])
            observation["raw_dy"] = float(outputs[1])

            return outputs

        def observed_wall_avoidance(
            observed_agent,
            dx,
            dy,
            width,
            height,
            margin=40.0,
            strength=1.5,
        ):
            observation["normalized_dx"] = float(dx)
            observation["normalized_dy"] = float(dy)

            result = original_wall_avoidance(
                observed_agent,
                dx,
                dy,
                width,
                height,
                margin=margin,
                strength=strength,
            )

            observation["wall_escape_after"] = bool(
                getattr(observed_agent, "wall_escape_active", False)
            )
            observation["wall_direction_changed"] = (
                not math.isclose(result[0], dx, abs_tol=1e-12)
                or not math.isclose(result[1], dy, abs_tol=1e-12)
            )

            return result

        # Solo durante esta decisión neuronal.
        network.forward = observed_forward
        controller_module.apply_wall_avoidance = observed_wall_avoidance

        try:
            result = original_controller(agent, network, environment)

            observation["final_dx"] = float(result[0])
            observation["final_dy"] = float(result[1])

            records.append(observation)
            return result
        finally:
            network.forward = original_forward
            controller_module.apply_wall_avoidance = original_wall_avoidance

    environment_module.calculate_neural_direction = observed_controller

    try:
        yield
    finally:
        environment_module.calculate_neural_direction = original_controller
        controller_module.apply_wall_avoidance = original_wall_avoidance