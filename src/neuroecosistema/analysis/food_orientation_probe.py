"""Deterministic probe of prey-network orientation toward a plant."""

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuroecosistema.agents.neural_controller import (
    calculate_neural_direction,
    normalize_direction,
)
from neuroecosistema.agents.perception import perceive_prey
from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.simulation.environment import (
    Environment,
    Plant,
)
from neuroecosistema.train import load_configs

WORLD_WIDTH = 1000
WORLD_HEIGHT = 700
PREY_POSITION = (500.0, 350.0)
PLANT_DISTANCE = 100.0
INITIAL_ENERGY = 100.0
NULL_VECTOR_EPSILON = 1e-8
# |logit| >= 3 means |tanh(logit)| >= 0.995, treated as saturated.
TANH_SATURATION_LOGIT_THRESHOLD = 3.0
FORWARD_OUTPUT_RTOL = 1e-7
FORWARD_OUTPUT_ATOL = 1e-8

MODEL_IDS = ("5G", "30G")
SCENARIO_DIRECTIONS = (
    ("north", -90.0),
    ("northeast", -45.0),
    ("east", 0.0),
    ("southeast", 45.0),
    ("south", 90.0),
    ("southwest", 135.0),
    ("west", 180.0),
    ("northwest", 225.0),
)


@dataclass(frozen=True)
class ProbeScenario:
    scenario_id: str
    prey_position: tuple[float, float]
    plant_position: tuple[float, float] | None


def project_root():
    """Return the repository root from this module's location."""

    return Path(__file__).resolve().parents[3]


def default_model_paths():
    """Return the tracked 5-generation and 30-generation prey models."""

    results_dir = project_root() / "results"
    return {
        "5G": results_dir
        / "coevolution_diagnostics_5gen"
        / "best_prey.npz",
        "30G": results_dir
        / "coevolution_diagnostics_30gen"
        / "best_prey.npz",
    }


def build_scenarios():
    """Build eight equidistant food directions and one no-food case."""

    center_x, center_y = PREY_POSITION
    scenarios = []

    for scenario_id, degrees in SCENARIO_DIRECTIONS:
        radians = math.radians(degrees)
        plant_position = (
            center_x + PLANT_DISTANCE * math.cos(radians),
            center_y + PLANT_DISTANCE * math.sin(radians),
        )
        scenarios.append(
            ProbeScenario(
                scenario_id=scenario_id,
                prey_position=PREY_POSITION,
                plant_position=plant_position,
            )
        )

    scenarios.append(
        ProbeScenario(
            scenario_id="no_plant",
            prey_position=PREY_POSITION,
            plant_position=None,
        )
    )
    return tuple(scenarios)


def observe_network_outputs(network, inputs):
    """Return final-layer logits and activation using the network's layers."""

    values = np.asarray(inputs, dtype=np.float32)
    if values.shape != (network.input_size,):
        raise ValueError(
            f"Se esperaban entradas de forma "
            f"({network.input_size},), pero se recibió {values.shape}."
        )

    final_logits = None

    for index, (weights, biases) in enumerate(
        zip(network.weights, network.biases)
    ):
        values = np.dot(values, weights.T) + biases

        if index == len(network.weights) - 1:
            final_logits = values.copy()
            values = network.output_activation(values)
        else:
            values = network.hidden_activation(values)

    if final_logits is None:
        raise ValueError("La red no contiene capas.")

    forward_outputs = network.forward(inputs)
    if not np.allclose(
        values,
        forward_outputs,
        rtol=FORWARD_OUTPUT_RTOL,
        atol=FORWARD_OUTPUT_ATOL,
    ):
        raise RuntimeError(
            "El observador no coincide con NeuralNetwork.forward()."
        )

    return final_logits, values.copy()


def angular_error_degrees(direction, ideal_direction):
    """Return the unsigned angle between two 2D vectors, or None if undefined."""

    if ideal_direction is None:
        return None

    observed = np.asarray(direction, dtype=np.float64)
    ideal = np.asarray(ideal_direction, dtype=np.float64)

    if observed.shape != (2,) or ideal.shape != (2,):
        raise ValueError("Las direcciones deben tener dos componentes.")

    if not np.all(np.isfinite(observed)) or not np.all(
        np.isfinite(ideal)
    ):
        raise ValueError("Las direcciones deben ser finitas.")

    observed_magnitude = float(np.linalg.norm(observed))
    ideal_magnitude = float(np.linalg.norm(ideal))

    if (
        observed_magnitude <= NULL_VECTOR_EPSILON
        or ideal_magnitude <= NULL_VECTOR_EPSILON
    ):
        return None

    cosine = float(
        np.dot(observed, ideal)
        / (observed_magnitude * ideal_magnitude)
    )
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def circular_angle_span_degrees(angles):
    """Return the smallest circular arc containing all supplied headings."""

    ordered = sorted(float(angle) % 360.0 for angle in angles)
    if not ordered:
        return None
    if len(ordered) == 1:
        return 0.0

    gaps = [
        next_angle - angle
        for angle, next_angle in zip(
            ordered,
            [*ordered[1:], ordered[0] + 360.0],
        )
    ]
    return 360.0 - max(gaps)


def sha256_file(path):
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_environment(scenario):
    environment = Environment(
        width=WORLD_WIDTH,
        height=WORLD_HEIGHT,
        initial_food=0,
        seed=42,
        food_respawn=0,
        max_food=0,
    )
    environment.generate_agents(
        prey_quantity=1,
        predator_quantity=0,
    )

    prey = environment.prey[0]
    prey.x, prey.y = scenario.prey_position
    prey.energy = INITIAL_ENERGY

    if scenario.plant_position is not None:
        environment.plants = [
            Plant(
                x=scenario.plant_position[0],
                y=scenario.plant_position[1],
            )
        ]

    return environment, prey


def _measure_scenario(
    model_id,
    model_path,
    model_hash,
    network,
    scenario,
):
    environment, prey = _create_environment(scenario)
    inputs = perceive_prey(
        prey,
        environment.plants,
        environment.predators,
        environment.width,
        environment.height,
    )
    logits, outputs = observe_network_outputs(network, inputs)

    output_magnitude = float(np.linalg.norm(outputs))
    vector_is_null = output_magnitude <= NULL_VECTOR_EPSILON
    normalized_output = normalize_direction(
        float(outputs[0]),
        float(outputs[1]),
    )

    effective_direction = calculate_neural_direction(
        prey,
        network,
        environment,
    )
    wall_avoidance_active = prey.wall_escape_active

    if scenario.plant_position is None:
        ideal_direction = None
        ideal_dx = None
        ideal_dy = None
    else:
        ideal_dx = (
            scenario.plant_position[0] - scenario.prey_position[0]
        )
        ideal_dy = (
            scenario.plant_position[1] - scenario.prey_position[1]
        )
        ideal_direction = (ideal_dx, ideal_dy)

    angular_error = (
        None
        if vector_is_null
        else angular_error_degrees(
            effective_direction,
            ideal_direction,
        )
    )
    output_heading = (
        None
        if vector_is_null
        else math.degrees(
            math.atan2(normalized_output[1], normalized_output[0])
        )
    )

    saturated_components = int(
        np.count_nonzero(
            np.abs(logits) >= TANH_SATURATION_LOGIT_THRESHOLD
        )
    )
    plant_dx, plant_dy, plant_distance = inputs[:3]
    predator_dx, predator_dy, predator_distance = inputs[3:6]

    return {
        "model_id": model_id,
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": model_hash,
        "scenario_id": scenario.scenario_id,
        "prey_x": scenario.prey_position[0],
        "prey_y": scenario.prey_position[1],
        "plant_x": (
            scenario.plant_position[0]
            if scenario.plant_position is not None
            else None
        ),
        "plant_y": (
            scenario.plant_position[1]
            if scenario.plant_position is not None
            else None
        ),
        "ideal_food_dx": ideal_dx,
        "ideal_food_dy": ideal_dy,
        "input_plant_dx": float(plant_dx),
        "input_plant_dy": float(plant_dy),
        "input_plant_distance": float(plant_distance),
        "input_predator_dx": float(predator_dx),
        "input_predator_dy": float(predator_dy),
        "input_predator_distance": float(predator_distance),
        "input_energy": float(inputs[6]),
        "input_wall_distance": float(inputs[7]),
        "linear_output_x": float(logits[0]),
        "linear_output_y": float(logits[1]),
        "tanh_output_x": float(outputs[0]),
        "tanh_output_y": float(outputs[1]),
        "normalized_output_x": normalized_output[0],
        "normalized_output_y": normalized_output[1],
        "normalized_output_heading_degrees": output_heading,
        "effective_direction_x": effective_direction[0],
        "effective_direction_y": effective_direction[1],
        "angular_error_degrees": angular_error,
        "wall_avoidance_active": wall_avoidance_active,
        "output_magnitude": output_magnitude,
        "vector_is_practically_null": vector_is_null,
        "tanh_saturated": saturated_components > 0,
        "tanh_saturated_components": saturated_components,
    }


def _summarize_model(model_id, rows):
    """Summarize food errors and counts across all nine probe cases."""

    food_rows = [
        row
        for row in rows
        if row["model_id"] == model_id
        and row["scenario_id"] != "no_plant"
    ]
    errors = [
        row["angular_error_degrees"]
        for row in food_rows
        if row["angular_error_degrees"] is not None
    ]
    model_rows = [
        row for row in rows if row["model_id"] == model_id
    ]
    food_headings = [
        row["normalized_output_heading_degrees"]
        for row in food_rows
        if row["normalized_output_heading_degrees"] is not None
    ]

    return {
        "model_id": model_id,
        "food_scenarios": len(food_rows),
        "angular_error_scenarios": len(errors),
        "mean_angular_error_degrees": (
            float(np.mean(errors)) if errors else None
        ),
        "median_angular_error_degrees": (
            float(np.median(errors)) if errors else None
        ),
        "max_angular_error_degrees": (
            float(np.max(errors)) if errors else None
        ),
        "errors_at_most_30_degrees": sum(
            error <= 30.0 for error in errors
        ),
        "errors_at_least_90_degrees": sum(
            error >= 90.0 for error in errors
        ),
        "food_output_heading_span_degrees": (
            circular_angle_span_degrees(food_headings)
        ),
        "saturated_output_components": sum(
            row["tanh_saturated_components"]
            for row in model_rows
        ),
        "practically_null_vectors": sum(
            row["vector_is_practically_null"]
            for row in model_rows
        ),
        "wall_avoidance_interventions": sum(
            row["wall_avoidance_active"]
            for row in model_rows
        ),
    }


def run_probe(model_paths, output_dir):
    """Load both models, run nine deterministic probes, and write two CSVs."""

    if set(model_paths) != set(MODEL_IDS):
        raise ValueError(
            "model_paths debe contener exactamente las claves 5G y 30G."
        )

    output_dir = Path(output_dir).resolve()
    model_paths = {
        model_id: Path(model_paths[model_id]).resolve()
        for model_id in MODEL_IDS
    }

    _, neural_config, _ = load_configs()

    expected_architecture = (8, 16, 12, 2)
    models = {}
    hashes_before = {}

    for model_id in MODEL_IDS:
        model_path = model_paths[model_id]
        hashes_before[model_id] = sha256_file(model_path)
        network = load_trained_network(model_path, neural_config)
        architecture = (
            network.input_size,
            *network.hidden_layers,
            network.output_size,
        )
        if architecture != expected_architecture:
            raise ValueError(
                f"Arquitectura inesperada para {model_id}: "
                f"{architecture}."
            )
        models[model_id] = network

    rows = []
    for model_id in MODEL_IDS:
        for scenario in build_scenarios():
            rows.append(
                _measure_scenario(
                    model_id=model_id,
                    model_path=model_paths[model_id],
                    model_hash=hashes_before[model_id],
                    network=models[model_id],
                    scenario=scenario,
                )
            )

    summaries = [
        _summarize_model(model_id, rows)
        for model_id in MODEL_IDS
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "food_orientation_diagnostics.csv"
    summary_path = output_dir / "food_orientation_summary.csv"
    output_paths = (diagnostics_path, summary_path)

    for path in output_paths:
        if path.exists() and path.resolve().parent != output_dir:
            raise ValueError(
                f"El archivo de salida apunta fuera del directorio: {path}"
            )

    with diagnostics_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    "N/A"
                    if key == "angular_error_degrees"
                    and value is None
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0]),
        )
        writer.writeheader()
        writer.writerows(summaries)

    for model_id in MODEL_IDS:
        if sha256_file(model_paths[model_id]) != hashes_before[model_id]:
            raise RuntimeError(
                f"El modelo {model_id} cambió durante el diagnóstico."
            )

    return {
        "diagnostics": diagnostics_path,
        "summary": summary_path,
        "rows": rows,
        "summaries": summaries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Sonda determinista de orientación de redes de presa "
            "hacia plantas."
        )
    )
    defaults = default_model_paths()
    parser.add_argument(
        "--model-5g",
        type=Path,
        default=defaults["5G"],
    )
    parser.add_argument(
        "--model-30g",
        type=Path,
        default=defaults["30G"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio donde se escribirán los dos CSV.",
    )
    args = parser.parse_args(argv)

    result = run_probe(
        model_paths={
            "5G": args.model_5g,
            "30G": args.model_30g,
        },
        output_dir=args.output_dir,
    )
    print(f"Diagnóstico: {result['diagnostics']}")
    print(f"Resumen: {result['summary']}")
    for summary in result["summaries"]:
        print(
            f"{summary['model_id']}: "
            f"error medio="
            f"{summary['mean_angular_error_degrees']:.2f}°, "
            f"mediano="
            f"{summary['median_angular_error_degrees']:.2f}°, "
            f"máximo="
            f"{summary['max_angular_error_degrees']:.2f}°"
        )


if __name__ == "__main__":
    main()
