
"""Analiza la sensibilidad local de los depredadores entrenados."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.train import load_configs


INPUT_NAMES = (
    "prey_1_dx",
    "prey_1_dy",
    "prey_1_distance",
    "prey_2_dx",
    "prey_2_dy",
    "prey_2_distance",
    "energy",
    "wall_distance",
)

MODEL_PATHS = {
    "5G": "models/coevolution_seed/best_predator.npz",
    "30G": "results/dagger_coevolution/best_predator.npz",
}

PERTURBATIONS = (0.0001, 0.001, 0.005)
ANGLE_THRESHOLD = 150.0


def calculate_angle(first, second):
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)

    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return None

    cosine = np.dot(first, second) / (
        first_norm * second_norm
    )

    return math.degrees(
        math.acos(float(np.clip(cosine, -1.0, 1.0)))
    )


def valid_bounds(index):
    if index in (0, 1, 3, 4):
        return -1.0, 1.0

    return 0.0, 1.0


def load_observations(csv_path):
    with Path(csv_path).open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    observations = {"5G": [], "30G": []}

    for row in rows:
        model = row["model"]

        if model not in observations:
            continue

        if row["agent_id"] != "predator_1":
            continue

        if int(row["step"]) < 150:
            continue

        inputs = np.asarray(
            json.loads(row["neural_inputs"]),
            dtype=np.float32,
        )

        if inputs.shape != (8,):
            raise ValueError(
                f"Entradas invalidas en {model}, "
                f"paso {row['step']}"
            )

        observations[model].append(
            (int(row["step"]), inputs)
        )

    return observations


def analyze_model(model_id, network, observations):
    results = []

    for step, inputs in observations:
        original = np.asarray(
            network.forward(inputs),
            dtype=np.float64,
        )

        for index, input_name in enumerate(INPUT_NAMES):
            lower, upper = valid_bounds(index)

            for delta in PERTURBATIONS:
                for sign in (-1, 1):
                    modified = inputs.copy()

                    proposed = float(inputs[index]) + sign * delta
                    new_value = max(
                        lower,
                        min(upper, proposed),
                    )

                    actual_delta = new_value - float(inputs[index])

                    if abs(actual_delta) < 1e-12:
                        continue

                    modified[index] = new_value

                    perturbed = np.asarray(
                        network.forward(modified),
                        dtype=np.float64,
                    )

                    angle = calculate_angle(
                        original,
                        perturbed,
                    )

                    output_change = float(
                        np.linalg.norm(perturbed - original)
                    )

                    results.append({
                        "model": model_id,
                        "step": step,
                        "input_index": index,
                        "input_name": input_name,
                        "requested_delta": sign * delta,
                        "actual_delta": actual_delta,
                        "original_magnitude": float(
                            np.linalg.norm(original)
                        ),
                        "output_change": output_change,
                        "angle_change": angle,
                        "inversion": (
                            angle is not None
                            and angle >= ANGLE_THRESHOLD
                        ),
                    })

    return results


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not results:
        raise ValueError("No se generaron resultados.")

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    for model in ("5G", "30G"):
        model_rows = [
            row for row in results
            if row["model"] == model
        ]

        print(f"\n=== MODELO {model} ===")

        for input_name in INPUT_NAMES:
            selected = [
                row for row in model_rows
                if row["input_name"] == input_name
            ]

            angles = [
                row["angle_change"]
                for row in selected
                if row["angle_change"] is not None
            ]

            mean_angle = (
                sum(angles) / len(angles)
                if angles else 0.0
            )

            mean_output_change = (
                sum(row["output_change"] for row in selected)
                / len(selected)
                if selected else 0.0
            )

            inversions = sum(
                row["inversion"] for row in selected
            )

            print(
                f"{input_name:<20} "
                f"angulo={mean_angle:8.3f} "
                f"cambio_raw={mean_output_change:.8f} "
                f"inversiones={inversions}"
            )


def main():
    _, neural_config, _ = load_configs()

    observations = load_observations(
        "results/pursuit_neural_inputs/neural_decisions.csv"
    )

    all_results = []

    for model_id, path in MODEL_PATHS.items():
        network = load_trained_network(
            path,
            neural_config,
        )

        results = analyze_model(
            model_id,
            network,
            observations[model_id],
        )

        all_results.extend(results)

    save_results(
        all_results,
        "results/pursuit_input_sensitivity/sensitivity.csv",
    )

    print_summary(all_results)

    print(
        "\nResultados guardados en "
        "results/pursuit_input_sensitivity/sensitivity.csv"
    )


if __name__ == "__main__":
    main()