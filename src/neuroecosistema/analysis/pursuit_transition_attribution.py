
"""Analiza cambios reales de entradas entre decisiones consecutivas."""

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


def angle(first, second):
    magnitude_first = np.linalg.norm(first)
    magnitude_second = np.linalg.norm(second)

    if magnitude_first <= 1e-8 or magnitude_second <= 1e-8:
        return None

    cosine = np.dot(first, second) / (
        magnitude_first * magnitude_second
    )

    return math.degrees(
        math.acos(float(np.clip(cosine, -1.0, 1.0)))
    )


def load_records(path):
    with Path(path).open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    result = {"5G": [], "30G": []}

    for row in rows:
        model = row["model"]

        if model not in result:
            continue

        if row["agent_id"] != "predator_1":
            continue

        if int(row["step"]) < 150:
            continue

        result[model].append({
            "step": int(row["step"]),
            "inputs": np.asarray(
                json.loads(row["neural_inputs"]),
                dtype=np.float32,
            ),
        })

    for model in result:
        result[model].sort(key=lambda row: row["step"])

    return result


def analyze_transition(network, previous, current, model):
    before = previous["inputs"]
    after = current["inputs"]

    original_output = np.asarray(
        network.forward(before),
        dtype=np.float64,
    )

    final_output = np.asarray(
        network.forward(after),
        dtype=np.float64,
    )

    total_angle = angle(original_output, final_output)

    total_raw_change = float(
        np.linalg.norm(final_output - original_output)
    )

    results = []

    for index, name in enumerate(INPUT_NAMES):
        modified = before.copy()
        modified[index] = after[index]

        output = np.asarray(
            network.forward(modified),
            dtype=np.float64,
        )

        results.append({
            "model": model,
            "previous_step": previous["step"],
            "current_step": current["step"],
            "input_name": name,
            "input_change": float(after[index] - before[index]),
            "total_angle": total_angle,
            "total_raw_change": total_raw_change,
            "individual_angle": angle(
                original_output,
                output,
            ),
            "individual_raw_change": float(
                np.linalg.norm(output - original_output)
            ),
            "is_reversal": (
                total_angle is not None
                and total_angle >= 150
            ),
        })

    return results


def main():
    _, neural_config, _ = load_configs()

    records = load_records(
        "results/pursuit_neural_inputs/neural_decisions.csv"
    )

    results = []

    for model, model_path in MODEL_PATHS.items():
        network = load_trained_network(
            model_path,
            neural_config,
        )

        rows = records[model]

        for previous, current in zip(rows, rows[1:]):
            if current["step"] != previous["step"] + 1:
                continue

            results.extend(
                analyze_transition(
                    network,
                    previous,
                    current,
                    model,
                )
            )

    output_path = Path(
        "results/pursuit_transition_attribution/attribution.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    for model in ("5G", "30G"):
        model_rows = [
            row for row in results
            if row["model"] == model
            and row["is_reversal"]
        ]

        print(f"\n=== MODELO {model} ===")
        print(
            "Inversiones analizadas:",
            len(model_rows) // len(INPUT_NAMES)
        )

        for name in INPUT_NAMES:
            selected = [
                row for row in model_rows
                if row["input_name"] == name
            ]

            mean_angle = np.mean([
                row["individual_angle"]
                for row in selected
                if row["individual_angle"] is not None
            ]) if selected else 0.0

            mean_raw_change = np.mean([
                row["individual_raw_change"]
                for row in selected
            ]) if selected else 0.0

            print(
                f"{name:<20} "
                f"angulo={mean_angle:8.3f} "
                f"cambio_raw={mean_raw_change:.8f}"
            )

    print(f"\nResultados guardados en {output_path}")


if __name__ == "__main__":
    main()
