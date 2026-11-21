
import argparse
import csv
from pathlib import Path

import numpy as np

from neuroecosistema.evolution.behavior_evaluator import (
    BehaviorEvaluator,
)
from neuroecosistema.train import load_configs

METRICS = (
    "mean_survival",
    "mean_food_eaten",
    "mean_captures",
    "mean_exploration",
    "mean_wall_ratio",
    "mean_distance",
    "mean_food_approach_progress",
    "mean_food_net_progress",
    "mean_food_target_switches",
)

def run_evaluations(
    simulation_config,
    neural_config,
    seeds=(42, 123, 456),
    model_dir=None,
    strategy=None,
    prey_strategy=None,
    predator_strategy=None,
):
    """Ejecuta varios episodios reproducibles."""

    if not seeds:
        raise ValueError(
            "Debes proporcionar al menos una semilla."
        )

    evaluator = BehaviorEvaluator(
        simulation_config=simulation_config,
        neural_config=neural_config,
    )

    results = []

    for seed in seeds:
        episode = evaluator.evaluate(
            seed=seed,
            model_dir=model_dir,
            strategy=strategy,
            prey_strategy=prey_strategy,
            predator_strategy=predator_strategy,
        )

        for species in ("prey", "predator"):
            results.append(episode[species])

        print(f"Evaluación completada: semilla {seed}")

    return results


def save_results(rows, output_dir):
    """Guarda resultados por episodio y resumen por especie."""

    if not rows:
        raise ValueError("No hay resultados para guardar.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = output_dir / "behavior_episodes.csv"
    summary_path = output_dir / "behavior_summary.csv"

    with episodes_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = []

    for species in ("prey", "predator"):
        species_rows = [
            row for row in rows
            if row["species"] == species
        ]

        if not species_rows:
            continue

        record = {
            "species": species,
            "episodes": len(species_rows),
        }

        for metric in METRICS:
            values = [
                row[metric] for row in species_rows
            ]
            record[metric] = float(np.mean(values))
            record[f"std_{metric}"] = float(np.std(values))

        summary.append(record)

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    return {
        "episodes": episodes_path,
        "summary": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación conductual de NeuroEcosistema"
    )

    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directorio de modelos; omitir para movimiento aleatorio",
    )

    parser.add_argument(
        "--strategy",
        choices=("random", "heuristic", "neural"),
        default=None,
        help="Estrategia de movimiento de los agentes.",
    )

    parser.add_argument(
        "--prey-strategy",
        choices=("random", "heuristic", "neural"),
        default=None,
        help="Estrategia específica para las presas.",
    )

    parser.add_argument(
        "--predator-strategy",
        choices=("random", "heuristic", "neural"),
        default=None,
        help="Estrategia específica para los depredadores.",
    )

    parser.add_argument(
        "--output",
        default="results/behavior",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    simulation, neural, _ = load_configs()

    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max-steps debe ser positivo")

        simulation["simulation"]["max_steps"] = args.max_steps

    rows = run_evaluations(
        simulation_config=simulation,
        neural_config=neural,
        seeds=args.seeds,
        model_dir=args.model_dir,
        strategy=args.strategy,
        prey_strategy=args.prey_strategy,
        predator_strategy=args.predator_strategy,
    )

    paths = save_results(rows, args.output)

    print("\nEvaluación conductual finalizada.")

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()