
import argparse
from pathlib import Path

import yaml

from neuroecosistema.evolution.trainer import EvolutionTrainer


def load_configs():
    root = Path(__file__).resolve().parents[2]

    with (root / "config/simulation.yaml").open(
        encoding="utf-8"
    ) as file:
        simulation_config = yaml.safe_load(file)

    with (root / "config/neural_network.yaml").open(
        encoding="utf-8"
    ) as file:
        neural_config = yaml.safe_load(file)["neural_network"]

    with (root / "config/evolution.yaml").open(
        encoding="utf-8"
    ) as file:
        evolution_config = yaml.safe_load(file)["evolution"]

    return simulation_config, neural_config, evolution_config


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento evolutivo de NeuroEcosistema"
    )

    parser.add_argument(
        "--species",
        choices=["prey", "predator"],
        default="prey",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default="results",
    )

    args = parser.parse_args()

    simulation, neural, evolution = load_configs()

    if args.generations is not None:
        evolution["generations"] = args.generations

    if args.population_size is not None:
        evolution["population_size"] = args.population_size

    if args.max_steps is not None:
        simulation["simulation"]["max_steps"] = args.max_steps

    trainer = EvolutionTrainer(
        simulation_config=simulation,
        neural_config=neural,
        evolution_config=evolution,
        species=args.species,
    )

    trainer.train()
    paths = trainer.save_results(args.output)

    print("\nEntrenamiento finalizado.")
    print(f"Mejor fitness: {trainer.best_fitness:.4f}")
    print(f"Mejor generación: {trainer.best_generation}")

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()