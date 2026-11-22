
import argparse
from pathlib import Path

import yaml

from neuroecosistema.evolution.coevolution_trainer import (
    CoevolutionTrainer,
)
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
        choices=["prey", "predator", "both"],
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

    parser.add_argument(
        "--record-diagnostics",
        action="store_true",
        help="Exportar diagnósticos de fitness de presas en coevolución.",
    )
    parser.add_argument(
        "--fitness-mode",
        choices=["baseline", "food_progress"],
        default="baseline",
        help="Señal de fitness para las presas.",
    )
    parser.add_argument(
        "--food-progress-weight",
        type=float,
        default=0.05,
        help="Peso de progreso alimentario en modo food_progress.",
    )
    parser.add_argument(
        "--pretrained-prey",
        default=None,
        help="Ruta a best_prey.npz para inicializar parte de la población.",
    )
    parser.add_argument(
        "--pretrained-predator",
        default=None,
        help=(
            "Ruta a best_predator.npz para inicializar parte "
            "de la población."
        ),
    )
    parser.add_argument(
        "--pretrained-fraction",
        type=float,
        default=0.5,
        help="Fracción máxima inicializada desde el modelo (menor que 1).",
    )
    parser.add_argument(
        "--pretrained-noise",
        type=float,
        default=0.02,
        help="Desviación del ruido de las variantes preentrenadas.",
    )
    parser.add_argument(
        "--hall-of-fame-size",
        type=int,
        default=3,
        help="Rivales históricos retenidos por especie en coevolución.",
    )
    parser.add_argument(
        "--historical-opponent-weight",
        type=float,
        default=0.2,
        help="Peso del episodio contra rivales históricos (0 a <1).",
    )

    args = parser.parse_args()

    if (
        args.fitness_mode == "food_progress"
        and args.species == "predator"
    ):
        parser.error(
            "--fitness-mode food_progress solo se admite para presas."
        )

    if args.record_diagnostics and args.species != "both":
        parser.error(
            "--record-diagnostics solo se puede usar con --species both"
        )
    if args.species == "prey" and args.pretrained_predator is not None:
        parser.error(
            "--pretrained-predator requiere --species both o predator."
        )
    if args.species == "predator" and args.pretrained_prey is not None:
        parser.error(
            "--pretrained-prey requiere --species both o prey."
        )
    if args.species != "both" and (
        args.hall_of_fame_size != 3
        or args.historical_opponent_weight != 0.2
    ):
        parser.error(
            "Las opciones del archivo de rivales solo aplican "
            "con --species both."
        )

    simulation, neural, evolution = load_configs()

    if args.generations is not None:
        evolution["generations"] = args.generations

    if args.population_size is not None:
        evolution["population_size"] = args.population_size

    if args.max_steps is not None:
        simulation["simulation"]["max_steps"] = args.max_steps

    
    if args.species == "both":
        trainer = CoevolutionTrainer(
            simulation_config=simulation,
            neural_config=neural,
            evolution_config=evolution,
            fitness_mode=args.fitness_mode,
            food_progress_weight=args.food_progress_weight,
            pretrained_prey_path=args.pretrained_prey,
            pretrained_predator_path=args.pretrained_predator,
            pretrained_fraction=args.pretrained_fraction,
            pretrained_noise=args.pretrained_noise,
            hall_of_fame_size=args.hall_of_fame_size,
            historical_opponent_weight=(
                args.historical_opponent_weight
            ),
        )

        if args.record_diagnostics:
            trainer.validate_output_directory(args.output)
            trainer.train(record_diagnostics=True)
        else:
            trainer.train()

        paths = trainer.save_results(args.output)

        print("\nCoevolución finalizada.")
        if args.record_diagnostics:
            print(
                "Diagnósticos de fitness: "
                f"{Path(args.output) / 'fitness_diagnostics_prey.csv'}"
            )

        for species, species_paths in paths.items():
            species_trainer = (
                trainer.prey_trainer
                if species == "prey"
                else trainer.predator_trainer
            )

            print(f"\nEspecie: {species}")
            print(
                f"Mejor fitness: "
                f"{species_trainer.best_fitness:.4f}"
            )
            print(
                f"Mejor generación: "
                f"{species_trainer.best_generation}"
            )

            for name, path in species_paths.items():
                print(f"{name}: {path}")

    else:
        trainer = EvolutionTrainer(
            simulation_config=simulation,
            neural_config=neural,
            evolution_config=evolution,
            species=args.species,
            fitness_mode=args.fitness_mode,
            food_progress_weight=args.food_progress_weight,
            pretrained_model_path=(
                args.pretrained_prey
                if args.species == "prey"
                else args.pretrained_predator
            ),
            pretrained_fraction=args.pretrained_fraction,
            pretrained_noise=args.pretrained_noise,
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