
import argparse
from pathlib import Path

from neuroecosistema.analysis.fitness_analysis import (
    load_fitness_history,
    plot_fitness_history,
    plot_species_comparison,
    summarize_history,
)


def main():
    parser = argparse.ArgumentParser(
        description="Análisis de entrenamiento de NeuroEcosistema"
    )

    parser.add_argument(
        "--results",
        default="results",
    )

    parser.add_argument(
        "--output",
        default="results/plots",
    )

    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)

    histories = {}

    for species in ("prey", "predator"):
        csv_path = (
            results_dir / f"history_{species}.csv"
        )

        history = load_fitness_history(csv_path)
        histories[species] = history

        plot_path = plot_fitness_history(
            history,
            species,
            output_dir / f"fitness_{species}.png",
        )

        summary = summarize_history(history)

        print(f"\nEspecie: {species}")
        print(
            f"Generaciones: {summary['generations']}"
        )
        print(
            f"Mejor fitness: {summary['best_fitness']:.4f}"
        )
        print(
            f"Mejor generación: {summary['best_generation']}"
        )
        print(
            f"Cambio fitness promedio: "
            f"{summary['mean_change']:+.4f}"
        )
        print(f"Gráfica: {plot_path}")

    comparison_path = plot_species_comparison(
        histories["prey"],
        histories["predator"],
        output_dir / "fitness_comparison.png",
    )

    print(
        f"\nComparación guardada: {comparison_path}"
    )


if __name__ == "__main__":
    main()