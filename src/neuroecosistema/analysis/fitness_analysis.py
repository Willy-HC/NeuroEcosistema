
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REQUIRED_COLUMNS = {
    "generation",
    "best_fitness",
    "mean_fitness",
    "worst_fitness",
    "std_fitness",
    "best_so_far",
}


def load_fitness_history(file_path):
    """Lee y valida el historial de entrenamiento."""

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el historial: {file_path}"
        )

    with file_path.open(
        newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        if not REQUIRED_COLUMNS.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "El historial no contiene las columnas requeridas."
            )

        history = []

        for row in reader:
            record = {
                "generation": int(row["generation"]),
            }

            for key in REQUIRED_COLUMNS - {"generation"}:
                record[key] = float(row[key])

            if record["generation"] < 1:
                raise ValueError(
                    "Las generaciones deben ser positivas."
                )

            if not np.all(
                np.isfinite([
                    record[key]
                    for key in REQUIRED_COLUMNS - {"generation"}
                ])
            ):
                raise ValueError(
                    "El historial contiene valores no finitos."
                )

            history.append(record)

    if not history:
        raise ValueError("El historial está vacío.")

    generations = [row["generation"] for row in history]

    if generations != list(
        range(1, len(history) + 1)
    ):
        raise ValueError(
            "Las generaciones deben ser consecutivas."
        )

    return history


def plot_fitness_history(
    history,
    species,
    output_path,
):
    """Grafica el fitness de una especie."""

    if not history:
        raise ValueError("El historial está vacío.")

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    generations = [
        row["generation"] for row in history
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = [
        ("best_fitness", "Mejor fitness"),
        ("mean_fitness", "Fitness promedio"),
        ("worst_fitness", "Peor fitness"),
        ("best_so_far", "Mejor histórico"),
    ]

    for key, label in metrics:
        ax.plot(
            generations,
            [row[key] for row in history],
            label=label,
            linewidth=2,
        )

    ax.set_title(
        f"Evolución del fitness - {species}"
    )
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def plot_species_comparison(
    prey_history,
    predator_history,
    output_path,
):
    """Compara la evolución histórica de ambas especies."""

    if not prey_history or not predator_history:
        raise ValueError(
            "Se requieren historiales de ambas especies."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for history, label in [
        (prey_history, "Presas"),
        (predator_history, "Depredadores"),
    ]:
        ax.plot(
            [row["generation"] for row in history],
            [row["best_so_far"] for row in history],
            label=label,
            linewidth=2,
        )

    ax.set_title(
        "Mejor fitness histórico por especie"
    )
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.text(
        0.5,
        0.01,
        "Las especies utilizan funciones de fitness distintas.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def summarize_history(history):
    """Resume los resultados del entrenamiento."""

    if not history:
        raise ValueError("El historial está vacío.")

    best_record = max(
        history,
        key=lambda row: row["best_fitness"],
    )

    return {
        "generations": len(history),
        "best_fitness": best_record["best_fitness"],
        "best_generation": best_record["generation"],
        "initial_mean": history[0]["mean_fitness"],
        "final_mean": history[-1]["mean_fitness"],
        "mean_change": (
            history[-1]["mean_fitness"]
            - history[0]["mean_fitness"]
        ),
    }