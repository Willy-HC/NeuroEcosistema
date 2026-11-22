"""Métricas observadas durante una ejecución de simulación."""

import csv
from pathlib import Path


class SimulationHistory:
    """Conserva las observaciones de una ejecución en memoria."""

    FIELDS = (
        "step",
        "prey",
        "predators",
        "plants",
        "captures",
        "food_consumed",
        "average_energy",
    )

    def __init__(self) -> None:
        self.rows: list[dict[str, int | float]] = []

    def record(self, environment) -> None:
        agents = (*environment.prey, *environment.predators)
        average_energy = (
            sum(agent.energy for agent in agents) / len(agents)
            if agents
            else 0.0
        )
        self.rows.append({
            "step": environment.step_count,
            "prey": len(environment.prey),
            "predators": len(environment.predators),
            "plants": len(environment.plants),
            "captures": environment.total_captures,
            "food_consumed": environment.total_food_consumed,
            "average_energy": average_energy,
        })

    def clear(self) -> None:
        self.rows.clear()

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        return path


def load_training_histories(directory: str | Path) -> list[dict]:
    """Carga solo métricas de entrenamiento que estén registradas en CSV."""
    directory = Path(directory)
    histories = []

    for species, label in (
        ("prey", "Presas"),
        ("predator", "Depredadores"),
    ):
        path = directory / f"history_{species}.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows and "generation" in rows[0]:
            histories.append({
                "label": f"{label} (generación)",
                "unit": "Generación",
                "x_field": "generation",
                "metrics": ("best_fitness", "mean_fitness"),
                "rows": _numeric_rows(
                    rows,
                    ("generation", "best_fitness", "mean_fitness"),
                ),
            })

    path = directory / "imitation_history.csv"
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for species, label in (
            ("prey", "Presas"),
            ("predator", "Depredadores"),
        ):
            species_rows = [
                row for row in rows
                if row.get("species") == species
            ]
            if species_rows and "epoch" in species_rows[0]:
                histories.append({
                    "label": f"{label} (época)",
                    "unit": "Época",
                    "x_field": "epoch",
                    "metrics": ("train_loss", "validation_loss"),
                    "rows": _numeric_rows(
                        species_rows,
                        ("epoch", "train_loss", "validation_loss"),
                    ),
                })

    return [history for history in histories if history["rows"]]


def _numeric_rows(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    numeric_rows = []
    for row in rows:
        values = {}
        for field in fields:
            try:
                values[field] = float(row[field])
            except (KeyError, TypeError, ValueError):
                break
        else:
            numeric_rows.append(values)
    return numeric_rows
