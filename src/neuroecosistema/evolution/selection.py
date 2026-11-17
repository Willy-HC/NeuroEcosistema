
import numpy as np


def tournament_selection(
    fitness_values,
    tournament_size=3,
    rng=None,
):
    """Selecciona el índice del individuo ganador de un torneo."""

    fitness_values = np.asarray(
        fitness_values,
        dtype=np.float64,
    )

    if fitness_values.ndim != 1 or fitness_values.size == 0:
        raise ValueError("La población no puede estar vacía.")

    if not np.all(np.isfinite(fitness_values)):
        raise ValueError("Los fitness deben ser valores finitos.")

    if not 1 <= tournament_size <= fitness_values.size:
        raise ValueError("Tamaño de torneo inválido.")

    if rng is None:
        rng = np.random.default_rng()

    candidates = rng.choice(
        fitness_values.size,
        size=tournament_size,
        replace=False,
    )

    winner = candidates[
        np.argmax(fitness_values[candidates])
    ]

    return int(winner)


def select_elites(genomes, fitness_values, elitism=5):
    """Devuelve copias de los mejores genomas."""

    if len(genomes) != len(fitness_values):
        raise ValueError(
            "La cantidad de genomas y fitness debe coincidir."
        )

    if not 0 <= elitism <= len(genomes):
        raise ValueError("Cantidad de élites inválida.")

    fitness_values = np.asarray(
        fitness_values,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(fitness_values)):
        raise ValueError("Los fitness deben ser finitos.")

    best_indices = np.argsort(fitness_values)[::-1][:elitism]

    return [
        np.asarray(genomes[index], dtype=np.float64).copy()
        for index in best_indices
    ]